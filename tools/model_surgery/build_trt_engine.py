#!/usr/bin/env python3
"""Build a TensorRT engine from an ONNX file.

Defaults to FP32 for parity checks. Add --fp16 for faster runtime.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import re
from pathlib import Path

import tensorrt as trt


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="Path to ONNX model")
    ap.add_argument("--engine", required=True, help="Output .engine path")
    ap.add_argument("--workspace-mb", type=int, default=4096, help="TensorRT workspace in MB")
    ap.add_argument(
        "--plugin-lib",
        action="append",
        default=[],
        help="Path to custom TensorRT plugin .so/.dll to dlopen before parsing/build.",
    )
    ap.add_argument("--fp16", action="store_true", help="Enable FP16 build")
    ap.add_argument(
        "--allow-tf32",
        action="store_true",
        help="Allow TF32 math in TensorRT (default: disabled for better parity).",
    )
    ap.add_argument(
        "--force-fp32-layer-name-regex",
        action="append",
        default=[],
        help=(
            "Force FP32 precision for layers whose names match regex. "
            "May be repeated."
        ),
    )
    ap.add_argument(
        "--force-fp32-layer-type",
        action="append",
        default=[],
        help=(
            "Force FP32 precision for layer types. May be repeated. "
            "Examples: GATHER, TOPK, SCATTER, SHUFFLE, MATRIX_MULTIPLY."
        ),
    )
    ap.add_argument(
        "--obey-precision-constraints",
        action="store_true",
        help="Enable OBEY/PREFER precision constraints when forcing FP32 layers.",
    )
    ap.add_argument(
        "--mark-output",
        action="append",
        default=[],
        help=(
            "Mark additional tensor(s) as network outputs for debug parity. "
            "May be repeated. Matches exact tensor names."
        ),
    )
    ap.add_argument(
        "--mark-output-regex",
        action="append",
        default=[],
        help=(
            "Mark additional tensors as network outputs if their names match regex. "
            "May be repeated."
        ),
    )
    ap.add_argument(
        "--list-layer-outputs",
        action="store_true",
        help="Print all layer output tensor names after ONNX parse (debug).",
    )
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()


def _collect_layer_output_tensors(network: trt.INetworkDefinition):
    seen = {}
    for li in range(network.num_layers):
        layer = network.get_layer(li)
        for oi in range(layer.num_outputs):
            t = layer.get_output(oi)
            if t is None:
                continue
            name = str(t.name) if t.name is not None else ""
            if not name:
                continue
            if name not in seen:
                seen[name] = t
    return seen


def _mark_additional_outputs(
    network: trt.INetworkDefinition,
    exact_names: list[str],
    regexes: list[str],
):
    tensor_map = _collect_layer_output_tensors(network)
    patterns = [re.compile(p) for p in regexes]
    to_mark = []
    missing = []

    for name in exact_names:
        t = tensor_map.get(name)
        if t is None:
            missing.append(name)
            continue
        to_mark.append(t)

    if patterns:
        for name, t in tensor_map.items():
            if any(p.search(name) for p in patterns):
                to_mark.append(t)

    unique = {}
    for t in to_mark:
        unique[str(t.name)] = t

    marked = []
    for name, t in sorted(unique.items(), key=lambda kv: kv[0]):
        if bool(getattr(t, "is_network_output", False)):
            continue
        network.mark_output(t)
        marked.append(name)

    return marked, missing, sorted(tensor_map.keys())


def _force_fp32_on_layers(
    network: trt.INetworkDefinition,
    name_regexes: list[str],
    type_names: list[str],
):
    patterns = [re.compile(p) for p in name_regexes]
    wanted_types = {t.strip().upper() for t in type_names if str(t).strip()}

    changed = []
    for li in range(network.num_layers):
        layer = network.get_layer(li)
        lname = str(layer.name) if layer.name is not None else ""
        ltype = str(layer.type).split(".")[-1].upper()

        hit_name = any(p.search(lname) for p in patterns) if patterns else False
        hit_type = ltype in wanted_types if wanted_types else False
        if not (hit_name or hit_type):
            continue

        try:
            layer.precision = trt.float32
        except Exception:
            pass

        # Only force float outputs to FP32. Do NOT touch index outputs
        # (e.g. TopK indices, Gather indices), which must remain integer types.
        for oi in range(layer.num_outputs):
            try:
                t = layer.get_output(oi)
                if t is None:
                    continue
                dtype = t.dtype
                float_like = {trt.DataType.FLOAT, trt.DataType.HALF}
                if hasattr(trt.DataType, "BF16"):
                    float_like.add(trt.DataType.BF16)
                if dtype in float_like:
                    layer.set_output_type(oi, trt.float32)
            except Exception:
                pass

        changed.append((li, lname, ltype))

    return changed


def main():
    args = parse_args()

    onnx_path = Path(args.onnx).expanduser().resolve()
    engine_path = Path(args.engine).expanduser().resolve()
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    log_level = trt.Logger.VERBOSE if args.verbose else trt.Logger.INFO
    logger = trt.Logger(log_level)
    for p in args.plugin_lib:
        lib_path = Path(p).expanduser().resolve()
        print(f"[build] loading plugin lib: {lib_path}")
        ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
    trt.init_libnvinfer_plugins(logger, "")

    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    print(f"[build] parsing ONNX: {onnx_path}")
    # TensorRT resolves external ONNX data files relative to CWD when only a
    # basename is stored in the ONNX external_data metadata.
    prev_cwd = os.getcwd()
    os.chdir(str(onnx_path.parent))
    try:
        onnx_bytes = onnx_path.read_bytes()
        ok = parser.parse(onnx_bytes)
    finally:
        os.chdir(prev_cwd)

    if not ok:
        print("[error] ONNX parse failed:")
        for i in range(parser.num_errors):
            print(" ", parser.get_error(i))
        raise SystemExit(2)

    marked, missing, all_layer_outputs = _mark_additional_outputs(
        network=network,
        exact_names=[str(x) for x in args.mark_output],
        regexes=[str(x) for x in args.mark_output_regex],
    )
    if args.list_layer_outputs:
        print(f"[debug] layer output tensors ({len(all_layer_outputs)}):")
        for name in all_layer_outputs:
            print(f"  {name}")
    if marked:
        print(f"[debug] marked additional outputs ({len(marked)}):")
        for name in marked:
            print(f"  {name}")
    if missing:
        print(f"[warn] requested --mark-output names not found ({len(missing)}):")
        for name in missing:
            print(f"  {name}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(args.workspace_mb) * 1024 * 1024)
    # Favor numerical parity with PyTorch over raw speed.
    try:
        config.clear_flag(trt.BuilderFlag.TF32)
        if args.allow_tf32:
            config.set_flag(trt.BuilderFlag.TF32)
    except Exception:
        pass

    if args.fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[build] precision: FP16")
        else:
            print("[warn] platform_has_fast_fp16=False, building FP32 instead")
    else:
        print("[build] precision: FP32")

    if args.obey_precision_constraints:
        # Prefer deterministic honoring of per-layer precision overrides.
        try:
            config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        except Exception:
            pass
        try:
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        except Exception:
            pass

    forced = _force_fp32_on_layers(
        network=network,
        name_regexes=[str(x) for x in args.force_fp32_layer_name_regex],
        type_names=[str(x) for x in args.force_fp32_layer_type],
    )
    if forced:
        print(f"[build] forced FP32 on {len(forced)} layers:")
        for li, lname, ltype in forced[:80]:
            print(f"  idx={li} type={ltype} name={lname}")
        if len(forced) > 80:
            print(f"  ... and {len(forced)-80} more")

    # Add dynamic profile only if any input has dynamic dims.
    has_dynamic = False
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        shape = tuple(int(x) for x in inp.shape)
        print(f"[build] input {i}: name={inp.name} shape={shape}")
        if any(d < 0 for d in shape):
            has_dynamic = True

    if has_dynamic:
        profile = builder.create_optimization_profile()
        for i in range(network.num_inputs):
            inp = network.get_input(i)
            s = [int(x) for x in inp.shape]
            # Fallback dynamic policy: replace negative dims with common defaults.
            # images -> [1,3,640,640], other tensors -> 1 for dynamic dims.
            if inp.name == "images":
                min_s = [1 if d < 0 else d for d in s]
                opt_s = [1 if d < 0 else d for d in s]
                max_s = [1 if d < 0 else d for d in s]
                if len(min_s) == 4:
                    min_s = [1, 3, 640, 640]
                    opt_s = [1, 3, 640, 640]
                    max_s = [1, 3, 640, 640]
            else:
                min_s = [1 if d < 0 else d for d in s]
                opt_s = [1 if d < 0 else d for d in s]
                max_s = [1 if d < 0 else d for d in s]

            profile.set_shape(inp.name, tuple(min_s), tuple(opt_s), tuple(max_s))
            print(f"[build] profile {inp.name}: min={tuple(min_s)} opt={tuple(opt_s)} max={tuple(max_s)}")

        config.add_optimization_profile(profile)

    print("[build] building TensorRT engine...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("TensorRT build failed: build_serialized_network returned None")

    engine_path.write_bytes(serialized_engine)
    print(f"[done] engine saved: {engine_path}")


if __name__ == "__main__":
    main()
