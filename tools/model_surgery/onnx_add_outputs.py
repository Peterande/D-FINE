#!/usr/bin/env python3
"""Create a debug ONNX by exposing additional internal tensors as outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import onnx
from onnx import TensorProto, helper


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-onnx", required=True, help="Input ONNX path")
    ap.add_argument("--out-onnx", required=True, help="Output ONNX path")
    ap.add_argument(
        "--add-output",
        action="append",
        default=[],
        help="Exact tensor name to expose as graph output. May be repeated.",
    )
    ap.add_argument(
        "--add-output-regex",
        action="append",
        default=[],
        help="Regex for tensor names to expose as graph outputs. May be repeated.",
    )
    return ap.parse_args()


def _all_value_infos(model: onnx.ModelProto) -> Dict[str, onnx.ValueInfoProto]:
    out: Dict[str, onnx.ValueInfoProto] = {}
    for vi in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        out[str(vi.name)] = vi
    return out


def _collect_tensor_names(model: onnx.ModelProto) -> List[str]:
    names = set()
    for node in model.graph.node:
        for n in node.output:
            if n:
                names.add(str(n))
    return sorted(names)


def main():
    args = parse_args()
    in_path = Path(args.in_onnx).expanduser().resolve()
    out_path = Path(args.out_onnx).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load_model(str(in_path), load_external_data=False)
    vi_map = _all_value_infos(model)
    existing_outputs = {str(o.name) for o in model.graph.output}
    all_tensor_names = _collect_tensor_names(model)

    import re

    regexes = [re.compile(p) for p in args.add_output_regex]
    requested = set(str(n) for n in args.add_output)
    if regexes:
        for name in all_tensor_names:
            if any(r.search(name) for r in regexes):
                requested.add(name)

    added = []
    missing = []
    for name in sorted(requested):
        if name in existing_outputs:
            continue
        if name in vi_map:
            model.graph.output.extend([vi_map[name]])
            added.append(name)
            continue
        if name in all_tensor_names:
            # Fallback when value_info is absent.
            model.graph.output.extend([helper.make_tensor_value_info(name, TensorProto.FLOAT, None)])
            added.append(name)
            continue
        missing.append(name)

    # Keep external-data behavior for large models.
    out_data = out_path.with_suffix(out_path.suffix + ".data").name
    onnx.save_model(
        model,
        str(out_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=out_data,
        size_threshold=1024,
    )

    print(f"[done] wrote debug ONNX: {out_path}")
    print(f"[info] original_outputs={len(existing_outputs)} added_outputs={len(added)} total_outputs={len(model.graph.output)}")
    if added:
        print("[info] added names:")
        for n in added:
            print(f"  {n}")
    if missing:
        print(f"[warn] requested names not found ({len(missing)}):")
        for n in missing:
            print(f"  {n}")


if __name__ == "__main__":
    main()

