#!/usr/bin/env python3
"""Rewrite GatherElements indices to INT32 for selected outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-onnx", required=True, help="Input ONNX path")
    ap.add_argument("--out-onnx", required=True, help="Output ONNX path")
    ap.add_argument(
        "--target-output",
        action="append",
        default=["gather_1"],
        help="GatherElements output tensor name to rewrite (repeatable).",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    in_path = Path(args.in_onnx).expanduser().resolve()
    out_path = Path(args.out_onnx).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load_model(str(in_path), load_external_data=False)
    graph = model.graph
    targets = set(str(x) for x in args.target_output)

    rewritten = 0
    new_nodes = []
    for idx, node in enumerate(graph.node):
        if node.op_type != "GatherElements":
            new_nodes.append(node)
            continue
        outs = [str(o) for o in node.output]
        if not any(o in targets for o in outs):
            new_nodes.append(node)
            continue
        if len(node.input) < 2:
            new_nodes.append(node)
            continue

        orig_idx_in = str(node.input[1])
        cast_out = f"{orig_idx_in}__int32_for_{outs[0]}"
        cast_node = helper.make_node(
            "Cast",
            inputs=[orig_idx_in],
            outputs=[cast_out],
            name=f"{node.name or f'GatherElements_{idx}'}__idx_cast_int32",
            to=int(TensorProto.INT32),
        )
        node.input[1] = cast_out
        new_nodes.append(cast_node)
        new_nodes.append(node)
        rewritten += 1

    if rewritten == 0:
        print("[warn] no GatherElements nodes were rewritten.")
    else:
        print(f"[info] rewritten GatherElements nodes: {rewritten}")

    del graph.node[:]
    graph.node.extend(new_nodes)

    out_data = out_path.with_suffix(out_path.suffix + ".data").name
    onnx.save_model(
        model,
        str(out_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=out_data,
        size_threshold=1024,
    )
    print(f"[done] wrote rewritten ONNX: {out_path}")


if __name__ == "__main__":
    main()

