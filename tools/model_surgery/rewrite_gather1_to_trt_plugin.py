#!/usr/bin/env python3
"""Rewrite gather_1 GatherElements node to custom TRT plugin op."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-onnx", required=True)
    ap.add_argument("--out-onnx", required=True)
    ap.add_argument(
        "--target-output",
        action="append",
        default=[],
        help="GatherElements output tensor name to rewrite (repeatable). "
        "If omitted and --rewrite-all-gather-elements is not set, defaults to gather_1.",
    )
    ap.add_argument(
        "--rewrite-all-gather-elements",
        action="store_true",
        help="Rewrite every GatherElements node in the graph to plugin op.",
    )
    ap.add_argument("--plugin-op", default="GatherElementsAxis1Plugin")
    return ap.parse_args()


def main():
    args = parse_args()
    in_path = Path(args.in_onnx).expanduser().resolve()
    out_path = Path(args.out_onnx).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load external tensor data so a rewritten model can be re-saved with a
    # fresh, self-consistent external data file next to out_path.
    model = onnx.load_model(str(in_path), load_external_data=True)
    targets = set(str(x) for x in args.target_output if str(x).strip())
    if not args.rewrite_all_gather_elements and not targets:
        targets = {"gather_1"}

    rewrote = 0
    for node in model.graph.node:
        if node.op_type != "GatherElements":
            continue
        if (not args.rewrite_all_gather_elements) and not any(
            str(x) in targets for x in node.output
        ):
            continue
        node.op_type = str(args.plugin_op)
        node.domain = ""
        # Plugin uses fixed axis=1 behavior. Remove attrs for deterministic parser mapping.
        del node.attribute[:]
        rewrote += 1

    if rewrote == 0:
        print("[warn] no matching GatherElements node was rewritten.")
    else:
        print(f"[info] rewritten nodes: {rewrote}")

    out_data = out_path.with_suffix(out_path.suffix + ".data").name
    onnx.save_model(
        model,
        str(out_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=out_data,
        size_threshold=1024,
    )
    print(f"[done] wrote ONNX: {out_path}")


if __name__ == "__main__":
    main()
