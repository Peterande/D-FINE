#!/usr/bin/env python3
"""Trace local ONNX neighborhood around seed tensors and list candidate tensors."""

from __future__ import annotations

import argparse
import re
from collections import deque
from pathlib import Path

import onnx


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="Input ONNX path")
    ap.add_argument(
        "--seed-tensor",
        action="append",
        default=[],
        help="Seed tensor name (repeatable), e.g. linear_8",
    )
    ap.add_argument(
        "--seed-regex",
        action="append",
        default=[],
        help="Regex to choose seed tensors, e.g. '^linear_(7|8|9|10|11|12)$'",
    )
    ap.add_argument(
        "--hops",
        type=int,
        default=2,
        help="Tensor-node-tensor hops from seed tensors.",
    )
    ap.add_argument(
        "--keep-regex",
        action="append",
        default=[r"^linear_", r"^bmm_", r"^view_"],
        help="Regex for output candidate tensor names (repeatable).",
    )
    ap.add_argument(
        "--out-names",
        default="",
        help="Optional path to write candidate names (one per line).",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    onnx_path = Path(args.onnx).expanduser().resolve()
    model = onnx.load_model(str(onnx_path), load_external_data=False)

    producers = {}  # tensor -> node idx
    consumers = {}  # tensor -> set(node idx)
    all_tensors = set()
    for i, node in enumerate(model.graph.node):
        for t in node.output:
            if not t:
                continue
            producers[str(t)] = i
            all_tensors.add(str(t))
        for t in node.input:
            if not t:
                continue
            consumers.setdefault(str(t), set()).add(i)
            all_tensors.add(str(t))

    seeds = set(str(x) for x in args.seed_tensor if x)
    for p in [re.compile(x) for x in args.seed_regex]:
        for t in all_tensors:
            if p.search(t):
                seeds.add(t)

    if not seeds:
        raise SystemExit("No seed tensors matched. Provide --seed-tensor or --seed-regex.")

    # BFS on tensor graph via producer/consumer nodes.
    q = deque((s, 0) for s in seeds)
    seen_tensors = set(seeds)
    while q:
        t, d = q.popleft()
        if d >= int(args.hops):
            continue
        near_nodes = set()
        p = producers.get(t)
        if p is not None:
            near_nodes.add(p)
        near_nodes.update(consumers.get(t, set()))
        for ni in near_nodes:
            node = model.graph.node[ni]
            for nt in list(node.input) + list(node.output):
                if not nt:
                    continue
                nt = str(nt)
                if nt in seen_tensors:
                    continue
                seen_tensors.add(nt)
                q.append((nt, d + 1))

    keep = [re.compile(x) for x in args.keep_regex]
    candidates = sorted([t for t in seen_tensors if any(p.search(t) for p in keep)])
    print(f"seed_tensors={len(seeds)} neighborhood_tensors={len(seen_tensors)} candidates={len(candidates)}")
    print("--- candidates ---")
    for n in candidates:
        print(n)

    if args.out_names:
        out = Path(args.out_names).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(f"{x}\n" for x in candidates), encoding="utf-8")
        print(f"[done] wrote names: {out}")


if __name__ == "__main__":
    main()

