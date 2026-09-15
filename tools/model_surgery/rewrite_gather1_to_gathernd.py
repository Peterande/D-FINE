#!/usr/bin/env python3
"""Rewrite one GatherElements(axis=1) node (e.g. gather_1) into GatherND logic.

This avoids TensorRT GatherElements path for a single sensitive tensor while
keeping graph semantics equivalent for tensors shaped like:
  data:    [B, N, C]
  indices: [B, M, C]
with output:
  out[b, m, c] = data[b, indices[b,m,c], c]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def _const_i64(name: str, values: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(
        name=name,
        data_type=TensorProto.INT64,
        dims=[len(values)],
        vals=values,
    )


def _scalar_i64(name: str, value: int) -> onnx.TensorProto:
    return helper.make_tensor(
        name=name,
        data_type=TensorProto.INT64,
        dims=[],
        vals=[value],
    )


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-onnx", required=True)
    ap.add_argument("--out-onnx", required=True)
    ap.add_argument("--target-output", default="gather_1")
    return ap.parse_args()


def main():
    args = parse_args()
    in_path = Path(args.in_onnx).expanduser().resolve()
    out_path = Path(args.out_onnx).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load_model(str(in_path), load_external_data=True)
    graph = model.graph

    new_nodes: list[onnx.NodeProto] = []
    rewritten = 0

    for i, node in enumerate(graph.node):
        if (
            node.op_type == "GatherElements"
            and args.target_output in [str(x) for x in node.output]
        ):
            axis = 0
            for a in node.attribute:
                if a.name == "axis":
                    axis = int(onnx.helper.get_attribute_value(a))
            if axis != 1:
                raise RuntimeError(
                    f"Target node {node.name or '<noname>'} has axis={axis}, expected axis=1."
                )
            if len(node.input) < 2:
                raise RuntimeError("GatherElements node has fewer than 2 inputs.")

            data = str(node.input[0])
            indices = str(node.input[1])
            out = str(node.output[0])
            base = f"{node.name or f'gather_elements_{i}'}__gnd"

            # Shape(indices) -> [B, M, C]
            shape_i = f"{base}_shape_i"
            new_nodes.append(
                helper.make_node(
                    "Shape",
                    inputs=[indices],
                    outputs=[shape_i],
                    name=f"{base}_shape",
                )
            )

            c0 = f"{base}_c0"
            c2 = f"{base}_c2"
            graph.initializer.extend(
                [
                    _const_i64(c0, [0]),
                    _const_i64(c2, [2]),
                ]
            )

            b_dim = f"{base}_b_dim"
            c_dim = f"{base}_c_dim"
            new_nodes.append(
                helper.make_node(
                    "Gather",
                    inputs=[shape_i, c0],
                    outputs=[b_dim],
                    name=f"{base}_gather_b",
                    axis=0,
                )
            )
            new_nodes.append(
                helper.make_node(
                    "Gather",
                    inputs=[shape_i, c2],
                    outputs=[c_dim],
                    name=f"{base}_gather_c",
                    axis=0,
                )
            )

            # Gather outputs are shape [1]; Range in TRT expects scalar shape [].
            sq_axes0 = f"{base}_sq_axes0"
            graph.initializer.extend([_const_i64(sq_axes0, [0])])
            b_dim_s = f"{base}_b_dim_s"
            c_dim_s = f"{base}_c_dim_s"
            new_nodes.append(
                helper.make_node(
                    "Squeeze",
                    inputs=[b_dim, sq_axes0],
                    outputs=[b_dim_s],
                    name=f"{base}_squeeze_b",
                )
            )
            new_nodes.append(
                helper.make_node(
                    "Squeeze",
                    inputs=[c_dim, sq_axes0],
                    outputs=[c_dim_s],
                    name=f"{base}_squeeze_c",
                )
            )

            # Scalars for Range.
            z = f"{base}_z"
            one = f"{base}_one"
            graph.initializer.extend(
                [
                    _scalar_i64(z, 0),
                    _scalar_i64(one, 1),
                ]
            )

            rb = f"{base}_range_b"
            rc = f"{base}_range_c"
            new_nodes.append(
                helper.make_node(
                    "Range",
                    inputs=[z, b_dim_s, one],
                    outputs=[rb],
                    name=f"{base}_range_b",
                )
            )
            new_nodes.append(
                helper.make_node(
                    "Range",
                    inputs=[z, c_dim_s, one],
                    outputs=[rc],
                    name=f"{base}_range_c",
                )
            )

            # Unsqueeze to [B,1,1] and [1,1,C].
            axes_01 = f"{base}_axes01"
            axes_00 = f"{base}_axes00"
            graph.initializer.extend(
                [
                    _const_i64(axes_01, [1, 2]),
                    _const_i64(axes_00, [0, 1]),
                ]
            )
            rb_u = f"{base}_rb_u"
            rc_u = f"{base}_rc_u"
            new_nodes.append(
                helper.make_node(
                    "Unsqueeze",
                    inputs=[rb, axes_01],
                    outputs=[rb_u],
                    name=f"{base}_unsq_rb",
                )
            )
            new_nodes.append(
                helper.make_node(
                    "Unsqueeze",
                    inputs=[rc, axes_00],
                    outputs=[rc_u],
                    name=f"{base}_unsq_rc",
                )
            )

            # Expand both to [B,M,C] (shape of indices).
            rb_e = f"{base}_rb_e"
            rc_e = f"{base}_rc_e"
            new_nodes.append(
                helper.make_node(
                    "Expand",
                    inputs=[rb_u, shape_i],
                    outputs=[rb_e],
                    name=f"{base}_expand_rb",
                )
            )
            new_nodes.append(
                helper.make_node(
                    "Expand",
                    inputs=[rc_u, shape_i],
                    outputs=[rc_e],
                    name=f"{base}_expand_rc",
                )
            )

            # Ensure indices are int64 for GatherND.
            idx_i64 = f"{base}_idx_i64"
            new_nodes.append(
                helper.make_node(
                    "Cast",
                    inputs=[indices],
                    outputs=[idx_i64],
                    name=f"{base}_cast_idx_i64",
                    to=int(TensorProto.INT64),
                )
            )

            # Build [B,M,C,3] index tensor.
            axes_last = f"{base}_axes_last"
            graph.initializer.extend([_const_i64(axes_last, [3])])
            rb_e_u = f"{base}_rb_e_u"
            idx_i64_u = f"{base}_idx_i64_u"
            rc_e_u = f"{base}_rc_e_u"
            new_nodes.append(
                helper.make_node(
                    "Unsqueeze",
                    inputs=[rb_e, axes_last],
                    outputs=[rb_e_u],
                    name=f"{base}_unsq_rb_e",
                )
            )
            new_nodes.append(
                helper.make_node(
                    "Unsqueeze",
                    inputs=[idx_i64, axes_last],
                    outputs=[idx_i64_u],
                    name=f"{base}_unsq_idx",
                )
            )
            new_nodes.append(
                helper.make_node(
                    "Unsqueeze",
                    inputs=[rc_e, axes_last],
                    outputs=[rc_e_u],
                    name=f"{base}_unsq_rc_e",
                )
            )

            idx3 = f"{base}_idx3"
            new_nodes.append(
                helper.make_node(
                    "Concat",
                    inputs=[rb_e_u, idx_i64_u, rc_e_u],
                    outputs=[idx3],
                    name=f"{base}_concat_idx3",
                    axis=3,
                )
            )

            # Final replacement.
            new_nodes.append(
                helper.make_node(
                    "GatherND",
                    inputs=[data, idx3],
                    outputs=[out],
                    name=f"{base}_gathernd",
                    batch_dims=0,
                )
            )
            rewritten += 1
            continue

        new_nodes.append(node)

    if rewritten == 0:
        print("[warn] no matching GatherElements node rewritten.")
    else:
        print(f"[info] rewritten nodes: {rewritten}")

    del graph.node[:]
    graph.node.extend(new_nodes)
    # Save inline to avoid external-data pointer issues in downstream tools.
    onnx.save_model(model, str(out_path), save_as_external_data=False)
    print(f"[done] wrote ONNX: {out_path}")


if __name__ == "__main__":
    main()
