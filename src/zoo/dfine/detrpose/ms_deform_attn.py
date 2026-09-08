"""
Multi-Scale Deformable Attention (PyTorch implementation).

Vendored from DETRPose:
  /DETRPose/src/models/detrpose/ms_deform_attn.py
"""

import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import constant_


def _spatial_shapes_to_list(value_spatial_shapes):
    """Normalize spatial shapes to a Python list of (H, W) ints.

    Using a Python list here avoids tensor iteration/unflatten patterns that are
    fragile during ONNX/TensorRT export for this module.
    """
    if isinstance(value_spatial_shapes, torch.Tensor):
        # Value is expected static for export (fixed image size).
        return [(int(h), int(w)) for h, w in value_spatial_shapes.detach().cpu().tolist()]
    return [(int(h), int(w)) for h, w in value_spatial_shapes]


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    # value: list[L] of tensors shaped [N_*M_, D_, H_*W_] (later unflattened)
    _, D_, _ = value[0].shape
    N_, Lq_, M_, L_, P_, _ = sampling_locations.shape

    sampling_grids = 2 * sampling_locations - 1
    sampling_grids = sampling_grids.transpose(1, 2).flatten(0, 1)

    sampling_value_list = []
    spatial_shapes_list = _spatial_shapes_to_list(value_spatial_shapes)
    for lid_, (H_, W_) in enumerate(spatial_shapes_list):
        # Avoid Tensor.unflatten with symbolic shape values in export paths.
        value_l_ = value[lid_].reshape(value[lid_].shape[0], D_, H_, W_)
        sampling_grid_l_ = sampling_grids[:, :, lid_]
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)

    attention_weights = attention_weights.transpose(1, 2).reshape(N_ * M_, 1, Lq_, L_ * P_)
    output = (torch.concat(sampling_value_list, dim=-1) * attention_weights).sum(-1).view(N_, M_ * D_, Lq_)
    return output.transpose(1, 2)


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, use_4D_normalizer=False):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads, got {d_model} and {n_heads}")

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.use_4D_normalizer = use_4D_normalizer

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i % 4 + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        if self.n_points % 4 != 0:
            constant_(self.sampling_offsets.bias, 0.0)
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)

    def forward(self, query, reference_points, value, input_spatial_shapes):
        N, Len_q, _ = query.shape

        sampling_offsets = self.sampling_offsets(query).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            N, Len_q, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points
        )

        # reference_points: [N, Len_q, n_levels, ..., 2/4] -> [N, Len_q*n_levels, ..., 2/4]
        reference_points = torch.transpose(reference_points, 2, 3).flatten(1, 2)

        if reference_points.shape[-1] == 2:
            # Keep tensor type/device consistent for export/runtime.
            if isinstance(input_spatial_shapes, torch.Tensor):
                offset_normalizer = input_spatial_shapes.to(device=query.device, dtype=query.dtype)
            else:
                offset_normalizer = torch.as_tensor(input_spatial_shapes, device=query.device, dtype=query.dtype)
            offset_normalizer = offset_normalizer.flip([1]).reshape(1, 1, 1, self.n_levels, 1, 2)
            sampling_locations = reference_points[:, :, None, :, None, :] + sampling_offsets / offset_normalizer
        elif reference_points.shape[-1] == 4:
            if self.use_4D_normalizer:
                if not isinstance(input_spatial_shapes, torch.Tensor):
                    input_spatial_shapes = torch.as_tensor(
                        input_spatial_shapes, device=query.device, dtype=query.dtype
                    )
                offset_normalizer = torch.stack(
                    [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1
                )
                sampling_locations = (
                    reference_points[:, :, None, :, None, :2]
                    + sampling_offsets
                    / offset_normalizer[None, None, None, :, None, :]
                    * reference_points[:, :, None, :, None, 2:]
                    * 0.5
                )
            else:
                sampling_locations = (
                    reference_points[:, :, None, :, None, :2]
                    + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
                )
        else:
            raise ValueError(
                f"Last dim of reference_points must be 2 or 4, got {reference_points.shape[-1]}"
            )

        output = ms_deform_attn_core_pytorch(value, input_spatial_shapes, sampling_locations, attention_weights)
        return output

