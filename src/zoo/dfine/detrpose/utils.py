"""
Small utilities vendored from DETRPose.

Source: /DETRPose/src/models/detrpose/utils.py
"""

import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor


def gen_encoder_output_proposals(memory: Tensor, spatial_shapes: Tensor):
    """
    Args:
      memory: [bs, sum(hw), d_model]
      spatial_shapes: [n_levels, 2] (H,W)
    Returns:
      output_memory: [bs, sum(hw), d_model]
      output_proposals: [bs, sum(hw), 2] (unsigmoid xy anchors)
    """
    N_, S_, C_ = memory.shape
    proposals = []
    for (H_, W_) in spatial_shapes:
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, H_ - 1, int(H_), dtype=torch.float32, device=memory.device),
            torch.linspace(0, W_ - 1, int(W_), dtype=torch.float32, device=memory.device),
            indexing="ij",
        )
        grid = torch.stack([grid_x, grid_y], -1)  # H,W,2
        grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / torch.tensor(
            [W_, H_], dtype=torch.float32, device=memory.device
        )
        proposal = grid.view(N_, -1, 2)
        proposals.append(proposal)
    output_proposals = torch.cat(proposals, 1)  # [bs, sum(hw), 2] in [0,1]
    output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
    output_proposals = torch.log(output_proposals / (1 - output_proposals))  # unsigmoid
    output_proposals = output_proposals.masked_fill(~output_proposals_valid, float("inf"))

    output_memory = memory.masked_fill(~output_proposals_valid, float(0))
    return output_memory, output_proposals


class MLP(nn.Module):
    """Very simple multi-layer perceptron (FFN)."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _get_activation_fn(activation, d_model=256, batch_dim=0):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    if activation == "selu":
        return F.selu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)

