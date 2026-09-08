"""
DETRPose-style transformer (decoder) vendored into this repo.

This implements:
- (K+1) tokens per query: 1 instance token + K keypoint tokens
- within-instance + across-instance self-attention (GroupPose-style)
- deformable cross-attention
- Pose-LQE using feature sampling at predicted keypoints
- pose denoising (cdn) producing positive/negative queries based on OKS noise

Source inspiration:
  /DETRPose/src/models/detrpose/transformer.py

Integration notes for this repo:
- We register DETRPoseTransformer so it can be used as DFINE.decoder in YAMLConfig.
- This module is pose-centric (no boxes). It outputs:
    out["pred_logits"]: [B, Q, num_classes]
    out["pred_keypoints"]: [B, Q, K*2] normalized (x,y) in [0,1] in resized image space.
"""

import copy
import math
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from ....core import register

from .ms_deform_attn import MSDeformAttn
from .dn_component import prepare_for_cdn, dn_post_process
from .utils import inverse_sigmoid, MLP, _get_activation_fn


def _get_clones(module, N, layer_share=False):
    if layer_share:
        return nn.ModuleList([module for _ in range(N)])
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def weighting_function(reg_max, up, reg_scale, deploy=False):
    if deploy:
        upper_bound1 = (abs(up[0]) * abs(reg_scale)).item()
        upper_bound2 = (abs(up[0]) * abs(reg_scale) * 2).item()
        step = (upper_bound1 + 1) ** (2 / (reg_max - 2))
        left_values = [-((step) ** i) + 1 for i in range(reg_max // 2 - 1, 0, -1)]
        right_values = [(step) ** i - 1 for i in range(1, reg_max // 2)]
        values = [-upper_bound2] + left_values + [torch.zeros_like(up[0][None])] + right_values + [upper_bound2]
        return torch.tensor([values], dtype=up.dtype, device=up.device)
    upper_bound1 = abs(up[0]) * abs(reg_scale)
    upper_bound2 = abs(up[0]) * abs(reg_scale) * 2
    step = (upper_bound1 + 1) ** (2 / (reg_max - 2))
    left_values = [-((step) ** i) + 1 for i in range(reg_max // 2 - 1, 0, -1)]
    right_values = [(step) ** i - 1 for i in range(1, reg_max // 2)]
    values = [-upper_bound2] + left_values + [torch.zeros_like(up[0][None])] + right_values + [upper_bound2]
    return torch.cat(values, 0)


def distance2pose(points, distance, reg_scale):
    reg_scale = abs(reg_scale)
    x1 = points[..., 0] + distance[..., 0] / reg_scale
    y1 = points[..., 1] + distance[..., 1] / reg_scale
    return torch.stack([x1, y1], -1)


class Gate(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Linear(2 * d_model, 2 * d_model)
        bias = float(-math.log((1 - 0.5) / 0.5))
        nn.init.constant_(self.gate.bias, bias)
        nn.init.constant_(self.gate.weight, 0)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x1, x2):
        gate_input = torch.cat([x1, x2], dim=-1)
        gates = torch.sigmoid(self.gate(gate_input))
        gate1, gate2 = gates.chunk(2, dim=-1)
        return self.norm(gate1 * x1 + gate2 * x2)


class Integral(nn.Module):
    def __init__(self, reg_max=32):
        super().__init__()
        self.reg_max = reg_max

    def forward(self, x, project):
        shape = x.shape
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, project.to(x.device)).reshape(-1, 4)
        return x.reshape(list(shape[:-1]) + [-1])


class LQE(nn.Module):
    def __init__(self, topk, hidden_dim, num_layers, num_body_points):
        super().__init__()
        self.k = topk
        self.reg_conf = MLP(num_body_points * (topk + 1), hidden_dim, 1, num_layers)
        nn.init.constant_(self.reg_conf.layers[-1].weight.data, 0)
        nn.init.constant_(self.reg_conf.layers[-1].bias.data, 0)
        self.num_body_points = num_body_points

    def forward(self, scores, pred_poses, feat):
        B, L = pred_poses.shape[:2]
        pred_poses = pred_poses.reshape(B, L, self.num_body_points, 2)
        sampling_values = (
            F.grid_sample(feat, 2 * pred_poses - 1, mode="bilinear", padding_mode="zeros", align_corners=False)
            .permute(0, 2, 3, 1)
        )
        prob_topk = sampling_values.topk(self.k, dim=-1)[0]
        stat = torch.cat([prob_topk, prob_topk.mean(dim=-1, keepdim=True)], dim=-1)
        quality_score = self.reg_conf(stat.reshape(B, L, -1))
        return scores + quality_score


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
    ):
        super().__init__()
        # within-instance self-attention
        self.within_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.within_dropout = nn.Dropout(dropout)
        self.within_norm = nn.LayerNorm(d_model)
        # across-instance self-attention
        self.across_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.across_dropout = nn.Dropout(dropout)
        self.across_norm = nn.LayerNorm(d_model)
        # deformable cross-attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        # gate
        self.gateway = Gate(d_model)
        # FFN
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn, batch_dim=1)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    @staticmethod
    def with_pos_embed(tensor, pos):
        if pos is not None:
            np_ = pos.shape[2]
            # Avoid in-place modification which breaks autograd when `tensor` requires_grad.
            # Create a new tensor for the addition to keep the computational graph intact.
            out = tensor.clone()
            out[:, :, -np_:] = out[:, :, -np_:] + pos
            return out
        return tensor

    def forward_FFN(self, tgt):
        tgt2 = self.linear2(self.dropout2(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm2(tgt.clamp(min=-65504, max=65504))
        return tgt

    def forward(
        self,
        tgt_pose: Optional[Tensor],
        tgt_pose_query_pos: Optional[Tensor],
        tgt_pose_reference_points: Optional[Tensor],
        attn_mask: Optional[Tensor] = None,
        memory: Optional[Tensor] = None,
        memory_spatial_shapes: Optional[Tensor] = None,
    ):
        bs, nq, num_kpt, d_model = tgt_pose.shape

        # within-instance self-attention
        q = k = self.with_pos_embed(tgt_pose, tgt_pose_query_pos).flatten(0, 1)
        tgt2 = self.within_attn(q, k, tgt_pose.flatten(0, 1))[0].reshape(bs, nq, num_kpt, d_model)
        tgt_pose = tgt_pose + self.within_dropout(tgt2)
        tgt_pose = self.within_norm(tgt_pose)

        # across-instance self-attention
        tgt_pose = tgt_pose.transpose(1, 2).flatten(0, 1)  # bs*num_kpt, nq, d
        tgt2_pose = self.across_attn(tgt_pose, tgt_pose, tgt_pose, attn_mask=attn_mask)[0].reshape(
            bs * num_kpt, nq, d_model
        )
        tgt_pose = tgt_pose + self.across_dropout(tgt2_pose)
        tgt_pose = (
            self.across_norm(tgt_pose).reshape(bs, num_kpt, nq, d_model).transpose(1, 2)
        )  # bs,nq,num_kpt,d

        # deformable cross-attention
        tgt2_pose = self.cross_attn(
            self.with_pos_embed(tgt_pose, tgt_pose_query_pos).flatten(1, 2),
            tgt_pose_reference_points,
            memory,
            memory_spatial_shapes,
        ).reshape(bs, nq, num_kpt, d_model)
        tgt_pose = self.gateway(tgt_pose, self.dropout1(tgt2_pose))
        tgt_pose = self.forward_FFN(tgt_pose)
        return tgt_pose


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False, hidden_dim=256, num_body_points=17):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers, layer_share=False) if num_layers > 0 else []
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_body_points = num_body_points
        self.return_intermediate = return_intermediate
        self.class_embed = None
        self.pose_embed = None
        self.half_pose_ref_point_head = MLP(hidden_dim, hidden_dim, hidden_dim, 2)
        self.eval_idx = num_layers - 1

        dim_t = torch.arange(hidden_dim // 2, dtype=torch.float32)
        dim_t = 10000 ** (2 * (dim_t // 2) / (hidden_dim // 2))
        self.register_buffer("dim_t", dim_t)
        self.scale = 2 * math.pi

    def sine_embedding(self, pos_tensor):
        x_embed = pos_tensor[..., 0:1] * self.scale
        y_embed = pos_tensor[..., 1:2] * self.scale
        pos_x = x_embed / self.dim_t
        pos_y = y_embed / self.dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=4).flatten(3)
        if pos_tensor.size(-1) == 2:
            return torch.cat((pos_y, pos_x), dim=3)
        raise ValueError(f"Unknown pos_tensor last dim {pos_tensor.size(-1)}")

    def forward(
        self,
        tgt,
        memory,
        refpoints_sigmoid,
        pre_pose_head,
        pose_head,
        class_head,
        lqe_head,
        feat_lqe,
        integral,
        up,
        reg_scale,
        reg_max,
        project,
        attn_mask=None,
        spatial_shapes: Optional[Tensor] = None,
    ):
        output = tgt
        refpoint_pose = refpoints_sigmoid
        output_pose_detach = pred_corners_undetach = 0

        dec_out_poses = []
        dec_out_logits = []
        dec_out_refs = []
        dec_out_pred_corners = []

        for layer_id, layer in enumerate(self.layers):
            refpoint_pose_input = refpoint_pose[:, :, None]
            refpoint_only_pose = refpoint_pose[:, :, 1:]
            pose_query_sine_embed = self.sine_embedding(refpoint_only_pose)
            pose_query_pos = self.half_pose_ref_point_head(pose_query_sine_embed)

            output = layer(
                tgt_pose=output,
                tgt_pose_query_pos=pose_query_pos,
                tgt_pose_reference_points=refpoint_pose_input,
                attn_mask=attn_mask,
                memory=memory,
                memory_spatial_shapes=spatial_shapes,
            )

            output_pose = output[:, :, 1:]
            output_instance = output[:, :, 0]

            if layer_id == 0:
                pre_poses = torch.sigmoid(pre_pose_head(output_pose) + inverse_sigmoid(refpoint_only_pose))
                pre_scores = class_head[0](output_instance)
                ref_pose_initial = pre_poses.detach()

            pred_corners = pose_head[layer_id](output_pose + output_pose_detach) + pred_corners_undetach
            refpoint_pose_without_center = distance2pose(ref_pose_initial, integral(pred_corners, project), reg_scale)

            refpoint_center_pose = torch.mean(refpoint_pose_without_center, dim=2, keepdim=True)
            refpoint_pose = torch.cat([refpoint_center_pose, refpoint_pose_without_center], dim=2)

            if self.training or layer_id == self.eval_idx:
                score = class_head[layer_id](output_instance)
                logit = lqe_head[layer_id](score, refpoint_pose_without_center, feat_lqe)
                dec_out_logits.append(logit)
                dec_out_poses.append(refpoint_pose_without_center)
                dec_out_pred_corners.append(pred_corners)
                dec_out_refs.append(ref_pose_initial)
                if not self.training:
                    break

            pred_corners_undetach = pred_corners
            if self.training:
                refpoint_pose = refpoint_pose.detach()
                output_pose_detach = output_pose.detach()

        return (
            torch.stack(dec_out_poses),
            torch.stack(dec_out_logits),
            torch.stack(dec_out_pred_corners),
            torch.stack(dec_out_refs),
            pre_poses,
            pre_scores,
        )


@register()
class DETRPoseTransformer(nn.Module):
    """
    Drop-in decoder for DFINE model: expects encoder features list (already 256-d).
    """

    __share__ = ["num_classes", "eval_spatial_size"]

    def __init__(
        self,
        num_classes=80,
        hidden_dim=256,
        num_queries=300,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.0,
        activation="relu",
        num_feature_levels=3,
        dec_n_points=4,
        nhead=8,
        aux_loss=True,
        num_body_points=17,
        feat_strides=(8, 16, 32),
        eval_spatial_size=None,
        reg_max=32,
        reg_scale=4.0,
        dn_number=20,
        dn_label_noise_ratio=0.5,
    ):
        super().__init__()
        self.num_feature_levels = int(num_feature_levels)
        self.num_decoder_layers = int(num_decoder_layers)
        self.num_queries = int(num_queries)
        self.num_classes = int(num_classes)
        self.aux_loss = bool(aux_loss)
        self.num_body_points = int(num_body_points)

        decoder_layer = DeformableTransformerDecoderLayer(
            hidden_dim, dim_feedforward, dropout, activation, self.num_feature_levels, nhead, dec_n_points
        )
        self.decoder = TransformerDecoder(
            decoder_layer,
            self.num_decoder_layers,
            return_intermediate=True,
            hidden_dim=hidden_dim,
            num_body_points=self.num_body_points,
        )

        # shared priors
        self.keypoint_embedding = nn.Embedding(self.num_body_points, hidden_dim)
        self.instance_embedding = nn.Embedding(1, hidden_dim)

        self.label_enc = nn.Embedding(self.num_classes + 1, hidden_dim)
        self.pose_enc = nn.Embedding(self.num_body_points, hidden_dim)

        # class head + pose heads
        _class_embed = nn.Linear(hidden_dim, self.num_classes)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        _class_embed.bias.data = torch.ones(self.num_classes) * bias_value

        _pre_point_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        nn.init.constant_(_pre_point_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_pre_point_embed.layers[-1].bias.data, 0)

        _point_embed = MLP(hidden_dim, hidden_dim, 2 * (reg_max + 1), 3)
        nn.init.constant_(_point_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_point_embed.layers[-1].bias.data, 0)

        _lqe_embed = LQE(4, hidden_dim, 2, self.num_body_points)

        self.class_embed = nn.ModuleList([copy.deepcopy(_class_embed) for _ in range(self.num_decoder_layers)])
        self.pose_embed = nn.ModuleList([copy.deepcopy(_point_embed) for _ in range(self.num_decoder_layers)])
        self.lqe_embed = nn.ModuleList([copy.deepcopy(_lqe_embed) for _ in range(self.num_decoder_layers)])
        self.pre_pose_embed = _pre_point_embed

        self.integral = Integral(reg_max)
        self.up = nn.Parameter(torch.tensor([1 / 2]), requires_grad=False)
        self.reg_max = int(reg_max)
        self.reg_scale = nn.Parameter(torch.tensor([reg_scale]), requires_grad=False)

        # two-stage encoder heads
        self.enc_output = nn.Linear(hidden_dim, hidden_dim)
        self.enc_output_norm = nn.LayerNorm(hidden_dim)
        self.enc_out_class_embed = copy.deepcopy(_class_embed)
        self.enc_pose_embed = MLP(hidden_dim, 2 * hidden_dim, 2 * self.num_body_points, 4)
        nn.init.constant_(self.enc_pose_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.enc_pose_embed.layers[-1].bias.data, 0)

        self.feat_strides = list(feat_strides) if feat_strides is not None else [8, 16, 32]
        self.eval_spatial_size = eval_spatial_size
        if self.eval_spatial_size:
            anchors, valid_mask = self._generate_anchors()
            self.register_buffer("anchors", anchors)
            self.register_buffer("valid_mask", valid_mask)

        self.dn_number = int(dn_number)
        self.dn_label_noise_ratio = float(dn_label_noise_ratio)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

    def _get_encoder_input(self, feats: List[torch.Tensor]):
        feat_flatten = []
        spatial_shapes = []
        split_sizes = []
        for feat in feats:
            _, _, h, w = feat.shape
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            spatial_shapes.append([h, w])
            split_sizes.append(h * w)
        feat_flatten = torch.concat(feat_flatten, 1)
        # Keep spatial shapes as Python list for export stability in DETRPose ms-deform-attn.
        return feat_flatten, spatial_shapes, split_sizes

    def _generate_anchors(self, spatial_shapes=None, device="cpu"):
        if spatial_shapes is None:
            spatial_shapes = []
            eval_h, eval_w = self.eval_spatial_size
            for s in self.feat_strides:
                spatial_shapes.append([int(eval_h / s), int(eval_w / s)])
        anchors = []
        for (H_, W_) in spatial_shapes:
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=device),
                torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=device),
                indexing="ij",
            )
            grid = torch.stack([grid_x, grid_y], -1)
            grid = (grid.unsqueeze(0).expand(1, -1, -1, -1) + 0.5) / torch.tensor(
                [W_, H_], dtype=torch.float32, device=device
            )
            anchors.append(grid.view(1, -1, 2))
        anchors = torch.cat(anchors, 1)
        valid_mask = ((anchors > 0.01) & (anchors < 0.99)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        return anchors, ~valid_mask

    def convert_to_deploy(self):
        self.project = weighting_function(self.reg_max, self.up, self.reg_scale, deploy=True)
        self.lqe_embed = nn.ModuleList([nn.Identity()] * (self.num_decoder_layers - 1) + [self.lqe_embed[-1]])

    def forward(self, feats, targets=None):
        memory, spatial_shapes, split_sizes = self._get_encoder_input(feats)

        if self.training:
            output_proposals, valid_mask = self._generate_anchors(spatial_shapes, memory.device)
            output_memory = memory.masked_fill(valid_mask, float(0))
            output_proposals = output_proposals.repeat(memory.size(0), 1, 1)
        else:
            output_proposals = self.anchors.repeat(memory.size(0), 1, 1)
            output_memory = memory.masked_fill(self.valid_mask, float(0))

        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        topk = self.num_queries
        enc_outputs_class_unselected = self.enc_out_class_embed(output_memory)
        topk_idx = torch.topk(enc_outputs_class_unselected.max(-1)[0], topk, dim=1)[1]

        topk_memory = output_memory.gather(dim=1, index=topk_idx.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]))
        topk_anchors = output_proposals.gather(dim=1, index=topk_idx.unsqueeze(-1).repeat(1, 1, 2))

        bs, nq = topk_memory.shape[:2]
        delta_unsig_keypoint = self.enc_pose_embed(topk_memory).reshape(bs, nq, self.num_body_points, 2)
        enc_outputs_pose_coord = torch.sigmoid(delta_unsig_keypoint + topk_anchors.unsqueeze(-2))
        enc_outputs_center_coord = torch.mean(enc_outputs_pose_coord, dim=2, keepdim=True)
        enc_outputs_pose_coord = torch.cat([enc_outputs_center_coord, enc_outputs_pose_coord], dim=2)
        refpoint_pose_sigmoid = enc_outputs_pose_coord.detach()

        tgt = topk_memory.detach().unsqueeze(-2)
        tgt_pose = self.keypoint_embedding.weight[None, None].expand(bs, topk, -1, -1) + tgt
        tgt_global = self.instance_embedding.weight[None, None].expand(bs, topk, -1, -1)
        tgt_pose = torch.cat([tgt_global, tgt_pose], dim=2)

        # Denoising (pose CDN)
        if self.training and targets is not None:
            input_query_label, input_query_pose, attn_mask, dn_meta = prepare_for_cdn(
                dn_args=(targets, self.dn_number, self.dn_label_noise_ratio),
                training=True,
                num_queries=self.num_queries,
                num_classes=self.num_classes,
                num_keypoints=self.num_body_points,
                hidden_dim=self.class_embed[0].in_features,
                label_enc=self.label_enc,
                pose_enc=self.pose_enc,
                device=feats[0].device,
            )
            tgt_pose = torch.cat([input_query_label, tgt_pose], dim=1)
            refpoint_pose_sigmoid = torch.cat([input_query_pose.sigmoid(), refpoint_pose_sigmoid], dim=1)
        else:
            attn_mask = None
            dn_meta = None

        # preprocess memory for deformable attention (same as DETRPose)
        value = memory.unflatten(2, (self.decoder.layers[0].cross_attn.n_heads, -1))
        value = value.permute(0, 2, 3, 1).flatten(0, 1).split(split_sizes, dim=-1)

        if not hasattr(self, "project"):
            project = weighting_function(self.reg_max, self.up, self.reg_scale)
        else:
            project = self.project

        out_poses, out_logits, out_corners, out_refs, out_pre_poses, out_pre_scores = self.decoder(
            tgt=tgt_pose,
            memory=value,
            refpoints_sigmoid=refpoint_pose_sigmoid,
            spatial_shapes=spatial_shapes,
            attn_mask=attn_mask,
            pre_pose_head=self.pre_pose_embed,
            pose_head=self.pose_embed,
            class_head=self.class_embed,
            lqe_head=self.lqe_embed,
            feat_lqe=feats[0],
            up=self.up,
            reg_max=self.reg_max,
            reg_scale=self.reg_scale,
            integral=self.integral,
            project=project,
        )

        # flatten keypoints to [B, Q, K*2] (no center)
        out_poses = out_poses.flatten(-2)  # [L,B,Q,K*2]

        if self.training and dn_meta is not None:
            out_pre_poses = out_pre_poses.flatten(-2)
            dn_out_poses, out_poses = torch.split(out_poses, [dn_meta["pad_size"], self.num_queries], dim=2)
            dn_out_logits, out_logits = torch.split(out_logits, [dn_meta["pad_size"], self.num_queries], dim=2)
            dn_out_corners, out_corners = torch.split(out_corners, [dn_meta["pad_size"], self.num_queries], dim=2)
            dn_out_refs, out_refs = torch.split(out_refs, [dn_meta["pad_size"], self.num_queries], dim=2)
            dn_out_pre_poses, out_pre_poses = torch.split(out_pre_poses, [dn_meta["pad_size"], self.num_queries], dim=1)
            dn_out_pre_scores, out_pre_scores = torch.split(out_pre_scores, [dn_meta["pad_size"], self.num_queries], dim=1)

        out = {"pred_logits": out_logits[-1], "pred_keypoints": out_poses[-1]}

        if self.training and self.aux_loss:
            out.update(
                {
                    "pred_corners": out_corners[-1],
                    "ref_points": out_refs[-1],
                    "up": self.up,
                    "reg_scale": self.reg_scale,
                    "reg_max": self.reg_max,
                }
            )
            out["aux_outputs"] = self._set_aux_loss2(
                out_logits[:-1],
                out_poses[:-1],
                out_corners[:-1],
                out_refs[:-1],
                out_corners[-1],
                out_logits[-1],
            )
            out["aux_pre_outputs"] = {"pred_logits": out_pre_scores, "pred_keypoints": out_pre_poses}
            if dn_meta is not None:
                out["dn_aux_outputs"] = self._set_aux_loss2(
                    dn_out_logits,
                    dn_out_poses,
                    dn_out_corners,
                    dn_out_refs,
                    dn_out_corners[-1],
                    dn_out_logits[-1],
                )
                out["dn_aux_pre_outputs"] = {"pred_logits": dn_out_pre_scores, "pred_keypoints": dn_out_pre_poses}
                out["dn_meta"] = dn_meta
        return out

    @torch.jit.unused
    def _set_aux_loss2(
        self,
        outputs_class,
        outputs_keypoints,
        outputs_corners,
        outputs_ref,
        teacher_corners=None,
        teacher_logits=None,
    ):
        return [
            {
                "pred_logits": a,
                "pred_keypoints": b,
                "pred_corners": c,
                "ref_points": d,
                "teacher_corners": teacher_corners,
                "teacher_logits": teacher_logits,
            }
            for a, b, c, d in zip(outputs_class, outputs_keypoints, outputs_corners, outputs_ref)
        ]

