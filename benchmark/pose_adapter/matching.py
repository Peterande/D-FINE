from __future__ import annotations

import torch
from scipy.optimize import linear_sum_assignment


def matched_distillation_loss(student: dict, teacher: dict, score_weight: float = 0.25) -> torch.Tensor:
    """Hungarian-match unordered pose queries, then distill coordinates and logits."""
    losses = []
    for batch in range(student["pred_keypoints"].shape[0]):
        student_kpt = student["pred_keypoints"][batch]
        teacher_kpt = teacher["pred_keypoints"][batch]
        student_score = student["pred_logits"][batch].sigmoid().amax(-1)
        teacher_score = teacher["pred_logits"][batch].sigmoid().amax(-1)
        coordinate_cost = torch.cdist(student_kpt.detach(), teacher_kpt.detach(), p=1) / student_kpt.shape[-1]
        score_cost = (student_score.detach()[:, None] - teacher_score.detach()[None, :]).abs()
        rows, cols = linear_sum_assignment((coordinate_cost + score_weight * score_cost).cpu().numpy())
        rows_t = torch.as_tensor(rows, device=student_kpt.device)
        cols_t = torch.as_tensor(cols, device=student_kpt.device)
        losses.append(
            torch.nn.functional.smooth_l1_loss(student_kpt[rows_t], teacher_kpt[cols_t])
            + score_weight * torch.nn.functional.smooth_l1_loss(
                student["pred_logits"][batch, rows_t], teacher["pred_logits"][batch, cols_t]
            )
        )
    return torch.stack(losses).mean()
