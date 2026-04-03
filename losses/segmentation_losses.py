from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


def dice_loss(probs: torch.Tensor, targets: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    probs = probs.contiguous().view(probs.size(0), -1)
    targets = targets.contiguous().view(targets.size(0), -1)
    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def tversky_loss(
    probs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    eps: float = 1e-8,
) -> torch.Tensor:
    probs = probs.contiguous().view(probs.size(0), -1)
    targets = targets.contiguous().view(targets.size(0), -1)
    tp = (probs * targets).sum(dim=1)
    fn = (targets * (1.0 - probs)).sum(dim=1)
    fp = ((1.0 - targets) * probs).sum(dim=1)
    score = (tp + eps) / (tp + alpha * fn + beta * fp + eps)
    return 1.0 - score.mean()


def compute_signed_distance_map(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if mask.max() == 0:
        return np.zeros_like(mask, dtype=np.float32)
    pos = distance_transform_edt(mask)
    neg = distance_transform_edt(~mask)
    sdt = neg - pos
    return sdt.astype(np.float32)


def boundary_loss(probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    device = probs.device
    batch_losses = []
    targets_np = targets.detach().cpu().numpy()
    for b in range(targets_np.shape[0]):
        sdt = compute_signed_distance_map(targets_np[b, 0] > 0.5)
        sdt_t = torch.from_numpy(sdt).to(device=device, dtype=probs.dtype).unsqueeze(0).unsqueeze(0)
        batch_losses.append((probs[b : b + 1] * sdt_t.abs()).mean())
    return torch.stack(batch_losses).mean()


def attention_consistency_loss(attention_map: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(attention_map, targets)


@dataclass
class LossConfig:
    dice_weight: float = 1.0
    tversky_weight: float = 0.7
    boundary_weight: float = 0.5
    attention_weight: float = 0.3
    tversky_alpha: float = 0.3
    tversky_beta: float = 0.7


class CompositeSegmentationLoss(nn.Module):
    def __init__(self, cfg: Dict) -> None:
        super().__init__()
        self.cfg = LossConfig(**cfg)

    def forward(
        self,
        probs: torch.Tensor,
        targets: torch.Tensor,
        attention_map: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        dl = dice_loss(probs, targets)
        tl = tversky_loss(probs, targets, self.cfg.tversky_alpha, self.cfg.tversky_beta)
        bl = boundary_loss(probs, targets)
        al = attention_consistency_loss(attention_map, targets)

        total = (
            self.cfg.dice_weight * dl
            + self.cfg.tversky_weight * tl
            + self.cfg.boundary_weight * bl
            + self.cfg.attention_weight * al
        )

        stats = {
            "dice_loss": float(dl.detach().cpu()),
            "tversky_loss": float(tl.detach().cpu()),
            "boundary_loss": float(bl.detach().cpu()),
            "attention_loss": float(al.detach().cpu()),
            "total_loss": float(total.detach().cpu()),
        }
        return total, stats
