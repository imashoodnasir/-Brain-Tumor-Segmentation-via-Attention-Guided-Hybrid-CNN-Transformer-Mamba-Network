from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt


def threshold_predictions(probs: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (probs >= threshold).float()


def dice_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    inter = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    score = (2 * inter + eps) / (union + eps)
    return float(score.mean().cpu())


def iou_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    inter = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - inter
    score = (inter + eps) / (union + eps)
    return float(score.mean().cpu())


def precision_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    tp = (pred * target).sum(dim=1)
    fp = (pred * (1 - target)).sum(dim=1)
    score = (tp + eps) / (tp + fp + eps)
    return float(score.mean().cpu())


def recall_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    tp = (pred * target).sum(dim=1)
    fn = ((1 - pred) * target).sum(dim=1)
    score = (tp + eps) / (tp + fn + eps)
    return float(score.mean().cpu())


def _surface_distances(mask_gt: np.ndarray, mask_pred: np.ndarray) -> np.ndarray:
    gt = mask_gt.astype(bool)
    pred = mask_pred.astype(bool)

    if gt.sum() == 0 and pred.sum() == 0:
        return np.array([0.0], dtype=np.float32)
    if gt.sum() == 0 or pred.sum() == 0:
        return np.array([np.inf], dtype=np.float32)

    gt_border = gt ^ binary_erosion(gt)
    pred_border = pred ^ binary_erosion(pred)

    dt_gt = distance_transform_edt(~gt_border)
    dt_pred = distance_transform_edt(~pred_border)

    dist_pred_to_gt = dt_gt[pred_border]
    dist_gt_to_pred = dt_pred[gt_border]

    distances = np.concatenate([dist_pred_to_gt, dist_gt_to_pred], axis=0)
    if distances.size == 0:
        distances = np.array([0.0], dtype=np.float32)
    return distances.astype(np.float32)


def hd95_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    vals = []
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    for b in range(pred_np.shape[0]):
        dists = _surface_distances(target_np[b, 0] > 0.5, pred_np[b, 0] > 0.5)
        vals.append(float(np.percentile(dists, 95)))
    return float(np.mean(vals))


def assd_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    vals = []
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    for b in range(pred_np.shape[0]):
        dists = _surface_distances(target_np[b, 0] > 0.5, pred_np[b, 0] > 0.5)
        vals.append(float(np.mean(dists)))
    return float(np.mean(vals))


def compute_segmentation_metrics(
    probs: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    pred = threshold_predictions(probs, threshold=threshold)
    metrics = {
        "dice": dice_score(pred, target),
        "iou": iou_score(pred, target),
        "precision": precision_score(pred, target),
        "recall": recall_score(pred, target),
        "hd95": hd95_score(pred, target),
        "assd": assd_score(pred, target),
    }
    return metrics
