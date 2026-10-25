"""Segmentation metrics (same formulas as the notebook)."""

import numpy as np
import torch


def calculate_metrics(pred_logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5):
    """Return (accuracy, dice, iou) for a batch of logits vs binary targets."""
    pred = (torch.sigmoid(pred_logits) > threshold).float()
    target = target.float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
    iou = (intersection + 1e-6) / (union - intersection + 1e-6)
    acc = (pred == target).sum() / torch.numel(pred)
    return acc.item(), dice.item(), iou.item()


def mask_metrics(pred_mask: np.ndarray, true_mask: np.ndarray):
    """Dice / IoU / accuracy for two binary numpy masks of the same shape."""
    p = pred_mask.astype(bool)
    t = true_mask.astype(bool)
    inter = np.logical_and(p, t).sum()
    total = p.sum() + t.sum()
    dice = (2 * inter + 1e-6) / (total + 1e-6)
    iou = (inter + 1e-6) / (total - inter + 1e-6)
    acc = (p == t).mean()
    return float(acc), float(dice), float(iou)
