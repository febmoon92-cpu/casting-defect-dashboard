"""Loss functions for classification."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class Focal Loss with optional class weights.

    Reference: Lin et al., "Focal Loss for Dense Object Detection" (2017).
    """

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("weight", weight if weight is not None else None, persistent=False)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        target_one_hot = F.one_hot(target, num_classes=logits.size(1)).float()
        log_pt = (log_probs * target_one_hot).sum(dim=1)
        pt = (probs * target_one_hot).sum(dim=1)
        focal = -((1.0 - pt) ** self.gamma) * log_pt
        if self.weight is not None:
            w = self.weight.to(logits.device)[target]
            focal = focal * w
        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


def build_loss(name: str, class_weights: torch.Tensor | None = None, focal_gamma: float = 2.0) -> nn.Module:
    name = name.lower()
    if name in {"ce", "cross_entropy", "crossentropy"}:
        return nn.CrossEntropyLoss(weight=class_weights)
    if name == "focal":
        return FocalLoss(gamma=focal_gamma, weight=class_weights)
    raise ValueError(f"Unknown loss: {name}")
