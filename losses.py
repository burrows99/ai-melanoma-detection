import torch
import torch.nn as nn
import torch.nn.functional as F

# Config-driven loss parameters
from config import (
    LOSS_FUNCTION_TYPE,
    FOCAL_LOSS_ALPHA,
    FOCAL_LOSS_GAMMA,
    FOCAL_LOSS_REDUCTION,
)


class FocalLoss(nn.Module):
    """
    Implementation of Focal Loss from Lin et al. (2017)
    Measures loss between raw logits (input) and binary targets.
    """
    def __init__(self, alpha=0.864, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # BCE with logits expects raw logits and targets of same shape
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        # alpha factor: alpha for positive class, (1-alpha) for negative
        alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_loss = alpha_factor * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        if self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def get_criterion(loss_type: str = LOSS_FUNCTION_TYPE) -> nn.Module:
    """
    Factory for criterion to allow extension without modifying callers.
    """
    if loss_type == 'FocalLoss':
        return FocalLoss(
            alpha=FOCAL_LOSS_ALPHA,
            gamma=FOCAL_LOSS_GAMMA,
            reduction=FOCAL_LOSS_REDUCTION,
        )
    raise ValueError(f"Unsupported LOSS_FUNCTION_TYPE: {loss_type}")
