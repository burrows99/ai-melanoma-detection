import torch
from typing import Union


def sigmoid_prob(logits: torch.Tensor) -> Union[float, torch.Tensor]:
    """Apply sigmoid to logits. If input is scalar-like, return float; else return tensor on CPU.
    """
    probs = torch.sigmoid(logits)
    if probs.numel() == 1:
        return probs.item()
    return probs.detach().cpu()


def binarize_probs(prob: Union[float, torch.Tensor], threshold: float = 0.5):
    """Binarize probability/probabilities with a threshold.
    Accepts a float or a tensor; returns int or tensor of ints accordingly.
    """
    if isinstance(prob, torch.Tensor):
        return (prob > threshold).int()
    return int(prob > threshold)
