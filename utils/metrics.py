from typing import List, Dict
import torch
import torchmetrics


def compute_classification_metrics(labels: List[int], preds: List[int]) -> Dict[str, float]:
    """Compute standard binary classification metrics using torchmetrics.
    Returns a dict with accuracy, recall, and f1.
    """
    # Convert to tensors with shape [N]
    labels_t = torch.tensor(labels).int()
    preds_t = torch.tensor(preds).int()

    acc = torchmetrics.classification.BinaryAccuracy()(preds_t, labels_t).item()
    rec = torchmetrics.classification.BinaryRecall()(preds_t, labels_t).item()
    f1 = torchmetrics.classification.BinaryF1Score()(preds_t, labels_t).item()
    return {
        "accuracy": acc,
        "recall": rec,
        "f1": f1,
    }
