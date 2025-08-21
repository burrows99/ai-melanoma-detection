import torch
import pytorch_lightning as pl
from torch import nn
from typing import Any
from configs.config import LEARNING_RATE
from utils.activations import sigmoid_prob
import torchmetrics


class MelanomaLightningModule(pl.LightningModule):
    def __init__(self, model: nn.Module, criterion: nn.Module, learning_rate: float = LEARNING_RATE):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.learning_rate = learning_rate

        # Metrics
        self.train_acc = torchmetrics.classification.BinaryAccuracy()
        self.train_rec = torchmetrics.classification.BinaryRecall()
        self.train_f1 = torchmetrics.classification.BinaryF1Score()
        self.val_acc = torchmetrics.classification.BinaryAccuracy()
        self.val_rec = torchmetrics.classification.BinaryRecall()
        self.val_f1 = torchmetrics.classification.BinaryF1Score()

    def forward(self, images: torch.Tensor, metadata: torch.Tensor):
        return self.model(images, metadata)

    def training_step(self, batch: Any, batch_idx: int):
        images, metadata, labels = batch
        logits = self(images, metadata)
        loss = self.criterion(logits, labels.unsqueeze(1).float())
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int()
        labels_i = labels.int().unsqueeze(1)

        # update metrics
        self.train_acc.update(preds, labels_i)
        self.train_rec.update(preds, labels_i)
        self.train_f1.update(preds, labels_i)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self) -> None:
        acc = self.train_acc.compute(); self.train_acc.reset()
        rec = self.train_rec.compute(); self.train_rec.reset()
        f1 = self.train_f1.compute(); self.train_f1.reset()
        self.log("train/accuracy", acc, prog_bar=True)
        self.log("train/recall", rec)
        self.log("train/f1", f1, prog_bar=True)

    def validation_step(self, batch: Any, batch_idx: int):
        images, metadata, labels = batch
        logits = self(images, metadata)
        loss = self.criterion(logits, labels.unsqueeze(1).float())
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int()
        labels_i = labels.int().unsqueeze(1)

        # update metrics
        self.val_acc.update(preds, labels_i)
        self.val_rec.update(preds, labels_i)
        self.val_f1.update(preds, labels_i)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        acc = self.val_acc.compute(); self.val_acc.reset()
        rec = self.val_rec.compute(); self.val_rec.reset()
        f1 = self.val_f1.compute(); self.val_f1.reset()
        self.log("val/accuracy", acc, prog_bar=True)
        self.log("val/recall", rec)
        # This key is used for checkpoint monitoring
        self.log("val/f1", f1, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
