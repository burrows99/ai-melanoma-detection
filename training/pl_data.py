import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Tuple
from data.dataset import get_data_loaders


class MelanomaDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self._train_loader = None
        self._val_loader = None
        self.num_metadata_features = None

    def setup(self, stage: str | None = None):
        # Reuse existing factory to keep behavior identical
        self._train_loader, self._val_loader, self.num_metadata_features = get_data_loaders()

    def train_dataloader(self) -> DataLoader:
        return self._train_loader

    def val_dataloader(self) -> DataLoader:
        return self._val_loader
