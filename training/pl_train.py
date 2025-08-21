import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb

from configs.config import LEARNING_RATE, TTA_ENABLED_EVAL
from models.model import get_model
from models.losses import get_criterion
from training.pl_data import MelanomaDataModule
from training.pl_module import MelanomaLightningModule


def main():
    # Data
    dm = MelanomaDataModule()
    dm.setup(None)

    # Model and criterion
    model = get_model(num_metadata_features=dm.num_metadata_features)
    criterion = get_criterion()

    lit_module = MelanomaLightningModule(model=model, criterion=criterion, learning_rate=LEARNING_RATE)

    # Logging with W&B (optional, will use same project)
    wandb_logger = WandbLogger(project="melanoma-classification", log_model=True)

    # Checkpoint on best val/f1
    checkpoint_cb = ModelCheckpoint(
        monitor="val/f1",
        mode="max",
        save_top_k=1,
        filename="best-{epoch:02d}-{val_f1:.4f}"
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        max_epochs=20,
        logger=wandb_logger,
        callbacks=[checkpoint_cb, lr_monitor],
        log_every_n_steps=10,
        deterministic=True,
    )

    trainer.fit(lit_module, datamodule=dm)

    # Finish W&B run
    wandb.finish()


if __name__ == "__main__":
    main()
