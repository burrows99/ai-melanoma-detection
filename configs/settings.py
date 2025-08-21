from pydantic import BaseSettings, Field
from typing import List, Dict
import os
import torch


class Settings(BaseSettings):
    # Core Training Parameters
    MODEL_ARCHITECTURE: str = Field('efficientnet_b0')
    NUM_CLASSES: int = Field(1)
    LEARNING_RATE: float = Field(1e-4)
    BATCH_SIZE: int = Field(32)
    NUM_EPOCHS: int = Field(20)
    IMAGE_SIZE: int = Field(256)

    # Metadata Configuration
    USE_METADATA: bool = Field(True)
    METADATA_COLS: List[str] = ['sex', 'age_approx', 'anatom_site_general_challenge']
    NUMERICAL_COLS: List[str] = ['age_approx']
    CATEGORICAL_COLS: List[str] = ['sex', 'anatom_site_general_challenge']

    # Dataset Paths
    TRAIN_DATA_DIR: str = Field(default_factory=lambda: os.path.join(os.path.dirname(__file__), 'data', 'train'))
    TRAIN_LABELS_PATH: str = Field(default_factory=lambda: os.path.join(os.path.dirname(__file__), 'data', 'train_labels.csv'))

    # Class Imbalance Handling
    CLASS_WEIGHTS: List[float] = [1.0, 4.0]
    LOSS_FUNCTION_TYPE: str = 'FocalLoss'
    FOCAL_LOSS_ALPHA: float = 0.864
    FOCAL_LOSS_GAMMA: float = 2.0
    FOCAL_LOSS_REDUCTION: str = 'mean'

    # Data Augmentation Parameters
    AUGMENTATION: Dict = {
        'rotation': 15,
        'horizontal_flip_prob': 0.5,
        'vertical_flip_prob': 0.5,
        'color_jitter': {
            'brightness': 0.2,
            'contrast': 0.2,
            'saturation': 0.1,
            'hue': 0.05,
        }
    }

    # Training Setup
    TRAIN_SPLIT: float = 0.8
    RANDOM_SEED: int = 42
    NUM_WORKERS: int = 4

    # Device Configuration
    DEVICE: str = Field(default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu')

    # TTA for Evaluation
    TTA_ENABLED_EVAL: bool = False

    class Config:
        env_file = '.env'
        case_sensitive = False
