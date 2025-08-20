# --- Configuration File ---

import torch
import os

# --- Core Training Parameters ---
MODEL_ARCHITECTURE = 'efficientnet_b0'
NUM_CLASSES = 1
LEARNING_RATE = 1e-4  # Original baseline
BATCH_SIZE = 32
NUM_EPOCHS = 20  # Original baseline
IMAGE_SIZE = 256

# --- Metadata Configuration ---
USE_METADATA = True  
METADATA_COLS = ['sex', 'age_approx', 'anatom_site_general_challenge']
NUMERICAL_COLS = ['age_approx']
CATEGORICAL_COLS = ['sex', 'anatom_site_general_challenge']

# --- Dataset Paths ---
# Update these paths to point to your actual dataset location
TRAIN_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "train")
TRAIN_LABELS_PATH = os.path.join(os.path.dirname(__file__), "data", "train_labels.csv")

# --- Class Imbalance Handling ---
CLASS_WEIGHTS = [1.0, 4.0]  # Kept for reference but not used with Focal Loss
LOSS_FUNCTION_TYPE = 'FocalLoss'  
FOCAL_LOSS_ALPHA = 0.864  
FOCAL_LOSS_GAMMA = 2.0    
FOCAL_LOSS_REDUCTION = 'mean'  

# --- Data Augmentation Parameters ---
# Original baseline augmentations
AUGMENTATION = {
    'rotation': 15,
    'horizontal_flip_prob': 0.5,
    'vertical_flip_prob': 0.5,
    'color_jitter': {          
        'brightness': 0.2,
        'contrast': 0.2,
        'saturation': 0.1,
        'hue': 0.05
    }
}

# --- Training Setup ---
TRAIN_SPLIT = 0.8
RANDOM_SEED = 42
NUM_WORKERS = 4

# --- Device Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Test Time Augmentation (TTA) for Evaluation ---
TTA_ENABLED_EVAL = False # Set to True to enable TTA during evaluation runs

