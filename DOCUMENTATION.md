- If you change configuration via environment variables, `configs/settings.py` (Pydantic) will pick them up automatically (supports `.env`).

## Model Variants, Metrics, Trade-offs, and Final Choice

We evaluated multiple backbones with the same metadata fusion head and consistent training setup. Below are summarized metrics from `base/*/metrics`.

### Metrics Summary (Validation)

- **EfficientNet-B0** (`base/b0_1/metrics`)
  - Acc: 97.24% | Recall: 88.64% | F1: 89.69% | Loss: 0.2757
- **DenseNet-121** (`base/dens_1/metrics`)
  - Acc: 97.10% | Recall: 88.93% | Loss: 0.2401 (F1 comparable to B0)
- **ResNet-50** (`base/res50_1/metrics`)
  - Acc: 96.25% | Recall: 90.01% | F1: 86.69% | Loss: 0.2757

Notes: Train metrics are very high for all models (≈99% acc), suggesting potential overfitting mitigated by metadata fusion and regularization.

### Trade-offs

- **EfficientNet-B0**
  - Pros: Strong accuracy/F1, lightweight, fast inference, widely available pretrained weights.
  - Cons: Slightly lower recall than ResNet-50 in this run.
- **DenseNet-121**
  - Pros: Competitive validation loss; robust features for medical imaging.
  - Cons: Heavier than B0; similar recall to B0, marginal overall gains.
- **ResNet-50**
  - Pros: Highest recall in this snapshot (reduces false negatives).
  - Cons: Lower F1/accuracy; heavier than B0; may be more sensitive to augmentation choices.

### Final Model Choice and Justification

We select **EfficientNet-B0 + Metadata MLP** as the default backbone for deployment:

- Best balance of validation F1 and accuracy with efficient compute and latency.
- Stable training with `timm` pretrained weights; fewer parameters than DenseNet/ResNet.
- Easier to optimize and deploy broadly (CPU-friendly), aligning with our Gradio + Docker serving setup.

If the clinical priority is to maximize sensitivity (recall), consider a ResNet-50 variant with threshold tuning and/or class weighting; however, overall decision-quality (F1) and operational efficiency favored EfficientNet-B0.

### Next Steps

- Add stratified cross-validation and confidence intervals for robust comparison.
- Tune thresholds per model to optimize recall vs. precision trade-offs.
- Evaluate additional backbones (e.g., ConvNeXt-Tiny, EfficientNetV2-S) and self-supervised pretraining.
- Add `.env` with environment-specific overrides for reproducibility.
# AI Melanoma Detection – Project Documentation

This document explains the purpose of each script in the repository and what the end-to-end system does, from training to serving predictions with explainability.

## Overview
- **Goal**: Detect melanoma from dermoscopy images with metadata, and provide visual explanations (Grad-CAM) via a Gradio UI.
- **Model**: CNN image backbone (e.g., EfficientNet via `timm`) fused with tabular metadata features.
- **Training**: PyTorch Lightning (`training/pl_train.py`, `training/pl_module.py`, `training/pl_data.py`) with TorchMetrics for metrics.
- **Serving**: Gradio web app (`app/app.py`) deployed via Gunicorn/Uvicorn inside Docker.
- **Explainability**: Uses `pytorch-grad-cam` (EigenCAM) to visualize important regions.

## Use Cases

- **Assistive triage**: Provide probabilities and heatmaps to help route cases for clinician review (non-diagnostic).
- **Research**: Inspect spurious correlations (rulers, borders, bubbles) with Grad-CAM; compare backbones and augmentation strategies.
- **Education**: Demonstrate metadata+image fusion and explainability techniques.

## Example Images & Screenshots

Place images under `docs/images/` and reference them here or in `README.md`:

![UI Home](docs/images/ui-home.png)
![Prediction Result](docs/images/prediction.png)
![Grad-CAM Heatmap](docs/images/gradcam.png)

See `docs/README.md` for how to capture these screenshots.

## Repository Structure
- `app/app.py`: Gradio app definition and serving logic with Grad-CAM visualization.
- `models/model.py`: Image+metadata fusion architecture and `get_model` factory.
- `models/losses.py`: Criterion factory (e.g., FocalLoss) from config.
- `data/dataset.py`: Datasets/dataloaders with `timm` transforms and metadata handling.
- `training/pl_train.py`: Lightning training entrypoint (W&B, checkpointing).
- `training/pl_module.py`: LightningModule (loss, optimizer, steps, TorchMetrics).
- `training/pl_data.py`: LightningDataModule wrapping dataloaders.
- `eval/evaluate.py`: Evaluation utilities (ROC, confusion matrix, optional TTA).
- `configs/settings.py`: Pydantic BaseSettings for typed, env-driven config.
- `configs/config.py`: Backward-compatible module-level constants sourced from Settings.
- `base/`: Pretrained experiment artifacts (best models, metrics, plots).
- `result/weights/`: Model weights used by the app (e.g., `gradcam.pth`).
- `requirements.txt`, `Dockerfile`, `docker-compose.yml`, `README.md`, `DOCUMENTATION.md`.

## Current Model Configuration

- **Backbone architecture**: `efficientnet_b0`
  - Set in `configs/config.py`: `MODEL_ARCHITECTURE = 'efficientnet_b0'`
  - Created in `models/model.py` `MetadataMelanomaModel` via `timm.create_model(MODEL_ARCHITECTURE, pretrained=True, num_classes=0)`

- **Metadata branch (MLP)**: hidden dims `[128, 64]` with BatchNorm + ReLU; dropout between hidden layers
  - Defined in `model.py` `MetadataMelanomaModel`

- **Classifier head / output**: linear to `NUM_CLASSES = 1` (binary logit)
  - `configs/config.py`: `NUM_CLASSES = 1`
  - `models/model.py`: `self.final_classifier = nn.Linear(..., NUM_CLASSES)`

- **Activation usage**:
  - Training: no activation (loss on raw logits)
  - Inference: `sigmoid` to convert logit → probability
    - `training/pl_module.py`: applies `sigmoid` to logits for metrics; threshold 0.5
    - `eval/evaluate.py`: computes probabilities per image and thresholds at 0.5
    - `app/app.py`: uses `sigmoid` for probabilities in the UI

- **Loss function**: Focal Loss
  - `models/losses.py`: class `FocalLoss` and `get_criterion()`
  - `configs/config.py`: `LOSS_FUNCTION_TYPE`, `FOCAL_LOSS_ALPHA`, `FOCAL_LOSS_GAMMA`, `FOCAL_LOSS_REDUCTION`

- **Optimizer**: Adam
  - `models/model.py` `get_optimizer()`: `torch.optim.Adam(model.parameters(), lr=learning_rate)`
  - `configs/config.py`: `LEARNING_RATE = 1e-4`

- **Key hyperparameters** (from `configs/config.py`):
  - `BATCH_SIZE = 32`
  - `NUM_EPOCHS = 20`
  - `IMAGE_SIZE = 256`
  - `DEVICE = 'cuda' if available else 'cpu'`

- **Preprocessing / normalization**:
  - `timm.data.create_transform` for train/val; ImageNet mean/std normalization
  - See `app/app.py` and `data/dataset.py`

- **TTA (Test-Time Augmentation)**:
  - App (`app/app.py`): PIL-based TTA (original, flips, rotations) averaged
  - Eval (`eval/evaluate.py`): optional tensor-based TTA (original + hflip) when enabled
  - `configs/config.py`: `TTA_ENABLED_EVAL = False` by default

- **Checkpoints in use**:
  - App loads `result/weights/gradcam.pth` (`app/app.py`: `CHECKPOINT_PATH`)
  - Lightning saves best to `result/weights/{run_name}_best_ep{N}.pth` (`training/pl_train.py`)

## Scripts in Detail

### `app/app.py`
- **Purpose**: Serve the trained model via a web UI and generate Grad-CAM visualizations.
- **Key elements**:
  - Imports model via `get_model()` from `model.py` and loads weights from `result/weights/gradcam.pth`.
  - Preprocessing: resize/normalize images to `IMAGE_SIZE` and prepare metadata (`prepare_metadata()`).
  - TTA (Test-Time Augmentation): simple flips/rotations to average predictions.
  - Grad-CAM: selects target CNN layer(s) and computes EigenCAM heatmaps using `pytorch-grad-cam`.
  - Gradio interface (`iface`): image input, metadata inputs (age/sex/site), target-layer selector, outputs (probability, heatmap, side-by-side visualization).
  - Exposes `asgi_app = iface.app` so Gunicorn can serve `app:asgi_app`.

### `models/model.py`
- **Purpose**: Define the image+metadata fusion architecture used for training and inference.
- **Typical contents**:
  - Image backbone from `timm` (e.g., EfficientNet-B0) for feature extraction.
  - Metadata encoder (e.g., MLP) to encode tabular inputs.
  - Fusion head combining image and metadata embeddings to output a single melanoma probability (sigmoid).
  - `get_model(num_metadata_features: int)` factory building the full network consistently across train/eval/app.

### `data/dataset.py`
- **Purpose**: Provide dataset/dataloader utilities with `timm` transforms and metadata handling.
- **Key elements**:
  - Metadata columns (e.g., `sex`, `age_approx`, `anatom_site_general_challenge`).
  - Albumentations pipelines for training/validation.
  - `MelanomaDataset` that returns `(image_tensor, metadata_tensor, label)` for each sample.

### `training/pl_train.py`
- **Purpose**: End-to-end training with PyTorch Lightning.
- **Typical flow**:
  - Build datasets/dataloaders using `dataset.py` and configs from `config.py`.
  - Instantiate model via `get_model()`; set optimizer, scheduler, loss (BCE with logits for binary classification).
  - Train/validate across epochs; save best checkpoint to `result/weights/` and log metrics.

### `eval/evaluate.py`
- **Purpose**: Evaluate a trained checkpoint.
- **Typical flow**:
  - Load model and weights.
  - Run inference on validation/test set; compute metrics (AUC, accuracy) and optionally save ROC curves.

### `configs/config.py` and `configs/settings.py`
- **Purpose**: Typed, environment-friendly configuration (`Settings`) with backward-compatible constants.
- **Examples**:
  - Paths: data directories, label CSV, weights output, image size.
  - Hyperparameters: batch size, learning rate, epochs, augmentation flags.
  - Device selection: `DEVICE = 'cuda' if available else 'cpu'`.

- **Purpose**: Python dependency list.
- **Notes**:
  - `timm`, `albumentations`, `opencv-python-headless`, `gradio` for UI.
  - `pytorch-grad-cam` installed via GitHub due to platform wheel availability.
  - `pandas` enabled as used by `app.py`.

### `Dockerfile`
- **Purpose**: Build a CPU-only image for serving the app.
- **Highlights**:
  - Base: `python:3.10-slim` (no CUDA required).
  - Installs minimal OS libs for OpenCV and health checks (`curl`, `libgl1`, `libglib2.0-0`, plus `git` to fetch Grad-CAM).
  - Installs PyTorch CPU (`torch==2.0.1`, `torchvision==0.15.2`, `torchaudio==2.0.2`), then Python deps from `requirements.txt`.
  - Starts Gunicorn/Uvicorn with `app:asgi_app`.

### `docker-compose.yml`
- **Purpose**: Run the app locally via Docker.
- **Highlights**:
  - Service `web` exposes port `7860:7860`.
  - Mounts `./base`, `./data`, `./result` into the container.
  - Healthcheck uses `curl http://localhost:7860`.

## End-to-End Workflow
1. **Prepare data**: Place images and labels under `data/` (paths set in `config.py`).
2. **Train** (optional if you use provided weights):
   - `python -m training.pl_train` to produce a checkpoint under `result/weights/`.
3. **Evaluate** (optional):
   - `python -m eval.evaluate` workflow or import helpers to compute metrics/plots.
4. **Serve**:
   - `docker compose up --build web`
   - Open `http://localhost:7860` and upload a lesion image.
   - Optionally adjust metadata (age/sex/site) and target layer for visualization.

## Explainability (EigenCAM)
- The app computes a class-agnostic activation map over the selected CNN layer.
- Heatmap is overlaid on the input image to highlight regions most influential for feature extraction.
- Earlier layers tend to show more spatial detail; later layers (e.g., `conv_head`) show higher-level semantics.

## Notes & Tips
- First launch downloads backbone weights into the container cache; subsequent runs are faster.
- If you change dependencies, rebuild the image: `docker compose build web`.
- To update `gradio`, bump the version in `requirements.txt` and rebuild.
- CPU inference is supported; for GPU you would need a CUDA base image and NVIDIA runtime.
