# AI Melanoma Detection – Project Documentation

This document explains the purpose of each script in the repository and what the end-to-end system does, from training to serving predictions with explainability.

## Overview
- **Goal**: Detect melanoma from dermoscopy images with metadata, and provide visual explanations (Grad-CAM) via a Gradio UI.
- **Model**: CNN image backbone (e.g., EfficientNet via `timm`) fused with tabular metadata features.
- **Serving**: Gradio web app deployed via Gunicorn/Uvicorn inside Docker.
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
- `app.py`: Gradio app definition and serving logic, wraps the trained model and Grad-CAM visualization.
- `model.py`: Model architecture and factory (`get_model`) to build the image+metadata fusion network.
- `dataset.py`: Dataset utilities and transforms for training/evaluation (Albumentations, metadata handling).
- `train.py`: Training script (loops, loss, optimizer, saving checkpoints/metrics).
- `evaluate.py`: Evaluation/inference script (metrics, ROC, etc.).
- `config.py`: Centralized configuration (paths, hyperparameters, device selection).
- `base/`: Pretrained experiment artifacts (e.g., alternate backbones, best models, metrics, plots).
- `data/`: Input data directory (images/labels) – not committed.
- `result/weights/`: Model weights used by the app (e.g., `gradcam.pth`).
- `requirements.txt`: Python dependencies.
- `Dockerfile`: Container image for CPU-only serving.
- `docker-compose.yml`: Orchestration to run the web service locally.
- `README.md`: Quick start and basic instructions.
- `DOCUMENTATION.md`: This file.

## Scripts in Detail

### `app.py`
- **Purpose**: Serve the trained model via a web UI and generate Grad-CAM visualizations.
- **Key elements**:
  - Imports model via `get_model()` from `model.py` and loads weights from `result/weights/gradcam.pth`.
  - Preprocessing: resize/normalize images to `IMAGE_SIZE` and prepare metadata (`prepare_metadata()`).
  - TTA (Test-Time Augmentation): simple flips/rotations to average predictions.
  - Grad-CAM: selects target CNN layer(s) and computes EigenCAM heatmaps using `pytorch-grad-cam`.
  - Gradio interface (`iface`): image input, metadata inputs (age/sex/site), target-layer selector, outputs (probability, heatmap, side-by-side visualization).
  - Exposes `asgi_app = iface.app` so Gunicorn can serve `app:asgi_app`.

### `model.py`
- **Purpose**: Define the image+metadata fusion architecture used for training and inference.
- **Typical contents**:
  - Image backbone from `timm` (e.g., EfficientNet-B0) for feature extraction.
  - Metadata encoder (e.g., MLP) to encode tabular inputs.
  - Fusion head combining image and metadata embeddings to output a single melanoma probability (sigmoid).
  - `get_model(num_metadata_features: int)` factory building the full network consistently across train/eval/app.

### `dataset.py`
- **Purpose**: Provide dataset/dataloader utilities with Albumentations transforms and metadata handling.
- **Key elements**:
  - Metadata columns (e.g., `sex`, `age_approx`, `anatom_site_general_challenge`).
  - Albumentations pipelines for training/validation.
  - `MelanomaDataset` that returns `(image_tensor, metadata_tensor, label)` for each sample.

### `train.py`
- **Purpose**: End-to-end training loop.
- **Typical flow**:
  - Build datasets/dataloaders using `dataset.py` and configs from `config.py`.
  - Instantiate model via `get_model()`; set optimizer, scheduler, loss (BCE with logits for binary classification).
  - Train/validate across epochs; save best checkpoint to `result/weights/` and log metrics.

### `evaluate.py`
- **Purpose**: Evaluate a trained checkpoint.
- **Typical flow**:
  - Load model and weights.
  - Run inference on validation/test set; compute metrics (AUC, accuracy) and optionally save ROC curves.

### `config.py`
- **Purpose**: Central configuration and constants.
- **Examples**:
  - Paths: data directories, label CSV, weights output, image size.
  - Hyperparameters: batch size, learning rate, epochs, augmentation flags.
  - Device selection: `DEVICE = 'cuda' if available else 'cpu'`.

### `requirements.txt`
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
   - `python train.py` to produce a checkpoint under `result/weights/`.
3. **Evaluate** (optional):
   - `python evaluate.py` to compute metrics/plots.
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
