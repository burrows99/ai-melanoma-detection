# Melanoma Detection Project

Deep learning model for melanoma detection from skin lesion images, incorporating patient metadata and Grad-CAM for explainability.

## Docker Setup

The provided Dockerfile builds a CPU-only image. Use docker compose to run services.

### Option 1: Using docker compose (Recommended)

1. **Run the web UI (Gradio):**
   ```bash
   docker compose up --build web
   ```
   Open http://localhost:7860

2. **Run training as an on-demand service:**
   ```bash
   docker compose run --rm --profile cli train
   ```

3. **Run evaluation as an on-demand service:**
   ```bash
   docker compose run --rm --profile cli eval
   ```

Notes:
- Volumes `./data`, `./base`, and `./result` are mounted into the containers.
- Healthcheck for the web service pings http://localhost:7860.

### Option 2: Manual Docker Commands

1. **Build the Docker image:**
   ```bash
   docker build -t melanoma-detection .
   ```

2. **Run the container:**
   - For CPU:
     ```bash
     docker run -p 7860:7860 -v $(pwd)/data:/app/data -v $(pwd)/result:/app/result melanoma-detection
     ```
   - For GPU (not configured in this image): you would need a CUDA base and NVIDIA runtime.

The app will be available at `http://localhost:7860`
- Mount your data directory to `/app/data`
- Model weights will be saved to `/app/result`

## Local Setup (without Docker)

The Docker workflow is recommended and already includes PyTorch CPU in the image. PyTorch is intentionally NOT listed in `requirements.txt`.

If you must run locally (no Docker), install PyTorch separately first, then install the rest of the requirements:

1.  **Install PyTorch (choose one):**
    ```bash
    # CPU-only
    pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu

    # (Optional) CUDA builds require a matching local CUDA setup; not needed for the provided Docker image
    ```

2.  **Install project dependencies (excluding PyTorch):**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Model Weights for `app.py`:**
    The Gradio application (`app.py`) expects pre-trained weights at `result/weights/gradcam.pth`. If training a new model, update this path in `app.py` or rename the saved model accordingly.

## Running the Application

To launch the Gradio interface for inference and Grad-CAM visualization:
```bash
python app.py
```

## Training the Model

1.  **Dataset:**
    Configure `TRAIN_DATA_DIR` (image folder) and `TRAIN_LABELS_PATH` (labels CSV) in `config.py`.

2.  **Configuration:**
    Adjust training parameters (model architecture, learning rate, etc.) in `config.py`.

3.  **Weights & Biases (W&B):**
    Training uses W&B for logging. You will be prompted to log in.
    To disable W&B, set the environment variable `WANDB_MODE=disabled` before running but will limits result output:
    ```bash
    # $env:WANDB_MODE="disabled"; python train.py
    ```

4.  **Run Training:**
    ```bash
    python train.py
    ```

## Key Files

*   `app.py`: Gradio application.
*   `train.py`: Model training script.
*   `evaluate.py`: Evaluation logic (metrics, TTA).
*   `dataset.py`: Data loading and augmentations.
*   `model.py`: Neural network definition.
*   `config.py`: Project configurations.
*   `result`: path where the weights are stored
*   `base`: path where the weights are stored





#