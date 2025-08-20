# Melanoma Detection Project

Deep learning model for melanoma detection from skin lesion images, incorporating patient metadata and Grad-CAM for explainability.

## Docker Setup

### Option 1: Using docker-compose (Recommended)

1. **Build and run with GPU (requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)):**
   ```bash
   docker-compose up --build
   ```

2. **For CPU-only mode, uncomment the CPU-specific lines in `docker-compose.yml` and run:**
   ```bash
   docker-compose up --build
   ```

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
   - For GPU:
     ```bash
     docker run -p 7860:7860 --gpus all -v $(pwd)/data:/app/data -v $(pwd)/result:/app/result melanoma-detection
     ```

The app will be available at `http://localhost:7860`
- Mount your data directory to `/app/data`
- Model weights will be saved to `/app/result`

## Local Setup

1.  **Environment:** Create a Python virtual environment and install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
   *( PyTorch nightly build for 12.8 CUDA is used.)*

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