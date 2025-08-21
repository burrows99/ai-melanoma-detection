FROM python:3.10-slim

# Install minimal system dependencies (for OpenCV and healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch (CPU builds) and project dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir \
       torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
  && pip install --no-cache-dir -r requirements.txt \
  && pip install --no-cache-dir gunicorn uvicorn

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/result/weights

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

# Expose the port the app runs on
EXPOSE 7860

# Start application
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--worker-class", "uvicorn.workers.UvicornWorker", "--timeout", "120", "app:asgi_app"]
