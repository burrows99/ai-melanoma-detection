# Use an official Python runtime as a parent image with Debian Bullseye for better compatibility
FROM python:3.9-slim-bullseye

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p /app/result/weights

# Download the model weights (uncomment and modify if you have a direct download link)
# RUN wget -O /app/result/weights/gradcam.pth <YOUR_MODEL_WEIGHTS_URL>

# Expose the port the app runs on
EXPOSE 7860

# Command to run the application
CMD ["python", "app.py"]
