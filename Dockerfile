# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV HF_HOME=/app/huggingface_cache # Cache for Hugging Face models
ENV RUNPOD_HANDLER_PATH=/app/handler.py # For RunPod to find the handler

# Install system dependencies
# ffmpeg is needed by imageio-ffmpeg for video processing
# git and git-lfs might be useful for some cases, but hf_hub_download should handle most model downloads.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    git-lfs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Using --no-cache-dir to reduce image size
# RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project directory into the container at /app
COPY . .

# Command to run the application
CMD ["python", "handler.py"]
