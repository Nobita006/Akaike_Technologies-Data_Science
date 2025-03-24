# Use an official Python 3.10 slim image
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Hugging Face Spaces: store large data/model caches in /data
# spaCy / Hugging Face / NLTK will download into these folders
ENV HF_HOME=/data/cache
ENV NLTK_DATA=/data/nltk_data

# Install system dependencies (build-essential, curl, dos2unix, bash, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    dos2unix \
    bash \
    && rm -rf /var/lib/apt/lists/*

# Create data dirs for HF/nltk and give open perms (for Spaces)
RUN mkdir -p /data/cache /data/nltk_data
RUN chmod -R 777 /data

# We'll keep our application code in /app
WORKDIR /app

# Copy requirements.txt into /app and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# (Optional) Install CPU-only PyTorch explicitly (no CUDA/GPU)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Download additional models/data into /data
# (Because HF_HOME, NLTK_DATA are set, they'll be stored in /data automatically.)
RUN python -m spacy download en_core_web_sm

# IMPORTANT: Explicitly install NLTK resources into /data/nltk_data
RUN python -m nltk.downloader punkt averaged_perceptron_tagger punkt_tab -d /data/nltk_data

# Copy the rest of your code (including run.sh) into /app
COPY . /app

# Convert run.sh to LF line endings (if created on Windows) and make it executable
RUN dos2unix /app/run.sh
RUN chmod +x /app/run.sh

# The container starts by executing run.sh in /app
CMD ["/app/run.sh"]
