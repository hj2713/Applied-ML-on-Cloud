FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Copy and install dependencies first (Docker layer caching optimization)
RUN pip install --no-cache-dir python-dotenv

# Copy the training script
COPY train.py .

# Default environment variables (can be overridden by K8s ConfigMap)
ENV EPOCHS=10
ENV BATCH_SIZE=64
ENV MODEL_DIR=/mnt/model

CMD ["python", "train.py"]
