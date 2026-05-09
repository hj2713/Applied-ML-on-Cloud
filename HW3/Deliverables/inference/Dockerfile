FROM python:3.10-slim

WORKDIR /app

# Copy and install dependencies first (Docker layer caching optimization)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the inference app and templates
COPY app.py .
COPY templates/ templates/

# Default environment variables (can be overridden by K8s ConfigMap/Secret)
ENV MODEL_DIR=/mnt/model
ENV PORT=5000
ENV API_KEY=change-me-in-production

EXPOSE 5000

CMD ["python", "app.py"]
