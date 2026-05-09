# Architecture & Implementation Report

**Course:** Applied Machine Learning in the Cloud  
**Homework:** HW3 - Simple DL Workflow in GKE/Kubernetes  
**Application:** MNIST digit classifier with interactive web inference

## 1. Executive Summary

This project implements an end-to-end deep learning workflow on Google Kubernetes Engine (GKE). The system has two separate cloud-native phases:

1. A GPU-backed training workload that trains a PyTorch CNN on MNIST and writes the trained weights to persistent storage.
2. A CPU-backed inference workload that loads those weights and serves predictions through a browser-accessible Flask application.

The final cluster successfully ran the training job, persisted `mnist_model.pt`, deployed inference, exposed the service through a LoadBalancer IP, and returned correct predictions for MNIST-style input images.

## 2. Application Architecture

```text
Training code + MNIST data
        |
        v
Kubernetes Job on GPU node
        |
        | writes /mnt/model/mnist_model.pt
        v
PersistentVolumeClaim: model-pvc
        |
        | mounted read-only by inference
        v
Kubernetes Deployment on CPU node
        |
        v
LoadBalancer Service
        |
        v
Browser UI / curl client
```

## 3. Kubernetes Resources

### 3.1 Training Job

File: `Deliverables/k8s/training-job.yaml`

The training phase uses a Kubernetes `Job` named `mnist-training-job`.

Why a `Job` was used:

- Training is a finite workload.
- It should run until completion and then stop.
- It uses an expensive GPU only while training is active.
- Kubernetes can retry the pod if the container fails.

The job runs image `hj2713/hw3-train:v3`, requests one GPU with `nvidia.com/gpu: 1`, and saves the model to `/mnt/model/mnist_model.pt`.

### 3.2 Inference Deployment

File: `Deliverables/k8s/deployment.yaml`

The inference phase uses a Kubernetes `Deployment` named `mnist-inference`.

Why a `Deployment` was used:

- Inference is a long-running web service.
- It must stay available after training finishes.
- Kubernetes restarts the pod automatically if it fails.
- It supports rolling updates when the inference image changes.

The deployment runs image `hj2713/hw3-inference:v2`, mounts the model PVC read-only, and exposes Flask on container port `5000`.

### 3.3 LoadBalancer Service

File: `Deliverables/k8s/service.yaml`

The `mnist-inference-service` resource is a Kubernetes `Service` of type `LoadBalancer`.

Why a `LoadBalancer` was used:

- The homework requires hosting a URL for interactive user input.
- GKE provisions an external IP that routes browser and curl requests to the inference pod.
- The service exposes port `80` externally and forwards traffic to Flask on port `5000`.

### 3.4 PersistentVolumeClaim

File: `Deliverables/k8s/pvc.yaml`

The `model-pvc` PVC provides persistent shared storage between training and inference.

Why a PVC was used:

- A training pod's local filesystem disappears when the pod completes.
- The model must survive after the training job exits.
- The inference deployment needs the same model file without baking weights into the image.

### 3.5 Kubernetes Secret

File: `Deliverables/k8s/secret.yaml`

The `inference-secret` resource stores the prediction API key. The Flask app reads it through the `API_KEY` environment variable, and `/predict` requires the matching `X-API-Key` header.

This demonstrates runtime secret injection and avoids hardcoding the production API key in the application code.

## 4. Containerization

### 4.1 Training Container

Folder: `Deliverables/training`

- Base image: `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime`
- Entrypoint: `python train.py`
- Important environment variables:
  - `EPOCHS=10`
  - `BATCH_SIZE=64`
  - `MODEL_DIR=/mnt/model`

The training image was built and pushed as:

```text
hj2713/hw3-train:v3
```

### 4.2 Inference Container

Folder: `Deliverables/inference`

- Base image: `python:3.10-slim`
- Framework: Flask
- Entrypoint: `python app.py`
- Port: `5000`
- Important environment variables:
  - `MODEL_DIR=/mnt/model`
  - `PORT=5000`
  - `API_KEY` from Kubernetes Secret

The inference image was built and pushed as:

```text
hj2713/hw3-inference:v2
```

## 5. Validation Results

The final working cluster state included:

- `mnist-training-job`: completed successfully.
- `model-pvc`: bound and used by both training and inference.
- `mnist-inference`: `1/1 Running`.
- `mnist-inference-service`: external LoadBalancer IP assigned.
- Browser UI: accepted controlled user image input and returned predictions.
- curl API: accepts authenticated image uploads using `X-API-Key`.

Training log evidence included:

```text
Using compute device: cuda
Starting training for 10 epochs...
Training complete! Model saved successfully to: /mnt/model/mnist_model.pt
```

## 6. Issues Faced and Resolutions

### 6.1 GCP Organization Policy and Private Nodes

The GCP project had an organization policy preventing VM instances from having external IPs. The cluster used private nodes, so outbound internet access required Cloud NAT.

Resolution:

- Created a Cloud Router and Cloud NAT in `us-east1`.
- Confirmed the training pod could download MNIST from the PyTorch fallback mirror.

### 6.2 GPU Region and Quota

Finding a region with available GPU quota took time. The final cluster used `us-east1` with an L4 GPU node.

Resolution:

- Used a GPU node pool for training.
- Used a separate CPU node pool for inference.
- Scheduled inference specifically on `cpu-pool` to avoid using GPU resources for serving.

### 6.3 `kubectl` Authentication Plugin

GKE required the `gke-gcloud-auth-plugin` for local `kubectl` authentication.

Resolution:

```bash
gcloud components install gke-gcloud-auth-plugin
```

### 6.4 Broken Training Image

The first training image failed with:

```text
python: can't open file '/app/train.py': [Errno 2] No such file or directory
```

Cause:

- The Kaniko build context did not include `train.py`.

Resolution:

- Recreated the training build ConfigMap with `Dockerfile`, `train.py`, and `requirements.txt`.
- Rebuilt the image as `hj2713/hw3-train:v3`.
- Added `imagePullPolicy: Always` to avoid cached bad images.

### 6.5 Inference Image Platform Mismatch

The first inference deployment failed with:

```text
no match for platform in manifest: not found
```

Cause:

- The image was likely built for the local Mac architecture instead of Linux/amd64.

Resolution:

- Rebuilt the inference image inside the GKE cluster with Kaniko.
- Deployed `hj2713/hw3-inference:v2`.

### 6.6 API Key Difference Between Local and Cluster

The app worked locally with `dev-secret-key`, but Kubernetes used the key from `inference-secret`.

Resolution:

- Used the Kubernetes Secret value for cluster testing:

```text
my-secure-api-key-2026
```

### 6.7 MNIST Input Distribution

The model predicts best on MNIST-style images: light digits on a dark background, centered and close to 28x28 formatting. Marker-on-paper photos may produce poor predictions because they differ from the MNIST training distribution.

Resolution:

- Validated the model using MNIST-like test images.
- Kept the model and preprocessing simple to align with the homework demo scope.

## 7. Conclusion

The project satisfies the homework requirements by providing containerized training and inference workloads on GKE, persistent model storage, a public interactive inference URL, and documentation of Kubernetes controller choices. The final system demonstrates the intended cloud-native ML pattern: use GPU-backed batch compute for training, persist model state separately, and serve predictions through a lightweight always-on web deployment.
