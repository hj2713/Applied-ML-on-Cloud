# HW3: Cloud-Native MNIST Training and Inference on GKE

This project implements the Homework 3 deep learning workflow on Google Kubernetes Engine (GKE). It trains a PyTorch CNN on MNIST in a Kubernetes `Job`, saves the trained model to a PersistentVolumeClaim (PVC), and serves interactive predictions through a Flask web UI running in a Kubernetes `Deployment`.

## What Was Built

- Training container: `Deliverables/training`
  - Trains a CNN on MNIST with PyTorch.
  - Uses GPU acceleration on the GKE GPU node.
  - Saves `mnist_model.pt` to `/mnt/model/mnist_model.pt`.
- Inference container: `Deliverables/inference`
  - Runs a Flask web server on port `5000`.
  - Loads the trained model from the shared PVC.
  - Provides a browser UI and `/predict` API endpoint.
  - Requires `X-API-Key` authentication for predictions.
- Kubernetes manifests: `Deliverables/k8s`
  - `pvc.yaml`: persistent model storage.
  - `training-job.yaml`: run-to-completion training workload.
  - `secret.yaml`: API key for inference authentication.
  - `deployment.yaml`: always-on inference server.
  - `service.yaml`: public LoadBalancer service.
- Build helper manifests: `HW Info and Logs/Kaniko_and_GCP_Logs`
  - Kaniko jobs used to build Linux/amd64 images inside the cluster.

## Final Running Images

- Training image: `hj2713/hw3-train:v3`
- Inference image: `hj2713/hw3-inference:v2`

These versions are the working versions used on GKE. Earlier image tags failed because of missing files or platform mismatch.

## Architecture

```text
MNIST data
   |
   v
Kubernetes Job: mnist-training-job
   |
   | saves model
   v
PVC: model-pvc (/mnt/model/mnist_model.pt)
   |
   | mounted read-only
   v
Kubernetes Deployment: mnist-inference
   |
   v
Service: mnist-inference-service (LoadBalancer)
   |
   v
Browser UI / curl request with X-API-Key
```

## Kubernetes Controllers Used

- `Job` for training: training is finite and should run to completion, then release the GPU.
- `Deployment` for inference: inference must stay available continuously and restart automatically if the pod fails.
- `Service` of type `LoadBalancer`: exposes the Flask app through a public IP.
- `PersistentVolumeClaim`: stores the trained model so it survives pod termination and can be shared between training and inference.
- `Secret`: injects the API key at runtime instead of hardcoding it into the application image.

## Local Development

From the repository root:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r Deliverables/inference/requirements.txt
pip install -r Deliverables/training/requirements.txt

python3 Deliverables/training/train.py
MODEL_DIR=./model_data API_KEY=dev-secret-key python3 Deliverables/inference/app.py
```

Local predictions can be tested at:

```text
http://localhost:5000
```

## GKE Deployment Summary

Apply the Kubernetes resources in this order:

```bash
kubectl apply -f Deliverables/k8s/pvc.yaml
kubectl apply -f Deliverables/k8s/training-job.yaml
kubectl logs -f job/mnist-training-job

kubectl apply -f Deliverables/k8s/secret.yaml
kubectl apply -f Deliverables/k8s/deployment.yaml
kubectl apply -f Deliverables/k8s/service.yaml
```

Check status:

```bash
kubectl get jobs,pods,pvc,svc
kubectl describe pod -l app=mnist-inference
```

The cluster reached the following expected state:

```text
job.batch/mnist-training-job   Complete   1/1
pod/mnist-inference-...        1/1        Running
service/mnist-inference-service LoadBalancer <external-ip>
```

## Testing Inference

Browser:

1. Open the LoadBalancer IP.
2. Enter the API key from `Deliverables/k8s/secret.yaml`.
3. Upload MNIST-style digit images: white digit on black background, centered.
4. Confirm predictions appear in the web UI history.

curl:

```bash
curl -X POST http://<EXTERNAL-IP>/predict \
  -H "X-API-Key: my-secure-api-key-2026" \
  -F "file=@Deliverables/test_images/4.png"
```

## Important Notes

- The model is trained on MNIST, so it predicts best on MNIST-like images. A black marker on white paper may fail because it does not match the training distribution.
- The API key protects `/predict`, but the browser UI itself is public. For production, HTTPS, rate limiting, and stronger identity/auth would be needed.
- Kaniko was used because building on a local Apple Silicon machine produced an inference image that GKE could not run on Linux/amd64 nodes.

## TA Navigation

For grading, start with:

- `Read this for easier navigation.md`: one-page map of the repository, rubric coverage, and links to every important file.
- `Deliverables/Screenshots/README.md`: index mapping screenshots/logs/video evidence to rubric items.
- `Deliverables/ARCHITECTURE_REPORT.md`: architecture and experience report.
- `Deliverables/PROJECT_COMMANDS_AND_DECISIONS.md`: reproducible command log and implementation decisions.
