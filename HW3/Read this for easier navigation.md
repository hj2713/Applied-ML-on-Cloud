# Read This for Easier Navigation

This document is a TA-friendly index for quickly reviewing the HW3 submission.

## GitHub Repository

Repository link:

https://github.com/hj2713/Applied-ML-on-Cloud/tree/main/HW3

## Recommended Review Order

| Step | File / Folder | GitHub link | What to check |
|---|---|---|---|
| 1 | Main README | [README.md](https://github.com/hj2713/Applied-ML-on-Cloud/tree/main/HW3/README.md) | High-level project overview, final image tags, run instructions, and submission summary |
| 2 | Architecture report | [Deliverables/ARCHITECTURE_REPORT.md](https://github.com/hj2713/Applied-ML-on-Cloud/tree/main/HW3/Deliverables/ARCHITECTURE_REPORT.md) | Explanation of Kubernetes controllers, architecture, issues faced, and decisions |
| 3 | Commands and decisions | [Deliverables/PROJECT_COMMANDS_AND_DECISIONS.md](https://github.com/hj2713/Applied-ML-on-Cloud/tree/main/HW3/Deliverables/PROJECT_COMMANDS_AND_DECISIONS.md) | Commands used for GCP, Docker/Kaniko, Kubernetes, inference testing, and debugging |
| 4 | Screenshot index | [Deliverables/Screenshots/README.md](https://github.com/hj2713/Applied-ML-on-Cloud/tree/main/HW3/Deliverables/Screenshots/README.md) | Maps screenshots, logs, and demo video to rubric items |
| 5 | Kubernetes manifests | [Deliverables/k8s](https://github.com/hj2713/Applied-ML-on-Cloud/tree/main/HW3/Deliverables/k8s) | Training Job, inference Deployment, Service, PVC, and Secret |
| 6 | Training code | [Deliverables/training](https://github.com/hj2713/Applied-ML-on-Cloud/tree/main/HW3/Deliverables/training) | PyTorch MNIST training script and training Dockerfile |
| 7 | Inference code | [Deliverables/inference](https://github.com/hj2713/Applied-ML-on-Cloud/tree/main/HW3/Deliverables/inference) | Flask inference app, UI, authentication, and inference Dockerfile |
| 8 | Evidence folder | [Deliverables/Screenshots](https://github.com/hj2713/Applied-ML-on-Cloud/tree/main/HW3/Deliverables/Screenshots) | Cluster proof, training logs, image push logs, service IP, curl test, and app demo |

## Rubric Coverage Summary

| Rubric category | Required proof | Where it is shown |
|---|---|---|
| Created k8s cluster | GKE cluster created and running | `01_gke_cluster_running_overview.png`, `02_gke_gpu_cpu_node_pools.png`, `03_kubectl_get_nodes_ready.png` |
| Training in k8s: Dockerfile | Training Dockerfile | `Deliverables/training/Dockerfile` |
| Training in k8s: docker push | Training image built and pushed | `10_docker_push_training_image.png`, `PROJECT_COMMANDS_AND_DECISIONS.md` |
| Training in k8s: train-yaml | Kubernetes training Job | `Deliverables/k8s/training-job.yaml` |
| Training in k8s: model saving | Model saved to PVC | `12_training_logs.txt`, `ARCHITECTURE_REPORT.md` |
| Inferencing in k8s: PVC creation | PVC exists and is bound | `Deliverables/k8s/pvc.yaml`, `06_kubectl_get_pvc_bound.png` |
| Inferencing in k8s: code augmentation | Inference web app loads model and accepts user input | `Deliverables/inference/app.py`, `Deliverables/inference/templates/index.html` |
| Inferencing in k8s: Dockerfile | Inference Dockerfile | `Deliverables/inference/Dockerfile` |
| Inferencing in k8s: infer-yaml | Deployment and Service YAML | `Deliverables/k8s/deployment.yaml`, `Deliverables/k8s/service.yaml` |
| Inferencing in k8s: infer-test | Browser and curl inference tests | `08_app_demo_browser_prediction.mp4`, `08_app_demo_browser_prediction_frame.png`, `09_curl_inference_test.png` |
| Writing | Clear docs and report | `README.md`, `ARCHITECTURE_REPORT.md`, `PROJECT_COMMANDS_AND_DECISIONS.md`, this file |

## Code Structure Table

| Path | Type | Purpose |
|---|---|---|
| `README.md` | Documentation | Main entry point with project overview, architecture summary, run instructions, and submission checklist |
| `Deliverables/ARCHITECTURE_REPORT.md` | Documentation/report | Architecture explanation, Kubernetes controller choices, issues faced, and final validation |
| `Deliverables/PROJECT_COMMANDS_AND_DECISIONS.md` | Documentation/report | Detailed command log and important implementation decisions |
| `Read this for easier navigation.md` | Documentation/report | TA-facing navigation map for the whole submission |
| `Deliverables/training/train.py` | Training code | Downloads MNIST, trains a PyTorch CNN, and saves `mnist_model.pt` to `MODEL_DIR` |
| `Deliverables/training/Dockerfile` | Training Dockerfile | Packages the PyTorch training workload into a GPU-capable container |
| `Deliverables/training/requirements.txt` | Training dependencies | Python package requirements for training |
| `Deliverables/inference/app.py` | Inference code | Flask app that loads the trained model, checks API key auth, and serves `/predict` |
| `Deliverables/inference/templates/index.html` | Web UI | Browser interface for entering API key and submitting images for prediction |
| `Deliverables/inference/Dockerfile` | Inference Dockerfile | Packages the Flask inference app into a container |
| `Deliverables/inference/requirements.txt` | Inference dependencies | Python package requirements for serving inference |
| `Deliverables/k8s/pvc.yaml` | Kubernetes YAML | Creates `model-pvc`, the shared persistent storage for the model |
| `Deliverables/k8s/training-job.yaml` | Kubernetes YAML | Runs the training container as a Kubernetes `Job` with GPU request |
| `Deliverables/k8s/deployment.yaml` | Kubernetes YAML | Runs the inference container as a Kubernetes `Deployment` on the CPU node pool |
| `Deliverables/k8s/service.yaml` | Kubernetes YAML | Exposes inference using a `LoadBalancer` service |
| `Deliverables/k8s/secret.yaml` | Kubernetes YAML | Stores the API key used by `/predict` authentication |
| `Deliverables/test_images` | Test inputs | MNIST-style digit images used for browser and curl inference tests |
| `Deliverables/Screenshots` | Evidence | Screenshots, logs, and app demo video proving the cluster and application worked |
| `HW Info and Logs/Kaniko_and_GCP_Logs/kaniko-job.yaml` | Build helper YAML | Kaniko job used to build and push `hj2713/hw3-train:v3` |
| `HW Info and Logs/Kaniko_and_GCP_Logs/kaniko-inference-job.yaml` | Build helper YAML | Kaniko job used to build and push `hj2713/hw3-inference:v2` |

## Core Files To Grade

| File | GitHub link | Why it is included |
|---|---|---|
| `README.md` | [link](https://github.com/hj2713/Applied-ML-on-Cloud/tree/main/HW3/README.md) | Main instructions and overview |
| `Deliverables/ARCHITECTURE_REPORT.md` | [link](https://github.com/hj2713/Applied-ML-on-Cloud/tree/main/HW3/Deliverables/ARCHITECTURE_REPORT.md) | Required writing/report component |
| `Deliverables/PROJECT_COMMANDS_AND_DECISIONS.md` | [link](https://github.com/hj2713/Applied-ML-on-Cloud/tree/main/HW3/Deliverables/PROJECT_COMMANDS_AND_DECISIONS.md) | Reproducibility and detailed command history |
| `Read this for easier navigation.md` | [link](https://github.com/hj2713/Applied-ML-on-Cloud/tree/main/HW3/Read%20this%20for%20easier%20navigation.md) | This TA navigation file |
| `Deliverables/training/train.py` | [link](https://github.com/hj2713/Applied-ML-on-Cloud/tree/main/HW3/Deliverables/training/train.py) | Training program |
| `Deliverables/training/Dockerfile` | [link](https://github.com/hj2713/Applied-ML-on-Cloud/tree/main/HW3/Deliverables/training/Dockerfile) | Training container definition |
| `Deliverables/training/requirements.txt` | [link](https://github.com/hj2713/Applied-ML-on-Cloud/tree/main/HW3/Deliverables/training/requirements.txt) | Training dependencies |
| `Deliverables/inference/app.py` | [link](https://github.com/hj2713/Applied-ML-on-Cloud/tree/main/HW3/Deliverables/inference/app.py) | Inference server and API key authentication |
| `Deliverables/inference/Dockerfile` | [link](https://github.com/hj2713/Applied-ML-on-Cloud/tree/main/HW3/Deliverables/inference/Dockerfile) | Inference container definition |
| `Deliverables/inference/requirements.txt` | [link](https://github.com/hj2713/Applied-ML-on-Cloud/tree/main/HW3/Deliverables/inference/requirements.txt) | Inference dependencies |
| `Deliverables/inference/templates/index.html` | [link](https://github.com/hj2713/Applied-ML-on-Cloud/tree/main/HW3/Deliverables/inference/templates/index.html) | Interactive browser UI |
| `Deliverables/k8s/pvc.yaml` | [link](https://github.com/hj2713/Applied-ML-on-Cloud/tree/main/HW3/Deliverables/k8s/pvc.yaml) | Persistent model storage |
| `Deliverables/k8s/training-job.yaml` | [link](https://github.com/hj2713/Applied-ML-on-Cloud/tree/main/HW3/Deliverables/k8s/training-job.yaml) | Training `Job` manifest |
| `Deliverables/k8s/deployment.yaml` | [link](https://github.com/hj2713/Applied-ML-on-Cloud/tree/main/HW3/Deliverables/k8s/deployment.yaml) | Inference `Deployment` manifest |
| `Deliverables/k8s/service.yaml` | [link](https://github.com/hj2713/Applied-ML-on-Cloud/tree/main/HW3/Deliverables/k8s/service.yaml) | Public `LoadBalancer` service |
| `Deliverables/k8s/secret.yaml` | [link](https://github.com/hj2713/Applied-ML-on-Cloud/tree/main/HW3/Deliverables/k8s/secret.yaml) | API key Secret |
| `Deliverables/Screenshots/README.md` | [link](https://github.com/hj2713/Applied-ML-on-Cloud/tree/main/HW3/Deliverables/Screenshots/README.md) | Screenshot/evidence index |

## Final Working Resources

| Resource | Final value |
|---|---|
| GKE cluster | `hw3-final` |
| Region/zone shown in evidence | `us-east1-c` |
| Training image | `hj2713/hw3-train:v3` |
| Inference image | `hj2713/hw3-inference:v2` |
| Training controller | Kubernetes `Job` |
| Inference controller | Kubernetes `Deployment` |
| Public exposure | Kubernetes `Service` of type `LoadBalancer` |
| Model storage | `model-pvc`, mounted at `/mnt/model` |
| Model path | `/mnt/model/mnist_model.pt` |
| Auth mechanism | Kubernetes `Secret` injected as `API_KEY`, checked through `X-API-Key` |

## Notes for Review

- The project uses a plain Kubernetes `Job` rather than a Kubeflow `PyTorchJob`. The lecture/homework guide states Kubeflow Training Operator is aligned with the theme but not strictly required; the homework asks for Kubernetes artifacts that perform training and inference on GKE.
- The final images were built with Kaniko inside the cluster to avoid local architecture mismatch and incomplete image build contexts.
- The app predicts best on MNIST-style images because the model is trained on MNIST.
