# Lecture 12 & Homework 3 — Complete Guide

## Part 1: Key Lecture Concepts

---

### 🐳 1. Containers — The Foundation

**What is a container?**
A container is a **lightweight, isolated sandbox** for running a process. Unlike a full Virtual Machine (VM) that includes an entire Guest OS (1–10 GB), a container **shares the host OS kernel** and only packages the application + its dependencies (often just 10s of MB).

| Feature | Containers | VMs |
|---|---|---|
| OS | Share host OS | Each has own Guest OS |
| Weight | Lightweight | Heavyweight |
| Performance | Near bare-metal | Some overhead |
| Startup | Milliseconds | Minutes |
| Isolation | Process-level | Full isolation |
| GPU support | Maturing | Mature |

**Key takeaway:** Containers are *not* replacing VMs — they coexist. Containers are ideal for packaging and deploying applications consistently across environments.

---

### 🏗️ 2. Docker — Container Tooling

Docker gives you three basic capabilities:

```
Build  →  Store  →  Run
```

1. **Build**: You write a `Dockerfile` that describes how to build your app image step-by-step (base image, install dependencies, copy code, set entry point)
2. **Store**: Push the built image to a **Container Registry** (Docker Hub, Google Container Registry / Artifact Registry)
3. **Run**: Pull the image and launch it as a container on any machine with a container runtime

**The Dockerfile** is the recipe. It looks like:
```dockerfile
FROM python:3.10-slim          # Base image
WORKDIR /app                   # Set working directory
COPY requirements.txt .        # Copy dependency list
RUN pip install -r requirements.txt  # Install dependencies
COPY . .                       # Copy your application code
CMD ["python", "train.py"]     # Default command to run
```

---

### ☸️ 3. Kubernetes (K8s) — Container Orchestration

**Problem:** If you have dozens or hundreds of containers across many servers, how do you manage them? Docker alone doesn't handle scheduling, scaling, failover, networking, and load balancing across a cluster.

**Answer: Kubernetes** — an open-source platform for managing containerized workloads.

#### K8s Architecture — The Building Blocks

Think of it as a hierarchy, from smallest to largest:

```
Container → Pod → ReplicaSet → Deployment → Service → Ingress
```

#### 3.1 Nodes & Pods

- **Node** = a physical or virtual machine in the cluster. Each node runs:
  - **Kubelet**: agent that communicates with the master
  - **Container runtime**: Docker, containerd, cri-o
  
- **Pod** = the smallest deployable unit. A pod wraps one or more containers that:
  - Share the same **network** (same IP address)
  - Share **storage volumes**
  - Always run on the **same node**

> [!IMPORTANT]
> You don't deploy containers directly — you deploy **Pods**. A Pod is the K8s wrapper around your container(s).

#### 3.2 ReplicaSets

- Ensures a **specified number of identical pods** are running at all times
- If a pod crashes, the ReplicaSet **automatically spawns a replacement**
- Enables horizontal scaling (run 3 copies of your inference server)

#### 3.3 Deployments

A **Deployment** is the most common K8s controller you'll use. It:
- Creates and manages a ReplicaSet for you
- Handles **rolling updates** (deploy new version without downtime)
- Handles **rollbacks** if something goes wrong
- Specifies ConfigMaps, Secrets, Volume Mounts

> [!TIP]
> For your homework: use a **Job** for training (runs once and completes) and a **Deployment** for inference (runs continuously).

#### 3.4 Services

A **Service** exposes your pods to network traffic. Types:

| Type | What it does |
|---|---|
| **ClusterIP** | Internal-only access within the cluster |
| **NodePort** | Exposes on a static port on each node's IP |
| **LoadBalancer** | Creates an external cloud load balancer with a public IP |

#### 3.5 Ingress

- Sits in front of Services
- Maps URL paths to different services
- Like a smart reverse proxy

#### 3.6 Volume Mounts

- **ConfigMaps**: hold configuration parameters (non-secret)
- **Secrets**: hold passwords, API keys, credentials
- **Persistent Volume Claims (PVC)**: request persistent storage that survives pod restarts — critical for storing your trained model

#### 3.7 Kubernetes Machinery (Control Plane)

| Component | Role |
|---|---|
| **Kube API Server** | Front door — all commands go through here |
| **Kube Scheduler** | Decides which node a pod runs on |
| **Controller Manager** | Ensures desired state = actual state |
| **Kubelet** | Agent on each node, manages pods |
| **Kube-proxy** | Network proxy, handles routing |

---

### 🤖 4. Kubeflow — ML on Kubernetes

**Kubeflow** = ML toolkit built on top of Kubernetes. It provides:
- End-to-end ML workflows (data prep → training → serving)
- Multi-framework support (PyTorch, TensorFlow, etc.)
- Scalable, portable, repeatable deployments

**Key component for HW3:** The **Kubeflow Training Operator** — lets you define training jobs as K8s resources (like `PyTorchJob`).

---

### 🌩️ 5. ML on Cloud — The Stack

```
┌──────────────────────────────────────────────┐
│ ML SaaS  — Watson APIs, AWS AI, Google AI    │  ← Most managed
│ ML PaaS  — SageMaker, Vertex AI, Azure ML    │
│ ML IaaS  — EC2 + GPU, GCE, Azure VMs         │  ← Most control
│ Hardware — GPUs, TPUs, FPGAs                  │
└──────────────────────────────────────────────┘
```

**HW3 operates at the IaaS level** — you're using GKE (Google Kubernetes Engine) to get GPU-backed infrastructure and managing everything yourself with K8s.

---

---

## Part 2: Homework 3 — What You Need To Do

---

### 📋 Objective

> Build a **simple DL workflow** on **GKE (Google Kubernetes Engine)** that:
> 1. **Trains** a deep learning model inside a container on K8s
> 2. **Serves inference** from that trained model via a web URL
> 3. Allows **interactive user input** to the inference engine

---

### 🏛️ Architecture Overview

Here is the complete pipeline you need to build:

```
┌─────────────────────── TRAINING PHASE ───────────────────────┐
│                                                               │
│  Training Data  →  Training Code  →  Trained Model            │
│                                                               │
│        ┌──────────┐    ┌──────────────┐                       │
│        │Dockerfile │    │ K8s YAML     │                       │
│        │(training) │    │ • Job/PyTorchJob                     │
│        └──────────┘    │ • Volume     │                       │
│                        └──────────────┘                       │
└───────────────────────────────────────────────────────────────┘
                              │
                              │ Model saved to shared storage
                              ▼
┌─────────────────────── INFERENCE PHASE ──────────────────────┐
│                                                               │
│  Model  →  Inference Code  →  Web Server  →  User Interacts   │
│                                                               │
│        ┌──────────┐    ┌──────────────┐                       │
│        │Dockerfile │    │ K8s YAML     │                       │
│        │(inference)│    │ • Deployment │                       │
│        └──────────┘    │ • Service    │                       │
│                        │ • Volume     │                       │
│                        └──────────────┘                       │
└───────────────────────────────────────────────────────────────┘
```

---

### 📁 Deliverables — File by File

You need to create **6–8 files total**:

#### A. Training Side

| # | File | Purpose |
|---|---|---|
| 1 | `train.py` | Python script that trains a DL model (e.g., MNIST classifier) and **saves the model** to a known path |
| 2 | `Dockerfile.train` | Container definition to package `train.py` + dependencies |
| 3 | `training-job.yaml` | K8s **Job** manifest that runs the training container once to completion |

#### B. Inference Side

| # | File | Purpose |
|---|---|---|
| 4 | `app.py` / `inference.py` | Python script that **loads the trained model** and serves predictions via a web endpoint (Flask/FastAPI) |
| 5 | `Dockerfile.inference` | Container definition to package the inference app |
| 6 | `inference-deployment.yaml` | K8s **Deployment** manifest (keeps inference pods running) |
| 7 | `inference-service.yaml` | K8s **Service** manifest (exposes the deployment externally via LoadBalancer) |

#### C. Optional but Recommended

| # | File | Purpose |
|---|---|---|
| 8 | `pvc.yaml` | Persistent Volume Claim — shared storage between training and inference |

#### D. Documentation

| # | File | Purpose |
|---|---|---|
| 9 | `README.md` / Report | Build/run instructions, screenshots, discussion of K8s controllers used and why |

---

### 🔧 Step-by-Step — What To Do

#### Step 1: Write the Training Script (`train.py`)

Pick a simple model. MNIST digit classifier is perfect:
- Use PyTorch or TensorFlow
- Train for a few epochs
- **Save the trained model** to a mounted path like `/mnt/model/model.pth`

> [!IMPORTANT]
> The model must be saved to a **persistent/shared location** so the inference container can access it.

#### Step 2: Write the Training Dockerfile (`Dockerfile.train`)

```dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
WORKDIR /app
COPY train.py .
COPY requirements.txt .
RUN pip install -r requirements.txt
CMD ["python", "train.py"]
```

#### Step 3: Write the Inference Script (`app.py`)

- Load the saved model from `/mnt/model/model.pth`
- Create a web server (Flask/FastAPI)
- Accept user input (image upload, number, etc.)
- Return the prediction

The key requirement: **"Host a URL to allow interaction with User"** and **"Give any input to the inference engine to show that the inference is happening in an interactive/controlled manner."**

This means you need a simple web UI or API endpoint.

#### Step 4: Write the Inference Dockerfile (`Dockerfile.inference`)

```dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
WORKDIR /app
COPY app.py .
COPY requirements.txt .
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python", "app.py"]
```

#### Step 5: Build & Push Images to Container Registry

You'll push both images to Google Container Registry (GCR) or Artifact Registry:

```bash
# Build and tag
docker build -f Dockerfile.train -t gcr.io/YOUR_PROJECT_ID/mnist-train:v1 .
docker build -f Dockerfile.inference -t gcr.io/YOUR_PROJECT_ID/mnist-inference:v1 .

# Push
docker push gcr.io/YOUR_PROJECT_ID/mnist-train:v1
docker push gcr.io/YOUR_PROJECT_ID/mnist-inference:v1
```

#### Step 6: Create GKE Cluster

```bash
gcloud container clusters create hw3-cluster \
  --zone us-central1-a \
  --num-nodes 1 \
  --machine-type e2-standard-4
```

> [!NOTE]
> GPU is optional for this homework. If you want GPU, you add a node pool with GPU accelerators and the `nvidia.com/gpu` resource limit in your pod spec.

#### Step 7: Write K8s YAML — Training Job

```yaml
# training-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: mnist-training-job
spec:
  template:
    spec:
      containers:
      - name: trainer
        image: gcr.io/YOUR_PROJECT_ID/mnist-train:v1
        volumeMounts:
        - name: model-storage
          mountPath: /mnt/model
      restartPolicy: Never
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc
  backoffLimit: 2
```

**Why a Job?** Training runs once and completes. A Job is the right K8s controller for run-to-completion tasks.

#### Step 8: Write K8s YAML — Inference Deployment + Service

```yaml
# inference-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mnist-inference
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mnist-inference
  template:
    metadata:
      labels:
        app: mnist-inference
    spec:
      containers:
      - name: inference
        image: gcr.io/YOUR_PROJECT_ID/mnist-inference:v1
        ports:
        - containerPort: 5000
        volumeMounts:
        - name: model-storage
          mountPath: /mnt/model
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc
---
# inference-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: mnist-inference-service
spec:
  type: LoadBalancer
  selector:
    app: mnist-inference
  ports:
  - port: 80
    targetPort: 5000
```

**Why a Deployment?** The inference server should run continuously and auto-restart if it crashes. A Deployment + ReplicaSet handles this.

**Why a Service with LoadBalancer?** This gives you an **external IP** that users can hit from their browser.

#### Step 9: Write PVC (Optional but Recommended)

```yaml
# pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
```

This creates persistent storage where the training job writes the model and the inference deployment reads it.

#### Step 10: Deploy Everything

```bash
# 1. Create PVC
kubectl apply -f pvc.yaml

# 2. Run training job
kubectl apply -f training-job.yaml

# 3. Wait for training to complete
kubectl get jobs --watch

# 4. Deploy inference
kubectl apply -f inference-deployment.yaml
kubectl apply -f inference-service.yaml

# 5. Get external IP
kubectl get svc mnist-inference-service
```

#### Step 11: Test & Screenshot

- Open the external IP in your browser
- Submit input (draw a digit, upload an image, enter a number)
- Take screenshots of:
  - The GKE cluster in GCP Console
  - `kubectl get pods` output
  - `kubectl get services` output
  - The web UI working with inference results

---

### 📝 Report: What To Discuss

The homework asks you to **"Document your work and report on your experiences"**, specifically:

1. **What K8s controllers did you use and why?**
   - **Job** for training — because training is a run-to-completion task
   - **Deployment** for inference — because it needs to run continuously, auto-restart, and support rolling updates
   - **Service (LoadBalancer)** — to expose inference externally with a public IP
   - **PVC** — to persist the trained model across pod lifecycles

2. **Workflow description** — explain the data flow from training → model storage → inference serving

3. **Challenges encountered** — anything related to GKE setup, GPU configuration, networking, etc.

---

### ⚡ Alternative: Using Kubeflow Training Operator

Instead of a plain K8s `Job`, you can use the **Kubeflow Training Operator** for a more sophisticated training setup:

```bash
# Install the operator
kubectl apply -k "github.com/kubeflow/training-operator/manifests/overlays/standalone"

# Then use PyTorchJob instead of Job
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: mnist-pytorch
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      template:
        spec:
          containers:
          - name: pytorch
            image: gcr.io/YOUR_PROJECT_ID/mnist-train:v1
```

This is more aligned with the "Kubeflow" theme of the homework but is **not strictly required**.

---

### 🎯 Summary Checklist

- [ ] Training Python script that saves model persistently
- [ ] Training Dockerfile
- [ ] Training Job YAML (K8s Job or Kubeflow PyTorchJob)
- [ ] Inference Python script with web server
- [ ] Inference Dockerfile
- [ ] Inference Deployment YAML
- [ ] Inference Service YAML (LoadBalancer type)
- [ ] PVC YAML (if using shared persistent storage)
- [ ] Build & push Docker images to GCR/Artifact Registry
- [ ] Create GKE cluster
- [ ] Deploy all YAML files with `kubectl apply`
- [ ] Test the web inference endpoint
- [ ] Take screenshots
- [ ] Write report explaining controllers and experiences
