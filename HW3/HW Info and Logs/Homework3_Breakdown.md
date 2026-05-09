# Homework #3 – Simple DL Workflow in Kubeflow

## 1. Overview of the Homework

The goal of this assignment is to build a complete, end-to-end Deep Learning (DL) pipeline on **Google Kubernetes Engine (GKE)**. You need to create a system that has two distinct phases:

1. **Training Phase**: A workload that trains a Deep Learning model and saves the trained model weights.
2. **Inference Phase**: A web server that loads the trained model and hosts a URL where users can interactively pass inputs (like numbers, text, or images) and receive predictions.

You are expected to wrap your code in **Docker containers** and deploy them using **Kubernetes (K8s)** orchestrations.

---

## 2. Core Concepts & Topics

To complete this assignment, you need to understand the following concepts:

### A. Docker & Containerization

* **Dockerfiles:** Scripts that define how to build a container image for your application (packaging your Python code, TensorFlow/PyTorch libraries, and dependencies into an isolated environment).
* **Container Registry:** A place to upload your built Docker images so that Kubernetes can download and run them (e.g., Docker Hub or Google Container Registry).

### B. Kubernetes (K8s) Controllers

Kubernetes manages your containers using different "Controllers." You need to choose the right controller for the right job:

* **Job (for Training):** A `Job` creates a pod that runs to completion and then stops. Since model training is a finite task (it starts, trains for N epochs, and finishes), a `Job` is the perfect controller for this.
* **Deployment (for Inference):** A `Deployment` ensures that a pod runs continuously. Since an inference server needs to be "always on" to listen for user web requests, you use a `Deployment`.

### C. Kubernetes Networking & Storage

* **Service:** A K8s `Service` exposes your Inference `Deployment` to the internet. You will likely use a `LoadBalancer` service to get a public URL/IP address so you can interact with your web server.
* **Persistent Volume Claim (PVC):** A way to attach persistent cloud storage to your pods. **Crucial Step:** Your Training Job will save the model file into this PVC. Later, your Inference Deployment will mount the same PVC to load the model.

---

## 3. Step-by-Step Execution Plan

Here is the architectural workflow you need to follow:

**Step 1: Write the Application Code**

* **Training Script:** Write a Python script that downloads data, trains a simple DL model, and saves the model to a specific local directory (e.g., `/mnt/data/model.pth`).
* **Inference Script:** Write a simple web server (using Flask or FastAPI) that loads the model from `/mnt/data/model.pth` and exposes an endpoint (e.g., `/predict`) that accepts user input and returns a prediction.

**Step 2: Containerize the Code**

* Create a `Dockerfile.train` for the training script. Build it and push it to a registry.
* Create a `Dockerfile.inference` for the web server. Build it and push it to a registry.

**Step 3: Setup GKE and Storage**

* Spin up a Google Kubernetes Engine (GKE) cluster.
* (Optional but recommended) Create a Persistent Volume (PVC) in GCP so both containers can share the model file.

**Step 4: Write K8s YAML Manifests**

* Create `training-job.yaml` (Kind: Job) that mounts the PVC and runs your training image.
* Create `inference-deployment.yaml` (Kind: Deployment) that mounts the same PVC and runs your web server.
* Create `inference-service.yaml` (Kind: Service) to expose your web server to a public IP.

**Step 5: Deploy and Test**

* Apply the PVC and Training Job. Wait for it to finish.
* Apply the Inference Deployment and Service.
* Hit the public URL with a test input (e.g., sending an image to get a classification) to prove it works.

---

## 4. Final Deliverables Checklist (Metrics for Success)

Based on the assignment description, your final submission **must** include the following artifacts:

- [ ] **Dockerfiles**: The definition files for both your Training container and your Inference container.
- [ ] **K8s YAML Files**: The manifest files used to deploy your K8s resources (Job, Deployment, Service, and potentially PVC).
- [ ] **Run Instructions**: A clear `README` or document explaining exactly how to build the images and run the services on a GCP-GKE cluster.
- [ ] **Screen Captures**: Visual proof that the app runs. Specifically, screenshots showing:
  * The user interacting with the hosted URL.
  * The controlled/interactive inference happening (e.g., passing an image ID or text and showing the output).
- [ ] **Architecture Report**: A document detailing your experiences.
  * *Mandatory Question to Answer in Report:* "What Kubernetes controllers did you use for training and inference and why?"

---

## 5. Teaching the Concepts: Why separate Training and Inference?

A key lesson from this homework is understanding **why we separate workloads in the Cloud.**

**1. Resource Efficiency:**
Training requires massive compute power, often needing GPUs and high memory. It runs at 100% utilization for a few hours and then stops.
Inference requires very little compute power (usually just a CPU) but needs to be highly available 24/7 to answer web requests.
If you put them in the same container, you would be paying for an expensive GPU to run 24/7 just to serve a simple web request. By separating them, you use a massive machine for a short `Job` (training), and a tiny, cheap machine for a long-running `Deployment` (inference).

**2. The Shared Volume (PVC) bridge:**
Because containers are isolated, when your Training Job finishes and dies, all the data inside it is deleted. To pass the trained model to the Inference container, we use a **Persistent Volume Claim (PVC)**. Think of a PVC as an external USB flash drive in the cloud.

1. The Training K8s Job "plugs in" the USB drive, saves the model to it, and shuts down.
2. The Inference K8s Deployment "plugs in" the same USB drive, reads the model, and starts serving predictions.

**3. Interactive User Control:**
The homework emphasizes *User Interaction*. This means your inference container shouldn't just run a batch prediction script. It must host a web server (like Flask) so that you can open a browser, type in a URL, upload a picture of a dog, and receive the text "Dog" back from the model. K8s handles the routing of that URL directly to your container via the `Service` object.

---

## 6. Detailed Action Plan (Steps for Proceeding)

To actively work on this homework and produce all required deliverables, follow these concrete steps in order:

### Phase 1: Local Development & Verification

Before involving Docker or Kubernetes, ensure your Python code works locally.

1. **Setup Project Directory**
   * Create two folders: `training/` and `inference/`.
2. **Write Training Script (`training/train.py`)**
   * Pick a simple dataset (e.g., MNIST for images, or a tiny tabular dataset).
   * Write code to train a basic PyTorch/TensorFlow model.
   * Make sure it saves the model to a local directory (e.g., `model.h5` or `model.pt`).
3. **Write Inference Script (`inference/app.py`)**
   * Create a simple Flask or FastAPI app.
   * Add a `/predict` POST endpoint that loads the saved model and processes input data (e.g., a JSON payload or image file) to return predictions.
4. **Test Locally**
   * Run `python train.py` and verify `model.pt` is generated.
   * Start `python app.py`, send a test request using `curl` or Postman, and verify it returns a valid prediction.

### Phase 2: Containerization (Docker)

Package your applications to run anywhere.

1. **Create Training Dockerfile (`training/Dockerfile`)**
   * Base image: `python:3.9-slim` or a PyTorch/TF base image.
   * Copy `train.py` and `requirements.txt`.
   * Install dependencies.
   * Set the default command: `CMD ["python", "train.py"]`.
2. **Create Inference Dockerfile (`inference/Dockerfile`)**
   * Base image: `python:3.9-slim`.
   * Copy `app.py` and `requirements.txt`.
   * Install dependencies (including Flask/FastAPI).
   * Expose the web server port (e.g., `EXPOSE 5000`).
   * Set the default command: `CMD ["python", "app.py"]`.
3. **Build & Push Images**
   * Run `docker build -t your-dockerhub-username/hw3-train:v1 ./training`
   * Run `docker build -t your-dockerhub-username/hw3-inference:v1 ./inference`
   * Run `docker push your-dockerhub-username/hw3-train:v1`
   * Run `docker push your-dockerhub-username/hw3-inference:v1`

### Phase 3: Infrastructure Setup (GCP / GKE)

Set up the cloud environment.

1. **Create GKE Cluster**
   * Use Google Cloud Console or `gcloud` CLI to create a basic Kubernetes cluster.
2. **Configure `kubectl`**
   * Connect your local terminal to the GKE cluster: `gcloud container clusters get-credentials <cluster-name> --zone <zone> --project <project-id>`.
3. **Set up Persistent Storage (Optional but Recommended)**
   * Create a `pvc.yaml` to request a PersistentVolumeClaim from GCP.
   * Apply it: `kubectl apply -f pvc.yaml`.

### Phase 4: Kubernetes Deployment (YAMLs)

Write the orchestrations to run your containers in the cloud.

1. **Deploy Training Job (`job.yaml`)**
   * Define `kind: Job`.
   * Set the image to `your-dockerhub-username/hw3-train:v1`.
   * Mount the PVC so it saves the model to the shared drive.
   * Run: `kubectl apply -f job.yaml`.
   * Verify it completed: `kubectl get jobs` and `kubectl logs <job-pod-name>`.
2. **Deploy Inference Web Server (`deployment.yaml` & `service.yaml`)**
   * Define `kind: Deployment`. Set image to `your-dockerhub-username/hw3-inference:v1`. Mount the PVC so it can read the saved model.
   * Define `kind: Service`. Set type to `LoadBalancer` to get a public IP address.
   * Run: `kubectl apply -f deployment.yaml` and `kubectl apply -f service.yaml`.
3. **Test the Cloud System**
   * Get the public IP: `kubectl get svc`.
   * Send a test request to `http://<PUBLIC-IP>:5000/predict`.
   * **Take screenshots** of this interaction for your deliverables!

### Phase 5: Documentation & Submission

1. Gather the 4 YAML files (`pvc.yaml`, `job.yaml`, `deployment.yaml`, `service.yaml`).
2. Gather the 2 Dockerfiles.
3. Write the `README.md` containing run instructions and your architectural choices report.
4. Bundle everything (including screenshots) as per the assignment's final instructions.
