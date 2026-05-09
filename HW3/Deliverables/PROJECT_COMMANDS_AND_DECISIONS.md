# Project Commands and Decisions

This file records the commands used to complete the HW3 GKE workflow and the main implementation decisions, issues, and fixes.

<details>
<summary><strong>Commands</strong></summary>

## 1. Local Setup and Verification

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r Deliverables/training/requirements.txt
pip install -r Deliverables/inference/requirements.txt
```

Run training locally:

```bash
python3 Deliverables/training/train.py
```

Run inference locally:

```bash
MODEL_DIR=./model_data API_KEY=dev-secret-key python3 Deliverables/inference/app.py
```

Test local inference with curl:

```bash
curl -X POST http://localhost:5000/predict \
  -H "X-API-Key: dev-secret-key" \
  -F "file=@Deliverables/test_images/4.png"
```

## 2. GCP and GKE Cluster Access

Install the GKE auth plugin if `kubectl` cannot authenticate:

```bash
gcloud components install gke-gcloud-auth-plugin
```

Get cluster credentials:

```bash
gcloud container clusters get-credentials hw3-final \
  --zone us-east1-c \
  --project coms-amlc
```

Confirm the active cluster context:

```bash
kubectl config current-context
```

Inspect nodes and node labels:

```bash
kubectl get nodes --show-labels
```

Expected node layout:

```text
GPU node pool: default-pool, g2-standard-4, nvidia-l4
CPU node pool: cpu-pool, e2-standard-4
```

## 3. Private Node Internet Access with Cloud NAT

Because the project policy blocks VM external IPs, private nodes need Cloud NAT for outbound internet access.

Create Cloud Router:

```bash
gcloud compute routers create hw3-router-east \
  --region=us-east1 \
  --network=default
```

Create Cloud NAT:

```bash
gcloud compute routers nats create hw3-nat-east \
  --router=hw3-router-east \
  --region=us-east1 \
  --auto-allocate-nat-external-ips \
  --nat-all-subnet-ip-ranges
```

Optional internet test pod:

```bash
kubectl run internet-test \
  --image=busybox \
  --restart=Never \
  -- wget -qO- https://www.google.com
```

Check the test:

```bash
kubectl logs internet-test
kubectl delete pod internet-test --ignore-not-found
```

## 4. Docker Commands

These are the normal local Docker build commands:

```bash
docker build -t hj2713/hw3-train:v3 Deliverables/training
docker build -t hj2713/hw3-inference:v2 Deliverables/inference
```

Push images:

```bash
docker push hj2713/hw3-train:v3
docker push hj2713/hw3-inference:v2
```

Important note:

On Apple Silicon Macs, local builds may create ARM images that do not run on GKE Linux/amd64 nodes. If building locally, force Linux/amd64:

```bash
docker buildx build --platform linux/amd64 \
  -t hj2713/hw3-inference:v2 \
  --push Deliverables/inference
```

Because we hit a platform mismatch, we used Kaniko inside GKE for the final working images.

## 5. Docker Hub Pull Secret for Kaniko

Create a Docker config JSON locally after Docker login, or use the existing `docker-config.json` artifact if available.

Create the Kubernetes Secret used by Kaniko:

```bash
kubectl create secret generic docker-config \
  --from-file=config.json="HW Info and Logs/Kaniko_and_GCP_Logs/docker-config.json"
```

If the secret already exists and needs to be recreated:

```bash
kubectl delete secret docker-config --ignore-not-found
kubectl create secret generic docker-config \
  --from-file=config.json="HW Info and Logs/Kaniko_and_GCP_Logs/docker-config.json"
```

## 6. Build Training Image with Kaniko

Create the training build context:

```bash
kubectl delete configmap build-context --ignore-not-found
kubectl create configmap build-context \
  --from-file=Dockerfile=Deliverables/training/Dockerfile \
  --from-file=train.py=Deliverables/training/train.py \
  --from-file=requirements.txt=Deliverables/training/requirements.txt
```

Run Kaniko:

```bash
kubectl delete job kaniko-build --ignore-not-found
kubectl apply -f "HW Info and Logs/Kaniko_and_GCP_Logs/kaniko-job.yaml"
kubectl wait --for=condition=complete job/kaniko-build --timeout=15m
kubectl logs job/kaniko-build
```

Final training image:

```text
hj2713/hw3-train:v3
```

## 7. Deploy and Run Training on GKE

Create the PVC:

```bash
kubectl apply -f Deliverables/k8s/pvc.yaml
```

Run the training job:

```bash
kubectl delete job mnist-training-job --ignore-not-found
kubectl apply -f Deliverables/k8s/training-job.yaml
```

Watch logs:

```bash
kubectl logs -f job/mnist-training-job
```

Check status:

```bash
kubectl get jobs,pods,pvc
kubectl describe job mnist-training-job
kubectl describe pod -l job-name=mnist-training-job
```

Expected successful log line:

```text
Training complete! Model saved successfully to: /mnt/model/mnist_model.pt
```

## 8. Build Inference Image with Kaniko

Create the inference build context:

```bash
kubectl delete configmap inference-build-context --ignore-not-found
kubectl create configmap inference-build-context \
  --from-file=Dockerfile=Deliverables/inference/Dockerfile \
  --from-file=app.py=Deliverables/inference/app.py \
  --from-file=requirements.txt=Deliverables/inference/requirements.txt \
  --from-file=index.html=Deliverables/inference/templates/index.html
```

Run Kaniko:

```bash
kubectl delete job kaniko-inference-build --ignore-not-found
kubectl apply -f "HW Info and Logs/Kaniko_and_GCP_Logs/kaniko-inference-job.yaml"
kubectl wait --for=condition=complete job/kaniko-inference-build --timeout=15m
kubectl logs job/kaniko-inference-build
```

Final inference image:

```text
hj2713/hw3-inference:v2
```

## 9. Deploy Inference on GKE

Apply the API key Secret:

```bash
kubectl apply -f Deliverables/k8s/secret.yaml
```

Deploy inference:

```bash
kubectl delete deployment mnist-inference --ignore-not-found
kubectl apply -f Deliverables/k8s/deployment.yaml
```

Expose it:

```bash
kubectl apply -f Deliverables/k8s/service.yaml
```

Check pod and service status:

```bash
kubectl get pods,svc
kubectl describe pod -l app=mnist-inference
kubectl get svc mnist-inference-service
```

Expected pod status:

```text
mnist-inference-...   1/1   Running
```

Expected service:

```text
mnist-inference-service   LoadBalancer   ...   <EXTERNAL-IP>   80:...
```

## 10. Test Inference

Browser test:

```text
http://<EXTERNAL-IP>
```

Use this API key in the UI:

```text
my-secure-api-key-2026
```

curl test:

```bash
curl -X POST http://<EXTERNAL-IP>/predict \
  -H "X-API-Key: my-secure-api-key-2026" \
  -F "file=@Deliverables/test_images/4.png"
```

More curl examples:

```bash
curl -X POST http://<EXTERNAL-IP>/predict \
  -H "X-API-Key: my-secure-api-key-2026" \
  -F "file=@Deliverables/test_images/8.jpeg"
```

```bash
curl -X POST http://<EXTERNAL-IP>/predict \
  -H "X-API-Key: my-secure-api-key-2026" \
  -F "file=@Deliverables/test_images/9.webp"
```

Verify authentication blocks missing keys:

```bash
curl -X POST http://<EXTERNAL-IP>/predict \
  -F "file=@Deliverables/test_images/4.png"
```

Expected unauthorized response:

```json
{ "error": "Unauthorized. Invalid or missing X-API-Key header." }
```

Port-forward fallback if the LoadBalancer IP is unavailable:

```bash
kubectl port-forward deployment/mnist-inference 5000:5000
```

Then test locally:

```bash
curl -X POST http://localhost:5000/predict \
  -H "X-API-Key: my-secure-api-key-2026" \
  -F "file=@Deliverables/test_images/4.png"
```

## 11. Useful Debugging Commands

General status:

```bash
kubectl get all
kubectl get jobs,pods,pvc,svc
```

Training logs:

```bash
kubectl logs job/mnist-training-job
```

Inference logs:

```bash
kubectl logs deployment/mnist-inference
```

Describe failed resources:

```bash
kubectl describe pod -l job-name=mnist-training-job
kubectl describe pod -l app=mnist-inference
kubectl describe svc mnist-inference-service
```

Decode the API key locally:

```bash
echo 'bXktc2VjdXJlLWFwaS1rZXktMjAyNg==' | base64 --decode
```

Clean up workloads if needed:

```bash
kubectl delete job mnist-training-job --ignore-not-found
kubectl delete deployment mnist-inference --ignore-not-found
kubectl delete svc mnist-inference-service --ignore-not-found
```

</details>

<details>
<summary><strong>Important Choices, Decisions, and Edge Cases</strong></summary>

## 1. Why MNIST?

MNIST was selected because the homework asks for a simple DL workflow, not a novel model. MNIST lets the project focus on the cloud-native workflow:

- containerized training
- persistent model handoff
- Kubernetes controllers
- hosted inference URL
- interactive user input

## 2. Why a Kubernetes Job for Training?

Training starts, runs for a fixed number of epochs, saves a model, and exits. A `Job` is designed for this lifecycle.

Benefits:

- run-to-completion semantics
- retry behavior through `backoffLimit`
- avoids keeping GPU resources allocated after training
- clean logs for training evidence

## 3. Why a Kubernetes Deployment for Inference?

Inference is a web service and should remain alive. A `Deployment` is appropriate because it manages a ReplicaSet and keeps the desired number of pods running.

Benefits:

- automatic restart
- stable desired state
- rolling updates
- health probes

## 4. Why a PVC?

Without a PVC, the model saved by the training pod would disappear when the pod exits. The PVC acts as the bridge between training and inference:

- training mounts it read-write
- inference mounts it read-only
- the model file survives pod lifecycle events

## 5. Why Separate GPU and CPU Node Pools?

Training benefits from GPU acceleration, but inference for MNIST is lightweight and runs well on CPU. Separating node pools avoids wasting GPU resources on the always-running inference service.

Final node usage:

- training: GPU node pool with NVIDIA L4
- inference: CPU node pool named `cpu-pool`

## 6. Why Kubernetes Secret?

The `/predict` route requires an API key. The key is stored in `inference-secret` and injected as an environment variable.

This avoids baking the real key into the image. It also demonstrates a standard Kubernetes pattern for runtime secrets.

Furthermore, since the inference service is exposed to the internet via a public URL, implementing this API key authentication ensures our endpoint remains secure.

Even if the public URL is discovered by an unauthorized party, they cannot hit the API or abuse our resources without the secret key, which is known only to us.

## 7. Input Images

The model is trained on MNIST images. It performs well on MNIST-like inputs:

- white or light digit
- dark background
- centered digit
- close to the 28x28 MNIST style

It can fail on real marker-on-paper photos because those images have a different distribution:

- black digit
- light paper background
- shadows or camera artifacts
- different crop/scale

For this homework, we kept the inference preprocessing simple and used MNIST-style test images for a fair demo.

## 8. Cloud NAT Issue

The GCP organization policy prevented VM external IPs:

```text
constraints/compute.vmExternalIpAccess
```

Because the GKE nodes were private, they needed Cloud NAT for outbound internet. This mattered because the training job downloads MNIST during execution.

Resolution:

- created Cloud Router
- created Cloud NAT
- confirmed the training job could download MNIST from the fallback S3 mirror

## 9. Using both GPU & CPU

- use GPU for training to demonstrate hardware acceleration
- schedule inference on CPU to reduce cost and resource usage

## 10. Training Image Missing `train.py`

Initial symptom:

```text
python: can't open file '/app/train.py': [Errno 2] No such file or directory
```

Cause:

- the training image was built from an incomplete Kaniko ConfigMap context

Fix:

- recreated the ConfigMap with `Dockerfile`, `train.py`, and `requirements.txt`
- rebuilt the image as `hj2713/hw3-train:v3`
- added `imagePullPolicy: Always`

## 11. Why Kaniko?

Kaniko builds container images inside Kubernetes without requiring Docker daemon access. It solved two project issues:

- ensured the build context included the correct files
- produced images compatible with GKE nodes

## 12. Why only 10 Epochs?

The model produced good predictions on MNIST-style images with 10 epochs. Increasing to 100 or 200 epochs is unnecessary for this assignment and would increase training time without improving the cloud architecture demonstration.

## 13. What We Would Improve in Production

For a real public service, the following would be added:

- HTTPS through Ingress
- rate limiting
- stronger user authentication
- secret rotation
- model versioning
- structured metrics
- a real model registry instead of a PVC
- preprocessing tuned for phone/camera uploads

</details>
