# Screenshot and Evidence Index

| File | Evidence shown | Rubric item |
|---|---|---|
| `01_gke_cluster_running_overview.png` | GKE cluster `hw3-final` running in Google Cloud Console | Created k8s cluster |
| `02_gke_gpu_cpu_node_pools.png` | CPU and GPU node pools visible in GKE | Created k8s cluster, training infrastructure |
| `03_kubectl_get_nodes_ready.png` | `kubectl get nodes` shows both nodes ready | Created k8s cluster |
| `04_kubectl_get_jobs_complete.png` | Kaniko build jobs and training job completed | Docker build/push, training in k8s |
| `05_kubectl_get_jobs_pods_status.png` | Training complete, inference running, build jobs complete | Training and inference in k8s |
| `06_kubectl_get_pvc_bound.png` | `model-pvc` exists and is bound | PVC creation/preparation |
| `07_kubectl_get_pods_svc_external_ip.png` | Inference pod running and LoadBalancer external IP assigned | Inference deployment/service |
| `08_app_demo_browser_prediction.mp4` | Full browser demo of interactive inference | Infer-test |
| `08_app_demo_browser_prediction_frame.png` | Still frame from the app demo showing prediction | Infer-test |
| `09_curl_inference_test.png` | curl prediction succeeds and wrong API key fails | Infer-test, authentication |
| `10_docker_push_training_image.png` | Kaniko pushes `hj2713/hw3-train:v3` | Training Docker image push |
| `11_inference_logs_docker_pushed.txt` | Kaniko pushes `hj2713/hw3-inference:v2` | Inference Docker image push |
| `12_training_logs.txt` | Training uses CUDA and saves `/mnt/model/mnist_model.pt` | Model saving |

Text log files are included where the relevant output was too long for one screenshot.
