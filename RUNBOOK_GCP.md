# GCP Runbook: Applied ML on Cloud Project

This guide is for first-time GCP usage and is aligned with the current benchmark script.

## 0) What this run produces

Each benchmark invocation creates a new folder under outputs without overwriting old runs.

- outputs/run_001_<run_name>
- outputs/run_002_<run_name>

Inside each run folder:

- metrics.csv
- run_plan.json
- status.log
- progress.jsonl
- profiler/*.trace.json
- profiler/*.ops.txt
- gpu_samples/*.json
- loss_history/*.txt

## 1) One-time local setup (your laptop)

Install Google Cloud CLI and initialize it.

1. Install CLI (macOS):

   brew install --cask google-cloud-sdk

2. Login and set defaults:

   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   gcloud config set compute/zone us-central1-a

3. Enable required APIs once:

   gcloud services enable compute.googleapis.com

4. Verify account and config:

   gcloud auth list
   gcloud config list

## 2) Set common shell variables (recommended)

Run on your local terminal:

PROJECT_ID=YOUR_PROJECT_ID
ZONE=us-central1-a
T4_NAME=aml-t4
A100_NAME=aml-a100
CPU_NAME=aml-cpu

## 3) Create instances

Create T4:

gcloud compute instances create "$T4_NAME" \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --maintenance-policy=TERMINATE \
  --boot-disk-size=100GB \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release

Create A100:

gcloud compute instances create "$A100_NAME" \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  --machine-type=a2-highgpu-1g \
  --maintenance-policy=TERMINATE \
  --boot-disk-size=100GB \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release

Create CPU VM:

gcloud compute instances create "$CPU_NAME" \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  --machine-type=c2-standard-8 \
  --boot-disk-size=100GB \
  --image-family=pytorch-latest-cpu \
  --image-project=deeplearning-platform-release

Check created VMs:

gcloud compute instances list --project="$PROJECT_ID"

## 4) Copy code and dataset to VM (first-time friendly)

From local project folder, copy files recursively to each VM home directory.

cd "/Users/himanshujhawar/Desktop/Subjects/Applied ML on Cloud/First Project"

gcloud compute scp benchmark_roofline.py requirements.txt RUNBOOK_GCP.md \
  --project="$PROJECT_ID" --zone="$ZONE" "$T4_NAME":~

gcloud compute scp --recurse imagenet_subset \
  --project="$PROJECT_ID" --zone="$ZONE" "$T4_NAME":~

Repeat for A100 and CPU:

gcloud compute scp benchmark_roofline.py requirements.txt RUNBOOK_GCP.md \
  --project="$PROJECT_ID" --zone="$ZONE" "$A100_NAME":~
gcloud compute scp --recurse imagenet_subset \
  --project="$PROJECT_ID" --zone="$ZONE" "$A100_NAME":~

gcloud compute scp benchmark_roofline.py requirements.txt RUNBOOK_GCP.md \
  --project="$PROJECT_ID" --zone="$ZONE" "$CPU_NAME":~
gcloud compute scp --recurse imagenet_subset \
  --project="$PROJECT_ID" --zone="$ZONE" "$CPU_NAME":~

Verify files on T4:

gcloud compute ssh "$T4_NAME" --project="$PROJECT_ID" --zone="$ZONE" \
  --command "ls -lah ~ && ls -lah ~/imagenet_subset | head"

## 5) First-time setup inside each VM (use venv)

SSH into T4:

gcloud compute ssh "$T4_NAME" --project="$PROJECT_ID" --zone="$ZONE"

Inside VM:

cd ~
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
nvidia-smi

Repeat similarly for A100. On CPU VM, nvidia-smi is not expected.

## 6) Script configuration to keep

In benchmark_roofline.py keep:

- USE_REAL_DATA = True
- USE_REAL_VALIDATION = True
- REAL_DATA_DIR = "imagenet_subset/train"
- REAL_VAL_DIR = "imagenet_subset/val"

Recommended runtime length for overnight profile-first runs:

- warmup_iterations = 10
- iterations = 40
- total per config = 50

This means:

- Training timing from real data
- Validation metrics from real data
- Automatic fallback to real training if class mismatch is detected

Note: quick-check still uses tiny FakeData by design for fast setup validation.

## 7) Mandatory smoke test before overnight run

On each VM (inside venv):

python benchmark_roofline.py --quick-check
python benchmark_roofline.py --model resnet50 --quick-check

Optional real-data smoke checks:

python benchmark_roofline.py --quick-check-real
python benchmark_roofline.py --model resnet50 --quick-check-real

Full workflow smoke across all models/configs (tiny real subsets, very short):

python benchmark_roofline.py --quick-check-real-all

Pass criteria:

- Run completes without exception
- New outputs/run_00x_* folder appears
- metrics.csv, status.log, progress.jsonl, profiler, loss_history present

## 8) Overnight full run

Use tmux so run survives disconnect.

tmux new -s aml-run
source ~/venv/bin/activate
cd ~
python benchmark_roofline.py

Detach tmux:

Ctrl+b then d

Reattach later:

tmux attach -t aml-run

## 9) Monitoring while running

Latest run folder:

ls -td outputs/* | head -n 1

Watch logs:

tail -f outputs/<latest_run_dir>/status.log
tail -f outputs/<latest_run_dir>/progress.jsonl

GPU watch (GPU VMs only):

watch -n 1 nvidia-smi

## 10) Nsight roofline quick capture (optional)

Run on T4 and A100 once:

ncu --set full --section SpeedOfLight_RooflineChart \
  --target-processes all \
  --csv --log-file outputs/nsight_roofline_resnet50.csv \
  python benchmark_roofline.py --model resnet50 --quick-check

## 11) Download results back to local

Find latest run folder on each VM:

gcloud compute ssh "$T4_NAME" --project="$PROJECT_ID" --zone="$ZONE" \
  --command "ls -td outputs/* | head -n 1"
gcloud compute ssh "$A100_NAME" --project="$PROJECT_ID" --zone="$ZONE" \
  --command "ls -td outputs/* | head -n 1"
gcloud compute ssh "$CPU_NAME" --project="$PROJECT_ID" --zone="$ZONE" \
  --command "ls -td outputs/* | head -n 1"

Copy full run folders (recommended):

mkdir -p results_t4 results_a100 results_cpu

LATEST_T4=$(gcloud compute ssh "$T4_NAME" --project="$PROJECT_ID" --zone="$ZONE" --command "ls -td ~/outputs/* | head -n 1")
LATEST_A100=$(gcloud compute ssh "$A100_NAME" --project="$PROJECT_ID" --zone="$ZONE" --command "ls -td ~/outputs/* | head -n 1")
LATEST_CPU=$(gcloud compute ssh "$CPU_NAME" --project="$PROJECT_ID" --zone="$ZONE" --command "ls -td ~/outputs/* | head -n 1")

gcloud compute scp --recurse "$T4_NAME":"$LATEST_T4" ./results_t4 \
  --project="$PROJECT_ID" --zone="$ZONE"
gcloud compute scp --recurse "$A100_NAME":"$LATEST_A100" ./results_a100 \
  --project="$PROJECT_ID" --zone="$ZONE"
gcloud compute scp --recurse "$CPU_NAME":"$LATEST_CPU" ./results_cpu \
  --project="$PROJECT_ID" --zone="$ZONE"

If you want only metrics files:

gcloud compute scp "$T4_NAME":"$LATEST_T4"/metrics.csv ./metrics_t4.csv \
  --project="$PROJECT_ID" --zone="$ZONE"
gcloud compute scp "$A100_NAME":"$LATEST_A100"/metrics.csv ./metrics_a100.csv \
  --project="$PROJECT_ID" --zone="$ZONE"
gcloud compute scp "$CPU_NAME":"$LATEST_CPU"/metrics.csv ./metrics_cpu.csv \
  --project="$PROJECT_ID" --zone="$ZONE"

## 12) Cost safety after completion

Stop instances when not running:

gcloud compute instances stop "$T4_NAME" --project="$PROJECT_ID" --zone="$ZONE"
gcloud compute instances stop "$A100_NAME" --project="$PROJECT_ID" --zone="$ZONE"
gcloud compute instances stop "$CPU_NAME" --project="$PROJECT_ID" --zone="$ZONE"

## 13) Troubleshooting quick map

- gcloud auth errors: rerun gcloud auth login and check gcloud config list
- scp fails: confirm instance name, zone, and that local path exists
- torchvision missing: activate venv, then pip install -r requirements.txt
- OOM on GPU: reduce DEFAULT_BATCH_SIZES in benchmark_roofline.py
- Validation path errors: verify imagenet_subset/val exists on VM
- Long run interrupted: use tmux and rerun

## 14) Final pre-overnight checklist

1. gcloud auth and project config verified
2. VMs created and reachable by ssh
3. Code and dataset copied to each VM
4. venv created and requirements installed on each VM
5. quick-check passed on each VM
6. tmux session started before overnight run
