# A100 Re-Run Instructions

> **Why are we re-running?** The previous `load_test.py` had a token counting bug — it counted SSE streaming chunks instead of actual tokens. With speculative decoding, each chunk contains multiple tokens, so all EAGLE-3 metrics (TPS, TPOT, Cost) were wrong. This has been fixed.

---

## Steps (Copy-paste in order)

### 1. SSH into your A100 instance
```bash
gcloud compute ssh <YOUR_INSTANCE_NAME> --zone=<YOUR_ZONE>
```

### 2. Go to the project directory
```bash
cd /mnt/disks/bigdisk/Applied-ML-on-Cloud/Project2/code
```

### 3. Kill any leftover GPU processes (prevents OOM)
```bash
pkill -9 -f vllm 2>/dev/null || true
pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
# Verify GPU memory is free:
nvidia-smi
# You should see ~0 MiB used. If not, run: kill -9 <PID> using the PID from nvidia-smi
```

### 4. Pull the fixed code
```bash
git pull origin main
```

### 5. Delete old (corrupted) A100 results only
```bash
rm -f results/raw/*_A100_*.jsonl
```

### 6. Activate venv
```bash
source .venv/bin/activate
# If .venv doesn't exist, run setup first:
# bash infra/setup_gcp.sh
```

### 7. Run the full experiment (~2 hours)
```bash
bash benchmark/run_experiment.sh A100
```

This will automatically:
- Start baseline server → run all sweeps → kill it
- Start EAGLE-3 server → run all sweeps → kill it
- Save all results to `results/raw/`

**You can leave it running and come back later.**

### 8. Push results
```bash
git add results/raw/
git commit -m "feat: add A100 benchmark results (fixed token counting)"
git push origin main
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `nvidia-smi` still shows memory used after pkill | Run `kill -9 <PID>` with the specific PID shown |
| Server fails to start (OOM) | Run step 3 again, wait 10 seconds, retry |
| `data/prompts/` is empty | Run `python3 data/prepare_datasets.py --n-samples 50` |
| Script errors on `git pull` | Run `git stash` first, then `git pull` |
