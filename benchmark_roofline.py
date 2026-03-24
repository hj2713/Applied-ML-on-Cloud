#!/usr/bin/env python3
"""Benchmark script for roofline-oriented training profiling.

This script runs short training workloads for vision models and records metrics
needed for later roofline analysis across different hardware environments.

Supported models:
- resnet50
- vit_b_16
- vgg16

Supported datasets:
- fake: torchvision FakeData (recommended for primary profiling)
- imagefolder: real data in ImageFolder layout

Dataset mode selection:
- Controlled by constants at the top of this file (`USE_REAL_DATA`, `REAL_DATA_DIR`)
- Quick-check always uses tiny FakeData regardless of `USE_REAL_DATA`

Run:
python benchmark_roofline.py
python benchmark_roofline.py --model resnet50
python benchmark_roofline.py --model vit_b_16
python benchmark_roofline.py --model vgg16
python benchmark_roofline.py --quick-check
python benchmark_roofline.py --quick-check-real         # it runs on default model , only 1
python benchmark_roofline.py --quick-check-real-all     # it runs on all models, but with tiny real-data subsets and short iterations for fast validation of end-to-end pipeline and artifact generation
python benchmark_roofline.py --model resnet50 --quick-check
python benchmark_roofline.py --model resnet50 --quick-check-real

Runtime modes:
- Full sweep: runs configured model x batch-size x precision matrix
- Single model: use `--model` to run one model only
- Quick check: fast smoke test (1 model, batch size 8, fp32, tiny FakeData,
  short iterations) and exports profiler artifacts
- Quick check real: fast smoke test on real train/val paths with tiny
    iterations to validate end-to-end data pipeline
- Quick check real all: runs all models/configs on tiny real-data subsets
    to validate complete artifact generation quickly

Per-run outputs (auto-generated in a sequential folder):
- metrics.csv
- run_plan.json
- status.log
- progress.jsonl
- profiler/*.trace.json and profiler/*.ops.txt
- gpu_samples/*.json (nvidia-smi samples per configuration)
- loss_history/*.txt (Epoch-style loss logs per configuration)

Configuration is controlled by constants at the top of this file.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import itertools
import json
import math
from contextlib import nullcontext
import platform
import random
import shutil
import socket
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms # type: ignore


SUPPORTED_MODELS = {
    "resnet50": models.resnet50,
    "vit_b_16": models.vit_b_16,
    "vgg16": models.vgg16,
}

DEFAULT_MODELS = ["resnet50", "vit_b_16", "vgg16"]
DEFAULT_BATCH_SIZES = [32, 128]
DEFAULT_PRECISIONS = ["fp32", "fp16"]
DEFAULT_OUTPUT_ROOT = "outputs"

# Dataset selection is constant-driven (no CLI flag needed).
# Full runs default to real-data training for simpler interpretation.
# Quick-check still uses tiny FakeData for fast smoke testing.
USE_REAL_DATA = True
REAL_DATA_DIR = "imagenet_subset/train"
FAKE_DATASET_SIZE = 12000
NUM_CLASSES = 1000
USE_REAL_VALIDATION = True
REAL_VAL_DIR = "imagenet_subset/val"
VALIDATION_MAX_BATCHES = 0
QUICK_CHECK_VALIDATION_MAX_BATCHES = 8
QUICK_CHECK_REAL_VALIDATION_MAX_BATCHES = 2
QUICK_CHECK_REAL_TRAIN_MAX_SAMPLES = 10
QUICK_CHECK_REAL_VAL_MAX_SAMPLES = 10
AUTO_FALLBACK_TO_REAL_TRAIN_ON_CLASS_MISMATCH = True


@dataclass(frozen=True)
class ExperimentSettings:
    models: List[str]
    batch_sizes: List[int]
    precisions: List[str]
    iterations: int
    warmup_iterations: int
    num_workers: int
    lr: float
    seed: int
    run_name: str
    output_root: str
    log_every: int
    force_cpu: bool
    max_runs: int


@dataclass(frozen=True)
class RuntimeOptions:
    model: str
    quick_check: bool
    quick_check_real: bool
    quick_check_real_all: bool


DEFAULT_SETTINGS = ExperimentSettings(
    models=DEFAULT_MODELS,
    batch_sizes=DEFAULT_BATCH_SIZES,
    precisions=DEFAULT_PRECISIONS,
    iterations=40,
    warmup_iterations=10,
    num_workers=4,
    lr=0.01,
    seed=42,
    run_name="baseline",
    output_root=DEFAULT_OUTPUT_ROOT,
    log_every=10,
    force_cpu=False,
    max_runs=0,
)


@dataclass(frozen=True)
class RunConfig:
    model_name: str
    batch_size: int
    precision: str


@dataclass
class GPUSample:
    utilization_gpu: float
    utilization_mem: float
    memory_used_mb: float
    memory_total_mb: float
    temperature_c: float


class TinyImageNetValDataset(Dataset):
    """Dataset adapter for Tiny-ImageNet val layout.

    Expected files:
    - <val_root>/images/*.JPEG
    - <val_root>/val_annotations.txt
    """

    def __init__(self, val_root: Path, class_to_idx: Dict[str, int], transform: transforms.Compose):
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []
        self.num_classes = 0

        images_dir = val_root / "images"
        annotations_file = val_root / "val_annotations.txt"
        if not images_dir.exists() or not annotations_file.exists():
            raise ValueError(f"Tiny-ImageNet val layout not found at: {val_root}")

        for line in annotations_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            file_name, wnid = parts[0], parts[1]
            label = class_to_idx.get(wnid)
            if label is None:
                continue
            image_path = images_dir / file_name
            if image_path.exists():
                self.samples.append((image_path, label))

        if not self.samples:
            raise ValueError(f"No validation samples found at: {val_root}")

        self.num_classes = len({label for _, label in self.samples})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, target = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(force_cpu: bool) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_gpu_name() -> str:
    if not torch.cuda.is_available():
        return "cpu-only"
    return torch.cuda.get_device_name(0)


def run_cmd(cmd: List[str]) -> Optional[str]:
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception:
        return None


def query_nvidia_smi() -> Optional[GPUSample]:
    if shutil.which("nvidia-smi") is None:
        return None

    out = run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    if not out:
        return None

    try:
        first = out.splitlines()[0]
        parts = [p.strip() for p in first.split(",")]
        return GPUSample(
            utilization_gpu=float(parts[0]),
            utilization_mem=float(parts[1]),
            memory_used_mb=float(parts[2]),
            memory_total_mb=float(parts[3]),
            temperature_c=float(parts[4]),
        )
    except Exception:
        return None


def build_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_val_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_dataset():
    tfm = build_transforms()

    if not USE_REAL_DATA:
        return datasets.FakeData(
            size=FAKE_DATASET_SIZE,
            image_size=(3, 224, 224),
            num_classes=NUM_CLASSES,
            transform=tfm,
        )

    if not REAL_DATA_DIR:
        raise ValueError("REAL_DATA_DIR is required when USE_REAL_DATA=True")

    train_dir = Path(REAL_DATA_DIR)
    if not train_dir.exists():
        raise ValueError(f"Data directory does not exist: {train_dir}")

    return datasets.ImageFolder(root=str(train_dir), transform=tfm)


def build_dataset_for_mode(use_real_data: bool, fake_size: int, fake_num_classes: int):
    tfm = build_transforms()

    if not use_real_data:
        return datasets.FakeData(
            size=fake_size,
            image_size=(3, 224, 224),
            num_classes=fake_num_classes,
            transform=tfm,
        )

    if not REAL_DATA_DIR:
        raise ValueError("REAL_DATA_DIR is required when USE_REAL_DATA=True")

    train_dir = Path(REAL_DATA_DIR)
    if not train_dir.exists():
        raise ValueError(f"Data directory does not exist: {train_dir}")

    return datasets.ImageFolder(root=str(train_dir), transform=tfm)


def build_real_train_dataset() -> datasets.ImageFolder:
    train_dir = Path(REAL_DATA_DIR)
    if not train_dir.exists():
        raise ValueError(f"Train data directory does not exist: {train_dir}")
    return datasets.ImageFolder(root=str(train_dir), transform=build_transforms())


def build_validation_dataset(class_to_idx: Optional[Dict[str, int]] = None):
    if not USE_REAL_VALIDATION:
        return None

    val_dir = Path(REAL_VAL_DIR)
    if not val_dir.exists():
        raise ValueError(f"Validation data directory does not exist: {val_dir}")

    val_tfm = build_val_transforms()
    tiny_images_dir = val_dir / "images"
    tiny_annotations = val_dir / "val_annotations.txt"
    if tiny_images_dir.exists() and tiny_annotations.exists():
        mapping = class_to_idx if class_to_idx is not None else build_real_train_dataset().class_to_idx
        return TinyImageNetValDataset(val_root=val_dir, class_to_idx=mapping, transform=val_tfm)

    return datasets.ImageFolder(root=str(val_dir), transform=val_tfm)


def infer_num_classes(dataset) -> int:
    if isinstance(dataset, Subset):
        return infer_num_classes(dataset.dataset)

    if hasattr(dataset, "classes"):
        return len(dataset.classes)

    if hasattr(dataset, "num_classes"):
        num_classes = getattr(dataset, "num_classes")
        if isinstance(num_classes, int) and num_classes > 0:
            return num_classes

    if hasattr(dataset, "samples"):
        try:
            labels = {int(sample[1]) for sample in dataset.samples}
            if labels:
                return len(labels)
        except Exception:
            pass

    return NUM_CLASSES


def limit_dataset_samples(dataset, max_samples: int):
    if max_samples <= 0:
        return dataset
    size = len(dataset)
    if size <= max_samples:
        return dataset
    return Subset(dataset, list(range(max_samples)))


def get_dataset_mode() -> str:
    return "imagefolder" if USE_REAL_DATA else "fake"


def get_dataset_dir() -> str:
    return REAL_DATA_DIR if USE_REAL_DATA else ""


def get_effective_dataset_mode(options: RuntimeOptions) -> str:
    if options.quick_check:
        return "fake"
    if options.quick_check_real or options.quick_check_real_all:
        return "imagefolder"
    return get_dataset_mode()


def get_validation_dataset_mode() -> str:
    return "real" if USE_REAL_VALIDATION else "none"


def get_effective_dataset_dir(options: RuntimeOptions) -> str:
    if options.quick_check:
        return ""
    if options.quick_check_real or options.quick_check_real_all:
        return REAL_DATA_DIR
    return get_dataset_dir()


def get_validation_dataset_dir() -> str:
    return REAL_VAL_DIR if USE_REAL_VALIDATION else ""


def evaluate_validation(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    precision: str,
    max_batches: int,
) -> Dict[str, float]:
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_samples = 0
    correct_top1 = 0
    correct_top5 = 0
    batches_done = 0

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break

            images = images.to(device, non_blocking=(device.type == "cuda"))
            targets = targets.to(device, non_blocking=(device.type == "cuda"))

            with get_autocast_context(device, precision):
                outputs = model(images)
                loss = criterion(outputs, targets)

            if device.type == "cuda":
                torch.cuda.synchronize()

            current_bs = targets.size(0)
            total_loss += float(loss.detach().cpu()) * current_bs
            total_samples += current_bs

            top_k = min(5, outputs.size(1))
            _, pred = outputs.topk(top_k, dim=1, largest=True, sorted=True)
            matches = pred.eq(targets.view(-1, 1))
            correct_top1 += int(matches[:, :1].sum().item())
            correct_top5 += int(matches.any(dim=1).sum().item())
            batches_done += 1

    if was_training:
        model.train()

    if total_samples == 0:
        return {
            "val_loss": float("nan"),
            "val_top1_acc": float("nan"),
            "val_top5_acc": float("nan"),
            "val_samples": 0.0,
            "val_batches": float(batches_done),
        }

    return {
        "val_loss": total_loss / total_samples,
        "val_top1_acc": (correct_top1 / total_samples) * 100.0,
        "val_top5_acc": (correct_top5 / total_samples) * 100.0,
        "val_samples": float(total_samples),
        "val_batches": float(batches_done),
    }


def export_profiler_artifacts(
    prof: torch.profiler.profile,
    profiler_dir: Path,
    model_name: str,
    batch_size: int,
    precision: str,
) -> None:
    profiler_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{model_name}_bs{batch_size}_{precision}"

    trace_file = profiler_dir / f"{stem}.trace.json"
    table_file = profiler_dir / f"{stem}.ops.txt"

    try:
        prof.export_chrome_trace(str(trace_file))
    except Exception:
        pass

    try:
        table = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=80)
        table_file.write_text(table, encoding="utf-8")
    except Exception:
        pass


def build_model(model_name: str, num_classes: int) -> nn.Module:
    model_ctor = SUPPORTED_MODELS.get(model_name)
    if model_ctor is None:
        raise ValueError(f"Unsupported model: {model_name}")

    model = model_ctor(weights=None)

    # Ensure classifier head matches chosen class count.
    if model_name.startswith("resnet"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name.startswith("vgg"):
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif model_name.startswith("vit"):
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

    return model


def maybe_amp_enabled(device: torch.device, precision: str) -> bool:
    return device.type == "cuda" and precision == "fp16"


def get_autocast_context(device: torch.device, precision: str):
    if maybe_amp_enabled(device, precision):
        amp_module = getattr(torch, "amp", None)
        if amp_module is not None and hasattr(amp_module, "autocast"):
            return amp_module.autocast("cuda", dtype=torch.float16)
        return torch.cuda.amp.autocast(dtype=torch.float16)
    return nullcontext()


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat()


def format_seconds(seconds: float) -> str:
    seconds = max(seconds, 0.0)
    mins, secs = divmod(int(seconds), 60)
    hrs, mins = divmod(mins, 60)
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    sorted_values = sorted(values)
    idx = max(0, min(len(sorted_values) - 1, int(math.ceil((p / 100.0) * len(sorted_values)) - 1)))
    return sorted_values[idx]


def emit(message: str, status_log: Optional[Path] = None) -> None:
    print(message)
    if status_log is None:
        return
    status_log.parent.mkdir(parents=True, exist_ok=True)
    with status_log.open("a", encoding="utf-8") as f:
        f.write(f"{utc_now_iso()} {message}\n")


def append_jsonl(path: Optional[Path], payload: Dict[str, object]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def create_next_run_dir(output_root: str, run_name: str) -> Path:
    """Create monotonic run folders to avoid overwrites and ease debugging.

    Folder format:
    - outputs/run_001_<run_name>
    - outputs/run_002_<run_name>
    """
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)

    max_idx = 0
    for child in root.iterdir():
        if not child.is_dir():
            continue
        parts = child.name.split("_", 2)
        if len(parts) < 2 or parts[0] != "run":
            continue
        if parts[1].isdigit():
            max_idx = max(max_idx, int(parts[1]))

    next_idx = max_idx + 1
    folder = root / f"run_{next_idx:03d}_{run_name}"
    folder.mkdir(parents=True, exist_ok=False)
    return folder


def write_gpu_samples(
    gpu_samples_dir: Optional[Path],
    model_name: str,
    batch_size: int,
    precision: str,
    samples: List[GPUSample],
) -> None:
    if gpu_samples_dir is None:
        return
    gpu_samples_dir.mkdir(parents=True, exist_ok=True)
    out_path = gpu_samples_dir / f"{model_name}_bs{batch_size}_{precision}.json"
    payload = {
        "model": model_name,
        "batch_size": batch_size,
        "precision": precision,
        "sample_count": len(samples),
        "samples": [
            {
                "utilization_gpu": s.utilization_gpu,
                "utilization_mem": s.utilization_mem,
                "memory_used_mb": s.memory_used_mb,
                "memory_total_mb": s.memory_total_mb,
                "temperature_c": s.temperature_c,
            }
            for s in samples
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_loss_history(
    loss_history_dir: Optional[Path],
    model_name: str,
    batch_size: int,
    precision: str,
    run_started_at: dt.datetime,
    losses: List[float],
    train_samples_measured: int,
    train_samples_total: int,
    val_samples_evaluated: int,
    val_loss: float,
    val_top1_acc: float,
    val_top5_acc: float,
) -> None:
    if loss_history_dir is None:
        return

    loss_history_dir.mkdir(parents=True, exist_ok=True)
    out_path = loss_history_dir / f"{model_name}_bs{batch_size}_{precision}.txt"

    if not out_path.exists():
        header = [
            f"Model: {model_name}",
            f"BatchSize: {batch_size}",
            f"Precision: {precision}",
            f"RunStartedAtUTC: {run_started_at.isoformat()}",
            "LiveEpochProgress:",
        ]
        out_path.write_text("\n".join(header) + "\n", encoding="utf-8")

    with out_path.open("a", encoding="utf-8") as f:
        f.write("\nFinalSummary:\n")
        f.write(f"MeasuredEpochs: {len(losses)}\n")
        f.write(f"TrainSamplesMeasured: {train_samples_measured}\n")
        f.write(f"TrainSamplesTotalWithWarmup: {train_samples_total}\n")
        f.write(f"ValSamplesEvaluated: {val_samples_evaluated}\n")
        f.write(f"ValLoss: {val_loss}\n")
        f.write(f"ValTop1Acc: {val_top1_acc}\n")
        f.write(f"ValTop5Acc: {val_top5_acc}\n")


def append_loss_history_progress(
    loss_history_dir: Optional[Path],
    model_name: str,
    batch_size: int,
    precision: str,
    epoch_index: int,
    loss_value: float,
    epoch_time_s: float,
    elapsed_from_model_start_s: float,
) -> None:
    if loss_history_dir is None:
        return

    loss_history_dir.mkdir(parents=True, exist_ok=True)
    out_path = loss_history_dir / f"{model_name}_bs{batch_size}_{precision}.txt"

    if not out_path.exists():
        header = [
            f"Model: {model_name}",
            f"BatchSize: {batch_size}",
            f"Precision: {precision}",
            "LiveEpochProgress:",
        ]
        out_path.write_text("\n".join(header) + "\n", encoding="utf-8")

    with out_path.open("a", encoding="utf-8") as f:
        f.write(
            f"Epoch:{epoch_index} Loss:{loss_value:.6f} EpochTimeS:{epoch_time_s:.4f} "
            f"ElapsedFromModelStartS:{elapsed_from_model_start_s:.4f}\n"
        )


def profile_run(
    settings: ExperimentSettings,
    options: RuntimeOptions,
    run_cfg: RunConfig,
    device: torch.device,
    output_csv: Path,
    run_started_at: dt.datetime,
    status_log: Optional[Path],
    progress_jsonl: Optional[Path],
    profiler_dir: Optional[Path],
    gpu_samples_dir: Optional[Path],
    loss_history_dir: Optional[Path],
) -> Dict[str, object]:
    precision = run_cfg.precision
    model_name = run_cfg.model_name
    batch_size = run_cfg.batch_size

    if precision == "fp16" and device.type != "cuda":
        raise RuntimeError("fp16 is only supported on CUDA devices in this script")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    use_real_data = options.quick_check_real or options.quick_check_real_all or (USE_REAL_DATA and not options.quick_check)
    fake_size = 256 if options.quick_check else FAKE_DATASET_SIZE
    val_dataset = None
    val_num_classes: Optional[int] = None

    if USE_REAL_VALIDATION:
        train_class_to_idx: Optional[Dict[str, int]] = None
        try:
            train_class_to_idx = build_real_train_dataset().class_to_idx
        except Exception:
            train_class_to_idx = None
        val_dataset = build_validation_dataset(class_to_idx=train_class_to_idx)
        val_num_classes = infer_num_classes(val_dataset)

    fake_num_classes = val_num_classes if (not use_real_data and val_num_classes is not None) else NUM_CLASSES
    dataset = build_dataset_for_mode(
        use_real_data=use_real_data,
        fake_size=fake_size,
        fake_num_classes=fake_num_classes,
    )

    train_num_classes = infer_num_classes(dataset)
    emit(
        f"[INFO] Class check: train_classes={train_num_classes}, val_classes={val_num_classes}",
        status_log,
    )
    fallback_used = False
    fallback_reason = ""
    if val_num_classes is not None and train_num_classes != val_num_classes:
        if not use_real_data and AUTO_FALLBACK_TO_REAL_TRAIN_ON_CLASS_MISMATCH:
            dataset = build_dataset_for_mode(
                use_real_data=True,
                fake_size=fake_size,
                fake_num_classes=NUM_CLASSES,
            )
            use_real_data = True
            train_num_classes = infer_num_classes(dataset)
            fallback_used = True
            fallback_reason = (
                f"train_classes={train_num_classes} did not match val_classes={val_num_classes}; switched to real train"
            )
            emit(f"[WARN] {fallback_reason}", status_log)
        else:
            raise RuntimeError(
                "Training/validation class mismatch. "
                f"train_classes={train_num_classes}, val_classes={val_num_classes}. "
                "Align class counts in train/validation data."
            )

    if options.quick_check_real or options.quick_check_real_all:
        dataset = limit_dataset_samples(dataset, QUICK_CHECK_REAL_TRAIN_MAX_SAMPLES)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=settings.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(settings.num_workers > 0),
        drop_last=not (options.quick_check_real or options.quick_check_real_all),
    )

    val_loader = None
    if options.quick_check_real or options.quick_check_real_all:
        val_max_batches = QUICK_CHECK_REAL_VALIDATION_MAX_BATCHES
    else:
        val_max_batches = QUICK_CHECK_VALIDATION_MAX_BATCHES if options.quick_check else VALIDATION_MAX_BATCHES
    if val_dataset is not None:
        if options.quick_check_real or options.quick_check_real_all:
            val_dataset = limit_dataset_samples(val_dataset, QUICK_CHECK_REAL_VAL_MAX_SAMPLES)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=settings.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(settings.num_workers > 0),
            drop_last=False,
        )

    model = build_model(model_name, train_num_classes).to(device)
    model.train()

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=settings.lr, momentum=0.9)
    amp_module = getattr(torch, "amp", None)
    if amp_module is not None and hasattr(amp_module, "GradScaler"):
        scaler = amp_module.GradScaler("cuda", enabled=maybe_amp_enabled(device, precision))
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=maybe_amp_enabled(device, precision))

    total_iterations = settings.warmup_iterations + settings.iterations
    iter_times: List[float] = []
    losses: List[float] = []
    gpu_samples: List[GPUSample] = []
    model_start_perf = time.perf_counter()

    prof = torch.profiler.profile(
        activities=(
            [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            if device.type == "cuda"
            else [torch.profiler.ProfilerActivity.CPU]
        ),
        acc_events=True,
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
        with_stack=False,
    )

    emit(
        f"\n[RUN START] model={model_name} batch_size={batch_size} precision={precision} "
        f"train_dataset={'imagefolder' if use_real_data else 'fake'} "
        f"val_dataset={get_validation_dataset_mode()} device={device.type}",
        status_log,
    )
    append_jsonl(
        progress_jsonl,
        {
            "timestamp_utc": utc_now_iso(),
            "event": "run_start",
            "model": model_name,
            "batch_size": batch_size,
            "precision": precision,
            "train_dataset": "imagefolder" if use_real_data else "fake",
            "val_dataset": get_validation_dataset_mode(),
            "device": device.type,
            "run_name": settings.run_name,
            "fallback_used": fallback_used,
        },
    )

    prof.__enter__()
    try:
        data_iter = iter(loader)
        for step in range(total_iterations):
            try:
                images, targets = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                images, targets = next(data_iter)

            iter_start = time.perf_counter()

            images = images.to(device, non_blocking=(device.type == "cuda"))
            targets = targets.to(device, non_blocking=(device.type == "cuda"))

            optimizer.zero_grad(set_to_none=True)

            with get_autocast_context(device, precision):
                outputs = model(images)
                loss = criterion(outputs, targets)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            if device.type == "cuda":
                torch.cuda.synchronize()

            iter_time = time.perf_counter() - iter_start

            if step >= settings.warmup_iterations:
                iter_times.append(iter_time)
                measured_loss = float(loss.detach().cpu())
                losses.append(measured_loss)
                measured_epoch = len(losses)
                elapsed_from_model_start_s = time.perf_counter() - model_start_perf
                append_loss_history_progress(
                    loss_history_dir=loss_history_dir,
                    model_name=model_name,
                    batch_size=batch_size,
                    precision=precision,
                    epoch_index=measured_epoch,
                    loss_value=measured_loss,
                    epoch_time_s=iter_time,
                    elapsed_from_model_start_s=elapsed_from_model_start_s,
                )
                sample = query_nvidia_smi()
                if sample is not None:
                    gpu_samples.append(sample)

            prof.step()

            if (step + 1) % settings.log_every == 0 or step == total_iterations - 1:
                completed = min(max(step + 1 - settings.warmup_iterations, 0), settings.iterations)
                avg = statistics.mean(iter_times) if iter_times else float("nan")
                eta = (settings.iterations - completed) * avg if iter_times else float("nan")
                emit(
                    "[PROGRESS] "
                    f"step={step + 1}/{total_iterations} "
                    f"measured={completed}/{settings.iterations} "
                    f"loss={float(loss.detach().cpu()):.4f} "
                    f"iter_s={iter_time:.4f} "
                    f"avg_s={avg:.4f} "
                    f"eta={format_seconds(eta) if not math.isnan(eta) else 'N/A'}",
                    status_log,
                )
                append_jsonl(
                    progress_jsonl,
                    {
                        "timestamp_utc": utc_now_iso(),
                        "event": "progress",
                        "model": model_name,
                        "batch_size": batch_size,
                        "precision": precision,
                        "step": step + 1,
                        "total_steps": total_iterations,
                        "measured_steps": completed,
                        "loss": float(loss.detach().cpu()),
                        "iter_s": iter_time,
                        "avg_s": avg,
                        "eta_s": eta,
                    },
                )
    finally:
        prof.__exit__(None, None, None)

    if profiler_dir is not None:
        export_profiler_artifacts(
            prof=prof,
            profiler_dir=profiler_dir,
            model_name=model_name,
            batch_size=batch_size,
            precision=precision,
        )

    if not iter_times:
        raise RuntimeError("No measured iterations captured. Increase --iterations.")

    avg_iter_s = statistics.mean(iter_times)
    median_iter_s = statistics.median(iter_times)
    p90_iter_s = sorted(iter_times)[max(0, int(0.9 * len(iter_times)) - 1)]
    images_per_s = batch_size / avg_iter_s

    peak_gpu_mem_mb = (
        torch.cuda.max_memory_allocated(device=device) / (1024.0 * 1024.0)
        if device.type == "cuda"
        else 0.0
    )

    # Profiler aggregates are helpful but may be incomplete for some kernels.
    events = prof.key_averages()
    total_cpu_time_us = float(sum(ev.cpu_time_total for ev in events))
    total_cuda_time_us = float(sum(getattr(ev, "cuda_time_total", 0.0) for ev in events))
    total_flops = float(sum(getattr(ev, "flops", 0.0) for ev in events))

    flops_time_s = total_cuda_time_us / 1e6 if device.type == "cuda" and total_cuda_time_us > 0 else avg_iter_s * settings.iterations
    achieved_tflops = (total_flops / flops_time_s / 1e12) if flops_time_s > 0 and total_flops > 0 else float("nan")

    gpu_util_values = [s.utilization_gpu for s in gpu_samples]
    gpu_mem_util_values = [s.utilization_mem for s in gpu_samples]
    gpu_util_mean = statistics.mean(gpu_util_values) if gpu_util_values else float("nan")
    gpu_util_max = max(gpu_util_values) if gpu_util_values else float("nan")
    gpu_util_p90 = percentile(gpu_util_values, 90.0)
    gpu_mem_util_mean = statistics.mean(gpu_mem_util_values) if gpu_mem_util_values else float("nan")
    gpu_mem_util_max = max(gpu_mem_util_values) if gpu_mem_util_values else float("nan")
    gpu_mem_util_p90 = percentile(gpu_mem_util_values, 90.0)
    gpu_temp_mean = statistics.mean([s.temperature_c for s in gpu_samples]) if gpu_samples else float("nan")
    write_gpu_samples(gpu_samples_dir, model_name, batch_size, precision, gpu_samples)

    params_count = sum(p.numel() for p in model.parameters())
    params_size_mb_fp32 = params_count * 4 / (1024.0 * 1024.0)

    # Approximate bytes moved for a training step for coarse roofline x-axis support.
    # Use this as a first-pass approximation and refine with Nsight Compute bytes if needed.
    bytes_per_image = 3 * 224 * 224 * (2 if precision == "fp16" else 4)
    approx_batch_input_bytes = bytes_per_image * batch_size
    approx_step_bytes = approx_batch_input_bytes + (params_count * (2 if precision == "fp16" else 4) * 3)
    approx_operational_intensity = (total_flops / max(approx_step_bytes * settings.iterations, 1)) if total_flops > 0 else float("nan")

    val_metrics = {
        "val_loss": float("nan"),
        "val_top1_acc": float("nan"),
        "val_top5_acc": float("nan"),
        "val_samples": 0.0,
        "val_batches": 0.0,
    }
    if val_loader is not None:
        emit("[INFO] Starting validation pass", status_log)
        val_metrics = evaluate_validation(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            precision=precision,
            max_batches=val_max_batches,
        )
        emit(
            "[INFO] Validation completed "
            f"val_loss={val_metrics['val_loss']:.4f} "
            f"val_top1={val_metrics['val_top1_acc']:.2f}% "
            f"val_top5={val_metrics['val_top5_acc']:.2f}%",
            status_log,
        )

    train_dataset_size = len(dataset)
    train_samples_measured = len(losses) * batch_size
    train_samples_total_with_warmup = total_iterations * batch_size
    val_dataset_size = len(val_dataset) if val_dataset is not None else 0
    val_samples_evaluated = int(val_metrics["val_samples"])
    write_loss_history(
        loss_history_dir=loss_history_dir,
        model_name=model_name,
        batch_size=batch_size,
        precision=precision,
        run_started_at=run_started_at,
        losses=losses,
        train_samples_measured=train_samples_measured,
        train_samples_total=train_samples_total_with_warmup,
        val_samples_evaluated=val_samples_evaluated,
        val_loss=float(val_metrics["val_loss"]),
        val_top1_acc=float(val_metrics["val_top1_acc"]),
        val_top5_acc=float(val_metrics["val_top5_acc"]),
    )

    row: Dict[str, object] = {
        "timestamp_utc": utc_now_iso(),
        "run_name": settings.run_name,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": get_gpu_name(),
        "device": device.type,
        "train_dataset": "imagefolder" if use_real_data else "fake",
        "train_data_dir": REAL_DATA_DIR if use_real_data else "",
        "val_dataset": get_validation_dataset_mode(),
        "val_data_dir": get_validation_dataset_dir(),
        "model": model_name,
        "batch_size": batch_size,
        "precision": precision,
        "num_workers": settings.num_workers,
        "train_dataset_size": train_dataset_size,
        "train_samples_measured": train_samples_measured,
        "train_samples_total_with_warmup": train_samples_total_with_warmup,
        "val_dataset_size": val_dataset_size,
        "warmup_iterations": settings.warmup_iterations,
        "iterations": settings.iterations,
        "avg_iter_s": avg_iter_s,
        "median_iter_s": median_iter_s,
        "p90_iter_s": p90_iter_s,
        "images_per_s": images_per_s,
        "avg_loss": statistics.mean(losses),
        "peak_gpu_mem_mb": peak_gpu_mem_mb,
        "gpu_util_mean_pct": gpu_util_mean,
        "gpu_util_max_pct": gpu_util_max,
        "gpu_util_p90_pct": gpu_util_p90,
        "gpu_mem_util_mean_pct": gpu_mem_util_mean,
        "gpu_mem_util_max_pct": gpu_mem_util_max,
        "gpu_mem_util_p90_pct": gpu_mem_util_p90,
        "gpu_temp_mean_c": gpu_temp_mean,
        "total_cpu_time_us": total_cpu_time_us,
        "total_cuda_time_us": total_cuda_time_us,
        "total_flops": total_flops,
        "achieved_tflops": achieved_tflops,
        "params_count": params_count,
        "params_size_mb_fp32": params_size_mb_fp32,
        "approx_step_bytes": approx_step_bytes,
        "approx_operational_intensity_flops_per_byte": approx_operational_intensity,
        "val_loss": val_metrics["val_loss"],
        "val_top1_acc": val_metrics["val_top1_acc"],
        "val_top5_acc": val_metrics["val_top5_acc"],
        "val_samples": val_metrics["val_samples"],
        "val_batches": val_metrics["val_batches"],
        "val_max_batches": val_max_batches,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "wall_clock_run_s": (utc_now() - run_started_at).total_seconds(),
    }

    append_row(output_csv, row)

    emit(
        "[RUN END] "
        f"model={model_name} batch_size={batch_size} precision={precision} "
        f"avg_iter_s={avg_iter_s:.4f} images_per_s={images_per_s:.2f} "
        f"peak_gpu_mem_mb={peak_gpu_mem_mb:.1f} achieved_tflops={achieved_tflops:.3f} "
        f"val_top1={val_metrics['val_top1_acc']:.2f}% val_top5={val_metrics['val_top5_acc']:.2f}%",
        status_log,
    )
    append_jsonl(
        progress_jsonl,
        {
            "timestamp_utc": utc_now_iso(),
            "event": "run_end",
            "model": model_name,
            "batch_size": batch_size,
            "precision": precision,
            "avg_iter_s": avg_iter_s,
            "images_per_s": images_per_s,
            "peak_gpu_mem_mb": peak_gpu_mem_mb,
            "achieved_tflops": achieved_tflops,
            "val_loss": val_metrics["val_loss"],
            "val_top1_acc": val_metrics["val_top1_acc"],
            "val_top5_acc": val_metrics["val_top5_acc"],
            "fallback_used": fallback_used,
        },
    )

    return row


def append_row(csv_path: Path, row: Dict[str, object]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(row.keys())
    file_exists = csv_path.exists()

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def dump_run_plan(settings: ExperimentSettings, run_cfgs: List[RunConfig], path: Path) -> None:
    payload = {
        "generated_at_utc": utc_now_iso(),
        "run_name": settings.run_name,
        "train_dataset": get_dataset_mode(),
        "train_data_dir": get_dataset_dir(),
        "val_dataset": get_validation_dataset_mode(),
        "val_data_dir": get_validation_dataset_dir(),
        "iterations": settings.iterations,
        "warmup_iterations": settings.warmup_iterations,
        "configs": [cfg.__dict__ for cfg in run_cfgs],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def validate_settings(settings: ExperimentSettings, options: RuntimeOptions) -> None:
    if not USE_REAL_DATA and not options.quick_check and not options.quick_check_real and not options.quick_check_real_all and FAKE_DATASET_SIZE < min(settings.batch_sizes):
        raise ValueError("FAKE_DATASET_SIZE must be >= smallest batch size")
    if (USE_REAL_DATA or options.quick_check_real or options.quick_check_real_all) and not options.quick_check and not REAL_DATA_DIR:
        raise ValueError("REAL_DATA_DIR must be set when USE_REAL_DATA=True")
    if USE_REAL_VALIDATION and not REAL_VAL_DIR:
        raise ValueError("REAL_VAL_DIR must be set when USE_REAL_VALIDATION=True")

    unknown_models = [m for m in settings.models if m not in SUPPORTED_MODELS]
    if unknown_models:
        raise ValueError(f"Unsupported models: {unknown_models}")

    unknown_precisions = [p for p in settings.precisions if p not in {"fp32", "fp16"}]
    if unknown_precisions:
        raise ValueError(f"Unsupported precisions: {unknown_precisions}")


def parse_runtime_options() -> RuntimeOptions:
    parser = argparse.ArgumentParser(description="Roofline benchmark runner")
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Optional single model override: resnet50 | vit_b_16 | vgg16",
    )
    parser.add_argument(
        "--quick-check",
        action="store_true",
        help="Run a short smoke test to catch setup issues quickly",
    )
    parser.add_argument(
        "--quick-check-real",
        action="store_true",
        help="Run a short smoke test on real data (1 warmup + 2 measured iterations)",
    )
    parser.add_argument(
        "--quick-check-real-all",
        action="store_true",
        help="Run a short real-data smoke test across all model/config combinations",
    )
    args = parser.parse_args()
    selected_quick_modes = sum([bool(args.quick_check), bool(args.quick_check_real), bool(args.quick_check_real_all)])
    if selected_quick_modes > 1:
        raise ValueError("Use only one quick mode flag: --quick-check | --quick-check-real | --quick-check-real-all")
    return RuntimeOptions(
        model=args.model.strip(),
        quick_check=bool(args.quick_check),
        quick_check_real=bool(args.quick_check_real),
        quick_check_real_all=bool(args.quick_check_real_all),
    )


def apply_runtime_options(settings: ExperimentSettings, options: RuntimeOptions) -> ExperimentSettings:
    updated = settings

    if options.quick_check_real_all and options.model:
        raise ValueError("--quick-check-real-all runs all models/configs; do not combine with --model")

    if options.model:
        if options.model not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported --model '{options.model}'. Choose from: {', '.join(SUPPORTED_MODELS.keys())}"
            )
        updated = replace(updated, models=[options.model])

    if options.quick_check:
        # Fast end-to-end smoke test with one model and tiny FakeData.
        quick_model = [options.model] if options.model else [DEFAULT_MODELS[0]]
        updated = replace(
            updated,
            models=quick_model,
            batch_sizes=[8],
            precisions=["fp32"],
            iterations=8,
            warmup_iterations=2,
            num_workers=2,
            max_runs=1,
            run_name=f"{updated.run_name}_quickcheck",
        )

    if options.quick_check_real:
        # Real-data smoke test for train/val path validation.
        quick_model = [options.model] if options.model else [DEFAULT_MODELS[0]]
        updated = replace(
            updated,
            models=quick_model,
            batch_sizes=[8],
            precisions=["fp32"],
            iterations=2,
            warmup_iterations=1,
            num_workers=2,
            max_runs=1,
            run_name=f"{updated.run_name}_quickcheck_real",
        )

    if options.quick_check_real_all:
        # Real-data smoke test across all configs with tiny iteration budget.
        updated = replace(
            updated,
            models=DEFAULT_MODELS,
            batch_sizes=DEFAULT_BATCH_SIZES,
            precisions=DEFAULT_PRECISIONS,
            iterations=2,
            warmup_iterations=1,
            num_workers=2,
            max_runs=0,
            run_name=f"{updated.run_name}_quickcheck_real_all",
        )

    return updated


def build_run_configs(settings: ExperimentSettings, device: torch.device) -> List[RunConfig]:
    configs: List[RunConfig] = []
    for model_name, batch_size, precision in itertools.product(
        settings.models,
        settings.batch_sizes,
        settings.precisions,
    ):
        if precision == "fp16" and device.type != "cuda":
            continue
        configs.append(RunConfig(model_name=model_name, batch_size=batch_size, precision=precision))
    return configs


def main() -> int:
    options = parse_runtime_options()
    settings = apply_runtime_options(DEFAULT_SETTINGS, options)
    set_seed(settings.seed)

    device = get_device(force_cpu=settings.force_cpu)
    validate_settings(settings, options)

    run_cfgs = build_run_configs(settings, device)
    if settings.max_runs > 0:
        run_cfgs = run_cfgs[: settings.max_runs]

    safe_run_name = "_".join(settings.run_name.strip().split()) or "baseline"
    run_dir = create_next_run_dir(settings.output_root, safe_run_name)

    output_csv = run_dir / "metrics.csv"
    plan_json = run_dir / "run_plan.json"
    status_log = run_dir / "status.log"
    progress_jsonl = run_dir / "progress.jsonl"
    profiler_dir = run_dir / "profiler"
    gpu_samples_dir = run_dir / "gpu_samples"
    loss_history_dir = run_dir / "loss_history"
    dump_run_plan(settings, run_cfgs, plan_json)

    emit("[INFO] Starting benchmark sweep", status_log)
    emit(f"[INFO] Quick check mode: {options.quick_check}", status_log)
    emit(f"[INFO] Quick check real mode: {options.quick_check_real}", status_log)
    emit(f"[INFO] Quick check real all mode: {options.quick_check_real_all}", status_log)
    if options.model:
        emit(f"[INFO] Single model override: {options.model}", status_log)
    emit(f"[INFO] Device: {device.type} | GPU: {get_gpu_name()}", status_log)
    emit(f"[INFO] Train dataset mode: {get_effective_dataset_mode(options)}", status_log)
    emit(f"[INFO] Validation dataset mode: {get_validation_dataset_mode()}", status_log)
    if USE_REAL_DATA and not options.quick_check:
        emit(f"[INFO] Real data dir: {get_dataset_dir()}", status_log)
    if USE_REAL_VALIDATION:
        emit(f"[INFO] Validation data dir: {get_validation_dataset_dir()}", status_log)
    if options.quick_check:
        emit("[INFO] Quick check uses tiny FakeData for training (size=256)", status_log)
    if options.quick_check_real:
        emit("[INFO] Quick check real uses real train/val with 1 warmup + 2 measured iterations", status_log)
    if options.quick_check_real_all:
        emit(
            "[INFO] Quick check real all uses all model/config combinations with tiny real subsets "
            f"(train={QUICK_CHECK_REAL_TRAIN_MAX_SAMPLES}, val={QUICK_CHECK_REAL_VAL_MAX_SAMPLES}) "
            "and 1 warmup + 2 measured iterations",
            status_log,
        )
    emit(f"[INFO] Planned runs: {len(run_cfgs)}", status_log)
    emit(f"[INFO] Output CSV: {output_csv}", status_log)
    emit(f"[INFO] Run plan JSON: {plan_json}", status_log)
    emit(f"[INFO] Profiler dir: {profiler_dir}", status_log)
    emit(f"[INFO] GPU samples dir: {gpu_samples_dir}", status_log)
    emit(f"[INFO] Loss history dir: {loss_history_dir}", status_log)
    emit(f"[INFO] Status log: {status_log}", status_log)
    emit(f"[INFO] Progress JSONL: {progress_jsonl}", status_log)
    emit(f"[INFO] Run directory: {run_dir}", status_log)

    failures: List[Tuple[RunConfig, str]] = []

    for idx, cfg in enumerate(run_cfgs, start=1):
        emit(f"\n[SWEEP] {idx}/{len(run_cfgs)}", status_log)
        started = utc_now()
        try:
            profile_run(
                settings=settings,
                options=options,
                run_cfg=cfg,
                device=device,
                output_csv=output_csv,
                run_started_at=started,
                status_log=status_log,
                progress_jsonl=progress_jsonl,
                profiler_dir=profiler_dir,
                gpu_samples_dir=gpu_samples_dir,
                loss_history_dir=loss_history_dir,
            )
        except torch.cuda.OutOfMemoryError as err:
            failures.append((cfg, f"OOM: {err}"))
            emit(
                "[WARN] OOM encountered and run skipped: "
                f"model={cfg.model_name} bs={cfg.batch_size} precision={cfg.precision}",
                status_log,
            )
            append_jsonl(
                progress_jsonl,
                {
                    "timestamp_utc": utc_now_iso(),
                    "event": "run_oom",
                    "model": cfg.model_name,
                    "batch_size": cfg.batch_size,
                    "precision": cfg.precision,
                    "reason": str(err),
                },
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except RuntimeError as err:
            # Keep sweep running if one run fails (e.g., non-OOM runtime error).
            failures.append((cfg, str(err)))
            emit(
                "[WARN] Run failed and will be skipped: "
                f"model={cfg.model_name} bs={cfg.batch_size} precision={cfg.precision} error={err}",
                status_log,
            )
            append_jsonl(
                progress_jsonl,
                {
                    "timestamp_utc": utc_now_iso(),
                    "event": "run_failed",
                    "model": cfg.model_name,
                    "batch_size": cfg.batch_size,
                    "precision": cfg.precision,
                    "reason": str(err),
                },
            )

    if failures:
        emit("\n[SUMMARY] Sweep completed with skipped/failed runs:", status_log)
        for cfg, reason in failures:
            emit(
                f"  - model={cfg.model_name}, bs={cfg.batch_size}, precision={cfg.precision}, reason={reason}",
                status_log,
            )
    else:
        emit("\n[SUMMARY] Sweep completed successfully with no failures.", status_log)

    emit("[DONE] Metrics collection complete.", status_log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
