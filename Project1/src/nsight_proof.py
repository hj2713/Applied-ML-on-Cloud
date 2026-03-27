#!/usr/bin/env python3
"""Nsight Compute Profiling Script for Deep Learning Workloads.

This script executes a short, precisely delimited training workload
specifically designed to be captured by NVIDIA Nsight Compute (`ncu`).
It utilizes Hardware NVTX markers to isolate the exact execution windows
of the forward pass, backward pass, and optimizer steps, preventing the
profiler from capturing framework initialization overhead.

Usage:
    ncu -o l4_trace \
        --set full \
        python src/nsight_proof.py --model vit_b_16 --batch-size 128 --precision fp16
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models # type: ignore

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


SUPPORTED_MODELS = {
    "resnet50": models.resnet50,
    "vit_b_16": models.vit_b_16,
    "vgg16": models.vgg16,
}


def parse_args() -> argparse.Namespace:
    """Parse robust CLI arguments for the Nsight profiler."""
    parser = argparse.ArgumentParser(description="Nsight Compute targeted training iterations.")
    parser.add_argument("--model", type=str, default="resnet50", choices=SUPPORTED_MODELS.keys(), help="Model Architecture")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for the forward pass")
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16"], help="Compute precision")
    parser.add_argument("--warmup", type=int, default=3, help="Number of un-profiled warmup iterations")
    parser.add_argument("--iterations", type=int, default=2, help="Number of profiled target iterations")
    return parser.parse_args()


def build_model(model_name: str, num_classes: int = 1000) -> nn.Module:
    """Instantiate the deep learning architecture."""
    model_ctor = SUPPORTED_MODELS.get(model_name)
    if not model_ctor:
        raise ValueError(f"Unsupported model: {model_name}")
    
    model = model_ctor(weights=None)
    
    # Align final classifier dimensions
    if model_name.startswith("resnet"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name.startswith("vgg"):
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif model_name.startswith("vit"):
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        
    return model


def main() -> int:
    args = parse_args()
    
    if not torch.cuda.is_available():
        logging.error("CUDA is absolutely required for Nsight Profiling, but is not available. Exiting.")
        return 1
        
    device = torch.device("cuda")
    
    logging.info(f"Initializing {args.model} on {device} (Batch Size: {args.batch_size}, Precision: {args.precision})")
    
    model = build_model(args.model).to(device)
    model.train()
    
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # Configure Mixed Precision dynamically
    use_amp = (args.precision == "fp16")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Synthesize fake ImageNet batch entirely on device to avoid dataloader multi-processing Overhead during profiling
    inputs_shape = (args.batch_size, 3, 224, 224)
    labels_shape = (args.batch_size,)
    
    logging.info("Executing Warmup Phase...")
    for i in range(args.warmup):
        inputs = torch.randn(*inputs_shape, device=device)
        labels = torch.randint(0, 1000, labels_shape, device=device)
        
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
    torch.cuda.synchronize()
    logging.info("Warmup complete. Starting Nsight Profile Window.")

    # Explicit Nsight Compute annotations
    torch.cuda.cudart().cudaProfilerStart()
    
    for i in range(args.iterations):
        logging.info(f"Executing Profiled Iteration {i+1}/{args.iterations}")
        torch.cuda.nvtx.range_push(f"Iteration_{i}")
        
        # Data Setup
        inputs = torch.randn(*inputs_shape, device=device)
        labels = torch.randint(0, 1000, labels_shape, device=device)
        optimizer.zero_grad(set_to_none=True)
        
        # Forward Pass
        torch.cuda.nvtx.range_push("Forward_Pass")
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        torch.cuda.nvtx.range_pop()
        
        # Backward Pass
        torch.cuda.nvtx.range_push("Backward_Pass")
        scaler.scale(loss).backward()
        torch.cuda.nvtx.range_pop()
        
        # Optimizer Step
        torch.cuda.nvtx.range_push("Optimizer_Step")
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.nvtx.range_pop()
        
        torch.cuda.nvtx.range_pop() # End Iteration
        
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    
    logging.info("Nsight Profiling Window Complete. Exiting gracefully.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
