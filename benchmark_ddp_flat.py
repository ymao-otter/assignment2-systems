#!/usr/bin/env python3
"""
Benchmark script comparing DDPIndividualParameters vs DDPFlattenedGradients.

This script measures the performance difference between:
1. DDPIndividualParameters: Issues one all-reduce per parameter tensor
2. DDPFlattenedGradients: Flattens all gradients and issues a single all-reduce

The benchmark measures the time taken for:
- Forward pass
- Backward pass
- Gradient synchronization
- Total time per step
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import time
import argparse
from typing import Optional
import numpy as np


class BenchmarkModel(nn.Module):
    """Model for benchmarking DDP implementations."""
    
    def __init__(self, num_layers: int = 10, hidden_size: int = 1024):
        super().__init__()
        layers = []
        layers.append(nn.Linear(hidden_size, hidden_size))
        for _ in range(num_layers - 1):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_size, hidden_size))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


def setup_process_group(rank: int, world_size: int, backend: str = "gloo"):
    """Initialize the distributed process group."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup_process_group():
    """Clean up the distributed process group."""
    dist.barrier()
    dist.destroy_process_group()


def benchmark_ddp(
    rank: int,
    world_size: int,
    ddp_class_name: str,
    num_layers: int,
    hidden_size: int,
    batch_size: int,
    num_warmup: int,
    num_steps: int,
    use_cuda: bool,
):
    """Benchmark a specific DDP implementation."""
    # Determine backend based on device
    backend = "nccl" if use_cuda else "gloo"
    setup_process_group(rank, world_size, backend)
    
    # Set device
    if use_cuda:
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create model
    model = BenchmarkModel(num_layers=num_layers, hidden_size=hidden_size).to(device)
    
    # Import DDP class
    from cs336_systems.ddp import DDPIndividualParameters, DDPFlattenedGradients, DDPIndividualParametersOverlapped
    
    if ddp_class_name == "individual":
        ddp_model = DDPIndividualParameters(model)
    elif ddp_class_name == "individual_overlapped":
        ddp_model = DDPIndividualParametersOverlapped(model)
    elif ddp_class_name == "flattened":
        ddp_model = DDPFlattenedGradients(model)
    else:
        raise ValueError(f"Unknown DDP class: {ddp_class_name}")
    
    # Create optimizer
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    
    # Generate random data
    torch.manual_seed(rank)
    x = torch.randn(batch_size, hidden_size, device=device)
    y = torch.randn(batch_size, hidden_size, device=device)
    
    # Warmup
    for _ in range(num_warmup):
        optimizer.zero_grad()
        output = ddp_model(x)
        loss = loss_fn(output, y)
        loss.backward()
        ddp_model.finish_gradient_synchronization()
        optimizer.step()
    
    # Synchronize before timing
    if use_cuda:
        torch.cuda.synchronize(device)
    dist.barrier()
    
    # Timing
    forward_times = []
    backward_times = []
    sync_times = []
    total_times = []
    
    for step in range(num_steps):
        step_start = time.perf_counter()
        
        # Forward pass
        optimizer.zero_grad()
        forward_start = time.perf_counter()
        output = ddp_model(x)
        loss = loss_fn(output, y)
        if use_cuda:
            torch.cuda.synchronize(device)
        forward_end = time.perf_counter()
        forward_times.append(forward_end - forward_start)
        
        # Backward pass
        backward_start = time.perf_counter()
        loss.backward()
        if use_cuda:
            torch.cuda.synchronize(device)
        backward_end = time.perf_counter()
        backward_times.append(backward_end - backward_start)
        
        # Gradient synchronization
        sync_start = time.perf_counter()
        ddp_model.finish_gradient_synchronization()
        if use_cuda:
            torch.cuda.synchronize(device)
        sync_end = time.perf_counter()
        sync_times.append(sync_end - sync_start)
        
        # Optimizer step
        optimizer.step()
        if use_cuda:
            torch.cuda.synchronize(device)
        
        step_end = time.perf_counter()
        total_times.append(step_end - step_start)
    
    # Gather timing statistics on rank 0
    timing_data = {
        'forward_mean': np.mean(forward_times),
        'forward_std': np.std(forward_times),
        'backward_mean': np.mean(backward_times),
        'backward_std': np.std(backward_times),
        'sync_mean': np.mean(sync_times),
        'sync_std': np.std(sync_times),
        'total_mean': np.mean(total_times),
        'total_std': np.std(total_times),
    }
    
    cleanup_process_group()
    
    return timing_data


def _benchmark_worker_wrapper(rank, world_size, ddp_class_name, num_layers, 
                              hidden_size, batch_size, num_warmup, num_steps, 
                              use_cuda, results_queue):
    """Worker function wrapper for multiprocessing (must be top-level for pickling)."""
    result = benchmark_ddp(
        rank, world_size, ddp_class_name, num_layers, hidden_size,
        batch_size, num_warmup, num_steps, use_cuda
    )
    # Put result in queue
    results_queue.put((rank, result))


def run_benchmark(
    world_size: int,
    ddp_class_name: str,
    num_layers: int,
    hidden_size: int,
    batch_size: int,
    num_warmup: int,
    num_steps: int,
    use_cuda: bool,
):
    """Run benchmark across multiple processes."""
    # Create a queue to collect results
    manager = mp.Manager()
    results_queue = manager.Queue()
    
    # Use mp.spawn for process management
    mp.spawn(
        _benchmark_worker_wrapper,
        args=(world_size, ddp_class_name, num_layers, hidden_size,
              batch_size, num_warmup, num_steps, use_cuda, results_queue),
        nprocs=world_size,
        join=True
    )
    
    # Collect results from queue
    results = {}
    while not results_queue.empty():
        rank, result = results_queue.get()
        results[rank] = result
    
    # Return rank 0 results
    return results[0]


def main():
    parser = argparse.ArgumentParser(description='Benchmark DDP implementations')
    parser.add_argument('--world-size', type=int, default=2,
                        help='Number of processes (default: 2)')
    parser.add_argument('--num-layers', type=int, default=10,
                        help='Number of layers in the model (default: 10)')
    parser.add_argument('--hidden-size', type=int, default=1024,
                        help='Hidden size (default: 1024)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size per rank (default: 32)')
    parser.add_argument('--num-warmup', type=int, default=5,
                        help='Number of warmup steps (default: 5)')
    parser.add_argument('--num-steps', type=int, default=20,
                        help='Number of timed steps (default: 20)')
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA (requires multiple GPUs)')
    
    args = parser.parse_args()
    
    if args.cuda and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.cuda = False
    
    if args.cuda and torch.cuda.device_count() < args.world_size:
        print(f"Not enough GPUs ({torch.cuda.device_count()}) for world_size={args.world_size}")
        print("Falling back to CPU")
        args.cuda = False
    
    print("=" * 80)
    print("DDP Implementation Benchmark")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  World size: {args.world_size}")
    print(f"  Number of layers: {args.num_layers}")
    print(f"  Hidden size: {args.hidden_size}")
    print(f"  Batch size per rank: {args.batch_size}")
    print(f"  Warmup steps: {args.num_warmup}")
    print(f"  Timed steps: {args.num_steps}")
    print(f"  Device: {'CUDA' if args.cuda else 'CPU'}")
    print()
    
    # Benchmark DDPIndividualParameters (truly naive - synchronous)
    print("Benchmarking DDPIndividualParameters (synchronous all-reduce per parameter)...")
    individual_results = run_benchmark(
        args.world_size, "individual", args.num_layers, args.hidden_size,
        args.batch_size, args.num_warmup, args.num_steps, args.cuda
    )
    
    # Benchmark DDPIndividualParametersOverlapped (with computation-communication overlap)
    print("Benchmarking DDPIndividualParametersOverlapped (async all-reduce with overlap)...")
    overlapped_results = run_benchmark(
        args.world_size, "individual_overlapped", args.num_layers, args.hidden_size,
        args.batch_size, args.num_warmup, args.num_steps, args.cuda
    )
    
    # Benchmark DDPFlattenedGradients
    print("Benchmarking DDPFlattenedGradients (single all-reduce for all gradients)...")
    flattened_results = run_benchmark(
        args.world_size, "flattened", args.num_layers, args.hidden_size,
        args.batch_size, args.num_warmup, args.num_steps, args.cuda
    )
    
    # Print results
    print("\n" + "=" * 100)
    print("Results (times in milliseconds)")
    print("=" * 100)
    print()
    
    print(f"{'Metric':<25} {'Naive (Sync)':<22} {'Overlapped (Async)':<22} {'Flattened (Batched)':<22} {'Best Speedup':<10}")
    print("-" * 100)
    
    def format_time(mean, std):
        return f"{mean*1000:.2f} ± {std*1000:.2f}"
    
    def calc_speedup(baseline, best):
        return f"{baseline/best:.2f}x"
    
    metrics = [
        ('Forward pass', 'forward'),
        ('Backward pass', 'backward'),
        ('Gradient sync', 'sync'),
        ('Total per step', 'total'),
    ]
    
    for label, key in metrics:
        ind_mean = individual_results[f'{key}_mean']
        ind_std = individual_results[f'{key}_std']
        ovr_mean = overlapped_results[f'{key}_mean']
        ovr_std = overlapped_results[f'{key}_std']
        flat_mean = flattened_results[f'{key}_mean']
        flat_std = flattened_results[f'{key}_std']
        
        ind_str = format_time(ind_mean, ind_std)
        ovr_str = format_time(ovr_mean, ovr_std)
        flat_str = format_time(flat_mean, flat_std)
        
        # Find best (lowest) time
        best_mean = min(ind_mean, ovr_mean, flat_mean)
        speedup_str = calc_speedup(ind_mean, best_mean)
        
        print(f"{label:<25} {ind_str:<22} {ovr_str:<22} {flat_str:<22} {speedup_str:<10}")
    
    print()
    print("=" * 100)
    print("Analysis:")
    print("=" * 100)
    print()
    
    # Calculate speedups relative to naive baseline
    overlapped_sync_speedup = individual_results['sync_mean'] / overlapped_results['sync_mean']
    flattened_sync_speedup = individual_results['sync_mean'] / flattened_results['sync_mean']
    overlapped_total_speedup = individual_results['total_mean'] / overlapped_results['total_mean']
    flattened_total_speedup = individual_results['total_mean'] / flattened_results['total_mean']
    
    print("Gradient Synchronization (compared to Naive):")
    print(f"  Overlapped:  {overlapped_sync_speedup:6.2f}x speedup")
    print(f"  Flattened:   {flattened_sync_speedup:6.2f}x speedup")
    print()
    
    print("Overall Training (compared to Naive):")
    print(f"  Overlapped:  {overlapped_total_speedup:6.2f}x speedup")
    print(f"  Flattened:   {flattened_total_speedup:6.2f}x speedup")
    print()
    
    # Determine winner
    if flattened_sync_speedup > overlapped_sync_speedup * 1.1:
        print("✓ Flattened approach wins! Batching communication is more effective than overlapping.")
        print(f"  Reason: Eliminating {args.num_layers * 2} all-reduce calls reduces overhead more than hiding latency.")
    elif overlapped_sync_speedup > flattened_sync_speedup * 1.1:
        print("✓ Overlapped approach wins! Hiding latency is more effective than batching.")
        print("  Reason: Async operations hide network latency better than reducing message count.")
    else:
        print("≈ Both approaches provide similar benefits on this configuration.")
    
    print()
    print("Key Insights:")
    print("  - Naive (Sync): Issues N synchronous all-reduce calls after backward completes")
    print("  - Overlapped: Issues N async all-reduce calls during backward (hides latency)")
    print("  - Flattened: Issues 1 all-reduce call after backward (eliminates overhead)")
    print()
    print("  On localhost with shared memory:")
    print(f"    - Fixed overhead per all-reduce: ~{individual_results['sync_mean']*1000/max(1, args.num_layers*2):.2f}ms")
    print(f"    - Overlapping can hide some latency but still pays N × overhead")
    print(f"    - Flattening eliminates (N-1) × overhead completely")
    print()
    print("  On real distributed systems (multi-node):")
    print("    - Network latency per message: milliseconds (vs microseconds here)")
    print("    - Overlapping would provide more benefit (hide network latency)")
    print("    - Flattening would provide even more benefit (eliminate protocol overhead)")
    print("    - PyTorch DDP combines both: buckets gradients AND overlaps!")
    print()


if __name__ == "__main__":
    main()
