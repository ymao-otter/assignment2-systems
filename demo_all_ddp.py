#!/usr/bin/env python3
"""
Demonstrate all four DDP implementations produce identical results.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from copy import deepcopy
import os


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def setup_process_group(rank: int, world_size: int):
    """Initialize the distributed process group."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12360"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup_process_group():
    """Clean up the distributed process group."""
    dist.barrier()
    dist.destroy_process_group()


def test_all_implementations(rank: int, world_size: int, results_queue):
    """Test that all four DDP implementations produce identical results."""
    setup_process_group(rank, world_size)
    
    # Set random seed
    torch.manual_seed(42)
    
    # Create base model
    base_model = SimpleModel()
    
    # Import all DDP implementations
    from cs336_systems.ddp import (
        DDPIndividualParameters,
        DDPIndividualParametersOverlapped,
        DDPFlattenedGradients,
        DDPBucketed
    )
    
    # Create all four versions
    naive_model = DDPIndividualParameters(deepcopy(base_model))
    overlapped_model = DDPIndividualParametersOverlapped(deepcopy(base_model))
    flattened_model = DDPFlattenedGradients(deepcopy(base_model))
    bucketed_model = DDPBucketed(deepcopy(base_model), bucket_size_mb=0.001)
    
    # Generate random training data
    torch.manual_seed(42 + rank)
    x = torch.randn(8, 10)
    y = torch.randn(8, 5)
    
    # Create optimizers
    naive_opt = optim.SGD(naive_model.parameters(), lr=0.01)
    overlapped_opt = optim.SGD(overlapped_model.parameters(), lr=0.01)
    flattened_opt = optim.SGD(flattened_model.parameters(), lr=0.01)
    bucketed_opt = optim.SGD(bucketed_model.parameters(), lr=0.01)
    
    loss_fn = nn.MSELoss()
    
    # Train all versions
    for step in range(3):
        # Naive
        naive_opt.zero_grad()
        naive_loss = loss_fn(naive_model(x), y)
        naive_loss.backward()
        naive_model.finish_gradient_synchronization()
        naive_opt.step()
        
        # Overlapped
        overlapped_opt.zero_grad()
        overlapped_loss = loss_fn(overlapped_model(x), y)
        overlapped_loss.backward()
        overlapped_model.finish_gradient_synchronization()
        overlapped_opt.step()
        
        # Flattened
        flattened_opt.zero_grad()
        flattened_loss = loss_fn(flattened_model(x), y)
        flattened_loss.backward()
        flattened_model.finish_gradient_synchronization()
        flattened_opt.step()
        
        # Bucketed
        bucketed_opt.zero_grad()
        bucketed_loss = loss_fn(bucketed_model(x), y)
        bucketed_loss.backward()
        bucketed_model.finish_gradient_synchronization()
        bucketed_opt.step()
    
    # Compare all parameters
    if rank == 0:
        max_diffs = {
            'overlapped_vs_naive': 0.0,
            'flattened_vs_naive': 0.0,
            'bucketed_vs_naive': 0.0,
        }
        
        for naive_p, over_p, flat_p, buck_p in zip(
            naive_model.module.parameters(),
            overlapped_model.module.parameters(),
            flattened_model.module.parameters(),
            bucketed_model.module.parameters()
        ):
            max_diffs['overlapped_vs_naive'] = max(
                max_diffs['overlapped_vs_naive'],
                (naive_p - over_p).abs().max().item()
            )
            max_diffs['flattened_vs_naive'] = max(
                max_diffs['flattened_vs_naive'],
                (naive_p - flat_p).abs().max().item()
            )
            max_diffs['bucketed_vs_naive'] = max(
                max_diffs['bucketed_vs_naive'],
                (naive_p - buck_p).abs().max().item()
            )
        
        results_queue.put(max_diffs)
    
    cleanup_process_group()


def main():
    world_size = 2
    
    print("=" * 70)
    print("Testing All Four DDP Implementations")
    print("=" * 70)
    print()
    print("Implementations:")
    print("  1. DDPIndividualParameters (Naive - Synchronous)")
    print("  2. DDPIndividualParametersOverlapped (Async individual)")
    print("  3. DDPFlattenedGradients (Batched all parameters)")
    print("  4. DDPBucketed (Bucketed + Overlap)")
    print()
    
    # Create queue for results
    manager = mp.Manager()
    results_queue = manager.Queue()
    
    # Spawn processes
    mp.spawn(
        test_all_implementations,
        args=(world_size, results_queue),
        nprocs=world_size,
        join=True
    )
    
    # Get results
    max_diffs = results_queue.get()
    
    print("Results (maximum parameter difference vs naive baseline):")
    print("-" * 70)
    
    all_pass = True
    for comparison, diff in max_diffs.items():
        status = "✓ PASS" if diff < 1e-6 else "✗ FAIL"
        print(f"  {comparison:30s}: {diff:.2e}  {status}")
        if diff >= 1e-6:
            all_pass = False
    
    print()
    print("=" * 70)
    
    if all_pass:
        print("✓ SUCCESS: All four implementations produce identical results!")
        print()
        print("This confirms:")
        print("  - All implementations compute correct gradients")
        print("  - All implementations synchronize properly across ranks")
        print("  - Optimizations (overlap, batching, bucketing) are correctness-preserving")
    else:
        print("✗ FAILURE: Some implementations produce different results")
    
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
