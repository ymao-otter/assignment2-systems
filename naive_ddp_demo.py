#!/usr/bin/env python3
"""
Demonstration of naive DDP implementation.

This script trains a small toy model using distributed data parallel training
and verifies that the weights match single-process training results.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from copy import deepcopy
import os


class ToyModel(nn.Module):
    """Simple toy model for testing DDP."""
    
    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def setup_process_group(rank: int, world_size: int):
    """Initialize the distributed process group."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup_process_group():
    """Clean up the distributed process group."""
    dist.barrier()
    dist.destroy_process_group()


def train_ddp(rank: int, world_size: int, num_steps: int = 10):
    """Train using DDP and compare with single-process training."""
    setup_process_group(rank, world_size)
    
    # Set random seed for reproducibility (different on each rank initially)
    torch.manual_seed(rank)
    
    # Create models
    non_parallel_model = ToyModel()
    ddp_base = deepcopy(non_parallel_model)
    
    # Import our DDP implementation
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from cs336_systems.ddp import DDPIndividualParametersOverlapped
    
    # Wrap with DDP
    ddp_model = DDPIndividualParametersOverlapped(ddp_base)
    
    # Generate random training data (same across all ranks)
    torch.manual_seed(42)
    batch_size = 20
    input_size = 10
    output_size = 5
    
    # Full dataset
    all_x = torch.randn(batch_size, input_size)
    all_y = torch.randn(batch_size, output_size)
    
    # Each rank gets a shard of the data
    local_batch_size = batch_size // world_size
    offset = rank * local_batch_size
    local_x = all_x[offset:offset + local_batch_size]
    local_y = all_y[offset:offset + local_batch_size]
    
    # Create optimizers
    non_parallel_optimizer = optim.SGD(non_parallel_model.parameters(), lr=0.01)
    ddp_optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
    
    loss_fn = nn.MSELoss()
    
    if rank == 0:
        print("=" * 60)
        print("Naive DDP Training Demonstration")
        print("=" * 60)
        print(f"\nTraining for {num_steps} steps...")
        print(f"World size: {world_size}")
        print(f"Total batch size: {batch_size}")
        print(f"Local batch size per rank: {local_batch_size}\n")
    
    # Training loop
    for step in range(num_steps):
        # Train non-parallel model on ALL data
        non_parallel_optimizer.zero_grad()
        non_parallel_output = non_parallel_model(all_x)
        non_parallel_loss = loss_fn(non_parallel_output, all_y)
        non_parallel_loss.backward()
        non_parallel_optimizer.step()
        
        # Train DDP model on LOCAL shard
        ddp_optimizer.zero_grad()
        ddp_output = ddp_model(local_x)
        ddp_loss = loss_fn(ddp_output, local_y)
        ddp_loss.backward()
        
        # Synchronize gradients across ranks
        ddp_model.finish_gradient_synchronization()
        
        ddp_optimizer.step()
        
        # Check that parameters match on rank 0
        if rank == 0 and (step % 5 == 0 or step == num_steps - 1):
            max_diff = 0.0
            for np_param, ddp_param in zip(
                non_parallel_model.parameters(), ddp_model.module.parameters()
            ):
                diff = (np_param - ddp_param).abs().max().item()
                max_diff = max(max_diff, diff)
            
            print(f"Step {step:2d} | "
                  f"Non-parallel loss: {non_parallel_loss.item():.6f} | "
                  f"DDP loss (local): {ddp_loss.item():.6f} | "
                  f"Max param diff: {max_diff:.2e}")
    
    # Final verification
    if rank == 0:
        print("\n" + "=" * 60)
        print("Final Verification")
        print("=" * 60)
        
        all_match = True
        for name, (np_param, ddp_param) in zip(
            ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"],
            zip(non_parallel_model.parameters(), ddp_model.module.parameters())
        ):
            matches = torch.allclose(np_param, ddp_param, rtol=1e-5, atol=1e-6)
            status = "✓ MATCH" if matches else "✗ MISMATCH"
            diff = (np_param - ddp_param).abs().max().item()
            print(f"{name:12s}: {status} (max diff: {diff:.2e})")
            if not matches:
                all_match = False
        
        print("\n" + "=" * 60)
        if all_match:
            print("SUCCESS: DDP weights match single-process training!")
        else:
            print("FAILURE: DDP weights do not match single-process training")
        print("=" * 60)
    
    cleanup_process_group()


if __name__ == "__main__":
    world_size = 2
    print("\nStarting naive DDP demonstration...")
    print("This will spawn 2 processes to simulate distributed training.\n")
    
    mp.spawn(
        train_ddp,
        args=(world_size, 10),
        nprocs=world_size,
        join=True
    )
    
    print("\nDemonstration complete!")
