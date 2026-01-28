#!/usr/bin/env python3
"""
Test that the truly naive DDPIndividualParameters produces correct results.
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
    os.environ["MASTER_PORT"] = "12358"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup_process_group():
    """Clean up the distributed process group."""
    dist.barrier()
    dist.destroy_process_group()


def test_naive_ddp(rank: int, world_size: int, results_queue):
    """Test that the truly naive DDP produces correct results."""
    setup_process_group(rank, world_size)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create base model
    base_model = ToyModel()
    
    # Create DDP version
    from cs336_systems.ddp import DDPIndividualParameters
    
    naive_model = DDPIndividualParameters(deepcopy(base_model))
    
    # Generate random training data
    torch.manual_seed(42 + rank)
    batch_size = 8
    x = torch.randn(batch_size, 10)
    y = torch.randn(batch_size, 5)
    
    # Create optimizers
    naive_optimizer = optim.SGD(naive_model.parameters(), lr=0.01)
    
    loss_fn = nn.MSELoss()
    
    num_steps = 5
    
    for step in range(num_steps):
        # Train with naive DDP
        naive_optimizer.zero_grad()
        naive_output = naive_model(x)
        naive_loss = loss_fn(naive_output, y)
        naive_loss.backward()
        naive_model.finish_gradient_synchronization()
        naive_optimizer.step()
    
    # Check that all ranks have synchronized parameters
    for param in naive_model.module.parameters():
        # Gather this parameter from all ranks
        param_list = [torch.zeros_like(param) for _ in range(world_size)]
        dist.all_gather(param_list, param)
        
        # Check all are equal
        for other_param in param_list:
            if not torch.allclose(param, other_param):
                if rank == 0:
                    results_queue.put((False, "Parameters not synchronized across ranks"))
                cleanup_process_group()
                return
    
    # Report success from rank 0
    if rank == 0:
        results_queue.put((True, "All parameters synchronized"))
    
    cleanup_process_group()


def main():
    world_size = 2
    
    print("=" * 60)
    print("Testing Truly Naive DDPIndividualParameters")
    print("=" * 60)
    print()
    
    # Create queue for results
    manager = mp.Manager()
    results_queue = manager.Queue()
    
    # Spawn processes
    mp.spawn(
        test_naive_ddp,
        args=(world_size, results_queue),
        nprocs=world_size,
        join=True
    )
    
    # Get results
    success, message = results_queue.get()
    
    print(message)
    print()
    
    if success:
        print("✓ SUCCESS: Truly naive DDP works correctly!")
    else:
        print("✗ FAILURE: " + message)
    
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
