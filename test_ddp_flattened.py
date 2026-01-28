#!/usr/bin/env python3
"""
Test that DDPFlattenedGradients produces the same results as DDPIndividualParameters.
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
    os.environ["MASTER_PORT"] = "12357"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup_process_group():
    """Clean up the distributed process group."""
    dist.barrier()
    dist.destroy_process_group()


def test_ddp_equivalence(rank: int, world_size: int, results_queue):
    """Test that both DDP implementations produce the same results."""
    setup_process_group(rank, world_size)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create base model
    base_model = ToyModel()
    
    # Create two copies for the two DDP implementations
    from cs336_systems.ddp import DDPIndividualParameters, DDPFlattenedGradients
    
    individual_model = DDPIndividualParameters(deepcopy(base_model))
    flattened_model = DDPFlattenedGradients(deepcopy(base_model))
    
    # Generate random training data
    torch.manual_seed(42 + rank)
    batch_size = 8
    x = torch.randn(batch_size, 10)
    y = torch.randn(batch_size, 5)
    
    # Create optimizers
    individual_optimizer = optim.SGD(individual_model.parameters(), lr=0.01)
    flattened_optimizer = optim.SGD(flattened_model.parameters(), lr=0.01)
    
    loss_fn = nn.MSELoss()
    
    num_steps = 5
    max_diff = 0.0
    
    for step in range(num_steps):
        # Train with individual parameters DDP
        individual_optimizer.zero_grad()
        individual_output = individual_model(x)
        individual_loss = loss_fn(individual_output, y)
        individual_loss.backward()
        individual_model.finish_gradient_synchronization()
        individual_optimizer.step()
        
        # Train with flattened gradients DDP
        flattened_optimizer.zero_grad()
        flattened_output = flattened_model(x)
        flattened_loss = loss_fn(flattened_output, y)
        flattened_loss.backward()
        flattened_model.finish_gradient_synchronization()
        flattened_optimizer.step()
        
        # Compare parameters
        for ip, fp in zip(individual_model.module.parameters(), 
                         flattened_model.module.parameters()):
            diff = (ip - fp).abs().max().item()
            max_diff = max(max_diff, diff)
    
    # Report results from rank 0
    if rank == 0:
        success = max_diff < 1e-6
        results_queue.put((success, max_diff))
    
    cleanup_process_group()


def main():
    world_size = 2
    
    print("=" * 60)
    print("Testing DDPFlattenedGradients Correctness")
    print("=" * 60)
    print()
    
    # Create queue for results
    manager = mp.Manager()
    results_queue = manager.Queue()
    
    # Spawn processes
    mp.spawn(
        test_ddp_equivalence,
        args=(world_size, results_queue),
        nprocs=world_size,
        join=True
    )
    
    # Get results
    success, max_diff = results_queue.get()
    
    print(f"Maximum parameter difference: {max_diff:.2e}")
    print()
    
    if success:
        print("✓ SUCCESS: DDPFlattenedGradients produces the same results")
        print("           as DDPIndividualParameters!")
    else:
        print("✗ FAILURE: Results do not match")
        print(f"           Max difference: {max_diff:.2e} (threshold: 1e-6)")
    
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
