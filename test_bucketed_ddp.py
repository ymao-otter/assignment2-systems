#!/usr/bin/env python3
"""
Test and demonstrate the bucketed DDP implementation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from copy import deepcopy
import os


class TestModel(nn.Module):
    """Model with known parameter sizes for testing bucketing."""
    
    def __init__(self):
        super().__init__()
        # Create layers with specific sizes
        self.fc1 = nn.Linear(100, 200)  # 20,000 + 200 = 20,200 params
        self.fc2 = nn.Linear(200, 300)  # 60,000 + 300 = 60,300 params
        self.fc3 = nn.Linear(300, 100)  # 30,000 + 100 = 30,100 params
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def setup_process_group(rank: int, world_size: int):
    """Initialize the distributed process group."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12359"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup_process_group():
    """Clean up the distributed process group."""
    dist.barrier()
    dist.destroy_process_group()


def test_bucketing(rank: int, world_size: int, results_queue):
    """Test that bucketing works correctly and produces correct results."""
    setup_process_group(rank, world_size)
    
    # Set random seed
    torch.manual_seed(42)
    
    # Create model
    model = TestModel()
    
    # Test different bucket sizes
    from cs336_systems.ddp import DDPBucketed
    
    # Small buckets (should create multiple buckets)
    ddp_small = DDPBucketed(deepcopy(model), bucket_size_mb=0.0001)  # ~100 bytes
    
    # Large buckets (should create 1 bucket)
    ddp_large = DDPBucketed(deepcopy(model), bucket_size_mb=1.0)  # 1 MB
    
    if rank == 0:
        info = {}
        info['small_buckets'] = len(ddp_small.buckets)
        info['large_buckets'] = len(ddp_large.buckets)
        
        # Calculate total parameters
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        info['total_params'] = total_params
        
        # Check bucket contents for small bucket version
        bucket_sizes = []
        for bucket in ddp_small.buckets:
            bucket_size = sum(p.numel() for p in bucket)
            bucket_sizes.append(bucket_size)
        info['bucket_sizes'] = bucket_sizes
        
        results_queue.put(('info', info))
    
    # Test correctness - train both and check they produce same results
    torch.manual_seed(42 + rank)
    x = torch.randn(16, 100)
    y = torch.randn(16, 100)
    
    optimizer_small = optim.SGD(ddp_small.parameters(), lr=0.01)
    optimizer_large = optim.SGD(ddp_large.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    
    for step in range(3):
        # Train small bucket version
        optimizer_small.zero_grad()
        output_small = ddp_small(x)
        loss_small = loss_fn(output_small, y)
        loss_small.backward()
        ddp_small.finish_gradient_synchronization()
        optimizer_small.step()
        
        # Train large bucket version
        optimizer_large.zero_grad()
        output_large = ddp_large(x)
        loss_large = loss_fn(output_large, y)
        loss_large.backward()
        ddp_large.finish_gradient_synchronization()
        optimizer_large.step()
    
    # Check that both versions produce the same parameters
    if rank == 0:
        max_diff = 0.0
        for p_small, p_large in zip(ddp_small.module.parameters(), 
                                     ddp_large.module.parameters()):
            diff = (p_small - p_large).abs().max().item()
            max_diff = max(max_diff, diff)
        
        success = max_diff < 1e-6
        results_queue.put(('result', (success, max_diff)))
    
    cleanup_process_group()


def main():
    world_size = 2
    
    print("=" * 70)
    print("Testing Bucketed DDP Implementation")
    print("=" * 70)
    print()
    
    # Create queue for results
    manager = mp.Manager()
    results_queue = manager.Queue()
    
    # Spawn processes
    mp.spawn(
        test_bucketing,
        args=(world_size, results_queue),
        nprocs=world_size,
        join=True
    )
    
    # Get results
    info = None
    result = None
    
    while not results_queue.empty():
        msg_type, data = results_queue.get()
        if msg_type == 'info':
            info = data
        elif msg_type == 'result':
            result = data
    
    if info:
        print("Bucketing Information:")
        print(f"  Total parameters: {info['total_params']:,}")
        print(f"  Small bucket size (0.0001 MB): {info['small_buckets']} buckets")
        print(f"  Large bucket size (1.0 MB): {info['large_buckets']} bucket(s)")
        print()
        
        print("Bucket sizes (small bucket configuration):")
        for i, size in enumerate(info['bucket_sizes']):
            print(f"  Bucket {i}: {size:,} parameters")
        print()
    
    if result:
        success, max_diff = result
        print(f"Correctness check:")
        print(f"  Maximum parameter difference: {max_diff:.2e}")
        print()
        
        if success:
            print("✓ SUCCESS: Bucketed DDP produces correct results!")
            print("  Both small and large bucket configurations produce identical parameters.")
        else:
            print("✗ FAILURE: Parameters differ between bucket configurations")
            print(f"  Max difference: {max_diff:.2e} (threshold: 1e-6)")
    
    print()
    print("=" * 70)
    print()
    
    # Verify bucketing behavior
    if info:
        if info['small_buckets'] > info['large_buckets']:
            print("✓ Bucketing works as expected:")
            print(f"  Small buckets (0.0001 MB) created {info['small_buckets']} buckets")
            print(f"  Large buckets (1.0 MB) created {info['large_buckets']} bucket(s)")
        else:
            print("⚠ Warning: Bucketing may not be working correctly")


if __name__ == "__main__":
    main()
