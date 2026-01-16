"""
Benchmarking script for Transformer model forward and backward passes.

This script supports:
- Initialization of a BasicsTransformerLM model with given hyperparameters
- Generation of random batch data
- Warm-up steps before timing
- Timing of forward-only or forward+backward passes
- CUDA synchronization after each step
"""

import argparse
import timeit
from typing import Optional

import torch
import torch.nn as nn
import numpy as np

# Import the model from cs336_basics
# Note: cs336_basics is a dependency declared in pyproject.toml
# Run with: uv run python3 -m cs336_systems.benchmark
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cs336-basics'))
from cs336_basics.model import BasicsTransformerLM  # type: ignore[import-not-found]


def create_random_batch(
    batch_size: int,
    sequence_length: int,
    vocab_size: int,
    device: str = "cpu"
) -> torch.Tensor:
    """Create a random batch of token IDs.
    
    Args:
        batch_size: Number of sequences in the batch
        sequence_length: Length of each sequence
        vocab_size: Size of the vocabulary (for random token generation)
        device: Device to create the batch on
    
    Returns:
        Random token IDs of shape (batch_size, sequence_length)
    """
    return torch.randint(
        0, vocab_size, 
        (batch_size, sequence_length), 
        device=device
    )


def benchmark_step(
    model: nn.Module,
    batch: torch.Tensor,
    include_backward: bool = False,
    device: str = "cpu"
) -> None:
    """Run a single benchmark step (forward and optionally backward pass).
    
    Args:
        model: The model to benchmark
        batch: Input batch
        include_backward: Whether to include backward pass
        device: Device being used
    """
    # Forward pass
    logits = model(batch)
    
    if include_backward:
        # Compute a simple loss and backward pass
        # Using a dummy target (same as input for simplicity)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            batch.view(-1)
        )
        loss.backward()
        
        # Zero gradients for next iteration
        model.zero_grad()
    
    # Synchronize CUDA if using GPU
    if device.startswith("cuda") or device == "gpu":
        torch.cuda.synchronize()


def run_benchmark(
    model: nn.Module,
    batch: torch.Tensor,
    num_steps: int,
    num_warmup: int,
    include_backward: bool,
    device: str
) -> dict:
    """Run the full benchmark with warmup and timing.
    
    Args:
        model: The model to benchmark
        batch: Input batch
        num_steps: Number of steps to time
        num_warmup: Number of warmup steps
        include_backward: Whether to include backward pass
        device: Device being used
    
    Returns:
        Dictionary with timing results including mean and std dev
    """
    print(f"\nRunning {num_warmup} warmup steps...")
    for _ in range(num_warmup):
        benchmark_step(model, batch, include_backward, device)
    
    print(f"Running {num_steps} timed steps...")
    
    # Track individual step times
    step_times = []
    
    for _ in range(num_steps):
        step_start = timeit.default_timer()
        benchmark_step(model, batch, include_backward, device)
        step_end = timeit.default_timer()
        step_times.append(step_end - step_start)
    
    # Calculate statistics
    step_times_array = np.array(step_times)
    total_time = np.sum(step_times_array)
    avg_time = np.mean(step_times_array)
    std_time = np.std(step_times_array)
    
    return {
        "total_time": total_time,
        "avg_time": avg_time,
        "std_time": std_time,
        "num_steps": num_steps,
        "throughput": num_steps / total_time,  # steps per second
        "step_times": step_times_array
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Transformer model forward and backward passes"
    )
    
    # Model hyperparameters
    parser.add_argument(
        "--vocab-size", type=int, default=8192,
        help="Vocabulary size (default: 8192)"
    )
    parser.add_argument(
        "--context-length", type=int, default=512,
        help="Maximum context length (default: 512)"
    )
    parser.add_argument(
        "--d-model", type=int, default=512,
        help="Model dimension (default: 512)"
    )
    parser.add_argument(
        "--num-layers", type=int, default=6,
        help="Number of transformer layers (default: 6)"
    )
    parser.add_argument(
        "--num-heads", type=int, default=8,
        help="Number of attention heads (default: 8)"
    )
    parser.add_argument(
        "--d-ff", type=int, default=2048,
        help="Feed-forward dimension (default: 2048)"
    )
    parser.add_argument(
        "--rope-theta", type=float, default=10000.0,
        help="RoPE theta parameter (default: 10000.0)"
    )
    
    # Batch parameters
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size (default: 8)"
    )
    parser.add_argument(
        "--sequence-length", type=int, default=256,
        help="Sequence length (default: 256)"
    )
    
    # Benchmark parameters
    parser.add_argument(
        "--num-steps", type=int, default=100,
        help="Number of steps to time (default: 100)"
    )
    parser.add_argument(
        "--num-warmup", type=int, default=10,
        help="Number of warmup steps (default: 10)"
    )
    parser.add_argument(
        "--backward", action="store_true",
        help="Include backward pass in benchmark"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        choices=["cpu", "cuda", "cuda:0", "cuda:1"],
        help="Device to run benchmark on (default: cuda)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed(args.seed)
    
    # Validate device
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = "cpu"
    
    print("=" * 80)
    print("TRANSFORMER MODEL BENCHMARK")
    print("=" * 80)
    
    # Print configuration
    print("\nModel Configuration:")
    print(f"  Vocabulary Size: {args.vocab_size}")
    print(f"  Context Length: {args.context_length}")
    print(f"  Model Dimension: {args.d_model}")
    print(f"  Number of Layers: {args.num_layers}")
    print(f"  Number of Heads: {args.num_heads}")
    print(f"  Feed-Forward Dimension: {args.d_ff}")
    print(f"  RoPE Theta: {args.rope_theta}")
    
    print("\nBatch Configuration:")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Sequence Length: {args.sequence_length}")
    
    print("\nBenchmark Configuration:")
    print(f"  Warmup Steps: {args.num_warmup}")
    print(f"  Timed Steps: {args.num_steps}")
    print(f"  Include Backward: {args.backward}")
    print(f"  Device: {device}")
    print(f"  Random Seed: {args.seed}")
    
    # Initialize model
    print("\nInitializing model...")
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    )
    
    # Move model to device
    model = model.to(device)
    
    # Set model to training mode if doing backward passes
    if args.backward:
        model.train()
    else:
        model.eval()
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,} ({num_params / 1e6:.2f}M)")
    
    # Generate random batch
    print("\nGenerating random batch...")
    batch = create_random_batch(
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        vocab_size=args.vocab_size,
        device=device
    )
    
    # Run benchmark
    results = run_benchmark(
        model=model,
        batch=batch,
        num_steps=args.num_steps,
        num_warmup=args.num_warmup,
        include_backward=args.backward,
        device=device
    )
    
    # Print results
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(f"\nTotal time: {results['total_time']:.4f} seconds")
    print(f"Average time per step: {results['avg_time']:.4f} seconds")
    print(f"Average time per step: {results['avg_time'] * 1000:.2f} ms")
    print(f"Standard deviation: {results['std_time']:.4f} seconds")
    print(f"Standard deviation: {results['std_time'] * 1000:.2f} ms")
    print(f"Coefficient of variation: {(results['std_time'] / results['avg_time'] * 100):.2f}%")
    print(f"Throughput: {results['throughput']:.2f} steps/second")
    
    # Calculate tokens per second
    tokens_per_step = args.batch_size * args.sequence_length
    tokens_per_second = tokens_per_step * results['throughput']
    print(f"\nTokens per step: {tokens_per_step:,}")
    print(f"Throughput: {tokens_per_second:,.0f} tokens/second")
    
    if device.startswith("cuda"):
        print(f"\nPeak memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        print(f"Peak memory reserved: {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
