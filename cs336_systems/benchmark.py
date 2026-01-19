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

# Import NVTX for profiling annotations
try:
    import torch.cuda.nvtx as nvtx  # type: ignore
    NVTX_AVAILABLE = True
except ImportError:
    NVTX_AVAILABLE = False
    # Provide no-op context manager if NVTX not available
    class nvtx:  # type: ignore
        @staticmethod
        def range(msg):
            class DummyContext:
                def __enter__(self): return self
                def __exit__(self, *args): pass
            return DummyContext()


# Global variable to track profiling phase (warmup vs timed)
# This allows detailed layer annotations to distinguish between phases
_profiling_phase = ""


class InstrumentedAttention(nn.Module):
    """
    Wrapper to add NVTX annotations to attention layers.
    
    This allows you to see which specific attention operations are taking time.
    """
    
    def __init__(self, attention_module, layer_idx: int = 0):
        super().__init__()
        self.attention = attention_module
        self.layer_idx = layer_idx
    
    def forward(self, x, *args, **kwargs):
        label = f"Layer {self.layer_idx} - Attention"
        if _profiling_phase:
            label = f"{label} ({_profiling_phase})"
        with nvtx.range(label):
            output = self.attention(x, *args, **kwargs)
        return output


class InstrumentedFFN(nn.Module):
    """
    Wrapper to add NVTX annotations to feed-forward layers.
    """
    
    def __init__(self, ffn_module, layer_idx: int = 0):
        super().__init__()
        self.ffn = ffn_module
        self.layer_idx = layer_idx
    
    def forward(self, x):
        label = f"Layer {self.layer_idx} - FFN"
        if _profiling_phase:
            label = f"{label} ({_profiling_phase})"
        with nvtx.range(label):
            output = self.ffn(x)
        return output


class InstrumentedLayerNorm(nn.Module):
    """
    Wrapper to add NVTX annotations to LayerNorm operations.
    """
    
    def __init__(self, ln_module, layer_idx: int = 0, position: str = ""):
        super().__init__()
        self.ln = ln_module
        self.layer_idx = layer_idx
        self.position = position
    
    def forward(self, x):
        label = f"Layer {self.layer_idx} - LayerNorm"
        if self.position:
            label += f" ({self.position})"
        if _profiling_phase:
            label = f"{label} [{_profiling_phase}]"
        with nvtx.range(label):
            output = self.ln(x)
        return output


def instrument_transformer_model(model):
    """
    Add detailed NVTX annotations to a transformer model's layers.
    
    This function wraps attention, FFN, and LayerNorm layers with NVTX ranges
    so you can see their individual contributions in the profiler.
    
    Args:
        model: A transformer model (e.g., BasicsTransformerLM)
    
    Returns:
        The instrumented model
    """
    if not NVTX_AVAILABLE:
        print("Warning: NVTX not available, model will not be instrumented")
        return model
    
    print("Instrumenting model with detailed NVTX annotations...")
    
    # Try different common attribute names for transformer layers
    layers_attr = None
    for attr_name in ['layers', 'transformer_layers', 'blocks', 'h']:
        if hasattr(model, attr_name):
            layers_attr = attr_name
            break
    
    if layers_attr is None:
        print(f"Warning: Could not find transformer layers in model. "
              f"Model attributes: {dir(model)}")
        return model
    
    layers = getattr(model, layers_attr)
    num_instrumented = 0
    
    for i, layer in enumerate(layers):
        # Instrument attention modules
        for attn_name in ['attention', 'self_attn', 'attn']:
            if hasattr(layer, attn_name):
                attn = getattr(layer, attn_name)
                if attn is not None and not isinstance(attn, InstrumentedAttention):
                    setattr(layer, attn_name, InstrumentedAttention(attn, i))
                    num_instrumented += 1
                break
        
        # Instrument FFN/MLP modules
        for ffn_name in ['ffn', 'mlp', 'feed_forward']:
            if hasattr(layer, ffn_name):
                ffn = getattr(layer, ffn_name)
                if ffn is not None and not isinstance(ffn, InstrumentedFFN):
                    setattr(layer, ffn_name, InstrumentedFFN(ffn, i))
                    num_instrumented += 1
                break
        
        # Instrument LayerNorm modules
        for ln_name, position in [('ln_1', 'pre-attn'), ('ln_2', 'pre-ffn'), 
                                   ('norm1', 'pre-attn'), ('norm2', 'pre-ffn')]:
            if hasattr(layer, ln_name):
                ln = getattr(layer, ln_name)
                if ln is not None and not isinstance(ln, InstrumentedLayerNorm):
                    setattr(layer, ln_name, InstrumentedLayerNorm(ln, i, position))
                    num_instrumented += 1
    
    print(f"Instrumented {num_instrumented} modules across {len(layers)} layers")
    return model

# Import the model from cs336_basics
# Note: cs336_basics is a dependency declared in pyproject.toml
# Run with: uv run python3 -m cs336_systems.benchmark
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cs336-basics'))
from cs336_basics.model import BasicsTransformerLM  # type: ignore[import-not-found]
from cs336_basics.optimizer import AdamW  # type: ignore[import-not-found]


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
    device: str = "cpu",
    step_idx: int = 0,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> None:
    """Run a single benchmark step (forward and optionally backward pass).
    
    Args:
        model: The model to benchmark
        batch: Input batch
        include_backward: Whether to include backward pass
        device: Device being used
        step_idx: Step index for NVTX annotation
        optimizer: Optimizer to use for parameter updates (if provided)
    """
    # Forward pass
    phase_label = f" ({_profiling_phase})" if _profiling_phase else ""
    with nvtx.range(f"Forward Pass{phase_label} (step {step_idx})"):
        logits = model(batch)
    
    if include_backward:
        # Compute a simple loss and backward pass
        # Using a dummy target (same as input for simplicity)
        with nvtx.range(f"Loss Computation{phase_label} (step {step_idx})"):
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                batch.view(-1)
            )
        
        with nvtx.range(f"Backward Pass{phase_label} (step {step_idx})"):
            loss.backward()
        
        # Optimizer step if provided
        if optimizer is not None:
            with nvtx.range(f"Optimizer Step{phase_label} (step {step_idx})"):
                optimizer.step()
        
        # Zero gradients for next iteration
        with nvtx.range(f"Zero Grad{phase_label} (step {step_idx})"):
            if optimizer is not None:
                optimizer.zero_grad()
            else:
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
    device: str,
    optimizer: Optional[torch.optim.Optimizer] = None
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
    global _profiling_phase
    
    print(f"\nRunning {num_warmup} warmup steps...")
    # Mark warmup steps with NVTX so they can be filtered out in the profiler
    _profiling_phase = "warmup"
    with nvtx.range("Warmup Steps"):
        for i in range(num_warmup):
            with nvtx.range(f"Warmup Step {i}"):
                benchmark_step(model, batch, include_backward, device, step_idx=i, optimizer=optimizer)
    
    print(f"Running {num_steps} timed steps...")
    
    # Track individual step times
    step_times = []
    
    # Mark timed steps with NVTX
    _profiling_phase = "timed"
    with nvtx.range("Timed Steps"):
        for i in range(num_steps):
            step_start = timeit.default_timer()
            with nvtx.range(f"Benchmark Step {i}"):
                benchmark_step(model, batch, include_backward, device, step_idx=i, optimizer=optimizer)
            step_end = timeit.default_timer()
            step_times.append(step_end - step_start)
    
    _profiling_phase = ""
    
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
    parser.add_argument(
        "--detailed-profile", action="store_true",
        help="Enable detailed NVTX profiling with per-layer annotations for attention, FFN, and LayerNorm"
    )
    parser.add_argument(
        "--optimizer", action="store_true",
        help="Include AdamW optimizer step in the benchmark (requires --backward)"
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
    print(f"  Include Optimizer: {args.optimizer}")
    print(f"  Device: {device}")
    print(f"  Random Seed: {args.seed}")
    print(f"  Detailed Profiling: {args.detailed_profile}")
    
    # Validate optimizer flag
    if args.optimizer and not args.backward:
        print("Warning: --optimizer requires --backward, enabling backward pass")
        args.backward = True
    
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
    
    # Apply detailed profiling instrumentation if requested
    if args.detailed_profile:
        model = instrument_transformer_model(model)
    
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
    
    # Initialize optimizer if requested
    optimizer = None
    if args.optimizer:
        print("\nInitializing AdamW optimizer...")
        optimizer = AdamW(
            model.parameters(),
            lr=1e-4,  # Default learning rate
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )
        print(f"Optimizer: AdamW (lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01)")
    
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
        device=device,
        optimizer=optimizer
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
