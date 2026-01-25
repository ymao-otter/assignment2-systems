"""
Benchmark script for scaled dot-product attention at different scales.

This script benchmarks the attention implementation with:
- Fixed batch size: 8
- No multihead attention (no head dimension)
- dmodel: [16, 32, 64, 128]
- sequence_length: [256, 1024, 4096, 8192, 16384]

Compares both uncompiled and torch.compile() versions.
"""

import torch
import time
from typing import Tuple, Optional
import pandas as pd
import sys
import os

# Add the cs336-basics directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cs336-basics'))

from cs336_basics.model import scaled_dot_product_attention

# Configure torch.compile to suppress errors and use appropriate backend
import torch._dynamo
torch._dynamo.config.suppress_errors = True


class AttentionModule(torch.nn.Module):
    """Wrapper module for scaled_dot_product_attention to make it compilable."""
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask=None):
        return scaled_dot_product_attention(Q, K, V, mask)


def benchmark_attention(
    batch_size: int,
    seq_len: int,
    d_model: int,
    num_iterations: int = 100,
    warmup_iterations: int = 10,
    device: str = 'cuda',
    use_compile: bool = False
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Benchmark attention forward and backward passes.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        d_model: Model dimension (head embedding dimension)
        num_iterations: Number of iterations to time
        warmup_iterations: Number of warmup iterations
        device: Device to run on
        use_compile: Whether to use torch.compile()

    Returns:
        Tuple of (forward_time_ms, backward_time_ms, memory_before_backward_mb)
        Returns (None, None, None) if OOM occurs
    """
    try:
        # Create the attention function (compiled or not)
        if use_compile:
            # Use a module wrapper for better compilation support
            attention_module = AttentionModule().to(device)
            attention_fn = torch.compile(attention_module)
        else:
            attention_fn = scaled_dot_product_attention

        # Create random inputs Q, K, V
        # Shape: (batch_size, seq_len, d_model)
        Q = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
        K = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
        V = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
        
        # Warmup
        for _ in range(warmup_iterations):
            output = attention_fn(Q, K, V)
            loss = output.sum()
            loss.backward()
            torch.cuda.synchronize()

            # Clear gradients
            Q.grad = None
            K.grad = None
            V.grad = None

        # Benchmark forward pass
        torch.cuda.synchronize()
        forward_start = time.time()

        for _ in range(num_iterations):
            output = attention_fn(Q, K, V)
            torch.cuda.synchronize()

        forward_end = time.time()
        forward_time_ms = (forward_end - forward_start) / num_iterations * 1000

        # Measure memory before backward pass
        # First run one forward pass to create the computation graph
        output = attention_fn(Q, K, V)
        torch.cuda.synchronize()

        # Measure memory
        memory_before_backward_bytes = torch.cuda.memory_allocated(device)
        memory_before_backward_mb = memory_before_backward_bytes / (1024 ** 2)

        # Benchmark backward pass
        torch.cuda.synchronize()
        backward_start = time.time()

        for i in range(num_iterations):
            if i > 0:
                # Need to recompute forward for each backward
                output = attention_fn(Q, K, V)
                torch.cuda.synchronize()

            loss = output.sum()
            loss.backward()
            torch.cuda.synchronize()

            # Clear gradients for next iteration
            Q.grad = None
            K.grad = None
            V.grad = None

        backward_end = time.time()
        backward_time_ms = (backward_end - backward_start) / num_iterations * 1000
        
        return forward_time_ms, backward_time_ms, memory_before_backward_mb
        
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None, None, None
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return None, None, None
        else:
            raise


def main():
    """Run the attention benchmark comparing compiled and uncompiled versions."""
    # Configuration
    batch_size = 8
    d_models = [16, 32, 64, 128]
    seq_lengths = [256, 1024, 4096, 8192, 16384]
    num_iterations = 100
    warmup_iterations = 10

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if device == 'cpu':
        print("WARNING: CUDA not available, running on CPU. Results may differ significantly.")

    # Store results
    results = []

    print("\nBenchmarking attention implementations (uncompiled vs compiled)...")
    print(f"Batch size: {batch_size}")
    print(f"Iterations: {num_iterations} (after {warmup_iterations} warmup)")
    print("=" * 80)

    # Run benchmarks for both uncompiled and compiled versions
    for use_compile in [False, True]:
        version = "Compiled" if use_compile else "Uncompiled"
        print(f"\n{version} Attention:")
        print("-" * 80)

        for d_model in d_models:
            for seq_len in seq_lengths:
                print(f"Testing d_model={d_model}, seq_len={seq_len}...", end=" ", flush=True)

                # Clear cache before each run
                if device == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()

                forward_time, backward_time, memory_mb = benchmark_attention(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    d_model=d_model,
                    num_iterations=num_iterations,
                    warmup_iterations=warmup_iterations,
                    device=device,
                    use_compile=use_compile
                )

                if forward_time is None:
                    print("OOM")
                    results.append({
                        'version': version,
                        'd_model': d_model,
                        'seq_len': seq_len,
                        'forward_time_ms': 'OOM',
                        'backward_time_ms': 'OOM',
                        'memory_mb': 'OOM'
                    })
                else:
                    print(f"Forward: {forward_time:.3f}ms, Backward: {backward_time:.3f}ms, Memory: {memory_mb:.2f}MB")
                    results.append({
                        'version': version,
                        'd_model': d_model,
                        'seq_len': seq_len,
                        'forward_time_ms': forward_time,
                        'backward_time_ms': backward_time,
                        'memory_mb': memory_mb
                    })

    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save full results
    output_file = 'attention_benchmark_results.csv'
    df.to_csv(output_file, index=False)
    print(f"\n\nFull results saved to: {output_file}")

    # Create comparison tables for the configuration: d_model=128 (as mentioned in the problem)
    print("\n" + "=" * 80)
    print("COMPARISON TABLE: Uncompiled vs Compiled Attention")
    print("Configuration: batch_size=8, d_model=128")
    print("=" * 80)

    # Filter for d_model=128
    df_128 = df[df['d_model'] == 128].copy()

    # Create comparison table
    comparison_data = []
    for seq_len in seq_lengths:
        row_data = {'seq_len': seq_len}

        # Get uncompiled times
        uncompiled = df_128[(df_128['seq_len'] == seq_len) & (df_128['version'] == 'Uncompiled')]
        if not uncompiled.empty:
            unc_fwd = uncompiled.iloc[0]['forward_time_ms']
            unc_bwd = uncompiled.iloc[0]['backward_time_ms']
            row_data['uncompiled_forward_ms'] = unc_fwd if unc_fwd != 'OOM' else 'OOM'
            row_data['uncompiled_backward_ms'] = unc_bwd if unc_bwd != 'OOM' else 'OOM'
        else:
            row_data['uncompiled_forward_ms'] = 'N/A'
            row_data['uncompiled_backward_ms'] = 'N/A'

        # Get compiled times
        compiled = df_128[(df_128['seq_len'] == seq_len) & (df_128['version'] == 'Compiled')]
        if not compiled.empty:
            comp_fwd = compiled.iloc[0]['forward_time_ms']
            comp_bwd = compiled.iloc[0]['backward_time_ms']
            row_data['compiled_forward_ms'] = comp_fwd if comp_fwd != 'OOM' else 'OOM'
            row_data['compiled_backward_ms'] = comp_bwd if comp_bwd != 'OOM' else 'OOM'
        else:
            row_data['compiled_forward_ms'] = 'N/A'
            row_data['compiled_backward_ms'] = 'N/A'

        # Calculate speedup if both are available
        if (isinstance(unc_fwd, (int, float)) and isinstance(comp_fwd, (int, float)) and
            unc_fwd != 'OOM' and comp_fwd != 'OOM'):
            row_data['forward_speedup'] = f"{unc_fwd / comp_fwd:.2f}x"
        else:
            row_data['forward_speedup'] = 'N/A'

        if (isinstance(unc_bwd, (int, float)) and isinstance(comp_bwd, (int, float)) and
            unc_bwd != 'OOM' and comp_bwd != 'OOM'):
            row_data['backward_speedup'] = f"{unc_bwd / comp_bwd:.2f}x"
        else:
            row_data['backward_speedup'] = 'N/A'

        comparison_data.append(row_data)

    comparison_df = pd.DataFrame(comparison_data)

    # Format the table nicely
    print("\nForward Pass Timing (ms):")
    print("-" * 80)
    print(f"{'Seq Len':<10} {'Uncompiled':<15} {'Compiled':<15} {'Speedup':<15}")
    print("-" * 80)
    for _, row in comparison_df.iterrows():
        seq = row['seq_len']
        unc = f"{row['uncompiled_forward_ms']:.3f}" if isinstance(row['uncompiled_forward_ms'], (int, float)) else row['uncompiled_forward_ms']
        comp = f"{row['compiled_forward_ms']:.3f}" if isinstance(row['compiled_forward_ms'], (int, float)) else row['compiled_forward_ms']
        speedup = row['forward_speedup']
        print(f"{seq:<10} {unc:<15} {comp:<15} {speedup:<15}")

    print("\nBackward Pass Timing (ms):")
    print("-" * 80)
    print(f"{'Seq Len':<10} {'Uncompiled':<15} {'Compiled':<15} {'Speedup':<15}")
    print("-" * 80)
    for _, row in comparison_df.iterrows():
        seq = row['seq_len']
        unc = f"{row['uncompiled_backward_ms']:.3f}" if isinstance(row['uncompiled_backward_ms'], (int, float)) else row['uncompiled_backward_ms']
        comp = f"{row['compiled_backward_ms']:.3f}" if isinstance(row['compiled_backward_ms'], (int, float)) else row['compiled_backward_ms']
        speedup = row['backward_speedup']
        print(f"{seq:<10} {unc:<15} {comp:<15} {speedup:<15}")

    print("\n" + "=" * 80)
    
    # Memory analysis
    print("\n" + "=" * 80)
    print("\nMEMORY ANALYSIS")
    print("=" * 80)
    
    # Find smallest OOM configuration
    oom_configs = df[df['forward_time_ms'] == 'OOM']
    if len(oom_configs) > 0:
        print("\nConfigurations that ran out of memory:")
        for _, row in oom_configs.iterrows():
            print(f"  d_model={row['d_model']}, seq_len={row['seq_len']}")
        
        # Analyze smallest OOM
        smallest_oom = oom_configs.nsmallest(1, ['seq_len', 'd_model']).iloc[0]
        d = smallest_oom['d_model']
        s = smallest_oom['seq_len']
        b = batch_size
        
        print(f"\nMemory accounting for smallest OOM (d_model={d}, seq_len={s}, batch={b}):")
        print(f"\nInputs (Q, K, V):")
        print(f"  3 × {b} × {s} × {d} × 4 bytes = {3 * b * s * d * 4 / (1024**2):.2f} MB")
        
        print(f"\nAttention scores (before softmax):")
        print(f"  {b} × {s} × {s} × 4 bytes = {b * s * s * 4 / (1024**2):.2f} MB")
        
        print(f"\nAttention weights (after softmax):")
        print(f"  {b} × {s} × {s} × 4 bytes = {b * s * s * 4 / (1024**2):.2f} MB")
        
        print(f"\nOutput:")
        print(f"  {b} × {s} × {d} × 4 bytes = {b * s * d * 4 / (1024**2):.2f} MB")
        
        print(f"\nGradients for backward (Q, K, V):")
        print(f"  3 × {b} × {s} × {d} × 4 bytes = {3 * b * s * d * 4 / (1024**2):.2f} MB")
        
        print(f"\nSaved activations for backward (attention scores/weights):")
        print(f"  2 × {b} × {s} × {s} × 4 bytes = {2 * b * s * s * 4 / (1024**2):.2f} MB")
        
        total_mb = (3 * b * s * d * 4 + 2 * b * s * s * 4 + b * s * d * 4 + 3 * b * s * d * 4 + 2 * b * s * s * 4) / (1024**2)
        print(f"\nEstimated total memory: ~{total_mb:.2f} MB")
        print(f"\nNote: The attention score matrices ({s}×{s}) dominate memory at large sequence lengths.")
        print(f"      Memory scales as O(batch × seq² × 4) for saved activations.")
    else:
        print("\nNo OOM errors encountered in this benchmark.")
    
    # Analyze how memory changes with sequence length
    print("\n" + "=" * 80)
    print("\nMEMORY SCALING WITH SEQUENCE LENGTH")
    print("=" * 80)
    
    # Filter successful runs and analyze memory scaling
    successful_runs = df[df['memory_mb'] != 'OOM'].copy()
    if len(successful_runs) > 0:
        successful_runs['memory_mb_float'] = successful_runs['memory_mb'].astype(float)
        
        print("\nFor fixed d_model, memory vs sequence length:")
        for d_model in d_models:
            subset = successful_runs[successful_runs['d_model'] == d_model]
            if len(subset) > 0:
                print(f"\n  d_model={d_model}:")
                for _, row in subset.iterrows():
                    seq_len = row['seq_len']
                    mem = row['memory_mb_float']
                    # Memory for attention matrix: b × s² × 4 bytes
                    attention_matrix_mb = batch_size * seq_len * seq_len * 4 / (1024**2)
                    print(f"    seq_len={seq_len:5d}: {mem:8.2f} MB (attention matrix alone: {attention_matrix_mb:.2f} MB)")
        
        print("\nKey observation:")
        print("  - Memory grows quadratically with sequence length (O(s²))")
        print("  - The attention score matrix (batch × seq × seq) is saved for backward pass")
        print("  - This is the main memory bottleneck in standard attention")
    
    print("\n" + "=" * 80)
    print("\nSOLUTIONS TO REDUCE MEMORY COST")
    print("=" * 80)
    print("""
1. Flash Attention / Memory-Efficient Attention:
   - Recompute attention scores during backward instead of saving them
   - Trades computation for memory (memory O(n) instead of O(n²))
   - Uses kernel fusion and tiling to maintain efficiency

2. Gradient Checkpointing:
   - Recompute activations during backward instead of storing them
   - Reduces memory at the cost of ~33% more computation

3. Sparse Attention:
   - Only compute attention for a subset of positions
   - Reduces both memory and computation from O(n²) to O(n√n) or O(n log n)

4. Approximate Attention (e.g., Linformer, Performer):
   - Use low-rank approximations of the attention matrix
   - Reduces complexity to O(n) with controlled accuracy tradeoff

5. Mixed Precision Training:
   - Use FP16/BF16 for activations (halves memory)
   - Though this doesn't change the O(n²) scaling
""")


if __name__ == "__main__":
    main()
