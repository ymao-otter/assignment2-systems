# Transformer Model Benchmarking

A comprehensive benchmarking script for the `BasicsTransformerLM` model to measure forward and backward pass performance.

## Quick Start

```bash
# Simple forward-only benchmark (uses GPU by default)
uv run python3 -m cs336_systems.benchmark

# With backward pass
uv run python3 -m cs336_systems.benchmark --backward

# Fast test (small model, few steps)
uv run python3 -m cs336_systems.benchmark \
    --d-model 128 --num-layers 2 --num-steps 5 --backward

# Run on CPU instead
uv run python3 -m cs336_systems.benchmark --device cpu
```

## What This Does

The benchmark script:
1. ✅ Initializes a Transformer model with configurable hyperparameters
2. ✅ Generates random input data
3. ✅ Runs warm-up steps (to stabilize performance)
4. ✅ Times forward passes (and optionally backward passes)
5. ✅ Uses `timeit.default_timer()` for high-resolution timing
6. ✅ Calls `torch.cuda.synchronize()` after each step (when using GPU)
7. ✅ Reports detailed timing and throughput metrics

## Command-Line Options

### Model Hyperparameters
- `--vocab-size`: Vocabulary size (default: 8192)
- `--context-length`: Maximum context length (default: 512)
- `--d-model`: Model embedding dimension (default: 512)
- `--num-layers`: Number of Transformer layers (default: 6)
- `--num-heads`: Number of attention heads (default: 8)
- `--d-ff`: Feed-forward network dimension (default: 2048)
- `--rope-theta`: RoPE theta parameter (default: 10000.0)

### Batch Configuration
- `--batch-size`: Number of sequences per batch (default: 8)
- `--sequence-length`: Length of each sequence (default: 256)

### Benchmark Configuration
- `--num-steps`: Number of timed iterations (default: 100)
- `--num-warmup`: Number of warm-up iterations before timing (default: 10)
- `--backward`: Include backward pass in benchmark (flag, default: False)
- `--device`: Device to run on: `cpu`, `cuda`, `cuda:0`, `cuda:1` (default: cuda)
- `--seed`: Random seed for reproducibility (default: 42)

## Usage Examples

### 1. Test Different Model Sizes

```bash
# Small model
uv run python3 -m cs336_systems.benchmark \
    --num-layers 2 --d-model 256

# Medium model
uv run python3 -m cs336_systems.benchmark \
    --num-layers 6 --d-model 512

# Large model
uv run python3 -m cs336_systems.benchmark \
    --num-layers 12 --d-model 768
```

### 2. Compare Forward vs Forward+Backward

```bash
# Forward only
uv run python3 -m cs336_systems.benchmark \
    --num-layers 4 --num-steps 50

# Forward + Backward
uv run python3 -m cs336_systems.benchmark \
    --num-layers 4 --num-steps 50 --backward
```

### 3. Test Different Batch Sizes

```bash
# Small batch
uv run python3 -m cs336_systems.benchmark --batch-size 2

# Large batch
uv run python3 -m cs336_systems.benchmark --batch-size 32
```

### 4. GPU Benchmarking (default)

```bash
# GPU is the default device
uv run python3 -m cs336_systems.benchmark \
    --batch-size 16 \
    --num-steps 100 \
    --backward

# Or explicitly specify
uv run python3 -m cs336_systems.benchmark \
    --device cuda \
    --batch-size 16 \
    --num-steps 100 \
    --backward
```

### 5. CPU Benchmarking

```bash
# Use --device cpu to override the default
uv run python3 -m cs336_systems.benchmark \
    --device cpu \
    --batch-size 8 \
    --num-steps 50
```

## Understanding the Output

```
================================================================================
BENCHMARK RESULTS
================================================================================

Total time: 1.0742 seconds           ← Total time for all timed steps
Average time per step: 0.1074 seconds ← Time per forward (+ backward) pass
Average time per step: 107.42 ms     ← Same, in milliseconds
Throughput: 9.31 steps/second        ← Steps processed per second

Tokens per step: 512                 ← batch_size × sequence_length
Throughput: 4,766 tokens/second      ← Total tokens processed per second
================================================================================
```

The benchmark reports:
1. **Configuration Summary**: All hyperparameters used
2. **Model Size**: Total parameter count
3. **Timing Statistics**: Total time, average time per step, throughput
4. **Token Throughput**: Tokens processed per step and per second
5. **Memory Usage** (GPU only): Peak memory allocated and reserved

## Implementation Details

### Timing Methodology

Uses `timeit.default_timer()` for high-resolution timing:

```python
import timeit

start_time = timeit.default_timer()
for _ in range(num_steps):
    benchmark_step(model, batch, include_backward, device)
end_time = timeit.default_timer()
```

**Why `timeit.default_timer()`?**
- Provides the system's highest resolution clock
- More accurate than `time.time()` for benchmarking
- Platform-independent

### CUDA Synchronization

Ensures accurate GPU timing by synchronizing after each step:

```python
def benchmark_step(model, batch, include_backward, device):
    logits = model(batch)
    
    if include_backward:
        loss = nn.functional.cross_entropy(...)
        loss.backward()
        model.zero_grad()
    
    # Critical: synchronize CUDA operations
    if device.startswith("cuda"):
        torch.cuda.synchronize()
```

**Why synchronization is important:**
- GPU operations are asynchronous by default
- Without sync, timing would only measure kernel launch overhead
- `torch.cuda.synchronize()` ensures all operations complete before timing

### Warm-up Steps

Run before timing to stabilize measurements:

```python
# Warm-up phase
for _ in range(num_warmup):
    benchmark_step(model, batch, include_backward, device)

# Timed phase
start_time = timeit.default_timer()
for _ in range(num_steps):
    benchmark_step(model, batch, include_backward, device)
```

**Purpose:**
- Allow JIT compilation to complete
- Warm up GPU kernels and caches
- Eliminate cold-start effects
- Ensure consistent measurements

### Backward Pass

When `--backward` is enabled:

```python
# Compute cross-entropy loss
loss = nn.functional.cross_entropy(
    logits.view(-1, logits.size(-1)),
    batch.view(-1)  # Use input as dummy target
)

# Compute gradients
loss.backward()

# Zero gradients (simulate training loop)
model.zero_grad()
```

## Advanced Usage

### Benchmarking Multiple Configurations

```bash
#!/bin/bash
# Benchmark different model sizes

for layers in 2 4 6 8; do
    echo "Benchmarking with $layers layers..."
    uv run python3 -m cs336_systems.benchmark \
        --num-layers $layers \
        --num-steps 50 \
        --backward
done
```

### Profiling Sequence Lengths

```bash
#!/bin/bash
# Test scaling with sequence length

for seq_len in 128 256 512 1024; do
    echo "Benchmarking sequence length $seq_len..."
    uv run python3 -m cs336_systems.benchmark \
        --sequence-length $seq_len \
        --num-steps 50 \
        --backward
done
```

### Programmatic Usage

Import and use the benchmark utilities in your own scripts:

```python
from cs336_systems.benchmark import create_random_batch, benchmark_step, run_benchmark
from cs336_basics.model import BasicsTransformerLM

# Initialize model
model = BasicsTransformerLM(
    vocab_size=8192,
    context_length=512,
    d_model=512,
    num_layers=6,
    num_heads=8,
    d_ff=2048,
    rope_theta=10000.0
).to("cuda")

# Create a batch
batch = create_random_batch(
    batch_size=8,
    sequence_length=256,
    vocab_size=8192,
    device="cuda"
)

# Run full benchmark
results = run_benchmark(
    model=model,
    batch=batch,
    num_steps=100,
    num_warmup=10,
    include_backward=True,
    device="cuda"
)

print(f"Average time: {results['avg_time']:.4f}s")
print(f"Throughput: {results['throughput']:.2f} steps/s")
```

## Tips for Effective Benchmarking

1. **Use adequate warm-up steps**: At least 10 for CPU, 20-50 for GPU
2. **Run enough timed steps**: At least 50-100 for stable measurements
3. **Run multiple times**: Performance can vary between runs
4. **Compare fairly**: Keep batch/sequence lengths constant when comparing configurations
5. **Monitor memory usage**: Watch for OOM errors with large configurations
6. **Consider device capabilities**: GPU benchmarks show very different characteristics than CPU
7. **Use `--seed` for reproducibility**: Ensures consistent results across runs

## Troubleshooting

### Out of Memory (OOM) Errors

```bash
# Reduce batch size
uv run python3 -m cs336_systems.benchmark --batch-size 2

# Reduce sequence length
uv run python3 -m cs336_systems.benchmark --sequence-length 128

# Reduce model size
uv run python3 -m cs336_systems.benchmark --d-model 256 --num-layers 4
```

### "CUDA not available" Warning

The script automatically falls back to CPU if CUDA is not available. This is normal if you don't have a GPU. You can also explicitly use CPU with `--device cpu`.

### Script Takes Too Long

```bash
# Reduce the number of steps
uv run python3 -m cs336_systems.benchmark --num-steps 10 --num-warmup 2
```

### Slow Performance

- GPU is the default; if it's slow, check GPU utilization
- If running on CPU, performance will be much slower
- Check that PyTorch is properly installed with CUDA support
- Verify no other processes are consuming GPU/CPU resources

## Validation

The benchmark has been tested and verified:

### Test 1: Forward-only pass
```bash
uv run python3 -m cs336_systems.benchmark \
    --vocab-size 1000 --d-model 128 --num-layers 2 --num-heads 4 \
    --batch-size 2 --sequence-length 64 --num-steps 5
```
✅ **Result**: ~5ms per step on CPU

### Test 2: Forward + backward pass
```bash
uv run python3 -m cs336_systems.benchmark \
    --vocab-size 1000 --d-model 128 --num-layers 2 --num-heads 4 \
    --batch-size 2 --sequence-length 64 --num-steps 5 --backward
```
✅ **Result**: ~15ms per step on CPU (3x slower than forward-only)

## Requirements Checklist

✅ Initialize model with given hyperparameters  
✅ Generate random batch data  
✅ Run w warm-up steps before timing  
✅ Time execution of n steps  
✅ Support forward-only or forward+backward  
✅ Use `timeit` module for timing  
✅ Call `torch.cuda.synchronize()` after each step  

## Help

```bash
# Show all options
uv run python3 -m cs336_systems.benchmark --help
```

---

**Implementation:** `cs336_systems/benchmark.py`
