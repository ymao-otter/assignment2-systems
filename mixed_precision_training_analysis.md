# Mixed Precision Training Analysis (BF16 vs FP32)

## Experimental Setup

**Models Tested:**
- Small: 126M params (d_model=768, num_layers=12, num_heads=12)
- Medium: 350M params (d_model=1024, num_layers=24, num_heads=16)
- Large: 774M params (d_model=1280, num_layers=36, num_heads=20) - OOM

**Configuration:**
- Context length: 512
- Batch size: 8
- Sequence length: 512
- Device: NVIDIA GPU (CUDA)
- Warmup steps: 10
- Timed steps: 50

**Modes:**
- Forward-only pass
- Forward + Backward pass

## Results

### Small Model (126M parameters)

| Mode | FP32 (ms) | BF16 (ms) | Speedup | Memory FP32 | Memory BF16 | Memory Reduction |
|------|-----------|-----------|---------|-------------|-------------|------------------|
| Forward | 117.1 | 80.9 | **1.45x** | 7.64 GB | 5.52 GB | 27.7% |
| Fwd+Bwd | 356.2 | 236.9 | **1.50x** | 8.05 GB | 5.85 GB | 27.3% |

### Medium Model (350M parameters)

| Mode | FP32 (ms) | BF16 (ms) | Speedup | Memory FP32 | Memory BF16 | Memory Reduction |
|------|-----------|-----------|---------|-------------|-------------|------------------|
| Forward | 350.6 | 212.2 | **1.65x** | 20.43 GB | 14.95 GB | 26.8% |
| Fwd+Bwd | 1040.4 | 653.2 | **1.59x** | 20.84 GB | 15.34 GB | 26.4% |

### Large Model (774M parameters)

Both FP32 and BF16 runs failed due to out-of-memory (OOM) errors with the current batch size and sequence length configuration.

## Key Findings and Commentary

**Deliverable (2-3 sentences):**

Mixed precision training with BF16 provides consistent speedups of 1.45-1.65x across model sizes, with larger models benefiting more (1.65x for medium forward-only vs 1.45x for small). Memory usage is reduced by approximately 27% consistently across both model sizes and modes, which is crucial for training larger models. The speedup trend shows that larger, more compute-intensive models benefit more from BF16's efficient tensor core utilization, suggesting that mixed precision becomes increasingly valuable as model size grows.

## Detailed Analysis

### 1. Speedup Trends

**Forward-only pass:**
- Small: 1.45x speedup
- Medium: 1.65x speedup
- **Trend:** Larger models show better speedup (+0.20x improvement)

**Forward+Backward pass:**
- Small: 1.50x speedup
- Medium: 1.59x speedup
- **Trend:** Similar improvement pattern, though slightly less pronounced

**Explanation:** Larger models spend proportionally more time in matrix multiplications (attention and FFN layers), which benefit significantly from BF16 tensor cores. Smaller models have relatively more overhead from non-compute operations (normalization, elementwise ops), diluting the speedup.

### 2. Memory Reduction

**Consistent ~27% memory reduction across all configurations:**
- Small forward: 27.7% reduction
- Small fwd+bwd: 27.3% reduction
- Medium forward: 26.8% reduction
- Medium fwd+bwd: 26.4% reduction

**Why not 50%?**
- Parameters remain in FP32 (required for optimizer stability)
- Gradients are accumulated in FP32
- Only intermediate activations are stored in BF16
- Some operations (LayerNorm, loss) remain in FP32 for numerical stability

### 3. Throughput Gains

**Tokens/second improvement:**
- Small fwd+bwd: 11,499 → 17,291 tokens/sec (1.50x)
- Medium fwd+bwd: 3,937 → 6,271 tokens/sec (1.59x)

The throughput gains directly translate to reduced training time and cost.

### 4. Scaling Implications

**Model Size vs Speedup:**
```
Small (126M):    1.45x - 1.50x
Medium (350M):   1.59x - 1.65x
Large (774M+):   Expected 1.7x+ (if memory permits)
```

This trend suggests that:
1. **Larger models benefit more** from mixed precision
2. The benefits compound with scale
3. For models like GPT-3 (175B), the speedup could be even more significant

### 5. Practical Implications

1. **Training Cost:** 1.5-1.65x speedup = 33-40% reduction in training time/cost
2. **Model Scale:** 27% memory reduction allows fitting larger batches or models
3. **Accessibility:** Enables training larger models on the same hardware
4. **Production:** Faster iteration cycles during development

### 6. Why BF16 Works Well for Transformers

1. **Compute-bound workloads:** Attention (O(n²)) and FFN layers dominate
2. **Tensor cores:** Modern GPUs have specialized BF16 hardware
3. **Numerical stability:** BF16's wider exponent range handles the dynamic range in transformers
4. **Minimal accuracy loss:** Studies show negligible difference vs FP32 for language models

## Implementation Details

The benchmark script was modified to support mixed precision via PyTorch's `torch.autocast`:

```python
# Create autocast context for BF16
autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

# Wrap forward and loss computation
with autocast_ctx:
    logits = model(batch)
    loss = nn.functional.cross_entropy(...)

# Backward pass runs in mixed precision automatically
loss.backward()
```

Key properties of autocast:
- Parameters stay in FP32
- Operations automatically choose FP32 or BF16 based on policy
- MatMul operations → BF16 (speed)
- Normalization/loss → FP32 (stability)
- Gradients accumulated in FP32

## Recommendations

1. **Always use mixed precision for production training** of models ≥100M parameters
2. **Pair with gradient scaling** (though BF16 needs it less than FP16)
3. **Monitor metrics early** to ensure no accuracy degradation
4. **Increase batch size** to take advantage of memory savings
5. **Benchmark your specific architecture** as speedup varies by model structure

## Files Generated

- `benchmark.py` - Modified with `--mixed-precision` flag
- `run_focused_comparison.py` - Comparison script for multiple models
- `mixed_precision_results.json` - Raw timing data
- `test_mixed_precision.py` - Quick validation script
- `mixed_precision_training_analysis.md` - This analysis document

## Reproducibility

To reproduce these results:

```bash
cd /home/ymao/cs336/assignment2-systems/cs336_systems
source ../.venv/bin/activate

# Run small model comparison
python benchmark.py --d-model 768 --d-ff 3072 --num-layers 12 --num-heads 12 \
    --context-length 512 --sequence-length 512 --backward --num-steps 50

python benchmark.py --d-model 768 --d-ff 3072 --num-layers 12 --num-heads 12 \
    --context-length 512 --sequence-length 512 --backward --num-steps 50 --mixed-precision

# Or run full comparison
python run_focused_comparison.py
```
