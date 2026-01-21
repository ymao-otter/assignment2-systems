# Mixed Precision (BF16) Benchmark Results - Summary

## Deliverable (2-3 sentences)

Mixed precision training with BF16 provides consistent speedups of 1.45-1.65x across model sizes, with larger models benefiting more (1.65x for medium forward-only vs 1.45x for small). Memory usage is reduced by approximately 27% consistently across both model sizes and modes, which is crucial for training larger models. The speedup trend shows that larger, more compute-intensive models benefit more from BF16's efficient tensor core utilization, suggesting that mixed precision becomes increasingly valuable as model size grows.

## Results Table

| Model | Params | Mode | FP32 (ms) | BF16 (ms) | Speedup | Memory Save |
|-------|--------|------|-----------|-----------|---------|-------------|
| Small | 126M | Forward | 117.1 | 80.9 | **1.45x** | 27.7% |
| Small | 126M | Fwd+Bwd | 356.2 | 236.9 | **1.50x** | 27.3% |
| Medium | 350M | Forward | 350.6 | 212.2 | **1.65x** | 26.8% |
| Medium | 350M | Fwd+Bwd | 1040.4 | 653.2 | **1.59x** | 26.4% |

## Key Observations

1. **Speedup increases with model size**: 1.45x (small) → 1.65x (medium) for forward pass
2. **Consistent memory savings**: ~27% reduction across all configurations
3. **Training cost reduction**: 33-40% less time for the same training
4. **Throughput gains**: Small model goes from 11.5K to 17.3K tokens/sec (1.50x)

## Configuration

- Context length: 512
- Batch size: 8
- Models: Small (126M), Medium (350M)
- Hardware: NVIDIA GPU with CUDA
- Implementation: PyTorch `torch.autocast` with `dtype=torch.bfloat16`

## LayerNorm in Mixed Precision

**Question:** What parts of layer normalization are sensitive to mixed precision? Do we still need to treat LayerNorm differently with BF16 vs FP16?

**Answer:** Layer normalization is sensitive to mixed precision because it involves variance computation (squaring and summing values), division by the square root of variance, and mean-variance subtractions, all of which can accumulate numerical errors or encounter underflow/overflow in reduced precision. PyTorch's autocast keeps LayerNorm in FP32 for **both FP16 and BF16**, because while BF16's wider dynamic range (same as FP32) prevents the overflow/underflow issues that plague FP16, its reduced mantissa precision (7 bits vs FP32's 23 bits) can still cause accuracy degradation in the sensitive variance and normalization calculations. The normalization operation is fundamentally more precision-sensitive than matrix multiplications, requiring the full FP32 precision to maintain training stability regardless of which reduced-precision format is used.

**Experimental Verification:**
- Tested LayerNorm with FP16, BF16, and FP32 autocast
- Result: **Both FP16 and BF16 autocast keep LayerNorm in FP32**
- Linear layers use reduced precision, but normalization stays FP32

**Key Insight:** BF16 solves the dynamic **range** problem but not the **precision** problem for LayerNorm.

## Files

- `benchmark.py` - Modified benchmark script with `--mixed-precision` flag
- `mixed_precision_training_analysis.md` - Detailed analysis
- `layernorm_mixed_precision_analysis.md` - LayerNorm sensitivity analysis
- `mixed_precision_results.json` - Raw benchmark data
- `run_focused_comparison.py` - Automated comparison script
- `test_layernorm_precision.py` - LayerNorm dtype verification script
