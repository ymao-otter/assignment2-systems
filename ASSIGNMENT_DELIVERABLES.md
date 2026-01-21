# Assignment 2 - Mixed Precision Training Deliverables

## Task 1: Mixed Precision Accumulation Test

**Script:** `cs336_systems/mixed_precision_accumulation.py`

**Analysis:** `mixed_precision_analysis.md`

Demonstrated numerical precision effects when accumulating small values (0.01 × 1000):
- Pure FP32: 10.0001 (0.001% error)
- Pure FP16: 9.9531 (0.469% error) - 47x worse
- Mixed FP32 accumulator + FP16 values: 10.0021 (0.021% error)

---

## Task 2: Autocast Data Types

**Question:** What are the data types of model components within autocast context?

**Script:** `test_autocast_dtypes.py`

**Analysis:** `autocast_mixed_precision_dtypes.md`

**Answer:**

| Component | Data Type | Reason |
|-----------|-----------|---------|
| Model parameters | **FP32** | Parameters unchanged by autocast |
| fc1 output | **FP16** | Linear ops run in FP16 |
| LayerNorm output | **FP32** | Normalization in FP32 for stability |
| fc2 output (logits) | **FP16** | Linear ops run in FP16 |
| Loss | **FP32** | Loss functions in FP32 for stability |
| Gradients | **FP32** | Gradients match parameter dtype |

---

## Task 3: Mixed Precision Benchmarking (BF16 vs FP32)

**Modified Script:** `cs336_systems/benchmark.py` (added `--mixed-precision` flag)

**Comparison Script:** `cs336_systems/run_focused_comparison.py`

**Results:** `cs336_systems/mixed_precision_results.json`

**Analysis:** `mixed_precision_training_analysis.md`, `MIXED_PRECISION_SUMMARY.md`

### Deliverable (2-3 sentences with timings and commentary)

Mixed precision training with BF16 provides consistent speedups of **1.45-1.65x** across model sizes, with larger models benefiting more (1.65x for medium forward-only vs 1.45x for small). Memory usage is reduced by approximately **27%** consistently across both model sizes and modes, which is crucial for training larger models. The speedup trend shows that larger, more compute-intensive models benefit more from BF16's efficient tensor core utilization, suggesting that mixed precision becomes increasingly valuable as model size grows.

### Results Table

| Model | Params | Mode | FP32 (ms) | BF16 (ms) | Speedup | Memory Save |
|-------|--------|------|-----------|-----------|---------|-------------|
| Small | 126M | Forward | 117.1 | 80.9 | **1.45x** | 27.7% |
| Small | 126M | Fwd+Bwd | 356.2 | 236.9 | **1.50x** | 27.3% |
| Medium | 350M | Forward | 350.6 | 212.2 | **1.65x** | 26.8% |
| Medium | 350M | Fwd+Bwd | 1040.4 | 653.2 | **1.59x** | 26.4% |

**Key Findings:**
- Speedup increases with model size (1.45x → 1.65x)
- Consistent ~27% memory reduction
- Training cost reduction: 33-40% less time
- Throughput: Small model 11.5K → 17.3K tokens/sec (1.50x)

---

## Task 4: LayerNorm Sensitivity in Mixed Precision

**Question:** What parts of layer normalization are sensitive to mixed precision? If we use BF16 instead of FP16, do we still need to treat layer normalization differently? Why or why not?

**Verification Script:** `test_layernorm_precision.py`

**Demonstration Script:** `demonstrate_layernorm_sensitivity.py`

**Analysis:** `layernorm_mixed_precision_analysis.md`, `LAYERNORM_ANSWER.md`

### Deliverable (2-3 sentences)

Layer normalization is sensitive to mixed precision because it involves variance computation (squaring and summing values), division by the square root of variance, and mean-variance subtractions, all of which can accumulate numerical errors or encounter underflow/overflow in reduced precision. PyTorch's autocast keeps LayerNorm in FP32 for **both FP16 and BF16**, because while BF16's wider dynamic range (same as FP32) prevents the overflow/underflow issues that plague FP16, its reduced mantissa precision (7 bits vs FP32's 23 bits) can still cause accuracy degradation in the sensitive variance and normalization calculations. The normalization operation is fundamentally more precision-sensitive than matrix multiplications, requiring the full FP32 precision to maintain training stability regardless of which reduced-precision format is used.

### Experimental Evidence

**Dtype behavior:**
```
FP16 Autocast:  Linear → FP16,  LayerNorm → FP32
BF16 Autocast:  Linear → BF16,  LayerNorm → FP32
```

**Numerical demonstration (extreme case with small variance):**
```
FP32:  Variance = 1e-8,  Output std = 1.0  ✓ Correct
FP16:  Variance = 0.0,   Output std = 0.0  ✗ 100% error (underflow)
BF16:  Variance = 0.0,   Output std = 0.0  ✗ 100% error (underflow)
```

**Gradient errors:**
```
FP16 gradient error vs FP32: 7,644%
BF16 gradient error vs FP32: 58,169%
```

**Key Insight:** BF16 solves the dynamic **range** problem but not the **precision** problem for LayerNorm. The variance computation requires the 23-bit mantissa precision of FP32, not just the 7-bit precision of BF16.

---

## Summary of All Files

### Analysis Documents
- `ASSIGNMENT_DELIVERABLES.md` - This summary
- `mixed_precision_analysis.md` - Accumulation test analysis
- `autocast_mixed_precision_dtypes.md` - Autocast data types analysis
- `mixed_precision_training_analysis.md` - Detailed BF16 vs FP32 analysis
- `MIXED_PRECISION_SUMMARY.md` - Quick reference summary
- `layernorm_mixed_precision_analysis.md` - LayerNorm sensitivity analysis
- `LAYERNORM_ANSWER.md` - Concise LayerNorm answer

### Scripts
- `cs336_systems/mixed_precision_accumulation.py` - Accumulation test
- `test_autocast_dtypes.py` - Autocast dtype verification
- `cs336_systems/benchmark.py` - **Modified with --mixed-precision flag**
- `cs336_systems/run_focused_comparison.py` - Automated comparison
- `cs336_systems/test_mixed_precision.py` - Quick test script
- `test_layernorm_precision.py` - LayerNorm dtype verification
- `demonstrate_layernorm_sensitivity.py` - Numerical sensitivity demo

### Data Files
- `cs336_systems/mixed_precision_results.json` - Raw benchmark results

---

## How to Reproduce Results

### Task 1: Accumulation Test
```bash
cd /home/ymao/cs336/assignment2-systems
source .venv/bin/activate
python cs336_systems/mixed_precision_accumulation.py
```

### Task 2: Autocast Types
```bash
python test_autocast_dtypes.py
```

### Task 3: Mixed Precision Benchmark
```bash
# Run full comparison
cd cs336_systems
python run_focused_comparison.py

# Or run individual benchmarks
python benchmark.py --d-model 768 --num-layers 12 --num-heads 12 --backward --num-steps 50
python benchmark.py --d-model 768 --num-layers 12 --num-heads 12 --backward --num-steps 50 --mixed-precision
```

### Task 4: LayerNorm Sensitivity
```bash
python test_layernorm_precision.py
python demonstrate_layernorm_sensitivity.py
```
