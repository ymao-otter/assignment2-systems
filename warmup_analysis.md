# Impact of Warmup Steps on Benchmark Results

## Quick Summary

### The Dramatic Effect of Skipping Warmup

**Forward Pass Performance:**
- Without warmup (0 steps): 141.72 ms ± 121.84 ms (CV: 85.97%)
- With proper warmup (5 steps): 52.42 ms ± 0.08 ms (CV: 0.16%)
- **Impact: 2.7x performance difference!**

**Backward Pass Performance:**
- Without warmup (0 steps): 457.38 ms ± 158.16 ms (CV: 34.58%)
- With proper warmup (5 steps): 155.68 ms ± 0.17 ms (CV: 0.11%)
- **Impact: 2.9x performance difference!**

### Key Takeaway

**Without proper warmup, you're measuring initialization overhead, not actual performance!** The first few GPU iterations are fundamentally different from steady-state execution due to JIT compilation, memory allocation, and auto-tuning. Always use adequate warmup (5-10+ steps) and verify coefficient of variation < 1%.

---

## Experimental Setup
- Model: Small (125.85M parameters: d_model=768, d_ff=3072, num_layers=12, num_heads=12)
- Batch size: 8
- Sequence length: 256
- Device: NVIDIA A10G (22GB)
- Measurement steps: 10 (consistent across all experiments)

## Results Summary

### Forward Pass Only

| Warmup Steps | Avg Time (ms) | Std Dev (ms) | CV (%) | Speedup vs 0 warmup |
|--------------|---------------|--------------|--------|---------------------|
| 0            | 141.72        | 121.84       | 85.97% | 1.0x (baseline)     |
| 1            | 90.79         | 26.66        | 29.36% | 1.56x               |
| 2            | 52.84         | 0.74         | 1.41%  | 2.68x               |
| 5            | 52.42         | 0.08         | 0.16%  | 2.70x               |

### Forward + Backward Pass

| Warmup Steps | Avg Time (ms) | Std Dev (ms) | CV (%)  | Speedup vs 0 warmup |
|--------------|---------------|--------------|---------|---------------------|
| 0            | 457.38        | 158.16       | 34.58%  | 1.0x (baseline)     |
| 1            | 415.46        | 71.55        | 17.22%  | 1.10x               |
| 2            | 260.00        | 142.94       | 54.98%  | 1.76x               |
| 3            | 344.77        | 0.71         | 0.21%   | 1.33x               |
| 4            | 326.62        | 54.39        | 16.65%  | 1.40x               |
| 5            | 155.68        | 0.17         | 0.11%   | 2.94x               |

## Key Findings

### 1. Dramatic Impact on Performance

**Without warmup (0 steps):**
- Forward pass is **2.7x slower** than with proper warmup
- Backward pass is **2.9x slower** than with proper warmup
- This represents a **massive performance penalty** for skipping warmup

**The true performance is only revealed after adequate warmup.**

### 2. Extreme Variability Without Warmup

**Coefficient of Variation (CV) Analysis:**

Forward pass:
- 0 warmup: **85.97% CV** - measurements range from ~20ms to ~260ms!
- 1 warmup: 29.36% CV - still highly variable
- 2 warmup: 1.41% CV - much more stable
- 5 warmup: **0.16% CV** - very stable

Backward pass:
- 0 warmup: **34.58% CV** - highly variable
- 1-4 warmup: 16-55% CV - inconsistent stabilization
- 5 warmup: **0.11% CV** - very stable

**Without warmup, the standard deviation is often larger than 50% of the mean!** This makes the measurements essentially meaningless for performance characterization.

### 3. Insufficient Warmup Still Shows Problems

**1-2 warmup steps for forward pass:**
- With 1 warmup: CV still 29.36% - unacceptable variability
- With 2 warmup: CV drops to 1.41% - better but not fully stable
- With 5 warmup: CV is 0.16% - properly stabilized

**1-4 warmup steps for backward pass:**
- The backward pass shows **inconsistent behavior** even with 2-4 warmup steps
- Performance varies wildly: 260ms (2 warmup) vs 344ms (3 warmup) vs 326ms (4 warmup)
- Only stabilizes at 5 warmup steps with 155.68ms ± 0.17ms
- The variability suggests different code paths or kernel selections during the first few iterations

### 4. Why This Happens

The performance differences are due to several **"cold start"** effects:

#### A. CUDA Kernel Compilation and JIT
- **First invocation**: CUDA kernels must be compiled from PTX to GPU machine code
- **JIT overhead**: The CUDA driver performs Just-In-Time compilation on first use
- **Kernel caching**: After first run, compiled kernels are cached in GPU memory
- **Impact**: First few iterations include compilation time, inflating measurements

#### B. GPU Memory Allocation and Initialization
- **Lazy allocation**: PyTorch uses lazy memory allocation - memory is allocated on first use
- **Memory pool warmup**: CUDA memory allocator needs to set up memory pools
- **Cache initialization**: GPU L1/L2 caches are cold on first access
- **Impact**: Initial iterations trigger memory allocations and cache misses

#### C. cuDNN/cuBLAS Library Initialization
- **Auto-tuning**: cuDNN benchmarks multiple algorithms on first invocation to find the fastest
- **Workspace allocation**: Libraries allocate workspace memory for operations
- **Algorithm selection**: Different algorithms may be tried before settling on the optimal one
- **Impact**: First few iterations may use sub-optimal algorithms or trigger auto-tuning overhead

#### D. PyTorch Autograd Graph Construction
- **Graph building**: First backward pass constructs the autograd computation graph
- **Backward kernel selection**: PyTorch selects and compiles backward kernels on first use
- **Gradient buffer allocation**: Gradient tensors are allocated on first backward pass
- **Impact**: Backward passes need more warmup due to additional graph construction overhead

#### E. GPU Clock Frequency Scaling
- **Power management**: GPUs throttle clock speeds when idle
- **Ramp-up time**: GPU needs time to reach full operating frequency
- **Thermal effects**: Initial cold state vs. steady-state operating temperature
- **Impact**: First few iterations run at lower clock speeds

### 5. Why 1-2 Warmup Steps May Still Be Insufficient

**For Forward Pass:**
- 1 warmup step: Covers basic initialization but not complete kernel optimization
- 2 warmup steps: Usually sufficient for forward pass to stabilize (CV: 1.41%)
- The rapid improvement suggests most overhead is in the first iteration

**For Backward Pass (more complex):**
- 1 warmup: Only initializes basic operations (CV: 17.22%)
- 2 warmup: Inconsistent - may still be switching algorithms (CV: 54.98%!)
- 3 warmup: Better but average time is suspiciously high (344ms vs. 155ms stable)
- 4 warmup: Still variable (CV: 16.65%)
- 5 warmup: Finally stable (CV: 0.11%)

**The backward pass requires more warmup because:**
1. More operations to compile (forward + backward kernels)
2. Auto-tuning must evaluate more algorithm choices
3. Gradient allocation and graph construction complexity
4. More complex memory access patterns
5. cuDNN/cuBLAS need to benchmark backward operations separately

The erratic behavior with 2-4 warmup steps for backward passes suggests that:
- **Different code paths** are being executed during warmup vs. measurement
- **Algorithm auto-tuning** may still be happening in the measurement phase
- **Memory fragmentation** patterns haven't stabilized
- **Kernel fusion optimizations** are still being discovered by the JIT compiler

### 6. Best Practices for Benchmarking

Based on these results:

✅ **Always use adequate warmup steps:**
- Minimum 5-10 warmup steps for forward-only benchmarks
- Minimum 10-20 warmup steps for training (forward+backward) benchmarks
- More warmup for larger models or complex operations

✅ **Monitor the coefficient of variation:**
- CV < 1% is good
- CV < 0.5% is excellent
- CV > 5% indicates insufficient warmup or system interference

✅ **Report both mean and standard deviation:**
- Mean alone can be misleading if there's high variability
- Standard deviation reveals measurement stability

✅ **Consider running multiple benchmark sessions:**
- Average across multiple separate runs (not just multiple steps)
- This accounts for process-level initialization effects

❌ **Never benchmark without warmup:**
- Results will be 2-3x slower than true performance
- Measurements will have 30-80% variability
- Comparisons will be meaningless

## Visualization of Results

### Forward Pass Convergence

```
Time (ms)
  150 |  *
      |  
  100 |     *
      |      
   50 |         * *  ← Stabilizes by 2 warmup steps
      |
    0 +--+--+--+--+--
      0  1  2  3  4  5  Warmup Steps
```

### Backward Pass Convergence

```
Time (ms)
  450 | *
      | |
  350 |     * *
      |        |
  250 |   *    
      |         
  150 |           * ← Stabilizes only at 5 warmup steps
      |
    0 +--+--+--+--+--+--
      0  1  2  3  4  5  Warmup Steps
```

## Conclusion

Warmup steps are **critical** for accurate GPU benchmarking:

1. **Performance Impact**: Skipping warmup makes code appear 2-3x slower than reality
2. **Measurement Stability**: Without warmup, CV can exceed 80%, making results meaningless
3. **Insufficient Warmup**: Even 1-2 warmup steps may not be enough, especially for backward passes
4. **Recommended Practice**: Use at least 5-10 warmup steps and verify CV < 1%
5. **Physical Reasons**: Kernel compilation, memory allocation, auto-tuning, and GPU frequency scaling all require warmup

**The first few iterations of GPU code are fundamentally different from steady-state performance**, and benchmarking must account for this to produce accurate, reproducible results.
