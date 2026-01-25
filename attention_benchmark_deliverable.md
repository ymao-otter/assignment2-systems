# Attention Benchmark Results and Analysis

## Benchmark Configuration

- **Batch size**: 8
- **Head dimension (dmodel)**: [16, 32, 64, 128]
- **Sequence length**: [256, 1024, 4096, 8192, 16384]
- **Iterations**: 100 forward and backward passes (after 10 warmup iterations)
- **Device**: CUDA

## Results Tables

### Forward Pass Time (milliseconds)

| d_model | seq=256 | seq=1024 | seq=4096 | seq=8192 | seq=16384 |
|---------|---------|----------|----------|----------|-----------|
| 16      | 0.260   | 0.998    | 14.103   | 54.825   | **OOM**   |
| 32      | 0.260   | 1.063    | 14.271   | 55.397   | **OOM**   |
| 64      | 0.260   | 1.082    | 14.404   | 56.638   | **OOM**   |
| 128     | 0.263   | 1.148    | 15.041   | 59.800   | **OOM**   |

### Backward Pass Time (milliseconds)

| d_model | seq=256 | seq=1024 | seq=4096 | seq=8192 | seq=16384 |
|---------|---------|----------|----------|----------|-----------|
| 16      | 0.944   | 3.498    | 49.721   | 193.061  | **OOM**   |
| 32      | 0.942   | 3.613    | 49.986   | 194.251  | **OOM**   |
| 64      | 0.959   | 3.652    | 50.352   | 196.926  | **OOM**   |
| 128     | 0.967   | 3.802    | 51.699   | 203.838  | **OOM**   |

### Memory Before Backward Pass (MB)

| d_model | seq=256 | seq=1024 | seq=4096 | seq=8192 | seq=16384 |
|---------|---------|----------|----------|----------|-----------|
| 16      | 20.77   | 82.34    | 1048.63  | 4129.00  | **OOM**   |
| 32      | 21.27   | 84.34    | 1056.63  | 4145.00  | **OOM**   |
| 64      | 22.27   | 88.34    | 1072.63  | 4177.00  | **OOM**   |
| 128     | 24.27   | 96.34    | 1104.63  | 4241.00  | **OOM**   |

## Out-of-Memory Analysis

### OOM Configurations

All configurations with **sequence length = 16384** ran out of memory, regardless of d_model:
- d_model=16, seq_len=16384
- d_model=32, seq_len=16384
- d_model=64, seq_len=16384
- d_model=128, seq_len=16384

### Memory Accounting for Smallest OOM Configuration

**Configuration**: batch=8, seq_len=16384, d_model=16

#### Forward Pass Memory:

1. **Input tensors (Q, K, V)**:
   - 3 × 8 × 16384 × 16 × 4 bytes = **24.00 MB**

2. **Attention scores matrix (Q @ K^T)**:
   - 8 × 16384 × 16384 × 4 bytes = **8,192.00 MB**

3. **Attention weights (softmax output)**:
   - 8 × 16384 × 16384 × 4 bytes = **8,192.00 MB**

4. **Output (attention @ V)**:
   - 8 × 16384 × 16 × 4 bytes = **8.00 MB**

#### Backward Pass Memory:

5. **Gradients for inputs (dQ, dK, dV)**:
   - 3 × 8 × 16384 × 16 × 4 bytes = **24.00 MB**

6. **Saved activations for backward** (attention scores/weights):
   - 2 × 8 × 16384 × 16384 × 4 bytes = **16,384.00 MB**

#### Total Estimated Memory:
**~32,824 MB (≈32 GB)**

The dominant memory cost is the **attention score/weight matrices** (batch × seq × seq), which must be saved for the backward pass. These alone account for **16,384 MB** in saved activations.

## Memory Scaling with Sequence Length

The following table shows how memory scales with sequence length for d_model=16:

| Sequence Length | Total Memory | Attention Matrix Alone |
|-----------------|--------------|------------------------|
| 256             | 20.77 MB     | 2.00 MB                |
| 1024            | 82.34 MB     | 32.00 MB               |
| 4096            | 1048.63 MB   | 512.00 MB              |
| 8192            | 4129.00 MB   | 2048.00 MB             |
| 16384           | OOM          | 8192.00 MB             |

**Key Observations**:

1. **Quadratic scaling**: Memory grows as O(batch × seq²) because of the attention score matrices
2. **Attention matrix dominates**: At seq_len=8192, the attention matrices (2048 MB) account for ~50% of total memory
3. **Rapid growth**: Doubling sequence length quadruples the attention matrix memory:
   - 256→512: 2 MB → 8 MB (4×)
   - 1024→2048: 32 MB → 128 MB (4×)
   - 4096→8192: 512 MB → 2048 MB (4×)
4. **d_model has minimal impact**: Changing d_model from 16 to 128 only adds ~100 MB at seq_len=8192, while the attention matrices still dominate at 2048 MB

## Solutions to Eliminate Memory Cost

### 1. **Flash Attention (Recommended)**

Flash Attention eliminates the O(seq²) memory bottleneck by recomputing attention scores during the backward pass instead of storing them. Key features:

- **Memory reduction**: O(seq²) → O(seq) for saved activations
- **Performance**: Despite recomputation, remains efficient through kernel fusion and tiling
- **Trade-off**: Trades additional computation for dramatic memory savings
- **Impact**: Would allow seq_len=16384 to run with ~100-200 MB instead of ~16 GB

### 2. **Gradient Checkpointing**

Selectively recompute activations during backward instead of storing all of them:

- **Memory reduction**: ~2× reduction in activation memory
- **Computational cost**: ~33% increase in training time
- **Trade-off**: More recomputation than Flash Attention, less memory efficient

### 3. **Sparse Attention Patterns**

Only compute attention for a subset of positions (e.g., local windows, strided patterns):

- **Memory reduction**: O(seq²) → O(seq × √seq) or O(seq × log seq)
- **Trade-off**: Reduced modeling capacity (can't attend to all positions)
- **Examples**: Longformer, BigBird, Sparse Transformers

### 4. **Low-Rank Approximations**

Approximate the attention matrix with low-rank factorizations:

- **Memory reduction**: O(seq²) → O(seq × k) where k << seq
- **Trade-off**: Approximation introduces modeling error
- **Examples**: Linformer, Performer, FMMformer

### 5. **Mixed Precision Training**

Use FP16/BF16 instead of FP32:

- **Memory reduction**: 2× reduction (halves all tensor sizes)
- **Trade-off**: Numerical precision (usually negligible with proper loss scaling)
- **Impact**: Would reduce attention matrices from 8192 MB to 4096 MB, but still O(seq²)

## Recommendation

**Flash Attention** is the best solution because it:
1. Eliminates the fundamental O(seq²) memory bottleneck
2. Maintains exact attention computation (no approximation)
3. Remains efficient through optimized CUDA kernels
4. Is widely adopted (PyTorch 2.0+, xformers, etc.)

With Flash Attention, the configuration that currently OOMs (seq_len=16384, d_model=16) would require only ~200 MB instead of ~32 GB, making it easily trainable even on modest GPUs.
