# XXL Model Memory and Scaling Analysis

## Model Configuration

- **d_model** = 16,384
- **d_ff** = 53,248
- **num_blocks** = 126
- **Architecture**: Each block contains 2 linear layers (d_model × d_ff and d_ff × d_model)

### Total Parameters

- Parameters per block = 2 × d_model × d_ff = 2 × 16,384 × 53,248 = 1,744,830,464
- **Total parameters** = 126 × 1,744,830,464 ≈ **219.85 billion parameters**

---

## (a) Single Device Memory Requirements

### FP32 Storage (4 bytes per parameter)

- **Master weights**: 219.85B × 4 bytes = 879.4 GB
- **Accumulated gradients**: 219.85B × 4 bytes = 879.4 GB
- **Optimizer state** (Adam: 2 states): 219.85B × 4 × 2 bytes = 1,758.8 GB
- **Total FP32**: 879.4 + 879.4 + 1,758.8 = **3,517.6 GB (3.52 TB)**

### BF16 Storage for Backward Pass

- **Gradients in BF16**: 219.85B × 2 bytes = 439.7 GB
- **Memory saved** vs FP32 gradients: 879.4 - 439.7 = **439.7 GB saved**

### H100 GPU Requirements

- GPUs needed: 3,517.6 GB ÷ 80 GB/GPU = **44 H100 80GB GPUs**

**Answer**: Storing FP32 master weights, gradients, and optimizer states requires 3,517.6 GB total, with 439.7 GB saved by using BF16 for backward pass gradients, requiring approximately 44 H100 80GB GPUs.

---

## (b) FSDP Sharding Analysis

### Memory Per Device Formula

Let P = 219.85B parameters, A = total activation memory in bytes

**Memory per device** = (Master weights + Optimizer state + Gradients) / N_FSDP + (Non-sharded activations) + (Sharded activations) / N_FSDP

```
Memory = 16P / N_FSDP + A/2 · (1 + 1/N_FSDP) bytes
       = 3,517.6 GB / N_FSDP + A/2 · (1 + 1/N_FSDP)
```

Where:
- **16P** comes from: 4P (master weights) + 8P (optimizer state) + 4P (gradients) in FP32
- **A/2** is half of activations (non-sharded)
- **A/(2·N_FSDP)** is the other half (sharded)

### Activation Memory Estimate

With per-device batch size b and sequence length L (in BF16):
- Intermediate activations: ≈ 126 × b × L × d_ff × 2 bytes
- A = 126 × b × L × 53,248 × 2 = 126 × b × L × 106,496 bytes

### Required N_FSDP for TPU v5p (95GB per device)

Considering the dominant term (parameters/gradients/optimizer):

```
3,517.6 / N_FSDP < 95 GB
N_FSDP > 37.03
```

**Answer**: Memory per device = 3,517.6 GB / N_FSDP + A/2(1 + 1/N_FSDP) where A is activation memory; N_FSDP needs to be at least **38 devices** to fit under 95GB per TPU (considering primarily parameter/gradient memory; activation memory would further constrain this).

---

## (c) Compute Bound Analysis

### Configuration

- **Bandwidth**: W_ici = 2 × 9 × 10^10 = 1.8 × 10^11 bytes/s
- **FLOPS**: C = 4.6 × 10^14 FLOPS/s
- **Mesh**: M_X = 2, M_Y = 1, X = 16 (FSDP), Y = 4 (TP)
- **Total devices**: 2 × 1 × 16 × 4 = 128 devices
- **Data parallel size**: 16 (along X dimension)

### Per-Device Computation (Forward Pass)

With tensor parallelism splitting d_ff across Y=4 devices:

- FLOPs per device per block = 4 × b × L × d_model × (d_ff/Y)
- = 4 × b × L × 16,384 × 13,312
- = b × L × 873,398,272
- **Total for 126 blocks**: 126 × b × L × 873,398,272 ≈ 1.1 × 10^11 × b × L FLOPs

**Compute time**: T_comp = (1.1 × 10^11 × b × L) / (4.6 × 10^14) ≈ 2.39 × 10^-4 × b × L seconds

### Communication Time (FSDP All-Gather)

- Parameter bytes per block (BF16): 2 × 2 × d_model × d_ff = 4 × 16,384 × 53,248 ≈ 3.49 × 10^9 bytes
- All-gather fraction: (N_FSDP - 1) / N_FSDP = 15/16
- Comm volume per block: (15/16) × 3.49 × 10^9 ≈ 3.27 × 10^9 bytes
- **Total for 126 blocks**: 126 × 3.27 × 10^9 ≈ 4.12 × 10^11 bytes

**Comm time**: T_comm = (4.12 × 10^11) / (1.8 × 10^11) ≈ 2.29 seconds

### Compute Bound Threshold

Model is compute-bound when: **T_comp > T_comm**

```
2.39 × 10^-4 × b × L > 2.29
b × L > 9,582
```

For sequence length L = 2,048:
- b > 4.68
- **Per-device batch size**: b ≥ 5
- **Overall batch size**: 16 × 5 = **80**

**Answer**: The model is compute-bound when per-device batch size × sequence length > 9,582; for sequence length 2,048, this requires per-device batch size ≥ 5, giving an overall batch size of 80 across 16 data-parallel replicas.

---

## (d) Techniques to Reduce Batch Size While Maintaining Throughput

To reduce batch size while remaining compute-bound and maintaining high throughput, several techniques can be employed:

### 1. Gradient Accumulation
**Gradient accumulation** is the most direct approach: run multiple micro-batches with smaller batch sizes, accumulating gradients before each optimizer step. This allows the effective batch size to be decoupled from the memory-constrained batch size (Narayanan et al., 2021, GPipe).

### 2. Pipeline Parallelism
**Pipeline parallelism** can be combined with gradient accumulation to overlap computation across pipeline stages using different micro-batches, maintaining high hardware utilization even with small per-device batches (Huang et al., 2019, GPipe; Narayanan et al., 2021, PipeDream).

### 3. Selective Activation Checkpointing
**Selective activation checkpointing** reduces activation memory by recomputing certain activations during backward pass rather than storing them, enabling larger effective batch sizes within memory constraints (Chen et al., 2016, Training Deep Nets with Sublinear Memory Cost).

### 4. Sequence Packing
**Sequence packing** increases effective tokens per batch by concatenating multiple sequences and using attention masks, reducing padding waste and improving arithmetic intensity (Krell et al., 2021).

### 5. Overlapping Computation and Communication
**Overlapping computation and communication** through bucketed gradient reduction ensures that communication happens asynchronously with computation, keeping compute resources busy even during necessary all-reduce operations (Li et al., 2020, PyTorch FSDP).

### Summary
These techniques allow practitioners to maintain compute efficiency while reducing batch sizes, enabling more flexible training configurations and potentially faster convergence through smaller effective batch sizes.

---

## References

- Chen, T., Xu, B., Zhang, C., & Guestrin, C. (2016). Training Deep Nets with Sublinear Memory Cost. arXiv preprint arXiv:1604.06174.
- Huang, Y., et al. (2019). GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism. NeurIPS.
- Krell, M. M., et al. (2021). Efficient Sequence Packing without Cross-contamination. arXiv preprint arXiv:2107.02027.
- Li, S., et al. (2020). PyTorch Distributed: Experiences on Accelerating Data Parallel Training. VLDB.
- Narayanan, D., et al. (2021). Memory-Efficient Pipeline-Parallel DNN Training. ICML.
