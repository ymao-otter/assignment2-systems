# Transformer Model Benchmark Results

## Configuration
- **Warmup steps**: 5
- **Measurement steps**: 10
- **Batch size**: 8
- **Sequence length**: 256
- **Device**: NVIDIA A10G (22GB)

## Model Configurations

| Size   | d_model | d_ff  | num_layers | num_heads | Parameters |
|--------|---------|-------|------------|-----------|------------|
| small  | 768     | 3072  | 12         | 12        | 125.85M    |
| medium | 1024    | 4096  | 24         | 16        | 419.48M    |
| large  | 1280    | 5120  | 36         | 20        | 964.78M    |
| xl     | 1600    | 6400  | 48         | 25        | ~1.9B      |
| 2.7B   | 2560    | 10240 | 32         | 32        | ~3.4B      |

## Benchmark Results

### Small Model (125.85M parameters)

**Forward Pass Only:**
- Average time: **52.42 ms**
- Standard deviation: **0.08 ms**
- Coefficient of variation: **0.16%** (very low variability)
- Throughput: 19.08 steps/second, 39,068 tokens/second
- Peak memory: 3.47 GB

**Forward + Backward Pass:**
- Average time: **155.68 ms**
- Standard deviation: **0.17 ms**
- Coefficient of variation: **0.11%** (very low variability)
- Throughput: 6.42 steps/second, 13,155 tokens/second
- Peak memory: 3.68 GB
- **Backward/Forward ratio: ~3.0x**

### Medium Model (419.48M parameters)

**Forward Pass Only:**
- Average time: **158.08 ms**
- Standard deviation: **0.35 ms**
- Coefficient of variation: **0.22%** (very low variability)
- Throughput: 6.33 steps/second, 12,956 tokens/second
- Peak memory: 9.44 GB

**Forward + Backward Pass:**
- Average time: **464.73 ms**
- Standard deviation: **0.53 ms**
- Coefficient of variation: **0.11%** (very low variability)
- Throughput: 2.15 steps/second, 4,407 tokens/second
- Peak memory: 9.65 GB
- **Backward/Forward ratio: ~2.9x**

### Large Model (964.78M parameters)

**Forward Pass Only:**
- Average time: **320.62 ms**
- Standard deviation: **0.33 ms**
- Coefficient of variation: **0.10%** (very low variability)
- Throughput: 3.12 steps/second, 6,388 tokens/second
- Peak memory: 18.45 GB

**Forward + Backward Pass:**
- Average time: **935.12 ms**
- Standard deviation: **0.94 ms**
- Coefficient of variation: **0.10%** (very low variability)
- Throughput: 1.07 steps/second, 2,190 tokens/second
- Peak memory: 18.66 GB
- **Backward/Forward ratio: ~2.9x**

### XL and 2.7B Models
- **Status**: Out of Memory (OOM)
- The NVIDIA A10G GPU has 22GB of memory, which was insufficient for these larger models
- The large model already used ~18.66 GB for forward+backward, leaving insufficient headroom for the XL model

## Analysis

### Timing Observations

1. **Forward Pass Duration**:
   - Small: ~52 ms
   - Medium: ~158 ms (3.0x slower than small)
   - Large: ~321 ms (6.1x slower than small, 2.0x slower than medium)
   
2. **Backward Pass Duration** (total forward+backward):
   - Small: ~156 ms
   - Medium: ~465 ms (3.0x slower than small)
   - Large: ~935 ms (6.0x slower than small, 2.0x slower than medium)

3. **Backward-to-Forward Ratio**:
   - Backward passes consistently take approximately **2.9-3.0x** longer than forward passes alone
   - This is expected because backpropagation needs to:
     - Recompute or store all intermediate activations
     - Compute gradients for all parameters
     - Perform the backward pass through all layers

### Variability Analysis

**Excellent Stability**: All measurements show **very low variability**:
- Coefficient of variation ranges from **0.10% to 0.22%**
- Standard deviations are typically **< 1 ms**
- This indicates:
  - Stable GPU performance
  - Effective warmup (5 steps was sufficient)
  - Minimal interference from other processes
  - Consistent CUDA kernel execution

The low variability (CV < 0.25%) demonstrates that:
1. The warmup phase effectively stabilized GPU kernels
2. The A10G GPU provides consistent performance
3. The benchmark methodology (CUDA synchronization) works correctly
4. Results are highly reproducible

### Memory Usage

Memory scales with model size:
- Small: ~3.7 GB
- Medium: ~9.7 GB (2.6x increase)
- Large: ~18.7 GB (5.1x increase from small, 1.9x from medium)

The memory increase is less than linear with parameter count because:
- Activations (batch-dependent) are constant across model sizes
- Only model weights and gradients scale with parameters

### Performance Scaling

As models grow larger, there's a superlinear increase in computation time:
- From Small to Medium: 3x more parameters → 3.0x slower
- From Medium to Large: 2.3x more parameters → 2.0x slower

This is reasonable given:
- Larger models have more layers (depth increases)
- Larger hidden dimensions (width increases)
- Both depth and width contribute to computational cost
