# NVTX Profiling Guide

This guide explains how to use NVIDIA Nsight Systems with NVTX annotations to profile your transformer model.

## Quick Start

### 1. Run Profiling with NVTX Annotations

```bash
cd cs336_systems
uv run nsys profile -o result python benchmark.py --num-steps 20 --num-warmup 5
```

For CUDA backtraces (adds overhead but provides Python call stacks):
```bash
uv run nsys profile --cudabacktrace=all -o result python benchmark.py --num-steps 20 --num-warmup 5
```

### 2. Convert to .nsys-rep Format

```bash
/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter result.qdstrm
```

This creates `result.nsys-rep` which you can open on your Mac.

### 3. Download and Open on Mac

```bash
# From your Mac:
scp ymao@<server-ip>:/home/ymao/cs336/assignment2-systems/cs336_systems/result.nsys-rep ~/Downloads/
```

Then open with NVIDIA Nsight Systems GUI.

## NVTX Annotations in benchmark.py

The benchmark script now includes the following NVTX ranges:

### Top-Level Ranges
- **"Warmup Steps"**: Contains all warmup iterations (filter these out in the profiler!)
- **"Timed Steps"**: Contains all measured benchmark iterations

### Per-Step Ranges
Each benchmark step is annotated with:
- **"Benchmark Step N"**: The entire step
- **"Forward Pass (step N)"**: Just the forward pass
- **"Loss Computation (step N)"**: Loss calculation (if backward enabled)
- **"Backward Pass (step N)"**: Backpropagation (if backward enabled)
- **"Zero Grad (step N)"**: Gradient zeroing (if backward enabled)

## Filtering in Nsight Systems

When you open the `.nsys-rep` file:

1. **Find the NVTX row**: Look for the "NVTX" row in the timeline
2. **Filter out warmup**: Right-click on the "Warmup Steps" range → "Filter" → "Hide this range"
3. **Focus on specific operations**: 
   - Right-click on "Forward Pass" ranges to see only forward pass kernels
   - Right-click on "Backward Pass" ranges to isolate backpropagation kernels

## Adding Fine-Grained Annotations

For more detailed profiling of attention layers, see `benchmark_detailed.py`.

### Example: Instrument Specific Model Components

```python
import torch.cuda.nvtx as nvtx

# In your attention forward pass:
def forward(self, x):
    with nvtx.range("QKV Projection"):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
    
    with nvtx.range("Attention Computation"):
        attn_output = self.attention(q, k, v)
    
    with nvtx.range("Output Projection"):
        output = self.out_proj(attn_output)
    
    return output
```

## Running with Backward Pass

To profile training (forward + backward):

```bash
uv run nsys profile -o result_backward python benchmark.py \
    --num-steps 10 \
    --num-warmup 3 \
    --backward
```

This will show you:
- Which kernels run during forward vs backward
- Memory allocation patterns
- Gradient computation costs

## Understanding the Profile

### Key Metrics to Look For

1. **Kernel Duration**: How long each CUDA kernel runs
2. **GPU Utilization**: Gaps between kernels indicate opportunities for optimization
3. **Memory Transfers**: Look for expensive CPU↔GPU transfers
4. **Attention Patterns**: Which attention operations are slowest

### Common Insights

- **Flash Attention**: If using flash attention, you should see fewer memory-bound operations
- **Gradient Accumulation**: Backward pass kernels should appear in reverse order of forward pass
- **Bottlenecks**: Long kernels or many small kernels may indicate optimization opportunities

## Notes on Python Backtraces

- `--cudabacktrace=all` adds overhead (~2-5x slower)
- On this system, CPU sampling is disabled (kernel paranoid level = 4)
- Python backtraces require CPU sampling, so they won't work here
- NVTX annotations are more reliable and lower overhead

## Advanced Usage

### Profile Specific Layers Only

Add conditional NVTX ranges:

```python
if layer_idx in [0, 3, 5]:  # Profile only certain layers
    with nvtx.range(f"Layer {layer_idx} - Attention"):
        output = self.attention(x)
```

### Color-Coded Ranges

```python
# Different colors for different operation types
nvtx.range_push("Forward", color="green")
# ... operations ...
nvtx.range_pop()

nvtx.range_push("Backward", color="red")
# ... operations ...
nvtx.range_pop()
```

## Troubleshooting

### "CUDA backtraces will not be collected"
- This is expected on systems with restrictive kernel paranoid levels
- NVTX annotations still work fine

### "Importer binary and its dependencies were not found"
- Use the manual import command provided above
- This is a known issue with some nsys installations

### Profile file too large
- Reduce `--num-steps` (10-20 is usually enough)
- Use shorter sequences or smaller batch sizes
- Focus on specific operations with targeted NVTX ranges

## Example Workflow

```bash
# 1. Profile with NVTX
cd /home/ymao/cs336/assignment2-systems/cs336_systems
uv run nsys profile -o profile_fwd python benchmark.py --num-steps 10

# 2. Convert to nsys-rep
/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter profile_fwd.qdstrm

# 3. Download to Mac (from Mac terminal)
scp ymao@<ip>:/home/ymao/cs336/assignment2-systems/cs336_systems/profile_fwd.nsys-rep .

# 4. Open in Nsight Systems GUI
# File → Open → profile_fwd.nsys-rep

# 5. In the GUI:
# - Find NVTX row
# - Right-click "Warmup Steps" → Hide
# - Zoom into "Timed Steps"
# - Analyze kernel execution patterns
```
