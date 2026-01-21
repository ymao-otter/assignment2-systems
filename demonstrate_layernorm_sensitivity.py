"""
Demonstrate why LayerNorm is numerically sensitive in mixed precision.
"""
import torch
import torch.nn as nn

def manual_layernorm(x, eps=1e-5):
    """Manual LayerNorm to show intermediate values."""
    mean = x.mean(dim=-1, keepdim=True)
    variance = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    std = torch.sqrt(variance + eps)
    normalized = (x - mean) / std
    return normalized, mean, variance, std

print("="*80)
print("LAYERNORM NUMERICAL SENSITIVITY DEMONSTRATION")
print("="*80)

# Create a tensor with small variance (common after training stabilizes)
torch.manual_seed(42)
x = torch.randn(1, 10) * 0.1 + 1.0  # Mean ≈ 1.0, small variance

print("\nInput values (FP32):")
print(x)

# Test in different precisions
print("\n" + "="*80)
print("COMPUTING LAYERNORM IN DIFFERENT PRECISIONS")
print("="*80)

# FP32
print("\n1. FP32 (Full Precision):")
x_fp32 = x.float()
norm_fp32, mean_fp32, var_fp32, std_fp32 = manual_layernorm(x_fp32)
print(f"   Mean: {mean_fp32.item():.8f}")
print(f"   Variance: {var_fp32.item():.8f}")
print(f"   Std: {std_fp32.item():.8f}")
print(f"   Output: {norm_fp32[0, :3].tolist()}")  # First 3 values

# FP16
print("\n2. FP16 (Half Precision):")
x_fp16 = x.half()
norm_fp16, mean_fp16, var_fp16, std_fp16 = manual_layernorm(x_fp16)
print(f"   Mean: {mean_fp16.item():.8f}")
print(f"   Variance: {var_fp16.item():.8f}")
print(f"   Std: {std_fp16.item():.8f}")
print(f"   Output: {norm_fp16[0, :3].tolist()}")

# BF16
print("\n3. BF16 (Brain Float):")
x_bf16 = x.bfloat16()
norm_bf16, mean_bf16, var_bf16, std_bf16 = manual_layernorm(x_bf16)
print(f"   Mean: {mean_bf16.item():.8f}")
print(f"   Variance: {var_bf16.item():.8f}")
print(f"   Std: {std_bf16.item():.8f}")
print(f"   Output: {norm_bf16[0, :3].tolist()}")

# Compare errors
print("\n" + "="*80)
print("ERROR ANALYSIS (compared to FP32)")
print("="*80)

print("\nVariance Error:")
print(f"   FP16: {abs(var_fp16.item() - var_fp32.item()) / var_fp32.item() * 100:.2f}%")
print(f"   BF16: {abs(var_bf16.item() - var_fp32.item()) / var_fp32.item() * 100:.2f}%")

print("\nOutput L2 Error:")
fp16_error = torch.norm(norm_fp16.float() - norm_fp32).item()
bf16_error = torch.norm(norm_bf16.float() - norm_fp32).item()
print(f"   FP16: {fp16_error:.6f}")
print(f"   BF16: {bf16_error:.6f}")

# Test with extreme case: very small variance
print("\n" + "="*80)
print("EXTREME CASE: VERY SMALL VARIANCE")
print("="*80)

x_small_var = torch.ones(1, 10) + torch.randn(1, 10) * 0.0001

print("\nInput (values very close to 1.0):")
print(x_small_var)

print("\nFP32:")
norm_fp32_sv, mean_fp32_sv, var_fp32_sv, std_fp32_sv = manual_layernorm(x_small_var.float())
print(f"   Variance: {var_fp32_sv.item():.10f}")
print(f"   Output std: {norm_fp32_sv.std().item():.6f} (should be ≈1.0)")

print("\nFP16:")
norm_fp16_sv, mean_fp16_sv, var_fp16_sv, std_fp16_sv = manual_layernorm(x_small_var.half())
print(f"   Variance: {var_fp16_sv.item():.10f}")
print(f"   Output std: {norm_fp16_sv.float().std().item():.6f} (should be ≈1.0)")
print(f"   Error: {abs(norm_fp16_sv.float().std().item() - 1.0) * 100:.2f}%")

print("\nBF16:")
norm_bf16_sv, mean_bf16_sv, var_bf16_sv, std_bf16_sv = manual_layernorm(x_small_var.bfloat16())
print(f"   Variance: {var_bf16_sv.item():.10f}")
print(f"   Output std: {norm_bf16_sv.float().std().item():.6f} (should be ≈1.0)")
print(f"   Error: {abs(norm_bf16_sv.float().std().item() - 1.0) * 100:.2f}%")

# Test gradient flow
print("\n" + "="*80)
print("GRADIENT SENSITIVITY TEST")
print("="*80)

print("\nComputing gradients through LayerNorm in different precisions...")

x_grad_test = torch.randn(4, 10, requires_grad=True)

# FP32
ln_fp32 = nn.LayerNorm(10)
out_fp32 = ln_fp32(x_grad_test)
loss_fp32 = out_fp32.sum()
loss_fp32.backward()
grad_fp32 = x_grad_test.grad.clone()
x_grad_test.grad = None

# FP16
x_grad_test_fp16 = x_grad_test.detach().half().requires_grad_(True)
ln_fp16 = nn.LayerNorm(10).half()
out_fp16 = ln_fp16(x_grad_test_fp16)
loss_fp16 = out_fp16.sum()
loss_fp16.backward()
grad_fp16 = x_grad_test_fp16.grad.clone()

# BF16
x_grad_test_bf16 = x_grad_test.detach().bfloat16().requires_grad_(True)
ln_bf16 = nn.LayerNorm(10).bfloat16()
out_bf16 = ln_bf16(x_grad_test_bf16)
loss_bf16 = out_bf16.sum()
loss_bf16.backward()
grad_bf16 = x_grad_test_bf16.grad.clone()

print(f"\nGradient L2 norm:")
print(f"   FP32: {grad_fp32.norm().item():.6f}")
print(f"   FP16: {grad_fp16.float().norm().item():.6f}")
print(f"   BF16: {grad_bf16.float().norm().item():.6f}")

print(f"\nGradient relative error vs FP32:")
fp16_grad_error = (grad_fp16.float() - grad_fp32).norm() / grad_fp32.norm() * 100
bf16_grad_error = (grad_bf16.float() - grad_fp32).norm() / grad_fp32.norm() * 100
print(f"   FP16: {fp16_grad_error.item():.2f}%")
print(f"   BF16: {bf16_grad_error.item():.2f}%")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("""
Both FP16 and BF16 show numerical errors in LayerNorm computation:
- FP16: Limited dynamic range can cause overflow/underflow
- BF16: Limited precision (7-bit mantissa) causes accuracy loss

While BF16 handles the range better, both suffer from precision issues
in the variance computation and normalization steps.

This is why PyTorch keeps LayerNorm in FP32 for BOTH FP16 and BF16 autocast.
""")
