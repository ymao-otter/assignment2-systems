import torch
import torch.nn as nn

class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, 10, bias=False)
    
    def forward(self, x):
        print(f"Input dtype: {x.dtype}")
        
        x = self.fc1(x)
        print(f"After fc1: {x.dtype}")
        
        x = self.ln(x)
        print(f"After LayerNorm: {x.dtype}")
        
        x = self.fc2(x)
        print(f"After fc2: {x.dtype}")
        
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TestModel().to(device)
x = torch.randn(2, 10, device=device)

print("="*80)
print("FP16 AUTOCAST")
print("="*80)
with torch.autocast(device_type=device.type, dtype=torch.float16):
    _ = model(x)

print("\n" + "="*80)
print("BF16 AUTOCAST")
print("="*80)
with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
    _ = model(x)

print("\n" + "="*80)
print("NO AUTOCAST (FP32)")
print("="*80)
_ = model(x)

# Now let's test numerical stability
print("\n" + "="*80)
print("NUMERICAL STABILITY TEST")
print("="*80)

# Create a scenario with small variances
x_small_var = torch.randn(100, 10, device=device) * 0.001  # Very small values

ln = nn.LayerNorm(10).to(device)

print("\nFP32 (no autocast):")
out_fp32 = ln(x_small_var)
print(f"Output mean: {out_fp32.mean().item():.6f}, std: {out_fp32.std().item():.6f}")
print(f"Output dtype: {out_fp32.dtype}")

print("\nFP16 autocast:")
with torch.autocast(device_type=device.type, dtype=torch.float16):
    out_fp16 = ln(x_small_var)
    print(f"Output mean: {out_fp16.mean().item():.6f}, std: {out_fp16.std().item():.6f}")
    print(f"Output dtype: {out_fp16.dtype}")

print("\nBF16 autocast:")
with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
    out_bf16 = ln(x_small_var)
    print(f"Output mean: {out_bf16.mean().item():.6f}, std: {out_bf16.std().item():.6f}")
    print(f"Output dtype: {out_bf16.dtype}")

# Test with large values
print("\n" + "="*80)
print("LARGE VALUES TEST")
print("="*80)

x_large = torch.randn(100, 10, device=device) * 1000  # Large values

print("\nFP32 (no autocast):")
out_fp32_large = ln(x_large)
print(f"Output mean: {out_fp32_large.mean().item():.6f}, std: {out_fp32_large.std().item():.6f}")

print("\nFP16 autocast:")
with torch.autocast(device_type=device.type, dtype=torch.float16):
    out_fp16_large = ln(x_large)
    print(f"Output mean: {out_fp16_large.mean().item():.6f}, std: {out_fp16_large.std().item():.6f}")
    print(f"Output dtype: {out_fp16_large.dtype}")

print("\nBF16 autocast:")
with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
    out_bf16_large = ln(x_large)
    print(f"Output mean: {out_bf16_large.mean().item():.6f}, std: {out_bf16_large.std().item():.6f}")
    print(f"Output dtype: {out_bf16_large.dtype}")
