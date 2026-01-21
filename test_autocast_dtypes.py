import torch
import torch.nn as nn

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        print(f"Input dtype: {x.dtype}")
        
        x = self.fc1(x)
        print(f"fc1 output dtype: {x.dtype}")
        
        x = self.relu(x)
        print(f"ReLU output dtype: {x.dtype}")
        
        x = self.ln(x)
        print(f"LayerNorm output dtype: {x.dtype}")
        
        x = self.fc2(x)
        print(f"fc2 output (logits) dtype: {x.dtype}")
        
        return x

# Create model and move to GPU (if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

model = ToyModel(in_features=5, out_features=3).to(device)

# Check parameter dtypes before autocast
print("=" * 60)
print("PARAMETER DTYPES (before and within autocast)")
print("=" * 60)
print(f"fc1.weight dtype (original): {model.fc1.weight.dtype}")
print(f"ln.weight dtype (original): {model.ln.weight.dtype}")
print(f"fc2.weight dtype (original): {model.fc2.weight.dtype}")
print()

# Create dummy input
x = torch.randn(2, 5, device=device)
target = torch.randint(0, 3, (2,), device=device)

# Test with autocast
print("=" * 60)
print("FORWARD PASS DTYPES (within autocast context)")
print("=" * 60)
with torch.autocast(device_type=device.type, dtype=torch.float16):
    # Check if parameters changed
    print(f"fc1.weight dtype (in autocast): {model.fc1.weight.dtype}")
    print(f"ln.weight dtype (in autocast): {model.ln.weight.dtype}")
    print(f"fc2.weight dtype (in autocast): {model.fc2.weight.dtype}")
    print()
    
    logits = model(x)
    
    # Compute loss
    loss = nn.functional.cross_entropy(logits, target)
    print(f"\nLoss dtype: {loss.dtype}")

print()
print("=" * 60)
print("GRADIENT DTYPES (after backward pass)")
print("=" * 60)
loss.backward()

print(f"fc1.weight.grad dtype: {model.fc1.weight.grad.dtype}")
print(f"ln.weight.grad dtype: {model.ln.weight.grad.dtype}")
print(f"ln.bias.grad dtype: {model.ln.bias.grad.dtype}")
print(f"fc2.weight.grad dtype: {model.fc2.weight.grad.dtype}")
