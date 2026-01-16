"""
Run benchmarks for all model sizes from the assignment.

Model configurations:
- small:  d_model=768,  d_ff=3072,  num_layers=12, num_heads=12
- medium: d_model=1024, d_ff=4096,  num_layers=24, num_heads=16
- large:  d_model=1280, d_ff=5120,  num_layers=36, num_heads=20
- xl:     d_model=1600, d_ff=6400,  num_layers=48, num_heads=25
- 2.7B:   d_model=2560, d_ff=10240, num_layers=32, num_heads=32
"""

import subprocess
import sys

# Model configurations
MODEL_CONFIGS = {
    "small": {
        "d_model": 768,
        "d_ff": 3072,
        "num_layers": 12,
        "num_heads": 12,
    },
    "medium": {
        "d_model": 1024,
        "d_ff": 4096,
        "num_layers": 24,
        "num_heads": 16,
    },
    "large": {
        "d_model": 1280,
        "d_ff": 5120,
        "num_layers": 36,
        "num_heads": 20,
    },
    "xl": {
        "d_model": 1600,
        "d_ff": 6400,
        "num_layers": 48,
        "num_heads": 25,
    },
    "2.7B": {
        "d_model": 2560,
        "d_ff": 10240,
        "num_layers": 32,
        "num_heads": 32,
    },
}

# Benchmark settings
NUM_WARMUP = 5
NUM_STEPS = 10
BATCH_SIZE = 8
SEQUENCE_LENGTH = 256
VOCAB_SIZE = 8192
CONTEXT_LENGTH = 512
DEVICE = "cuda"

def run_benchmark(model_name, config, include_backward):
    """Run a single benchmark configuration."""
    pass_type = "forward+backward" if include_backward else "forward-only"
    
    print("\n" + "=" * 100)
    print(f"BENCHMARKING: {model_name.upper()} - {pass_type.upper()}")
    print("=" * 100)
    
    cmd = [
        "uv", "run", "python3", "-m", "cs336_systems.benchmark",
        "--d-model", str(config["d_model"]),
        "--d-ff", str(config["d_ff"]),
        "--num-layers", str(config["num_layers"]),
        "--num-heads", str(config["num_heads"]),
        "--num-warmup", str(NUM_WARMUP),
        "--num-steps", str(NUM_STEPS),
        "--batch-size", str(BATCH_SIZE),
        "--sequence-length", str(SEQUENCE_LENGTH),
        "--vocab-size", str(VOCAB_SIZE),
        "--context-length", str(CONTEXT_LENGTH),
        "--device", DEVICE,
    ]
    
    if include_backward:
        cmd.append("--backward")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running benchmark for {model_name} ({pass_type}): {e}")
        return False
    
    return True

def main():
    print("=" * 100)
    print("TRANSFORMER MODEL BENCHMARK SUITE")
    print("=" * 100)
    print("\nConfiguration:")
    print(f"  Warmup steps: {NUM_WARMUP}")
    print(f"  Measurement steps: {NUM_STEPS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Sequence length: {SEQUENCE_LENGTH}")
    print(f"  Device: {DEVICE}")
    print("\nModel sizes to benchmark:")
    for name, config in MODEL_CONFIGS.items():
        print(f"  {name:8s}: d_model={config['d_model']:4d}, d_ff={config['d_ff']:5d}, "
              f"layers={config['num_layers']:2d}, heads={config['num_heads']:2d}")
    
    # Optionally skip some models to avoid OOM
    models_to_run = list(MODEL_CONFIGS.keys())
    
    # You can comment out models that may cause OOM
    # models_to_run = ["small", "medium"]  # Run only smaller models
    
    print("\n" + "=" * 100)
    print("Starting benchmarks...")
    print("=" * 100)
    
    results = {}
    
    for model_name in models_to_run:
        config = MODEL_CONFIGS[model_name]
        
        # Run forward-only benchmark
        print(f"\n{'*' * 100}")
        print(f"Running {model_name} - FORWARD ONLY")
        print(f"{'*' * 100}")
        success = run_benchmark(model_name, config, include_backward=False)
        
        if not success:
            print(f"Skipping backward pass for {model_name} due to forward pass failure")
            continue
        
        # Run forward+backward benchmark
        print(f"\n{'*' * 100}")
        print(f"Running {model_name} - FORWARD + BACKWARD")
        print(f"{'*' * 100}")
        run_benchmark(model_name, config, include_backward=True)
    
    print("\n" + "=" * 100)
    print("ALL BENCHMARKS COMPLETE")
    print("=" * 100)

if __name__ == "__main__":
    main()
