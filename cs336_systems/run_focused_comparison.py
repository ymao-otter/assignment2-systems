#!/usr/bin/env python3
"""
Run focused mixed precision comparison on small, medium, and large models.
"""

import subprocess
import json

# Test small, medium, large (skip xl and 2.7B to save time)
MODEL_CONFIGS = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
}

# Benchmark parameters
CONTEXT_LENGTH = 512
SEQUENCE_LENGTH = 512
BATCH_SIZE = 8
NUM_WARMUP = 10
NUM_STEPS = 50
VOCAB_SIZE = 8192

def parse_benchmark_output(output):
    """Parse benchmark output to extract metrics."""
    avg_time_ms = None
    std_time_ms = None
    tokens_per_sec = None
    peak_memory_gb = None
    
    for line in output.split('\n'):
        if "Average time per step:" in line and "ms" in line:
            try:
                avg_time_ms = float(line.split(':')[1].strip().split()[0])
            except:
                pass
        elif "Standard deviation:" in line and "ms" in line:
            try:
                std_time_ms = float(line.split(':')[1].strip().split()[0])
            except:
                pass
        elif "Throughput:" in line and "tokens/second" in line:
            try:
                tokens_per_sec = float(line.split(':')[1].strip().replace(',', '').split()[0])
            except:
                pass
        elif "Peak memory allocated:" in line:
            try:
                peak_memory_gb = float(line.split(':')[1].strip().split()[0])
            except:
                pass
    
    return {
        "avg_time_ms": avg_time_ms,
        "std_time_ms": std_time_ms,
        "tokens_per_sec": tokens_per_sec,
        "peak_memory_gb": peak_memory_gb,
    }

def run_benchmark(model_name, config, use_mixed_precision, include_backward):
    """Run a single benchmark."""
    precision = "BF16" if use_mixed_precision else "FP32"
    mode = "Fwd+Bwd" if include_backward else "Fwd"
    
    print(f"\nRunning: {model_name:8s} | {precision:4s} | {mode:7s}...", end=" ", flush=True)
    
    cmd = [
        "python", "benchmark.py",
        "--d-model", str(config["d_model"]),
        "--d-ff", str(config["d_ff"]),
        "--num-layers", str(config["num_layers"]),
        "--num-heads", str(config["num_heads"]),
        "--context-length", str(CONTEXT_LENGTH),
        "--sequence-length", str(SEQUENCE_LENGTH),
        "--batch-size", str(BATCH_SIZE),
        "--num-warmup", str(NUM_WARMUP),
        "--num-steps", str(NUM_STEPS),
        "--vocab-size", str(VOCAB_SIZE),
        "--device", "cuda",
        "--seed", "42",
    ]
    
    if include_backward:
        cmd.append("--backward")
    
    if use_mixed_precision:
        cmd.append("--mixed-precision")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False)
        
        if result.returncode == 0:
            metrics = parse_benchmark_output(result.stdout)
            print(f"{metrics['avg_time_ms']:.1f} ms")
            return {
                "success": True,
                "model": model_name,
                "precision": precision,
                "mode": mode,
                **metrics
            }
        else:
            print("FAILED")
            return {"success": False, "model": model_name, "precision": precision, "mode": mode, "error": "Non-zero exit"}
    
    except subprocess.TimeoutExpired:
        print("TIMEOUT")
        return {"success": False, "model": model_name, "precision": precision, "mode": mode, "error": "Timeout"}
    except Exception as e:
        print(f"ERROR: {e}")
        return {"success": False, "model": model_name, "precision": precision, "mode": mode, "error": str(e)}


def main():
    print("="*80)
    print("MIXED PRECISION (BF16) vs FULL PRECISION (FP32) COMPARISON")
    print("="*80)
    print(f"Models: {list(MODEL_CONFIGS.keys())}")
    print(f"Context Length: {CONTEXT_LENGTH}, Batch Size: {BATCH_SIZE}")
    print(f"Warmup: {NUM_WARMUP}, Timed Steps: {NUM_STEPS}")
    print("="*80)
    
    all_results = []
    
    # Test both forward-only and forward+backward
    for include_backward in [False, True]:
        mode_name = "Forward+Backward" if include_backward else "Forward-only"
        print(f"\n{'='*80}")
        print(f"Mode: {mode_name}")
        print(f"{'='*80}")
        
        for model_name, config in MODEL_CONFIGS.items():
            # FP32
            fp32 = run_benchmark(model_name, config, False, include_backward)
            all_results.append(fp32)
            
            # BF16
            bf16 = run_benchmark(model_name, config, True, include_backward)
            all_results.append(bf16)
            
            # Print comparison
            if fp32["success"] and bf16["success"]:
                speedup = fp32["avg_time_ms"] / bf16["avg_time_ms"]
                mem_reduction = (fp32["peak_memory_gb"] - bf16["peak_memory_gb"]) / fp32["peak_memory_gb"] * 100
                print(f"  → Speedup: {speedup:.2f}x, Memory reduction: {mem_reduction:.1f}%")
    
    # Save results
    with open("mixed_precision_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'Model':<10} {'Mode':<8} {'FP32 (ms)':<12} {'BF16 (ms)':<12} {'Speedup':<10} {'Mem Save':<10}")
    print("-"*80)
    
    for include_backward in [False, True]:
        mode_str = "Fwd+Bwd" if include_backward else "Fwd"
        for model_name in MODEL_CONFIGS.keys():
            fp32 = next((r for r in all_results if r["model"] == model_name and r["precision"] == "FP32" and 
                        ("Bwd" in r["mode"]) == include_backward), None)
            bf16 = next((r for r in all_results if r["model"] == model_name and r["precision"] == "BF16" and 
                        ("Bwd" in r["mode"]) == include_backward), None)
            
            if fp32 and bf16 and fp32["success"] and bf16["success"]:
                speedup = fp32["avg_time_ms"] / bf16["avg_time_ms"]
                mem_save = (fp32["peak_memory_gb"] - bf16["peak_memory_gb"]) / fp32["peak_memory_gb"] * 100
                print(f"{model_name:<10} {mode_str:<8} {fp32['avg_time_ms']:<12.1f} {bf16['avg_time_ms']:<12.1f} {speedup:<10.2f}x {mem_save:<9.1f}%")
    
    print("="*80)
    print("Results saved to: mixed_precision_results.json")
    print("="*80)


if __name__ == "__main__":
    main()
