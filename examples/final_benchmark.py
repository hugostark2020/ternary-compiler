"""
Final comprehensive benchmark for the Ternary Transformer Compiler.

Tests all configurations:
1. FP16 baseline
2. Static quantization (native)
3. Static quantization (CPU backend)
4. Dynamic quantization
5. Memory usage comparison

Usage:
    python examples/final_benchmark.py
    python examples/final_benchmark.py --model distilgpt2 --samples 10 --runs 5
"""
import sys
import os
import time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.compiler import TernaryCompiler
from src.kernels import get_backend, get_cpu_backend
from src.quantizer import replace_with_dynamic_layers


def benchmark_all_configs(model_name="distilgpt2", num_samples=10, num_runs=5):
    """Benchmark all quantization configurations."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print(f"  TERNARY TRANSFORMER COMPILER — FINAL BENCHMARK")
    print("=" * 70)
    print(f"  Model:        {model_name}")
    print(f"  Device:       {device}")
    print(f"  GPU Backend:  {get_backend()}")
    print(f"  Samples:      {num_samples}")
    print(f"  Runs:         {num_runs}")
    print("=" * 70)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt = "The future of artificial intelligence is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    results = {}

    # ── 1. FP16 Baseline ──
    print("\n[1/5] FP16 Baseline...")
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).to(device)
    model_fp16.eval()

    with torch.no_grad():
        for _ in range(2):
            _ = model_fp16.generate(**inputs, max_new_tokens=30)

    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model_fp16.generate(**inputs, max_new_tokens=30)
    if device == "cuda":
        torch.cuda.synchronize()
    fp16_time = time.time() - start
    fp16_mem = (
        torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0
    )
    results["FP16"] = {
        "time": fp16_time,
        "avg_time": fp16_time / num_runs,
        "memory_gb": fp16_mem,
    }
    print(f"    Time: {fp16_time:.2f}s ({fp16_time/num_runs:.3f}s/run)")
    if device == "cuda":
        print(f"    Memory: {fp16_mem:.2f} GB")

    # ── 2. Compile (Static Quantization) ──
    print("\n[2/5] Compiling model (static quantization)...")
    compiler = TernaryCompiler(
        model_name=model_name,
        num_calibration_samples=num_samples,
        device=device,
        target_accuracy_loss=0.01,
    )
    quant_model, quant_info = compiler.compile()
    quant_model.eval()

    # Count quantization types
    type_counts = {}
    for name, info in quant_info.items():
        t = info if isinstance(info, str) else info[0]
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"    Quantization: {type_counts}")

    # Benchmark static quantized
    with torch.no_grad():
        for _ in range(2):
            _ = quant_model.generate(**inputs, max_new_tokens=30)

    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = quant_model.generate(**inputs, max_new_tokens=30)
    if device == "cuda":
        torch.cuda.synchronize()
    static_time = time.time() - start
    static_mem = (
        torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0
    )
    results["Static Quantized"] = {
        "time": static_time,
        "avg_time": static_time / num_runs,
        "memory_gb": static_mem,
    }
    print(f"    Time: {static_time:.2f}s ({static_time/num_runs:.3f}s/run)")
    if device == "cuda":
        print(f"    Memory: {static_mem:.2f} GB")

    # ── 3. CPU Backend (if on CPU) ──
    if device == "cpu":
        print("\n[3/5] CPU Backend (Numba JIT)...")
        cpu_backend = get_cpu_backend()

        # Warm-up
        with torch.no_grad():
            for _ in range(2):
                _ = quant_model.generate(**inputs, max_new_tokens=30)

        start = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = quant_model.generate(**inputs, max_new_tokens=30)
        cpu_time = time.time() - start
        results["CPU Backend"] = {
            "time": cpu_time,
            "avg_time": cpu_time / num_runs,
            "memory_gb": 0,
        }
        print(f"    Time: {cpu_time:.2f}s ({cpu_time/num_runs:.3f}s/run)")

    # ── 4. Dynamic Quantization ──
    print("\n[4/5] Dynamic Quantization...")
    # Reload model and re-quantize with dynamic layers
    compiler2 = TernaryCompiler(
        model_name=model_name,
        num_calibration_samples=num_samples,
        device=device,
        target_accuracy_loss=0.01,
    )
    dyn_model, dyn_quant_info = compiler2.compile()
    dyn_model = replace_with_dynamic_layers(dyn_model, dyn_quant_info)
    dyn_model.eval()

    with torch.no_grad():
        for _ in range(2):
            _ = dyn_model.generate(**inputs, max_new_tokens=30)

    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = dyn_model.generate(**inputs, max_new_tokens=30)
    if device == "cuda":
        torch.cuda.synchronize()
    dyn_time = time.time() - start
    dyn_mem = (
        torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0
    )
    results["Dynamic Quantized"] = {
        "time": dyn_time,
        "avg_time": dyn_time / num_runs,
        "memory_gb": dyn_mem,
    }
    print(f"    Time: {dyn_time:.2f}s ({dyn_time/num_runs:.3f}s/run)")
    if device == "cuda":
        print(f"    Memory: {dyn_mem:.2f} GB")

    # ── 5. Results Summary ──
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print(f"  {'Configuration':<25} {'Total Time':<12} {'Avg Time':<12} {'Speedup':<10} {'Memory':<10}")
    print("-" * 70)

    baseline_time = results["FP16"]["time"]
    for config, data in results.items():
        speedup = baseline_time / data["time"] if data["time"] > 0 else 0
        mem_str = f"{data['memory_gb']:.2f} GB" if data['memory_gb'] > 0 else "N/A"
        print(
            f"  {config:<25} {data['time']:<12.2f} {data['avg_time']:<12.3f} {speedup:<10.2f}x {mem_str:<10}"
        )

    print("-" * 70)
    print(f"  Quantization strategy: {type_counts}")
    print(f"  GPU Backend: {get_backend()}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Final benchmark for Ternary Transformer Compiler"
    )
    parser.add_argument(
        "--model", type=str, default="distilgpt2",
        help="Model name (default: distilgpt2)"
    )
    parser.add_argument(
        "--samples", type=int, default=10,
        help="Calibration samples (default: 10)"
    )
    parser.add_argument(
        "--runs", type=int, default=5,
        help="Benchmark runs (default: 5)"
    )
    args = parser.parse_args()

    benchmark_all_configs(
        model_name=args.model,
        num_samples=args.samples,
        num_runs=args.runs,
    )