"""
Benchmark script for the Ternary Transformer Compiler.
Measures inference latency and memory before/after quantization.

Usage:
    python examples/benchmark.py --model distilgpt2 --samples 10 --runs 20
    python examples/benchmark.py --model distilgpt2 --samples 10 --runs 20 --compile
    python examples/benchmark.py --model meta-llama/Llama-2-7b-hf --samples 100 --runs 20
"""
import sys
import os
import time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.compiler import TernaryCompiler
from src.kernels import get_backend


def benchmark_model(
    model_name="distilgpt2",
    num_calibration_samples=10,
    num_runs=10,
    use_torch_compile=False,
):
    """Benchmark FP16 vs quantized model performance."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Model: {model_name}")
    print(f"Calibration samples: {num_calibration_samples}")
    print(f"Benchmark runs: {num_runs}")
    print(f"torch.compile: {'enabled' if use_torch_compile else 'disabled'}")
    print(f"Kernel backend: {get_backend()}")
    print("=" * 60)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt = "The future of artificial intelligence is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # --- Baseline FP16 ---
    print("\n[1/4] Loading FP16 baseline model...")
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).to(device)
    model_fp16.eval()

    # Warm-up
    print("Warming up...")
    with torch.no_grad():
        for _ in range(3):
            _ = model_fp16.generate(**inputs, max_new_tokens=30)

    # Measure FP16 latency
    print(f"Benchmarking FP16 ({num_runs} runs)...")
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
    print(f"  FP16 total time: {fp16_time:.2f}s")
    print(f"  FP16 avg time: {fp16_time/num_runs:.3f}s")
    print(f"  FP16 peak memory: {fp16_mem:.2f} GB" if device == "cuda" else "")

    # --- Quantized Model ---
    print("\n[2/4] Running Ternary Compiler pipeline...")
    compiler = TernaryCompiler(
        model_name=model_name,
        calibration_data="c4",
        num_calibration_samples=num_calibration_samples,
        device=device,
        target_accuracy_loss=0.01,
    )
    quant_model, quant_info = compiler.compile()
    quant_model.eval()

    # Measure quantized latency
    print(f"\n[3/4] Benchmarking quantized model ({num_runs} runs)...")
    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = quant_model.generate(**inputs, max_new_tokens=30)
    if device == "cuda":
        torch.cuda.synchronize()
    quant_time = time.time() - start
    quant_mem = (
        torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0
    )
    print(f"  Quantized total time: {quant_time:.2f}s")
    print(f"  Quantized avg time: {quant_time/num_runs:.3f}s")
    print(f"  Quantized peak memory: {quant_mem:.2f} GB" if device == "cuda" else "")

    # --- Results ---
    print("\n[4/4] Results Summary")
    print("=" * 60)
    speedup = fp16_time / quant_time if quant_time > 0 else 0
    print(f"  FP16 time:       {fp16_time:.2f}s")
    print(f"  Quantized time:  {quant_time:.2f}s")
    print(f"  Speedup:         {speedup:.2f}x")
    if device == "cuda":
        mem_savings = (fp16_mem - quant_mem) / fp16_mem * 100 if fp16_mem > 0 else 0
        print(f"  FP16 memory:     {fp16_mem:.2f} GB")
        print(f"  Quantized memory:{quant_mem:.2f} GB")
        print(f"  Memory savings:  {mem_savings:.1f}%")

    # Print quantization info
    print("\nQuantization strategy:")
    type_counts = {}
    for name, info in quant_info.items():
        t = info if isinstance(info, str) else info[0]
        type_counts[t] = type_counts.get(t, 0) + 1
    for t, count in sorted(type_counts.items()):
        print(f"  {t}: {count} layers")

    return speedup, quant_info


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark Ternary Transformer Compiler")
    parser.add_argument(
        "--model", type=str, default="distilgpt2",
        help="Model name (default: distilgpt2)"
    )
    parser.add_argument(
        "--samples", type=int, default=10,
        help="Number of calibration samples (default: 10)"
    )
    parser.add_argument(
        "--runs", type=int, default=10,
        help="Number of benchmark runs (default: 10)"
    )
    parser.add_argument(
        "--compile", action="store_true",
        help="Enable torch.compile optimization (auto-detected if available)"
    )
    args = parser.parse_args()

    benchmark_model(
        model_name=args.model,
        num_calibration_samples=args.samples,
        num_runs=args.runs,
        use_torch_compile=args.compile,
    )
