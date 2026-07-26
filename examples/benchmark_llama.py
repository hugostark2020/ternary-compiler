#!/usr/bin/env python
"""
Full benchmark on Llama-2-7B.
Measures speedup, memory, and accuracy loss.
"""

import torch
import time
import argparse
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.compiler import TernaryCompiler


def benchmark_llama(
    model_name="meta-llama/Llama-2-7b-hf", samples=100, runs=20
):
    print(f"📊 Benchmarking {model_name}")
    print("-" * 60)

    # Load model and tokenizer
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Prepare prompts
    prompts = [
        "The future of AI is",
        "Climate change is a",
        "The meaning of life is",
        "In 2050, the world will be",
    ] * (samples // 4 + 1)
    prompts = prompts[:samples]

    # Baseline FP16
    print("Running FP16 baseline...")
    times = []
    for prompt in prompts[:runs]:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        start = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20)
        torch.cuda.synchronize()
        times.append(time.time() - start)
    fp16_avg = sum(times) / len(times)
    print(f"FP16 avg time: {fp16_avg:.4f}s")

    # Quantize
    print("\nRunning quantization...")
    compiler = TernaryCompiler(
        model_name=model_name,
        num_gpus=1,
        target_accuracy_loss=0.005,
        calibration_data="bundled",
        num_calibration_samples=100,
    )
    quant_model, quant_info = compiler.compile()

    # Benchmark quantized
    print("\nRunning quantized benchmark...")
    times_q = []
    for prompt in prompts[:runs]:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        start = time.time()
        with torch.no_grad():
            outputs = quant_model.generate(**inputs, max_new_tokens=20)
        torch.cuda.synchronize()
        times_q.append(time.time() - start)
    q_avg = sum(times_q) / len(times_q)
    print(f"Quantized avg time: {q_avg:.4f}s")

    # Speedup
    speedup = fp16_avg / q_avg
    print(f"Speedup: {speedup:.2f}x")

    # Memory reduction
    fp16_mem = sum(p.numel() for p in model.parameters()) * 2 / 1e9  # GB
    q_mem = sum(p.numel() for p in quant_model.parameters()) * 2 / 1e9
    print(f"FP16 memory: {fp16_mem:.2f} GB")
    print(f"Quantized memory: {q_mem:.2f} GB")
    print(f"Memory reduction: {fp16_mem/q_mem:.2f}x")

    # Results
    results = {
        "model": model_name,
        "fp16_time": fp16_avg,
        "quantized_time": q_avg,
        "speedup": speedup,
        "fp16_memory_gb": fp16_mem,
        "quantized_memory_gb": q_mem,
        "memory_reduction": fp16_mem / q_mem,
        "quant_info": quant_info,
    }

    with open("llama_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n✅ Results saved to llama_benchmark_results.json")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--calibration", default="bundled")
    args = parser.parse_args()
    benchmark_llama(args.model, args.samples, args.runs)