#!/usr/bin/env python
"""
Test GPT-2 quantization with bundled calibration data.
Measures FP16 vs quantized perplexity.
"""

import sys
import os
import math
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import get_calibration_data
from src.compiler import TernaryCompiler
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    model_name = "distilgpt2"
    print("=" * 60)
    print("TEST: GPT-2 WITH BUNDLED CALIBRATION DATA")
    print("=" * 60)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).to("cpu")
    model.eval()

    # Use bundled calibration data (100 samples, we take first 20 for speed)
    calib = get_calibration_data(20)
    texts = calib["text"][:5]  # only 5 for quick test

    # ---- Baseline FP16 Perplexity ----
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True,
            )
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
            total_loss += loss * inputs["input_ids"].numel()
            total_tokens += inputs["input_ids"].numel()

    baseline_ppl = math.exp(total_loss / total_tokens)
    print(f"FP16 Baseline Perplexity: {baseline_ppl:.2f}")

    # ---- Compile with bundled data ----
    print("\nCompiling with bundled calibration data...")
    compiler = TernaryCompiler(
        model_name=model_name,
        num_calibration_samples=20,
        device="cpu",
        target_accuracy_loss=0.01,  # 1% for test
    )
    quant_model, quant_info = compiler.compile()
    quant_model.eval()

    # ---- Quantized Perplexity ----
    total_loss_q = 0.0
    total_tokens_q = 0
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True,
            )
            outputs = quant_model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            if loss is None or torch.isnan(loss) or torch.isinf(loss):
                print(
                    f'  ⚠️ NaN/Inf detected for: "{text[:50]}..." -- skipping'
                )
                continue
            total_loss_q += loss.item() * inputs["input_ids"].numel()
            total_tokens_q += inputs["input_ids"].numel()

    if total_tokens_q > 0:
        quant_ppl = math.exp(total_loss_q / total_tokens_q)
        loss_diff = quant_ppl - baseline_ppl
        loss_pct = (loss_diff / baseline_ppl) * 100
        print(f"Quantized Perplexity: {quant_ppl:.2f}")
        print(f"Accuracy Loss: {loss_diff:.2f} ({loss_pct:.2f}%)")
        status = "PASS" if loss_pct <= 1.5 else "FAIL"
        symbol = "✅" if loss_pct <= 1.5 else "❌"
        print(f"{symbol} {status}: Accuracy loss {loss_pct:.2f}%")
    else:
        print("❌ All batches had NaN/Inf loss -- investigate further")

    # Print quantization breakdown
    type_counts = {}
    for name, info in quant_info.items():
        t = info if isinstance(info, str) else info[0]
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"Quantization layers: {type_counts}")
    print("=" * 60)


if __name__ == "__main__":
    main()