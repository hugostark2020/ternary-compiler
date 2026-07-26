#!/usr/bin/env python
"""
Test GPT-2 quantization with bundled calibration data (GPU version).
Designed for Google Colab with T4 GPU.
"""
import sys
import os
import math
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.data import get_calibration_data
from src.compiler import TernaryCompiler
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "distilgpt2"
    print("=" * 60)
    print(f"TEST: GPT-2 WITH BUNDLED CALIBRATION DATA (device: {device})")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).to(device)
    model.eval()

    calib = get_calibration_data(20)
    texts = calib["text"][:5]

    # Baseline FP16
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
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
            total_loss += loss * inputs["input_ids"].numel()
            total_tokens += inputs["input_ids"].numel()

    baseline_ppl = math.exp(total_loss / total_tokens)
    print(f"FP16 Baseline Perplexity: {baseline_ppl:.2f}")

    # Compile
    print("\nCompiling with bundled calibration data...")
    compiler = TernaryCompiler(
        model_name=model_name,
        num_calibration_samples=20,
        device=device,
        target_accuracy_loss=0.01,
    )
    quant_model, quant_info = compiler.compile()
    quant_model.eval()

    # Quantized
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
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = quant_model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            if loss is None or torch.isnan(loss) or torch.isinf(loss):
                print(
                    f'  ⚠️ NaN/Inf detected for: "{text[:50]}..." — skipping'
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
        print("❌ All batches had NaN/Inf loss — investigate further")

    type_counts = {}
    for name, info in quant_info.items():
        t = info if isinstance(info, str) else info[0]
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"Quantization layers: {type_counts}")
    print("=" * 60)


if __name__ == "__main__":
    main()