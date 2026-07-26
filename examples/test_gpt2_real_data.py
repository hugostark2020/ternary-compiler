"""
Test the Ternary Compiler on GPT-2 with real calibration data.
Measures perplexity before and after quantization.
"""
import sys
import os
import math
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.compiler import TernaryCompiler
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    model_name = "gpt2"
    cal_file = os.path.join(
        os.path.dirname(__file__), "..", "calibration_data.txt"
    )

    print("=" * 60)
    print("  TESTING ON GPT-2 WITH REAL CALIBRATION DATA")
    print("=" * 60)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).to("cpu")
    model.eval()

    # Load calibration texts
    with open(cal_file, "r") as f:
        texts = [l.strip() for l in f if l.strip()]
    print(f"Loaded {len(texts)} calibration texts")

    # Compute baseline perplexity
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for text in texts[:10]:
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=128
            )
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
            total_loss += loss * inputs["input_ids"].numel()
            total_tokens += inputs["input_ids"].numel()

    baseline_ppl = math.exp(total_loss / total_tokens)
    print(f"FP16 Baseline Perplexity: {baseline_ppl:.2f}")

    # Compile with real calibration data
    print("\nCompiling with real calibration data...")
    compiler = TernaryCompiler(
        model_name=model_name,
        calibration_data=cal_file,
        num_calibration_samples=20,
        device="cpu",
        target_accuracy_loss=0.01,
    )
    quant_model, quant_info = compiler.compile()
    quant_model.eval()

    # Compute quantized perplexity
    total_loss_q = 0.0
    total_tokens_q = 0
    with torch.no_grad():
        for text in texts[:10]:
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=128
            )
            outputs = quant_model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
            total_loss_q += loss * inputs["input_ids"].numel()
            total_tokens_q += inputs["input_ids"].numel()

    quant_ppl = math.exp(total_loss_q / total_tokens_q)
    loss_diff = quant_ppl - baseline_ppl
    loss_pct = (loss_diff / baseline_ppl) * 100

    print(f"\nResults:")
    print(f"  FP16 Perplexity:      {baseline_ppl:.2f}")
    print(f"  Quantized Perplexity: {quant_ppl:.2f}")
    print(f"  Accuracy Loss:        {loss_diff:.2f} ({loss_pct:.2f}%)")

    # Count quantization types
    type_counts = {}
    for name, info in quant_info.items():
        t = info if isinstance(info, str) else info[0]
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"  Quantization: {type_counts}")

    # Pass/fail
    if loss_pct <= 1.5:
        print(f"\n✅ PASSED: Accuracy loss {loss_pct:.2f}% ≤ 1.5% target")
    else:
        print(f"\n❌ FAILED: Accuracy loss {loss_pct:.2f}% > 1.5% target")

    print("=" * 60)


if __name__ == "__main__":
    main()