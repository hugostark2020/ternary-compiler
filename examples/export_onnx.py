"""
Export a quantized model to ONNX format for production deployment.
Also demonstrates vLLM-compatible saving.

Usage:
    python examples/export_onnx.py
    python examples/export_onnx.py --model meta-llama/Llama-2-7b-hf
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.compiler import TernaryCompiler


def main():
    model_name = "distilgpt2"
    print(f"Loading and compiling model: {model_name}")
    compiler = TernaryCompiler(
        model_name=model_name,
        num_calibration_samples=10,
        device="cuda" if __import__("torch").cuda.is_available() else "cpu",
        target_accuracy_loss=0.01,
    )

    print("Running compilation pipeline...")
    quantized_model, quant_info = compiler.compile()

    # 1. Export to ONNX
    print("\n--- Exporting to ONNX ---")
    onnx_path = compiler.export_onnx(
        quantized_model,
        compiler.tokenizer,
        save_path="./quantized_model/model.onnx",
        max_seq_len=128,
        dynamic_axes=True,
    )
    print(f"ONNX model saved to: {onnx_path}")

    # 2. Save for vLLM
    print("\n--- Saving for vLLM ---")
    vllm_path = compiler.deploy_vllm(
        quantized_model, save_path="./quantized_model"
    )
    print(f"vLLM-compatible model saved to: {vllm_path}")

    # 3. Verify ONNX (if onnx is installed)
    try:
        import onnx

        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model validation: PASSED")
    except ImportError:
        print(
            "Install onnx to validate: pip install onnx"
        )

    print("\nDone! Files produced:")
    for f in os.listdir("./quantized_model"):
        if not f.endswith(".bin") and not os.path.isdir(
            os.path.join("./quantized_model", f)
        ):
            print(f"  - quantized_model/{f}")


if __name__ == "__main__":
    main()