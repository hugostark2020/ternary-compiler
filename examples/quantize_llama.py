import sys
import os

# Add the parent directory to the path so we can import src package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.compiler import TernaryCompiler


def main():
    """Example script to quantize a small model using the Ternary Compiler."""
    # Use a small model for testing (distilgpt2 is ~82M params)
    # For production, use "meta-llama/Llama-2-7b-hf"
    model_name = "distilgpt2"

    print(f"Initializing Ternary Compiler for model: {model_name}")
    compiler = TernaryCompiler(
        model_name=model_name,
        calibration_data="c4",
        num_calibration_samples=10,  # Small for testing
        device="cuda" if __import__("torch").cuda.is_available() else "cpu",
        target_accuracy_loss=0.01,
    )

    print("Running compilation pipeline...")
    quantized_model, quant_info = compiler.compile()

    print("\nQuantization info:")
    for name, info in quant_info.items():
        print(f"  {name}: {info}")

    # Save the quantized model
    output_dir = "./quantized_model"
    os.makedirs(output_dir, exist_ok=True)
    quantized_model.save_pretrained(output_dir)
    print(f"\nQuantized model saved to {output_dir}")


if __name__ == "__main__":
    main()