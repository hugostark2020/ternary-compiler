import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantizer import MixedPrecisionQuantizer


def test_ternary_quantization():
    """Test ternary quantization produces values in {-1, 0, +1}."""
    quantizer = MixedPrecisionQuantizer()
    weight = torch.randn(32, 64)
    ternary = quantizer.quantize_ternary(weight)

    assert ternary.shape == weight.shape, "Shape should be preserved"
    unique_vals = torch.unique(ternary)
    for v in unique_vals:
        assert v.item() in [-1, 0, 1], f"Value {v.item()} not in {{-1, 0, +1}}"

    print("test_ternary_quantization PASSED")


def test_int8_quantization():
    """Test INT8 quantization produces values in [-127, 127]."""
    quantizer = MixedPrecisionQuantizer()
    weight = torch.randn(32, 64)
    int8, scale = quantizer.quantize_int8(weight)

    assert int8.shape == weight.shape, "Shape should be preserved"
    assert int8.min().item() >= -127, f"Min value {int8.min().item()} < -127"
    assert int8.max().item() <= 127, f"Max value {int8.max().item()} > 127"
    assert scale.shape == (32, 1), f"Scale shape {scale.shape} should be (32, 1)"

    print("test_int8_quantization PASSED")


def test_mixed_precision():
    """Test mixed precision quantization on a simple model."""
    quantizer = MixedPrecisionQuantizer()

    # Create a simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(64, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 10),
    )

    # Create dummy sensitivity map
    sensitivity = {}
    for name, param in model.named_parameters():
        if "weight" in name:
            sensitivity[name] = 0.5  # Medium sensitivity

    quantized, quant_info = quantizer.apply_mixed_precision(model, sensitivity)

    assert len(quantized) > 0, "Quantized state dict should not be empty"
    assert len(quant_info) > 0, "Quant info should not be empty"

    print("test_mixed_precision PASSED")


if __name__ == "__main__":
    test_ternary_quantization()
    test_int8_quantization()
    test_mixed_precision()