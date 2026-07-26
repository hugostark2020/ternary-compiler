"""
Dynamic Quantized Linear layer.

Quantizes activations per-batch at runtime to adapt to changing input
distributions, improving accuracy without retraining.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicQuantizedLinear(nn.Module):
    """
    Linear layer with dynamic activation quantization.

    Computes per-batch scaling factors for activations, quantizes to INT8,
    performs matrix multiply with quantized weights, and dequantizes output.

    Args:
        weight: FP16/FP32 weight tensor
        bias: Optional bias tensor
        activation_bits: Bit-width for activation quantization (default: 8)
        weight_bits: Bit-width for weight quantization (default: 2 for ternary)
    """

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor = None,
        activation_bits: int = 8,
        weight_bits: int = 2,
    ):
        super().__init__()
        self.weight = nn.Parameter(weight.clone(), requires_grad=False)
        self.bias = nn.Parameter(bias.clone()) if bias is not None else None
        self.activation_bits = activation_bits
        self.weight_bits = weight_bits

        # Pre-compute weight scale for ternary/int8 weights
        if weight_bits == 2:
            # Ternary: scale = mean(|w|)
            self.weight_scale = torch.mean(torch.abs(self.weight))
            if self.weight_scale < 1e-8:
                self.weight_scale = 1.0
        elif weight_bits == 8:
            # INT8: per-channel scale
            self.weight_scale = (
                torch.max(torch.abs(self.weight), dim=1, keepdim=True)[0] / 127.0
            )
            self.weight_scale = torch.clamp(self.weight_scale, min=1e-8)
        else:
            self.weight_scale = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with dynamic activation quantization.

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            Output tensor of shape (..., out_features)
        """
        # Dynamic quantization of activations
        if self.activation_bits == 8:
            # Compute per-batch scale
            orig_dtype = x.dtype
            max_abs = torch.max(torch.abs(x), dim=-1, keepdim=True)[0]
            scale = max_abs / 127.0
            scale = torch.clamp(scale, min=1e-8)

            # Quantize activations to INT8
            q = torch.clamp(torch.round(x / scale), -127, 127).to(orig_dtype)

            # Matrix multiply with quantized weights
            if self.weight_bits == 2:
                # Ternary weights: use scale
                out = torch.matmul(q, self.weight.t()) * self.weight_scale
            else:
                out = torch.matmul(q, self.weight.t())

            # Dequantize activations
            out = out * scale
        else:
            # No activation quantization
            out = torch.matmul(x, self.weight.t())

        if self.bias is not None:
            out += self.bias

        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.weight.shape[1]}, "
            f"out_features={self.weight.shape[0]}, "
            f"activation_bits={self.activation_bits}, "
            f"weight_bits={self.weight_bits}"
        )