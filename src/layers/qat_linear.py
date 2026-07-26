"""
Quantization-Aware Training (QAT) Linear layer.
Uses straight-through estimator for ternary weight gradients during training.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class QATLinear(nn.Module):
    """
    Linear layer with Quantization-Aware Training support.

    During forward pass, weights are quantized to ternary {-1,0,+1} or INT8.
    During backward pass, gradients flow through the straight-through estimator
    to update the underlying full-precision weights.

    Args:
        in_features: Size of input features
        out_features: Size of output features
        bias: Whether to use bias
        weight_bits: 2 for ternary, 8 for INT8
    """

    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, weight_bits: int = 2
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_bits = weight_bits

        # Full-precision weights for training
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def quantize_weights(self) -> torch.Tensor:
        """Quantize weights with straight-through estimator."""
        if self.weight_bits == 2:
            # Ternary quantization: {-1, 0, +1}
            scale = torch.mean(torch.abs(self.weight))
            if scale < 1e-8:
                return torch.zeros_like(self.weight)
            q = torch.clamp(torch.round(self.weight / scale), -1, 1)
            return q * scale  # Dequantize for forward pass
        else:
            # INT8 quantization
            scale = torch.max(torch.abs(self.weight)) / 127.0
            scale = torch.clamp(scale, min=1e-8)
            q = torch.clamp(torch.round(self.weight / scale), -127, 127)
            return q * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with quantized weights and straight-through gradients.

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            Output tensor of shape (..., out_features)
        """
        # Quantize weights
        q_weight = self.quantize_weights()

        # Straight-through estimator:
        # Forward uses quantized weights, backward uses full-precision gradients
        weight = q_weight.detach() + self.weight - self.weight.detach()

        return F.linear(x, weight, self.bias)