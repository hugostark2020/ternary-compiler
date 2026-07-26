import torch
import torch.nn as nn
import numpy as np


class MixedPrecisionQuantizer:
    """
    Applies mixed-precision quantization: ternary for V/FFN, INT8 for Q/K, FP16 for embeddings/heads.
    """

    def __init__(self, sensitivity_threshold=0.3):
        self.sensitivity_threshold = sensitivity_threshold

    def quantize_ternary(self, weight):
        """Quantize weight to {-1, 0, +1} using absolute mean scaling."""
        scale = torch.mean(torch.abs(weight))
        scale = torch.clamp(scale, min=1e-8)  # Prevent division by zero
        ternary = torch.clamp(torch.round(weight / scale), -1, 1)
        return ternary

    def quantize_int8(self, weight):
        """Quantize weight to INT8 with per-channel scaling."""
        scale = torch.max(torch.abs(weight), dim=1, keepdim=True)[0] / 127.0
        scale = torch.clamp(scale, min=1e-8)  # Prevent division by zero
        int8 = torch.clamp(torch.round(weight / scale), -127, 127)
        return int8, scale

    def apply_mixed_precision(self, model, sensitivity: dict):
        """
        Apply quantization based on sensitivity map.
        - Embeddings & output head: FP16 (no change)
        - Q/K: INT8
        - V & FFN: Ternary
        """
        quantized_state_dict = {}
        quant_info = {}

        for name, param in model.named_parameters():
            # Skip if not a weight matrix (e.g., bias)
            if 'weight' not in name or param.dim() < 2:
                quantized_state_dict[name] = param.data.clone()
                continue

            # Determine layer type
            sens = sensitivity.get(name, 0.5)

            if 'embed' in name or 'lm_head' in name or 'out_proj' in name:
                # Keep FP16 for sensitive output layers
                quantized_state_dict[name] = param.data.clone().half()
                quant_info[name] = 'FP16'
            elif 'q_proj' in name or 'k_proj' in name or sens > self.sensitivity_threshold:
                # INT8 for Q/K and highly sensitive layers
                int8, scale = self.quantize_int8(param.data)
                # Store as float32 to avoid dtype issues
                quantized_state_dict[name] = int8.float() * scale.float()
                quant_info[name] = ('INT8', scale)
            else:
                # Ternary for V, FFN, and low-sensitivity
                ternary = self.quantize_ternary(param.data)
                quantized_state_dict[name] = ternary
                quant_info[name] = 'Ternary'

        return quantized_state_dict, quant_info

    def _classify_layer(self, name):
        """Heuristic layer classification."""
        if 'q_proj' in name:
            return 'Q'
        if 'k_proj' in name:
            return 'K'
        if 'v_proj' in name:
            return 'V'
        if 'mlp' in name or 'ffn' in name or 'fc' in name:
            return 'FFN'
        if 'embed' in name or 'lm_head' in name:
            return 'Embed'
        return 'Other'


def replace_with_dynamic_layers(model, quant_info: dict):
    """
    Replace linear layers in a model with DynamicQuantizedLinear layers
    for runtime activation quantization.

    Args:
        model: The quantized model
        quant_info: Dict mapping layer names to quantization info

    Returns:
        Model with DynamicQuantizedLinear layers
    """
    from .layers import DynamicQuantizedLinear

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue

        # Determine weight bits from quant_info
        weight_bits = 16  # default FP16
        if name in quant_info:
            info = quant_info[name]
            if info == 'Ternary':
                weight_bits = 2
            elif isinstance(info, tuple) and info[0] == 'INT8':
                weight_bits = 8

        # Create dynamic quantized layer
        dyn_layer = DynamicQuantizedLinear(
            weight=module.weight.data,
            bias=module.bias.data if module.bias is not None else None,
            activation_bits=8,
            weight_bits=weight_bits,
        )

        # Replace the module in the parent
        parent_name = '.'.join(name.split('.')[:-1])
        child_name = name.split('.')[-1]
        if parent_name:
            parent = dict(model.named_modules())[parent_name]
        else:
            parent = model

        setattr(parent, child_name, dyn_layer)

    return model