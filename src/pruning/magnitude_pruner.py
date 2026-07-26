"""
Magnitude-based pruning for sparse ternary quantization.
Combines pruning with ternary quantization for 10×+ compression.
"""
import torch
import torch.nn as nn
from typing import Dict, Optional


class MagnitudePruner:
    """
    Applies magnitude-based pruning before ternary quantization.
    Supports global and layer-wise sparsity.

    Args:
        sparsity_ratio: Fraction of weights to prune (0.0 to 1.0)
        global_pruning: If True, prune globally across all layers.
                        If False, prune each layer independently.
    """

    def __init__(self, sparsity_ratio: float = 0.3, global_pruning: bool = True):
        self.sparsity_ratio = sparsity_ratio
        self.global_pruning = global_pruning
        self.masks: Dict[str, torch.Tensor] = {}

    def compute_masks(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Compute pruning masks for all linear layers."""
        if self.global_pruning:
            return self._global_masks(model)
        else:
            return self._layerwise_masks(model)

    def _global_masks(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """
        Global pruning: keep top (1-sparsity_ratio) weights across all layers.
        This is more aggressive but can cause layer collapse.
        """
        all_weights = []
        weight_names = []
        for name, param in model.named_parameters():
            if "weight" in name and param.dim() >= 2:
                all_weights.append(param.data.abs().flatten())
                weight_names.append(name)

        if not all_weights:
            return {}

        all_weights = torch.cat(all_weights)
        threshold = torch.quantile(all_weights, self.sparsity_ratio)

        masks = {}
        for name, param in model.named_parameters():
            if "weight" in name and param.dim() >= 2:
                mask = (param.data.abs() > threshold).float()
                masks[name] = mask
        return masks

    def _layerwise_masks(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """
        Layer-wise pruning: each layer keeps (1-sparsity_ratio) of its weights.
        This is safer and preserves layer capacity.
        """
        masks = {}
        for name, param in model.named_parameters():
            if "weight" in name and param.dim() >= 2:
                flat = param.data.abs().flatten()
                threshold = torch.quantile(flat, self.sparsity_ratio)
                mask = (param.data.abs() > threshold).float()
                masks[name] = mask
        return masks

    def apply_pruning(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Apply pruning masks to the model."""
        masks = self.compute_masks(model)
        for name, param in model.named_parameters():
            if name in masks:
                param.data *= masks[name]
        self.masks = masks
        return masks

    def apply_sparse_ternary(self, model: nn.Module) -> nn.Module:
        """
        Apply pruning first, then ternary quantization.
        Result: sparse ternary weights with most weights = 0.

        Args:
            model: The model to prune and quantize

        Returns:
            Model with sparse ternary weights
        """
        # Step 1: Prune
        masks = self.apply_pruning(model)

        # Step 2: Quantize remaining weights to ternary
        for name, param in model.named_parameters():
            if "weight" in name and param.dim() >= 2 and name in masks:
                scale = torch.mean(torch.abs(param.data))
                if scale > 1e-8:
                    ternary = torch.clamp(
                        torch.round(param.data / scale), -1, 1
                    )
                    # Apply mask: pruned weights stay 0
                    param.data = ternary * scale * masks[name]

        return model