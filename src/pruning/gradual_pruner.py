"""
Gradual magnitude pruning with regrowth (GMPR).
Supports sparsity up to 90% while maintaining accuracy.
"""
import torch
import torch.nn as nn
import math
from typing import Dict, Optional


class GradualPruner:
    """
    Gradual magnitude pruning with regrowth (GMPR).

    Gradually increases sparsity from 0 to target_sparsity over prune_steps,
    with periodic regrowth of important pruned weights.

    Args:
        target_sparsity: Target sparsity ratio (0.0 to 1.0)
        schedule: Pruning schedule ('linear', 'cosine', 'exponential')
        prune_every_n_steps: Update pruning mask every N steps
        regrowth_interval: Regrow pruned weights every N steps
        regrowth_ratio: Fraction of pruned weights to regrow
    """

    def __init__(
        self,
        target_sparsity: float = 0.9,
        schedule: str = "cosine",
        prune_every_n_steps: int = 10,
        regrowth_interval: int = 100,
        regrowth_ratio: float = 0.5,
    ):
        self.target_sparsity = target_sparsity
        self.schedule = schedule
        self.prune_every_n_steps = prune_every_n_steps
        self.regrowth_interval = regrowth_interval
        self.regrowth_ratio = regrowth_ratio
        self.current_sparsity = 0.0
        self.masks: Dict[str, torch.Tensor] = {}

    def get_sparsity_at_step(self, step: int) -> float:
        """Compute target sparsity at a given step based on schedule."""
        total_steps = self.prune_every_n_steps
        progress = min(step / total_steps, 1.0) if total_steps > 0 else 1.0

        if self.schedule == "linear":
            return self.target_sparsity * progress
        elif self.schedule == "cosine":
            return self.target_sparsity * (1 - math.cos(progress * math.pi / 2))
        elif self.schedule == "exponential":
            return self.target_sparsity * (1 - math.exp(-5 * progress))
        else:
            return self.target_sparsity * progress

    def apply_pruning(self, model: nn.Module, step: int) -> Dict[str, torch.Tensor]:
        """
        Apply gradual pruning and regrowth at the given step.

        Args:
            model: The model to prune
            step: Current training/pruning step

        Returns:
            Dict of weight name -> pruning mask
        """
        # Compute current sparsity target
        target = self.get_sparsity_at_step(step)

        # Collect all weight magnitudes for global thresholding
        all_magnitudes = []
        weight_names = []
        for name, param in model.named_parameters():
            if "weight" in name and param.dim() >= 2:
                all_magnitudes.append(param.data.abs().flatten())
                weight_names.append(name)

        if not all_magnitudes:
            return self.masks

        all_magnitudes = torch.cat(all_magnitudes)
        threshold = torch.quantile(all_magnitudes, target)

        # Create/update masks
        for name, param in model.named_parameters():
            if name not in weight_names:
                continue

            # Base mask: keep weights above threshold
            mask = (param.data.abs() > threshold).float()

            # Apply regrowth: unprune some weights that were pruned earlier
            if (
                name in self.masks
                and step > 0
                and step % self.regrowth_interval == 0
            ):
                pruned_mask = self.masks[name] == 0
                if pruned_mask.any():
                    # Regrow a fraction of pruned weights with highest magnitude
                    pruned_magnitudes = param.data.abs()[pruned_mask]
                    if pruned_magnitudes.numel() > 0:
                        regrowth_count = max(
                            1,
                            int(pruned_magnitudes.numel() * self.regrowth_ratio),
                        )
                        # Get indices of largest pruned weights
                        _, top_indices = torch.topk(
                            pruned_magnitudes.flatten(),
                            min(regrowth_count, pruned_magnitudes.numel()),
                        )
                        # Convert flat indices back to original shape
                        flat_mask = mask.flatten()
                        pruned_flat = pruned_mask.flatten()
                        pruned_positions = pruned_flat.nonzero(as_tuple=True)[0]
                        for idx in top_indices:
                            if idx < len(pruned_positions):
                                flat_mask[pruned_positions[idx]] = 1.0
                        mask = flat_mask.reshape(mask.shape)

            self.masks[name] = mask
            param.data *= mask

        # Update current sparsity
        if len(all_magnitudes) > 0:
            self.current_sparsity = 1.0 - (all_magnitudes > threshold).float().mean().item()

        return self.masks