import torch
import torch.nn as nn
from transformers import PreTrainedModel
from tqdm import tqdm
from typing import Dict, List


class SensitivityProfiler:
    """
    Identifies layer-wise sensitivity to quantization.
    """

    def __init__(self, model: PreTrainedModel, calibration_loader, device='cuda'):
        self.model = model.to(device)
        self.calibration_loader = calibration_loader
        self.device = device
        self.hooks = []
        self.activations = {}

    def _register_hooks(self):
        """Register forward hooks to capture layer outputs."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                hook = module.register_forward_hook(
                    lambda m, inp, out, name=name: self._capture_activation(name, out)
                )
                self.hooks.append(hook)

    def _capture_activation(self, name, output):
        self.activations[name] = output.detach().cpu()

    def compute_sensitivity(self, num_batches=10):
        """Compute Hessian trace approximation for each layer."""
        self._register_hooks()
        self.model.eval()
        sensitivity = {}

        # First, run forward passes to capture activations
        for batch_idx, batch in enumerate(tqdm(self.calibration_loader, desc="Forward passes")):
            if batch_idx >= num_batches:
                break
            inputs = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                self.model(**inputs)

        # Compute sensitivity = trace of Hessian approximation
        for name, act in self.activations.items():
            if act.dim() == 2:  # Linear layer: (batch, features)
                act_norm = torch.norm(act, dim=0)
                sensitivity[name] = torch.mean(act_norm).item()
            elif act.dim() == 3:  # Attention or sequence data
                act_norm = torch.norm(act, dim=(0, 1))
                sensitivity[name] = torch.mean(act_norm).item()
            else:
                sensitivity[name] = 1.0

        # Normalize sensitivities
        max_sens = max(sensitivity.values()) if sensitivity else 1.0
        for name in sensitivity:
            sensitivity[name] /= max_sens

        # Remove hooks
        for hook in self.hooks:
            hook.remove()

        return sensitivity