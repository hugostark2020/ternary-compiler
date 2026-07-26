import torch
import torch.nn as nn
from transformers import PreTrainedModel


class SmoothQuantCalibrator:
    """
    Applies SmoothQuant activation smoothing to reduce outliers.
    """

    def __init__(self, model: PreTrainedModel, calibration_loader, device='cuda'):
        self.model = model.to(device)
        self.calibration_loader = calibration_loader
        self.device = device

    def collect_activations(self, num_batches=10):
        """Collect activation statistics."""
        activation_stats = {}
        hooks = []

        def hook_fn(name):
            def fn(module, inp, out):
                if isinstance(out, torch.Tensor):
                    activation_stats[name] = out.detach().cpu()
            return fn

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                hooks.append(module.register_forward_hook(hook_fn(name)))

        self.model.eval()
        for batch_idx, batch in enumerate(self.calibration_loader):
            if batch_idx >= num_batches:
                break
            inputs = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                self.model(**inputs)

        for hook in hooks:
            hook.remove()

        return activation_stats

    def compute_smoothing_factors(self, activation_stats, alpha=0.5):
        """
        Compute per-channel scaling factors to smooth activations.
        """
        smoothing_factors = {}
        for name, act in activation_stats.items():
            # Compute per-channel max and moving average
            if act.dim() == 2:
                max_vals = torch.max(torch.abs(act), dim=0)[0]
            elif act.dim() == 3:
                max_vals = torch.amax(torch.abs(act), dim=(0, 1))
            else:
                continue
            # Smooth: scale = (max)^alpha
            scale = max_vals ** alpha
            scale = torch.clamp(scale, min=1e-8)
            smoothing_factors[name] = scale
        return smoothing_factors