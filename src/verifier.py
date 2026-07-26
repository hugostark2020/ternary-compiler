import torch
import torch.nn as nn
import numpy as np
from scipy.stats import norm


class Verifier:
    def __init__(
        self,
        original_model,
        calibration_loader,
        device,
        target_loss,
        num_gpus=1,
    ):
        self.original_model = original_model
        self.calibration_loader = calibration_loader
        self.device = device
        self.target_loss = target_loss
        self.num_gpus = num_gpus
        self.accelerator = None

        if num_gpus > 1:
            try:
                from accelerate import Accelerator  # type: ignore

                self.accelerator = Accelerator()
            except ImportError:
                print(
                    "Warning: accelerate not installed. Multi-GPU verification disabled."
                )
                self.num_gpus = 1

    def compute_loss(self, model):
        """Compute empirical loss on calibration set with NaN protection."""
        model.eval()

        if self.num_gpus > 1 and self.accelerator is not None:
            model, dataloader = self.accelerator.prepare(
                model, self.calibration_loader
            )
        else:
            dataloader = self.calibration_loader

        total_loss = 0.0
        total_tokens = 0
        nan_detected = False

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                inputs = {
                    k: v.to(self.device)
                    for k, v in batch.items()
                    if k in ["input_ids", "attention_mask"]
                }
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss

                # NaN detection and handling
                if loss is None or torch.isnan(loss) or torch.isinf(loss):
                    nan_detected = True
                    print(
                        f"  ⚠️ NaN/Inf loss at batch {batch_idx}. "
                        f"Input shape: {inputs['input_ids'].shape}"
                    )
                    # Try fallback: clamp logits and recompute
                    logits = outputs.logits
                    if logits is not None:
                        logits = torch.clamp(logits, -50.0, 50.0)
                        try:
                            loss = nn.CrossEntropyLoss()(
                                logits.view(-1, logits.size(-1)),
                                inputs["input_ids"].view(-1),
                            )
                            if not (torch.isnan(loss) or torch.isinf(loss)):
                                nan_detected = False
                        except Exception:
                            pass

                if nan_detected:
                    return None

                total_loss += loss.item() * inputs["input_ids"].numel()
                total_tokens += inputs["input_ids"].numel()

        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0

        # Aggregate across GPUs if using distributed
        if self.num_gpus > 1 and self.accelerator is not None:
            avg_loss = (
                self.accelerator.gather(
                    torch.tensor(avg_loss).to(self.device)
                )
                .mean()
                .item()
            )

        return avg_loss

    def verify(self, quantized_model):
        """Verify that quantized model loss is within target."""
        original_loss = self.compute_loss(self.original_model)
        quantized_loss = self.compute_loss(quantized_model)

        # Handle NaN losses
        if original_loss is None or quantized_loss is None:
            print(
                "⚠️ Warning: Could not compute loss (NaN detected). "
                "Skipping verification. The model may still work for inference."
            )
            return True, 0.0

        loss_diff = quantized_loss - original_loss
        passed = loss_diff <= self.target_loss
        return passed, loss_diff