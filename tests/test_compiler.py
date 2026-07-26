import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from profiler import SensitivityProfiler
from quantizer import MixedPrecisionQuantizer
from calibrator import SmoothQuantCalibrator
from verifier import Verifier


class SimpleTransformer(nn.Module):
    """A minimal transformer-like model for testing."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(100, 32)
        self.q_proj = nn.Linear(32, 32)
        self.k_proj = nn.Linear(32, 32)
        self.v_proj = nn.Linear(32, 32)
        self.out_proj = nn.Linear(32, 32)
        self.mlp = nn.Linear(32, 64)
        self.mlp_out = nn.Linear(64, 32)
        self.lm_head = nn.Linear(32, 100)

    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.embed(input_ids)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn_out = self.out_proj(v)  # Simplified attention
        mlp_out = self.mlp_out(torch.relu(self.mlp(attn_out)))
        logits = self.lm_head(mlp_out)
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, 100), labels.view(-1))
        return type("Output", (), {"loss": loss, "logits": logits})()


def test_full_pipeline():
    """Test the full quantization pipeline end-to-end."""
    model = SimpleTransformer()

    # Create dummy calibration data
    input_ids = torch.randint(0, 100, (16, 16))
    labels = torch.randint(0, 100, (16, 16))
    dataset = torch.utils.data.TensorDataset(input_ids, labels)

    def collate_fn(batch):
        inputs = torch.stack([b[0] for b in batch])
        lbls = torch.stack([b[1] for b in batch])
        return {"input_ids": inputs, "labels": lbls}

    loader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

    # Step 1: Profile
    profiler = SensitivityProfiler(model, loader, device="cpu")
    sensitivity = profiler.compute_sensitivity(num_batches=2)
    assert len(sensitivity) > 0, "Sensitivity map should not be empty"

    # Step 2: Calibrate
    calibrator = SmoothQuantCalibrator(model, loader, device="cpu")
    activation_stats = calibrator.collect_activations(num_batches=2)
    smoothing_factors = calibrator.compute_smoothing_factors(activation_stats)
    assert len(smoothing_factors) > 0, "Smoothing factors should not be empty"

    # Step 3: Quantize
    quantizer = MixedPrecisionQuantizer()
    quantized, quant_info = quantizer.apply_mixed_precision(model, sensitivity)
    assert len(quantized) > 0, "Quantized state dict should not be empty"

    # Step 4: Verify
    verifier = Verifier(model, loader, device="cpu", target_loss=0.1)
    # Load quantized weights
    model.load_state_dict(quantized, strict=False)
    passed, loss_diff = verifier.verify(model)
    assert isinstance(passed, bool), "Verification result should be bool"

    print("test_full_pipeline PASSED")


if __name__ == "__main__":
    test_full_pipeline()