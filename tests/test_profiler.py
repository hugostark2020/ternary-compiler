import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from profiler import SensitivityProfiler


class DummyModel(nn.Module):
    """A simple model for testing the profiler."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.fc1(input_ids)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(x, labels)
        return type("Output", (), {"loss": loss, "logits": x})()


def test_sensitivity_profiler():
    """Test that the profiler can compute sensitivities for a dummy model."""
    model = DummyModel()
    # Create dummy calibration data
    data = torch.randn(16, 64)
    labels = torch.randint(0, 10, (16,))
    dataset = torch.utils.data.TensorDataset(data, labels)

    def collate_fn(batch):
        inputs = torch.stack([b[0] for b in batch])
        lbls = torch.tensor([b[1] for b in batch])
        return {"input_ids": inputs, "labels": lbls}

    loader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

    profiler = SensitivityProfiler(model, loader, device="cpu")
    sensitivity = profiler.compute_sensitivity(num_batches=2)

    assert isinstance(sensitivity, dict), "Sensitivity should be a dict"
    assert len(sensitivity) > 0, "Sensitivity should not be empty"
    for name, sens in sensitivity.items():
        assert 0.0 <= sens <= 1.0, f"Sensitivity {name} should be in [0, 1], got {sens}"

    print("test_sensitivity_profiler PASSED")


if __name__ == "__main__":
    test_sensitivity_profiler()