"""
GPU-accelerated QAT with LoRA adapters.
Uses mixed precision to speed up training on GPU.
"""
import torch
import torch.nn as nn
from tqdm import tqdm
import gc


class QATTrainer:
    """
    Quantization-Aware Training with LoRA adapters.

    Freezes the quantized model weights and adds LoRA adapters to
    sensitive layers. Trains only the adapters with mixed precision.

    Args:
        model: The quantized model to fine-tune
        calibration_loader: DataLoader with calibration samples
        device: Device to train on ('cuda' or 'cpu')
        lora_r: LoRA rank
        lora_alpha: LoRA alpha scaling
        target_modules: List of module names to attach LoRA
        learning_rate: Learning rate for training
        epochs: Number of training epochs
        mixed_precision: Use AMP (only effective on CUDA)
    """

    def __init__(
        self,
        model: nn.Module,
        calibration_loader,
        device: str = "cuda",
        lora_r: int = 8,
        lora_alpha: int = 16,
        target_modules=None,
        learning_rate: float = 1e-4,
        epochs: int = 5,
        mixed_precision: bool = True,
    ):
        self.model = model
        self.calibration_loader = calibration_loader
        self.device = device
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.target_modules = target_modules or ["q_proj", "v_proj"]
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.mixed_precision = mixed_precision and device == "cuda"

    def train(self) -> nn.Module:
        """
        Run QAT with LoRA adapters.

        Returns:
            Model with merged LoRA weights
        """
        self.model.to(self.device)

        # Configure LoRA
        try:
            from peft import LoraConfig, get_peft_model, TaskType  # type: ignore

            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                target_modules=self.target_modules,
                lora_dropout=0.1,
                bias="none",
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        except ImportError:
            print("Warning: peft not installed. Skipping LoRA. Using full fine-tuning.")
            self.model.train()

        # Optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate
        )

        # Mixed precision scaler
        scaler = torch.amp.GradScaler(enabled=self.mixed_precision)

        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            pbar = tqdm(
                self.calibration_loader,
                desc=f"QAT Epoch {epoch+1}/{self.epochs}",
            )
            for batch in pbar:
                inputs = {
                    k: v.to(self.device)
                    for k, v in batch.items()
                    if k in ["input_ids", "attention_mask"]
                }

                # Forward pass with mixed precision
                with torch.amp.autocast(
                    device_type=self.device, enabled=self.mixed_precision
                ):
                    outputs = self.model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss

                # Backward pass
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_loss = total_loss / len(self.calibration_loader)
            print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")

            # Clear GPU memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Merge LoRA weights back into base model
        try:
            self.model = self.model.merge_and_unload()
        except (AttributeError, NameError):
            pass

        return self.model