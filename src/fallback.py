from peft import LoraConfig, get_peft_model, TaskType  # type: ignore
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import List


def _detect_attention_modules(model) -> List[str]:
    """Auto-detect attention projection module names in the model."""
    all_module_names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Look for common attention projection patterns
            if any(
                p in name
                for p in [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "query",
                    "key",
                    "value",
                    "out_proj",
                    "c_attn",
                    "c_proj",
                    "qkv",
                    "Wq",
                    "Wk",
                    "Wv",
                    "Wo",
                    "fc1",
                    "fc2",
                    "fc3",
                ]
            ):
                all_module_names.append(name)
    return all_module_names


class LoRAFallback:
    def __init__(self, model, calibration_loader, device, num_gpus=1):
        self.model = model
        self.calibration_loader = calibration_loader
        self.device = device
        self.num_gpus = num_gpus
        self.accelerator = None

        if num_gpus > 1:
            try:
                from accelerate import Accelerator  # type: ignore

                self.accelerator = Accelerator()
            except ImportError:
                print(
                    "Warning: accelerate not installed. Multi-GPU LoRA disabled."
                )
                self.num_gpus = 1

    def run(self, quantized_state_dict, epochs=3, lr=1e-4):
        # Load quantized weights into model
        self.model.load_state_dict(quantized_state_dict, strict=False)

        # Auto-detect target modules
        target_modules = _detect_attention_modules(self.model)
        if not target_modules:
            # Fallback: use common layer names for GPT-2 / OPT / LLaMA
            target_modules = ["c_attn", "c_proj", "fc1", "fc2"]

        print(f"LoRA target modules: {target_modules[:6]}...")

        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none",
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        # Prepare for distributed training if multi-GPU
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        if self.num_gpus > 1 and self.accelerator is not None:
            self.model, optimizer, dataloader = self.accelerator.prepare(
                self.model, optimizer, self.calibration_loader
            )
        else:
            dataloader = self.calibration_loader

        # Train
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in tqdm(
                dataloader, desc=f"LoRA Epoch {epoch+1}"
            ):
                inputs = {
                    k: v.to(self.device)
                    for k, v in batch.items()
                    if k in ["input_ids", "attention_mask"]
                }
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss

                if self.num_gpus > 1 and self.accelerator is not None:
                    self.accelerator.backward(loss)
                else:
                    loss.backward()

                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            if self.num_gpus > 1 and self.accelerator is not None:
                avg_loss = (
                    self.accelerator.gather(
                        torch.tensor(avg_loss).to(self.device)
                    )
                    .mean()
                    .item()
                )
            print(f"Epoch {epoch+1} loss: {avg_loss:.4f}")

        # Merge LoRA weights back into base model
        self.model = self.model.merge_and_unload()
        return self.model