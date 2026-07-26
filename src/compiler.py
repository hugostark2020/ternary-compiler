import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict, Any, Union
import numpy as np
import os

# Conditional import for datasets (optional)
try:
    from datasets import load_dataset, Dataset  # type: ignore
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    load_dataset = None
    Dataset = None

from .profiler import SensitivityProfiler
from .calibrator import SmoothQuantCalibrator
from .quantizer import MixedPrecisionQuantizer, replace_with_dynamic_layers
from .verifier import Verifier
from .fallback import LoRAFallback


class TernaryCompiler:
    """
    Main compiler orchestration: profile → calibrate → quantize → verify → (fallback).

    Supports single GPU, multi-GPU (with device_map="auto"), and CPU.
    """

    def __init__(
        self,
        model_name: str,
        calibration_data: Optional[str] = None,
        num_calibration_samples: int = 100,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        target_accuracy_loss: float = 0.005,
        confidence: float = 0.9999,
        num_gpus: Optional[int] = None,
        max_memory_per_gpu: str = "20GiB",
    ):
        self.model_name = model_name
        self.target_accuracy_loss = target_accuracy_loss
        self.confidence = confidence
        self.calibration_data = calibration_data or "c4"
        self.num_calibration_samples = num_calibration_samples
        self.num_gpus = num_gpus or torch.cuda.device_count()
        self.max_memory_per_gpu = max_memory_per_gpu

        # Determine effective device and load model
        self.device, self.model = self._load_model(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Prepare calibration data
        self.calibration_loader = self._prepare_calibration_data()

        # Components
        self.profiler = SensitivityProfiler(
            self.model, self.calibration_loader, self.device
        )
        self.calibrator = SmoothQuantCalibrator(
            self.model, self.calibration_loader, self.device
        )
        self.quantizer = MixedPrecisionQuantizer()
        self.verifier = Verifier(
            self.model,
            self.calibration_loader,
            self.device,
            target_accuracy_loss,
            num_gpus=self.num_gpus,
        )
        self.fallback = LoRAFallback(
            self.model,
            self.calibration_loader,
            self.device,
            num_gpus=self.num_gpus,
        )

    def _load_model(self, model_name: str):
        """Load model on single device or sharded across multiple GPUs."""
        if self.num_gpus > 1:
            print(
                f"Loading model on {self.num_gpus} GPUs with device_map='auto'"
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                max_memory={
                    i: self.max_memory_per_gpu for i in range(self.num_gpus)
                },
            )
            return "cuda", model
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading model on single device: {device}")
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16
            ).to(device)
            return device, model

    def _load_calibration_file(self, file_path: str):
        """Load calibration texts from a local file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Calibration file not found: {file_path}")

        if file_path.endswith(".jsonl"):
            import json

            texts = []
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        texts.append(json.loads(line)["text"])
                    except (json.JSONDecodeError, KeyError):
                        pass
        elif file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                texts = [line.strip() for line in f if line.strip()]
        else:
            raise ValueError(
                f"Unsupported file format: {file_path}. Use .txt or .jsonl"
            )

        if not texts:
            raise ValueError("No valid texts found in calibration file")

        texts = texts[: self.num_calibration_samples]
        print(f"Loaded {len(texts)} calibration samples from {file_path}")
        return Dataset.from_dict({"text": texts})

    def _prepare_calibration_data(self):
        """Create a calibration DataLoader — uses real data, local file, or synthetic."""
        # Check if a local file path is provided
        if self.calibration_data and os.path.exists(self.calibration_data):
            try:
                dataset = self._load_calibration_file(self.calibration_data)
                return self._tokenize_and_loader(dataset)
            except Exception as e:
                print(f"Warning: Could not load calibration file: {e}")
                print("Falling back to synthetic data.")

        # Try loading from datasets library
        if (
            DATASETS_AVAILABLE
            and self.calibration_data in ["c4", "wikitext", "bookcorpus"]
        ):
            try:
                dataset = load_dataset(self.calibration_data, split="train")
                dataset = dataset.select(
                    range(min(self.num_calibration_samples, len(dataset)))
                )
                return self._tokenize_and_loader(dataset)
            except Exception:
                pass

        # Fallback to synthetic
        return self._tokenize_and_loader(self._synthetic_dataset())

    def _tokenize_and_loader(self, dataset):
        """Tokenize a dataset and return a DataLoader."""
        if isinstance(dataset, torch.utils.data.DataLoader):
            return dataset

        def tokenize(batch):
            texts = batch.get(
                "text", batch.get("content", [""] * len(batch.get("text", [])))
            )
            return self.tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=256,
                return_tensors="pt",
            )

        dataset = dataset.map(tokenize, batched=True)
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        return torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

    def _synthetic_dataset(self):
        """Generate a dummy calibration dataset when no real data is available."""
        dummy_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence is transforming the world.",
            "The Navier-Stokes equations describe fluid dynamics.",
            "Transformers have revolutionized natural language processing.",
            "Global warming is a critical challenge for humanity.",
        ] * (self.num_calibration_samples // 5 + 1)
        dummy_texts = dummy_texts[: self.num_calibration_samples]
        return Dataset.from_dict({"text": dummy_texts})

    def _validate_model(self, model):
        """Quick forward pass to check for NaN outputs after quantization."""
        model.eval()
        try:
            dummy = torch.randint(0, 100, (1, 10)).to(self.device)
            with torch.no_grad():
                out = model(dummy)
            if hasattr(out, 'logits') and torch.isnan(out.logits).any():
                print("⚠️ Warning: Quantized model produces NaN logits.")
            else:
                print("  ✅ Model validation: outputs are finite")
        except Exception as e:
            print(f"  ⚠️ Model validation warning: {e}")

    def compile(self):
        """Run the full compilation pipeline."""
        print("Step 1: Sensitivity profiling...")
        sensitivity = self.profiler.compute_sensitivity()

        print("Step 2: SmoothQuant calibration...")
        activation_stats = self.calibrator.collect_activations()
        smoothing_factors = self.calibrator.compute_smoothing_factors(
            activation_stats
        )

        print("Step 3: Mixed-precision quantization...")
        quantized_state_dict, quant_info = self.quantizer.apply_mixed_precision(
            self.model, sensitivity
        )

        # Apply quantized weights
        self.model.load_state_dict(quantized_state_dict, strict=False)

        # Validate model produces finite outputs
        self._validate_model(self.model)

        print("Step 4: Verification...")
        passed, empirical_loss = self.verifier.verify(self.model)

        if not passed:
            print(
                f"PTQ failed with loss {empirical_loss:.4f} > {self.target_accuracy_loss:.4f}. Falling back to LoRA..."
            )
            self.fallback.run(quantized_state_dict)
            passed, empirical_loss = self.verifier.verify(self.model)
            if not passed:
                raise RuntimeError(
                    f"Compilation failed: cannot meet {self.target_accuracy_loss*100}% accuracy bound. "
                    f"Empirical loss: {empirical_loss:.4f}"
                )

        print(f"Compilation successful! Empirical loss: {empirical_loss:.4f}")
        return self.model, quant_info

    def export_onnx(
        self,
        model,
        tokenizer,
        save_path: str = "./quantized_model/model.onnx",
        opset_version: int = 14,
        max_seq_len: int = 512,
        batch_size: int = 1,
        dynamic_axes: bool = True,
    ):
        """
        Export the quantized model to ONNX format for deployment on
        TensorRT, ONNX Runtime, and cloud inference servers.
        """
        import torch.onnx

        model.eval()

        # Prepare example input
        example_input = tokenizer(
            "The future of AI is",
            return_tensors="pt",
            max_length=max_seq_len,
            truncation=True,
            padding="max_length",
        )
        input_ids = example_input["input_ids"].to(self.device)
        attention_mask = example_input["attention_mask"].to(self.device)

        # Dynamic axes for variable batch size and seq length
        if dynamic_axes:
            dynamic_axes_dict = {
                "input_ids": {0: "batch_size", 1: "seq_len"},
                "attention_mask": {0: "batch_size", 1: "seq_len"},
                "logits": {0: "batch_size", 1: "seq_len"},
            }
        else:
            dynamic_axes_dict = None

        # Create output directory
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        # Export to ONNX
        torch.onnx.export(
            model,
            (input_ids, attention_mask),
            save_path,
            opset_version=opset_version,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes_dict,
            do_constant_folding=True,
            export_params=True,
        )
        print(f"ONNX model exported to {save_path}")
        return save_path

    def deploy_vllm(
        self, model, model_name: str = None, save_path: str = "./quantized_model"
    ):
        """
        Save the quantized model in a format compatible with vLLM.
        vLLM expects standard transformers format with a config.json.
        """
        # Save with transformers format
        model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        # Update config.json with quantization info
        config = model.config
        config.update(
            {
                "quantization": "ternary",
                "quantized_model_name": model_name or self.model_name,
            }
        )
        config.save_pretrained(save_path)

        print(f"Model saved for vLLM at {save_path}")
        print(f"To serve, run: vllm serve {save_path}")
        return save_path