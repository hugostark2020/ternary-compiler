"""
Architecture detection and quantization rules for different model types.
Supports GPT-2, LLaMA, Mistral, Falcon, Mamba, Phi, Qwen, and more.
"""
from typing import Dict, List


class ArchitectureDetector:
    """
    Detect model architecture and apply quantization rules.

    Each architecture has specific layer naming conventions and
    quantization sensitivity patterns.
    """

    ARCHITECTURE_PATTERNS = {
        "gpt2": [
            "transformer.h.",
            "wte",
            "wpe",
            "ln_f",
            "c_attn",
            "c_fc",
            "c_proj",
        ],
        "llama": [
            "model.layers.",
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ],
        "mistral": [
            "model.layers.",
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ],
        "falcon": [
            "transformer.h.",
            "self_attention.query_key_value",
            "self_attention.dense",
            "mlp.dense_h_to_4h",
            "mlp.dense_4h_to_h",
        ],
        "mamba": [
            "mamba.layers.",
            "ssm",
            "conv1d",
            "x_proj",
            "dt_proj",
            "out_proj",
        ],
        "phi": [
            "model.layers.",
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.dense",
            "mlp.fc1",
            "mlp.fc2",
        ],
        "qwen": [
            "model.layers.",
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ],
    }

    @classmethod
    def detect(cls, model) -> str:
        """
        Detect architecture from model config or module names.

        Args:
            model: A HuggingFace model or any nn.Module

        Returns:
            Architecture name string (e.g., "llama", "gpt2", "unknown")
        """
        # Check config first
        if hasattr(model, "config") and hasattr(model.config, "model_type"):
            return model.config.model_type

        # Check module names
        module_names = set()
        for name, _ in model.named_modules():
            parts = name.split(".")
            for part in parts:
                if part and part[0].islower():
                    module_names.add(part)

        module_str = str(module_names)
        for arch, patterns in cls.ARCHITECTURE_PATTERNS.items():
            if any(p in module_str for p in patterns[:2]):
                return arch

        return "unknown"

    @classmethod
    def get_quantization_rules(cls, arch: str) -> Dict[str, str]:
        """
        Get quantization rules for a specific architecture.

        Returns a dict mapping layer patterns to quantization types:
        - "FP16": Keep as FP16 (no quantization)
        - "INT8": Quantize to INT8
        - "TERNARY": Quantize to ternary {-1, 0, +1}

        Args:
            arch: Architecture name from detect()

        Returns:
            Dict of pattern -> quantization type
        """
        # Default rules
        rules: Dict[str, str] = {
            "embed": "FP16",
            "lm_head": "FP16",
            "embed_tokens": "FP16",
            "embed_positions": "FP16",
            "ln": "FP16",
            "layer_norm": "FP16",
            "norm": "FP16",
            "q_proj": "INT8",
            "k_proj": "INT8",
            "v_proj": "TERNARY",
            "o_proj": "TERNARY",
            "out_proj": "TERNARY",
            "down_proj": "TERNARY",
            "up_proj": "TERNARY",
            "gate_proj": "TERNARY",
            "fc1": "TERNARY",
            "fc2": "TERNARY",
            "fc3": "TERNARY",
            "c_fc": "TERNARY",
            "c_proj": "TERNARY",
            "c_attn": "INT8",
            "dense": "TERNARY",
            "dense_h_to_4h": "TERNARY",
            "dense_4h_to_h": "TERNARY",
            "query_key_value": "INT8",
        }

        # Architecture-specific overrides
        if arch in ("gpt2",):
            rules["c_attn"] = "INT8"  # QKV combined in GPT-2
            rules["c_fc"] = "TERNARY"
            rules["c_proj"] = "TERNARY"
        elif arch in ("llama", "mistral", "qwen"):
            rules["q_proj"] = "INT8"
            rules["k_proj"] = "INT8"
            rules["v_proj"] = "TERNARY"
            rules["gate_proj"] = "TERNARY"
        elif arch == "falcon":
            rules["query_key_value"] = "INT8"
            rules["dense"] = "TERNARY"
        elif arch == "mamba":
            rules["ssm"] = "INT8"
            rules["conv1d"] = "INT8"
            rules["x_proj"] = "TERNARY"
            rules["dt_proj"] = "TERNARY"
            rules["out_proj"] = "TERNARY"
        elif arch == "phi":
            rules["q_proj"] = "INT8"
            rules["k_proj"] = "INT8"
            rules["v_proj"] = "TERNARY"
            rules["fc1"] = "TERNARY"
            rules["fc2"] = "TERNARY"

        return rules