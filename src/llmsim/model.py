"""Model architecture configuration and parameter counting for LLM inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    """Architectural parameters of a Transformer-based Large Language Model.

    Attributes:
        name: Model identifier (e.g. 'llama3_8b').
        layers: Number of transformer layers (L).
        hidden: Hidden dimension size (h).
        heads: Number of query attention heads (n_h).
        kv_heads: Number of key/value attention heads (n_kv).
        ffn_dim: Feed-forward intermediate dimension (d_ffn).
        vocab: Vocabulary size (V).
        num_experts: Total number of routed experts in MoE (E). Defaults to 1 for dense models.
        top_k: Number of active routed experts per token (k). Defaults to 1 for dense models.
        shared_experts: Number of shared experts always active in MoE. Defaults to 0.
    """

    name: str
    layers: int
    hidden: int
    heads: int
    kv_heads: int
    ffn_dim: int
    vocab: int
    num_experts: int = 1
    top_k: int = 1
    shared_experts: int = 0

    def __post_init__(self) -> None:
        """Validate model configuration parameters."""
        if not self.name or not isinstance(self.name, str):
            raise ValueError("Model name must be a non-empty string.")
        if self.layers <= 0:
            raise ValueError(f"layers must be positive, got {self.layers}.")
        if self.hidden <= 0:
            raise ValueError(f"hidden must be positive, got {self.hidden}.")
        if self.heads <= 0:
            raise ValueError(f"heads must be positive, got {self.heads}.")
        if self.kv_heads <= 0:
            raise ValueError(f"kv_heads must be positive, got {self.kv_heads}.")
        if self.ffn_dim <= 0:
            raise ValueError(f"ffn_dim must be positive, got {self.ffn_dim}.")
        if self.vocab <= 0:
            raise ValueError(f"vocab must be positive, got {self.vocab}.")
        if self.num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {self.num_experts}.")
        if self.top_k <= 0:
            raise ValueError(f"top_k must be positive, got {self.top_k}.")
        if self.shared_experts < 0:
            raise ValueError(f"shared_experts must be non-negative, got {self.shared_experts}.")

        if self.hidden % self.heads != 0:
            raise ValueError(
                f"hidden ({self.hidden}) must be divisible by heads ({self.heads})."
            )
        if self.heads % self.kv_heads != 0:
            raise ValueError(
                f"heads ({self.heads}) must be divisible by kv_heads ({self.kv_heads}) for GQA/MHA."
            )
        if self.top_k > self.num_experts:
            raise ValueError(
                f"top_k ({self.top_k}) cannot exceed num_experts ({self.num_experts})."
            )

    @property
    def head_dim(self) -> int:
        """Dimension per attention head: d = h / n_h."""
        return self.hidden // self.heads

    @property
    def attention_params_per_layer(self) -> int:
        """Attention weights per layer: Wq (h*h) + Wk (h*n_kv*d) + Wv (h*n_kv*d) + Wo (h*h)."""
        h = self.hidden
        d = self.head_dim
        n_kv = self.kv_heads
        # Wq = h * h, Wk = h * n_kv * d, Wv = h * n_kv * d, Wo = h * h
        return 2 * (h * h) + 2 * (h * n_kv * d)

    @property
    def ffn_params_per_layer(self) -> int:
        """Total FFN weights per layer including all routed and shared experts.

        For SwiGLU: 3 * h * ffn_dim per expert (gate, up, down projections).
        For MoE: router (h * num_experts) + num_experts * (3 * h * ffn_dim) + shared experts.
        """
        h = self.hidden
        expert_ffn = 3 * h * self.ffn_dim
        if self.num_experts <= 1:
            return expert_ffn
        router = h * self.num_experts
        routed = self.num_experts * expert_ffn
        shared = self.shared_experts * expert_ffn
        return router + routed + shared

    @property
    def active_ffn_params_per_layer(self) -> int:
        """Active FFN weights evaluated per token per layer."""
        h = self.hidden
        expert_ffn = 3 * h * self.ffn_dim
        if self.num_experts <= 1:
            return expert_ffn
        router = h * self.num_experts
        active_routed = self.top_k * expert_ffn
        shared = self.shared_experts * expert_ffn
        return router + active_routed + shared

    @property
    def embedding_params(self) -> int:
        """Embedding and LM head parameters: 2 * vocab * hidden."""
        return 2 * self.vocab * self.hidden

    @property
    def params_per_layer(self) -> int:
        """Total parameters in a single transformer layer."""
        return self.attention_params_per_layer + self.ffn_params_per_layer

    @property
    def active_params_per_layer(self) -> int:
        """Active parameters evaluated per token in a single transformer layer."""
        return self.attention_params_per_layer + self.active_ffn_params_per_layer

    @property
    def total_params(self) -> int:
        """Total parameter count of the model (including all experts and embeddings)."""
        return self.embedding_params + self.layers * self.params_per_layer

    @property
    def active_params(self) -> int:
        """Active parameter count evaluated per token (active experts and embeddings)."""
        return self.embedding_params + self.layers * self.active_params_per_layer

    @property
    def is_moe(self) -> bool:
        """Whether the model utilizes a Mixture-of-Experts architecture."""
        return self.num_experts > 1


def load_model(path: str | Path) -> ModelConfig:
    """Load and validate a model configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Validated ModelConfig instance.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If configuration values are invalid or required fields are missing.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Model configuration file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        data: Any = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML content in {file_path}; expected dictionary.")

    required_keys = {"name", "layers", "hidden", "heads", "kv_heads", "ffn_dim", "vocab"}
    missing_keys = required_keys - set(data.keys())
    if missing_keys:
        raise ValueError(f"Missing required model configuration fields: {sorted(missing_keys)}")

    return ModelConfig(
        name=str(data["name"]),
        layers=int(data["layers"]),
        hidden=int(data["hidden"]),
        heads=int(data["heads"]),
        kv_heads=int(data["kv_heads"]),
        ffn_dim=int(data["ffn_dim"]),
        vocab=int(data["vocab"]),
        num_experts=int(data.get("num_experts", 1)),
        top_k=int(data.get("top_k", 1)),
        shared_experts=int(data.get("shared_experts", 0)),
    )
