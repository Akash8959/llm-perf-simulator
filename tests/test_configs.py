"""Tests for hardware and model configurations and loading."""

from pathlib import Path

import pytest

from llmsim.hardware import GPUConfig, load_gpu
from llmsim.model import ModelConfig, load_model

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def test_load_gpu_valid() -> None:
    """Verify loading valid GPU configuration files."""
    a100 = load_gpu(CONFIGS_DIR / "hardware" / "a100_80gb.yaml")
    assert a100.name == "a100_80gb"
    assert a100.sm_count == 108
    assert a100.memory_capacity_gb == 80.0
    assert a100.memory_capacity_bytes == 80 * 1024**3
    assert a100.hbm_bandwidth_gbps == 2039.0

    h100 = load_gpu(CONFIGS_DIR / "hardware" / "h100_sxm.yaml")
    assert h100.name == "h100_sxm"
    assert h100.sm_count == 132
    assert h100.peak_tflops_fp8 == 1978.0


def test_load_gpu_invalid(tmp_path: Path) -> None:
    """Verify that invalid GPU configurations raise appropriate errors."""
    non_existent = tmp_path / "non_existent.yaml"
    with pytest.raises(FileNotFoundError):
        load_gpu(non_existent)

    # Missing field
    bad_yaml = tmp_path / "missing_field.yaml"
    bad_yaml.write_text("name: bad_gpu\nsm_count: 10\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing required hardware configuration fields"):
        load_gpu(bad_yaml)

    # Invalid values
    with pytest.raises(ValueError, match="sm_count must be positive"):
        GPUConfig(
            name="test",
            sm_count=0,
            peak_tflops_fp16=100.0,
            peak_tflops_bf16=100.0,
            peak_tflops_fp8=100.0,
            peak_tflops_int8=100.0,
            hbm_bandwidth_gbps=1000.0,
            memory_capacity_gb=80.0,
            l2_cache_mb=40.0,
            interconnect_bandwidth_gbps=600.0,
            interconnect_latency_us=4.0,
        )


def test_load_models_valid() -> None:
    """Verify loading all standard model configurations."""
    llama8b = load_model(CONFIGS_DIR / "models" / "llama3_8b.yaml")
    llama70b = load_model(CONFIGS_DIR / "models" / "llama3_70b.yaml")
    qwen7b = load_model(CONFIGS_DIR / "models" / "qwen2_7b.yaml")
    mixtral = load_model(CONFIGS_DIR / "models" / "mixtral_8x7b.yaml")
    deepseek = load_model(CONFIGS_DIR / "models" / "deepseek_moe.yaml")

    assert llama8b.layers == 32
    assert llama70b.layers == 80
    assert qwen7b.heads == 28
    assert mixtral.num_experts == 8
    assert deepseek.num_experts == 64


def test_llama3_8b_params() -> None:
    """Verify LLaMA-3-8B total parameter count is within 2% of 8.0B."""
    llama8b = load_model(CONFIGS_DIR / "models" / "llama3_8b.yaml")
    total = llama8b.total_params
    expected = 8.0e9
    diff_pct = abs(total - expected) / expected
    assert diff_pct < 0.02, f"LLaMA-3-8B param count {total} differs from 8.0B by {diff_pct:.4%}"
    assert llama8b.active_params == total  # Dense model has active == total


def test_llama3_70b_params() -> None:
    """Verify LLaMA-3-70B total parameter count is within 2% of 70.0B."""
    llama70b = load_model(CONFIGS_DIR / "models" / "llama3_70b.yaml")
    total = llama70b.total_params
    expected = 70.0e9
    diff_pct = abs(total - expected) / expected
    assert diff_pct < 0.02, f"LLaMA-3-70B param count {total} differs from 70.0B by {diff_pct:.4%}"
    assert llama70b.active_params == total  # Dense model


def test_moe_params() -> None:
    """Verify MoE parameter calculations for active vs total."""
    mixtral = load_model(CONFIGS_DIR / "models" / "mixtral_8x7b.yaml")
    assert mixtral.is_moe
    # Total params ~46.7B, active params ~12.9B
    assert 45e9 < mixtral.total_params < 48e9
    assert 12e9 < mixtral.active_params < 14e9
    assert mixtral.active_params < mixtral.total_params

    deepseek = load_model(CONFIGS_DIR / "models" / "deepseek_moe.yaml")
    assert deepseek.is_moe
    assert deepseek.active_params < deepseek.total_params


def test_model_invalid_config(tmp_path: Path) -> None:
    """Verify that invalid model configurations raise errors."""
    with pytest.raises(ValueError, match="divisible by heads"):
        ModelConfig(
            name="invalid",
            layers=12,
            hidden=4095,  # Not divisible by 32
            heads=32,
            kv_heads=8,
            ffn_dim=11008,
            vocab=32000,
        )

    with pytest.raises(ValueError, match="divisible by kv_heads"):
        ModelConfig(
            name="invalid",
            layers=12,
            hidden=4096,
            heads=32,
            kv_heads=7,  # 32 not divisible by 7
            ffn_dim=11008,
            vocab=32000,
        )

    with pytest.raises(ValueError, match="cannot exceed num_experts"):
        ModelConfig(
            name="invalid",
            layers=12,
            hidden=4096,
            heads=32,
            kv_heads=8,
            ffn_dim=11008,
            vocab=32000,
            num_experts=4,
            top_k=8,  # top_k > num_experts
        )
