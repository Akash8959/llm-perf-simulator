"""Tests for analytical roofline estimator, quantization, and memory capacity checks."""

from dataclasses import replace
from pathlib import Path

import pytest

from llmsim.estimator import (
    decode_estimate,
    kv_cache_bytes,
    memory_fit_check,
    prefill_estimate,
    ridge_point,
)
from llmsim.hardware import GPUConfig, load_gpu
from llmsim.model import ModelConfig, load_model
from llmsim.quantization import Precision

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


@pytest.fixture
def h100_gpu() -> GPUConfig:
    """Fixture providing loaded H100 SXM5 GPUConfig."""
    return load_gpu(CONFIGS_DIR / "hardware" / "h100_sxm.yaml")


@pytest.fixture
def llama3_8b() -> ModelConfig:
    """Fixture providing loaded LLaMA-3-8B ModelConfig."""
    return load_model(CONFIGS_DIR / "models" / "llama3_8b.yaml")


@pytest.fixture
def mixtral() -> ModelConfig:
    """Fixture providing loaded Mixtral 8x7B ModelConfig."""
    return load_model(CONFIGS_DIR / "models" / "mixtral_8x7b.yaml")


def test_kv_cache_hand_formula(llama3_8b: ModelConfig) -> None:
    """Verify KV cache bytes match hand calculation for LLaMA-3-8B.

    L=32, n_kv=8, d=128, C=2048, B=1, b_kv=2 (FP16)
    Formula: 2 * L * n_kv * d * C * B * b_kv
    Calculation: 2 * 32 * 8 * 128 * 2048 * 1 * 2 = 268,435,456 bytes (256 MB)
    """
    bytes_fp16 = kv_cache_bytes(llama3_8b, batch=1, context_len=2048, kv_precision=Precision.FP16)
    assert bytes_fp16 == 268_435_456

    # In INT8, should be exactly half (128 MB)
    bytes_int8 = kv_cache_bytes(llama3_8b, batch=1, context_len=2048, kv_precision=Precision.INT8)
    assert bytes_int8 == 134_217_728


def test_decode_memory_bound_at_batch_1(llama3_8b: ModelConfig, h100_gpu: GPUConfig) -> None:
    """Verify decode phase is memory-bound at batch size 1."""
    est = decode_estimate(llama3_8b, h100_gpu, batch=1, context_len=2048)
    assert est.bound == "memory"
    assert est.memory_time > est.compute_time
    assert est.arithmetic_intensity < ridge_point(
        h100_gpu, Precision.FP16, eff_compute=0.6, eff_memory=0.8
    )


def test_prefill_compute_bound_large_batch(llama3_8b: ModelConfig, h100_gpu: GPUConfig) -> None:
    """Verify prefill phase is compute-bound at S=4096 with large batch."""
    est = prefill_estimate(llama3_8b, h100_gpu, batch=16, prompt_len=4096)
    assert est.bound == "compute"
    assert est.compute_time > est.memory_time
    assert est.arithmetic_intensity > ridge_point(
        h100_gpu, Precision.FP16, eff_compute=0.6, eff_memory=0.8
    )


def test_higher_hbm_bw_never_increases_latency(
    llama3_8b: ModelConfig, h100_gpu: GPUConfig
) -> None:
    """Verify that increasing HBM bandwidth never increases decode or prefill latency."""
    base_gpu = h100_gpu
    fast_gpu = replace(base_gpu, hbm_bandwidth_gbps=base_gpu.hbm_bandwidth_gbps * 2.0)

    # Test decode (memory bound)
    dec_base = decode_estimate(llama3_8b, base_gpu, batch=1, context_len=2048)
    dec_fast = decode_estimate(llama3_8b, fast_gpu, batch=1, context_len=2048)
    assert dec_fast.latency <= dec_base.latency

    # Test prefill (compute bound or memory bound)
    pref_base = prefill_estimate(llama3_8b, base_gpu, batch=1, prompt_len=512)
    pref_fast = prefill_estimate(llama3_8b, fast_gpu, batch=1, prompt_len=512)
    assert pref_fast.latency <= pref_base.latency


def test_int8_weights_reduce_decode_latency(
    llama3_8b: ModelConfig, h100_gpu: GPUConfig
) -> None:
    """Verify INT8 weights reduce decode latency compared to FP16 in memory-bound regime."""
    dec_fp16 = decode_estimate(
        llama3_8b, h100_gpu, batch=1, context_len=2048, weight_precision=Precision.FP16
    )
    dec_int8 = decode_estimate(
        llama3_8b, h100_gpu, batch=1, context_len=2048, weight_precision=Precision.INT8
    )
    assert dec_int8.latency < dec_fp16.latency
    assert dec_int8.bytes < dec_fp16.bytes


def test_moe_decode_bytes_grow_with_batch(mixtral: ModelConfig, h100_gpu: GPUConfig) -> None:
    """Verify MoE decode bytes transferred per step increase monotonically as batch size grows."""
    dec_b1 = decode_estimate(mixtral, h100_gpu, batch=1, context_len=512)
    dec_b2 = decode_estimate(mixtral, h100_gpu, batch=2, context_len=512)
    dec_b8 = decode_estimate(mixtral, h100_gpu, batch=8, context_len=512)

    assert dec_b1.bytes < dec_b2.bytes < dec_b8.bytes


def test_memory_fit_check(llama3_8b: ModelConfig, h100_gpu: GPUConfig) -> None:
    """Verify memory fit check, OOM detection, and capacity limit calculations."""
    fit_ok = memory_fit_check(llama3_8b, h100_gpu, batch=1, context_len=2048)
    assert fit_ok.fits
    assert fit_ok.max_batch_size > 0
    assert fit_ok.max_context_length > 0

    # Simulate insufficient memory (e.g. tiny 4GB GPU)
    small_gpu = replace(h100_gpu, memory_capacity_gb=4.0)
    fit_fail = memory_fit_check(llama3_8b, small_gpu, batch=1, context_len=2048)
    assert not fit_fail.fits
    assert fit_fail.memory_utilization > 1.0


def test_precision_enum() -> None:
    """Verify Precision parsing, byte counts, and error handling."""
    assert Precision.from_str("fp16") == Precision.FP16
    assert Precision.from_str("BF16") == Precision.BF16
    assert Precision.from_str("int4") == Precision.INT4
    assert Precision.INT4.bytes_per_param == 0.5

    with pytest.raises(ValueError, match="Invalid precision"):
        Precision.from_str("fp32_not_supported")
