"""Tests for distributed parallelism models, communication time, and memory sharding."""

from dataclasses import replace
from pathlib import Path

import pytest

from llmsim.estimator import decode_estimate, memory_fit_check, prefill_estimate
from llmsim.hardware import GPUConfig, load_gpu
from llmsim.model import ModelConfig, load_model
from llmsim.parallelism import ParallelConfig, all_to_all_time, ring_allreduce_time

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


def test_tp1_zero_communication(llama3_8b: ModelConfig, h100_gpu: GPUConfig) -> None:
    """Verify TP=1 results in strictly zero communication overhead."""
    # Ring all-reduce function directly
    t_comm = ring_allreduce_time(
        message_bytes=1024 * 1024,
        num_gpus=1,
        link_bw_bytes_sec=900e9,
        link_latency_sec=2e-6,
    )
    assert t_comm == 0.0

    # Estimator integration
    p_tp1 = ParallelConfig(tp=1)
    pref = prefill_estimate(llama3_8b, h100_gpu, batch=1, prompt_len=512, parallel=p_tp1)
    dec = decode_estimate(llama3_8b, h100_gpu, batch=1, context_len=512, parallel=p_tp1)

    assert pref.comm_time == 0.0
    assert pref.comm_share == 0.0
    assert dec.comm_time == 0.0
    assert dec.comm_share == 0.0


def test_comm_time_rises_as_link_bw_drops(llama3_8b: ModelConfig, h100_gpu: GPUConfig) -> None:
    """Verify communication time increases as interconnect link bandwidth decreases."""
    fast_gpu = h100_gpu  # 900 GB/s NVLink
    slow_gpu = replace(h100_gpu, interconnect_bandwidth_gbps=100.0)  # 100 GB/s PCIe / slow link

    p_tp4 = ParallelConfig(tp=4)
    pref_fast = prefill_estimate(llama3_8b, fast_gpu, batch=4, prompt_len=1024, parallel=p_tp4)
    pref_slow = prefill_estimate(llama3_8b, slow_gpu, batch=4, prompt_len=1024, parallel=p_tp4)

    assert pref_slow.comm_time > pref_fast.comm_time
    assert pref_slow.latency > pref_fast.latency
    assert pref_slow.comm_share > pref_fast.comm_share


def test_tp8_cuts_per_gpu_weight_memory_8x(llama3_8b: ModelConfig, h100_gpu: GPUConfig) -> None:
    """Verify TP=8 cuts per-GPU weight memory footprint by a factor of 8."""
    fit_tp1 = memory_fit_check(llama3_8b, h100_gpu, batch=1, context_len=2048, tp=1)
    fit_tp8 = memory_fit_check(llama3_8b, h100_gpu, batch=1, context_len=2048, tp=8)

    ratio = fit_tp1.weight_memory_bytes / fit_tp8.weight_memory_bytes
    assert abs(ratio - 8.0) < 1e-4


def test_dp4_throughput_scaling() -> None:
    """Verify Data Parallel (DP) configuration scales aggregate throughput 4x."""
    p_dp1 = ParallelConfig(dp=1)
    p_dp4 = ParallelConfig(dp=4)

    assert p_dp4.dp == 4 * p_dp1.dp
    assert p_dp4.world_size == 4 * p_dp1.world_size


def test_ep_only_affects_moe(
    llama3_8b: ModelConfig, mixtral: ModelConfig, h100_gpu: GPUConfig
) -> None:
    """Verify EP only impacts MoE architectures and does not introduce comm overhead in dense."""
    p_base = ParallelConfig(tp=1, ep=1)
    p_ep4 = ParallelConfig(tp=1, ep=4)

    # Dense model (LLaMA-3-8B): EP has no effect
    dense_base = decode_estimate(llama3_8b, h100_gpu, batch=4, context_len=512, parallel=p_base)
    dense_ep4 = decode_estimate(llama3_8b, h100_gpu, batch=4, context_len=512, parallel=p_ep4)
    assert dense_base.comm_time == 0.0
    assert dense_ep4.comm_time == 0.0
    assert dense_base.latency == dense_ep4.latency

    # MoE model (Mixtral 8x7B): EP introduces all-to-all communication
    moe_base = decode_estimate(mixtral, h100_gpu, batch=4, context_len=512, parallel=p_base)
    moe_ep4 = decode_estimate(mixtral, h100_gpu, batch=4, context_len=512, parallel=p_ep4)
    assert moe_base.comm_time == 0.0
    assert moe_ep4.comm_time > 0.0
    assert moe_ep4.latency > moe_base.latency


def test_all_to_all_zero_for_single_gpu() -> None:
    """Verify all-to-all returns 0.0 for num_gpus <= 1."""
    assert all_to_all_time(1000, 1, 900e9, 2e-6) == 0.0
    assert all_to_all_time(1000, 0, 900e9, 2e-6) == 0.0
