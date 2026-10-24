"""Tests for end-to-end inference simulator and metrics aggregation."""

from dataclasses import replace
from pathlib import Path

import pytest

from llmsim.hardware import GPUConfig, load_gpu
from llmsim.model import ModelConfig, load_model
from llmsim.parallelism import ParallelConfig
from llmsim.quantization import Precision
from llmsim.simulator import simulate

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


@pytest.fixture
def h100_gpu() -> GPUConfig:
    """Fixture providing loaded H100 SXM5 GPUConfig."""
    return load_gpu(CONFIGS_DIR / "hardware" / "h100_sxm.yaml")


@pytest.fixture
def llama3_8b() -> ModelConfig:
    """Fixture providing loaded LLaMA-3-8B ModelConfig."""
    return load_model(CONFIGS_DIR / "models" / "llama3_8b.yaml")


def test_simulate_e2e_relation(llama3_8b: ModelConfig, h100_gpu: GPUConfig) -> None:
    """Verify e2e latency equals TTFT + output_len * TPOT."""
    batch = 4
    prompt_len = 512
    output_len = 64

    res = simulate(
        gpu=h100_gpu,
        model=llama3_8b,
        batch=batch,
        prompt_len=prompt_len,
        output_len=output_len,
        weight_precision=Precision.FP16,
    )

    expected_e2e = res.ttft + (output_len * res.tpot)
    assert abs(res.e2e_latency - expected_e2e) < 1e-9
    assert res.ttft > 0.0
    assert res.tpot > 0.0


def test_simulate_throughput(llama3_8b: ModelConfig, h100_gpu: GPUConfig) -> None:
    """Verify generation throughput equals (batch * output_len) / e2e for DP=1."""
    batch = 8
    prompt_len = 256
    output_len = 32

    res = simulate(
        gpu=h100_gpu,
        model=llama3_8b,
        batch=batch,
        prompt_len=prompt_len,
        output_len=output_len,
        parallel=ParallelConfig(dp=1),
    )

    expected_throughput = (batch * output_len) / res.e2e_latency
    assert abs(res.tokens_per_sec - expected_throughput) < 1e-5


def test_simulate_oom_flagged(llama3_8b: ModelConfig, h100_gpu: GPUConfig) -> None:
    """Verify OOM flag is raised when memory requirements exceed physical capacity."""
    # Sufficient memory (H100 80GB)
    res_fits = simulate(
        gpu=h100_gpu,
        model=llama3_8b,
        batch=4,
        prompt_len=512,
        output_len=64,
    )
    assert not res_fits.is_oom
    assert res_fits.memory_utilization < 1.0

    # Constrained memory (GPU with only 8GB capacity)
    small_gpu = replace(h100_gpu, memory_capacity_gb=8.0)
    res_oom = simulate(
        gpu=small_gpu,
        model=llama3_8b,
        batch=4,
        prompt_len=512,
        output_len=64,
    )
    assert res_oom.is_oom
    assert res_oom.memory_utilization > 1.0


def test_simulate_to_dict(llama3_8b: ModelConfig, h100_gpu: GPUConfig) -> None:
    """Verify SimulationResult converts cleanly to a dictionary with all required keys."""
    res = simulate(
        gpu=h100_gpu,
        model=llama3_8b,
        batch=1,
        prompt_len=128,
        output_len=16,
    )
    data = res.to_dict()
    assert isinstance(data, dict)
    expected_keys = {
        "ttft",
        "tpot",
        "e2e_latency",
        "tokens_per_sec",
        "memory_per_gpu_gb",
        "bound_prefill",
        "bound_decode",
        "comm_share",
        "is_oom",
        "memory_utilization",
        "batch",
        "prompt_len",
        "output_len",
        "world_size",
    }
    assert expected_keys.issubset(data.keys())
