"""Tests for parametric sweeps, DataFrame structures, monotonicity, and visualizations."""

from pathlib import Path

import pytest

from llmsim.hardware import GPUConfig, load_gpu
from llmsim.model import ModelConfig, load_model
from llmsim.plots import (
    plot_latency_vs_batch,
    plot_roofline,
    plot_throughput_vs_tp,
    plot_tpot_vs_hbm,
)
from llmsim.recommend import generate_recommendation
from llmsim.sweep import (
    sweep_batch_size,
    sweep_hbm_bandwidth,
    sweep_quantization,
    sweep_sequence_length,
    sweep_sm_count,
    sweep_tp_interconnect,
)

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


@pytest.fixture
def h100_gpu() -> GPUConfig:
    """Fixture providing loaded H100 SXM5 GPUConfig."""
    return load_gpu(CONFIGS_DIR / "hardware" / "h100_sxm.yaml")


@pytest.fixture
def llama3_8b() -> ModelConfig:
    """Fixture providing loaded LLaMA-3-8B ModelConfig."""
    return load_model(CONFIGS_DIR / "models" / "llama3_8b.yaml")


def test_sweep_hbm_bandwidth_monotonic(llama3_8b: ModelConfig, h100_gpu: GPUConfig) -> None:
    """Verify HBM bandwidth sweep DataFrame shape and monotonic decrease in TPOT."""
    bw_vals = [1000.0, 2000.0, 3000.0, 4000.0]
    df = sweep_hbm_bandwidth(h100_gpu, llama3_8b, batch=1, bandwidth_values=bw_vals)

    assert len(df) == len(bw_vals)
    expected_cols = {"hbm_bandwidth_gbps", "ttft_ms", "tpot_ms", "tokens_per_sec", "is_oom"}
    assert expected_cols.issubset(df.columns)

    # In memory-bound decode at B=1, TPOT must monotonically decrease with higher HBM bandwidth
    tpots = df["tpot_ms"].tolist()
    for i in range(len(tpots) - 1):
        assert tpots[i + 1] <= tpots[i]


def test_sweep_batch_size_monotonic(llama3_8b: ModelConfig, h100_gpu: GPUConfig) -> None:
    """Verify batch size sweep DataFrame shape and monotonic throughput increase."""
    batch_sizes = [1, 2, 4, 8, 16]
    df = sweep_batch_size(
        h100_gpu, llama3_8b, prompt_len=256, output_len=32, batch_sizes=batch_sizes
    )

    assert len(df) == len(batch_sizes)

    # Throughput (tokens/sec) increases with batch size
    tps = df["tokens_per_sec"].tolist()
    for i in range(len(tps) - 1):
        assert tps[i + 1] > tps[i]

    # Memory requirement increases with batch size
    mems = df["memory_per_gpu_gb"].tolist()
    for i in range(len(mems) - 1):
        assert mems[i + 1] > mems[i]


def test_sweep_sequence_and_sm(llama3_8b: ModelConfig, h100_gpu: GPUConfig) -> None:
    """Verify sequence length sweep and SM count sweep behaviors."""
    # Sequence length sweep: TTFT increases with prompt length
    df_seq = sweep_sequence_length(
        h100_gpu, llama3_8b, batch=1, prompt_lengths=[256, 512, 1024], output_len=16
    )
    assert len(df_seq) == 3
    assert df_seq["ttft_ms"].iloc[0] < df_seq["ttft_ms"].iloc[1] < df_seq["ttft_ms"].iloc[2]

    # SM count sweep: compute capacity scales
    df_sm = sweep_sm_count(
        h100_gpu, llama3_8b, batch=16, scale_factors=[0.5, 1.0, 2.0], output_len=16
    )
    assert len(df_sm) == 3
    assert df_sm["peak_tflops_fp16"].iloc[0] < df_sm["peak_tflops_fp16"].iloc[2]


def test_sweep_quantization_monotonic(llama3_8b: ModelConfig, h100_gpu: GPUConfig) -> None:
    """Verify quantization sweep monotonically reduces memory footprint from FP16 to INT4."""
    df = sweep_quantization(h100_gpu, llama3_8b, batch=1, prompt_len=512, output_len=32)

    assert len(df) == 3
    fp16_mem = df[df["precision"] == "fp16"]["memory_per_gpu_gb"].iloc[0]
    int8_mem = df[df["precision"] == "int8"]["memory_per_gpu_gb"].iloc[0]
    int4_mem = df[df["precision"] == "int4"]["memory_per_gpu_gb"].iloc[0]

    assert fp16_mem > int8_mem > int4_mem

    fp16_tpot = df[df["precision"] == "fp16"]["tpot_ms"].iloc[0]
    int8_tpot = df[df["precision"] == "int8"]["tpot_ms"].iloc[0]
    int4_tpot = df[df["precision"] == "int4"]["tpot_ms"].iloc[0]

    assert fp16_tpot > int8_tpot >= int4_tpot


def test_sweep_tp_interconnect_shape(llama3_8b: ModelConfig, h100_gpu: GPUConfig) -> None:
    """Verify TP degree crossed with interconnect bandwidth produces correct Cartesian product."""
    tps = [1, 2, 4]
    bws = [100.0, 300.0]
    df = sweep_tp_interconnect(
        h100_gpu, llama3_8b, batch=2, prompt_len=256, output_len=16,
        tp_degrees=tps, interconnect_bandwidths=bws
    )

    assert len(df) == len(tps) * len(bws)
    assert set(df["tp_degree"].unique()) == set(tps)
    assert set(df["interconnect_bandwidth_gbps"].unique()) == set(bws)


def test_plots_generation(tmp_path: Path, llama3_8b: ModelConfig, h100_gpu: GPUConfig) -> None:
    """Verify all plotting routines generate valid, non-empty PNG files."""
    r_path = plot_roofline(h100_gpu, llama3_8b, batch=1, out_path=tmp_path / "roofline.png")
    assert r_path.is_file() and r_path.stat().st_size > 0

    df_b = sweep_batch_size(h100_gpu, llama3_8b, batch_sizes=[1, 2], output_len=16)
    b_path = plot_latency_vs_batch(df_b, out_path=tmp_path / "batch.png")
    assert b_path.is_file() and b_path.stat().st_size > 0

    df_h = sweep_hbm_bandwidth(h100_gpu, llama3_8b, bandwidth_values=[1000.0, 2000.0])
    h_path = plot_tpot_vs_hbm(df_h, out_path=tmp_path / "hbm.png")
    assert h_path.is_file() and h_path.stat().st_size > 0

    df_t = sweep_tp_interconnect(
        h100_gpu, llama3_8b, tp_degrees=[1, 2], interconnect_bandwidths=[100.0, 300.0],
        output_len=16
    )
    t_path = plot_throughput_vs_tp(df_t, out_path=tmp_path / "tp.png")
    assert t_path.is_file() and t_path.stat().st_size > 0


def test_recommendation_generation(llama3_8b: ModelConfig, h100_gpu: GPUConfig) -> None:
    """Verify recommendation generation contains expected diagnostic keywords."""
    df_h = sweep_hbm_bandwidth(h100_gpu, llama3_8b, bandwidth_values=[1000.0, 2000.0])
    rec = generate_recommendation({"hbm": df_h})
    assert isinstance(rec, str)
    assert "memory-bound" in rec.lower()
    assert "hbm bandwidth" in rec.lower()
