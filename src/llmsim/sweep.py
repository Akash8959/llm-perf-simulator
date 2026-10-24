"""Parametric sweep utilities returning structured pandas DataFrames."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

import pandas as pd

from llmsim.parallelism import ParallelConfig
from llmsim.quantization import Precision
from llmsim.simulator import simulate

if TYPE_CHECKING:
    from llmsim.hardware import GPUConfig
    from llmsim.model import ModelConfig


def sweep_hbm_bandwidth(
    gpu: GPUConfig,
    model: ModelConfig,
    batch: int = 1,
    prompt_len: int = 2048,
    output_len: int = 128,
    bandwidth_values: Sequence[float] | None = None,
    parallel: ParallelConfig | None = None,
    weight_precision: Precision | str = Precision.FP16,
    kv_precision: Precision | str = Precision.FP16,
) -> pd.DataFrame:
    """Sweep HBM bandwidth to analyze memory bandwidth sensitivity.

    Args:
        gpu: Base GPU configuration.
        model: Model architecture configuration.
        batch: Batch size.
        prompt_len: Input prompt length in tokens.
        output_len: Output generation length in tokens.
        bandwidth_values: Sequence of HBM bandwidth values in GB/s.
        parallel: Parallelism strategy configuration.
        weight_precision: Weight data precision.
        kv_precision: KV cache precision.

    Returns:
        DataFrame containing simulation results across varied HBM bandwidths.
    """
    if bandwidth_values is None:
        bandwidth_values = [1000.0, 1500.0, 2039.0, 2500.0, 3350.0, 4000.0, 5000.0]

    records = []
    for bw in bandwidth_values:
        swept_gpu = replace(gpu, hbm_bandwidth_gbps=float(bw))
        res = simulate(
            gpu=swept_gpu,
            model=model,
            batch=batch,
            prompt_len=prompt_len,
            output_len=output_len,
            parallel=parallel,
            weight_precision=weight_precision,
            kv_precision=kv_precision,
        )
        records.append({
            "hbm_bandwidth_gbps": bw,
            "ttft_ms": res.ttft * 1e3,
            "tpot_ms": res.tpot * 1e3,
            "e2e_latency_s": res.e2e_latency,
            "tokens_per_sec": res.tokens_per_sec,
            "memory_per_gpu_gb": res.memory_per_gpu_gb,
            "bound_prefill": res.bound_prefill,
            "bound_decode": res.bound_decode,
            "comm_share": res.comm_share,
            "is_oom": res.is_oom,
        })
    return pd.DataFrame(records)


def sweep_sm_count(
    gpu: GPUConfig,
    model: ModelConfig,
    batch: int = 16,
    prompt_len: int = 2048,
    output_len: int = 128,
    scale_factors: Sequence[float] | None = None,
    parallel: ParallelConfig | None = None,
    weight_precision: Precision | str = Precision.FP16,
    kv_precision: Precision | str = Precision.FP16,
) -> pd.DataFrame:
    """Sweep SM count and scale peak TFLOPs to evaluate compute sensitivity.

    Args:
        gpu: Base GPU configuration.
        model: Model architecture configuration.
        batch: Batch size.
        prompt_len: Input prompt length in tokens.
        output_len: Output generation length in tokens.
        scale_factors: Multiplicative factors scaling base SM count and TFLOPs.
        parallel: Parallelism strategy configuration.
        weight_precision: Weight data precision.
        kv_precision: KV cache precision.

    Returns:
        DataFrame containing simulation results across varied SM and compute capacities.
    """
    if scale_factors is None:
        scale_factors = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

    records = []
    for factor in scale_factors:
        swept_gpu = replace(
            gpu,
            sm_count=int(gpu.sm_count * factor),
            peak_tflops_fp16=gpu.peak_tflops_fp16 * factor,
            peak_tflops_bf16=gpu.peak_tflops_bf16 * factor,
            peak_tflops_fp8=gpu.peak_tflops_fp8 * factor,
            peak_tflops_int8=gpu.peak_tflops_int8 * factor,
        )
        res = simulate(
            gpu=swept_gpu,
            model=model,
            batch=batch,
            prompt_len=prompt_len,
            output_len=output_len,
            parallel=parallel,
            weight_precision=weight_precision,
            kv_precision=kv_precision,
        )
        records.append({
            "sm_scale_factor": factor,
            "sm_count": swept_gpu.sm_count,
            "peak_tflops_fp16": swept_gpu.peak_tflops_fp16,
            "ttft_ms": res.ttft * 1e3,
            "tpot_ms": res.tpot * 1e3,
            "e2e_latency_s": res.e2e_latency,
            "tokens_per_sec": res.tokens_per_sec,
            "bound_prefill": res.bound_prefill,
            "bound_decode": res.bound_decode,
            "is_oom": res.is_oom,
        })
    return pd.DataFrame(records)


def sweep_batch_size(
    gpu: GPUConfig,
    model: ModelConfig,
    prompt_len: int = 1024,
    output_len: int = 128,
    batch_sizes: Sequence[int] | None = None,
    parallel: ParallelConfig | None = None,
    weight_precision: Precision | str = Precision.FP16,
    kv_precision: Precision | str = Precision.FP16,
) -> pd.DataFrame:
    """Sweep batch sizes to observe throughput scaling and transition to compute-bound decode.

    Args:
        gpu: Base GPU configuration.
        model: Model architecture configuration.
        prompt_len: Input prompt length in tokens.
        output_len: Output generation length in tokens.
        batch_sizes: Sequence of batch sizes to evaluate.
        parallel: Parallelism strategy configuration.
        weight_precision: Weight data precision.
        kv_precision: KV cache precision.

    Returns:
        DataFrame containing performance metrics across batch sizes.
    """
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]

    records = []
    for b in batch_sizes:
        res = simulate(
            gpu=gpu,
            model=model,
            batch=b,
            prompt_len=prompt_len,
            output_len=output_len,
            parallel=parallel,
            weight_precision=weight_precision,
            kv_precision=kv_precision,
        )
        records.append({
            "batch_size": b,
            "ttft_ms": res.ttft * 1e3,
            "tpot_ms": res.tpot * 1e3,
            "e2e_latency_s": res.e2e_latency,
            "tokens_per_sec": res.tokens_per_sec,
            "memory_per_gpu_gb": res.memory_per_gpu_gb,
            "memory_utilization": res.memory_utilization,
            "bound_prefill": res.bound_prefill,
            "bound_decode": res.bound_decode,
            "is_oom": res.is_oom,
        })
    return pd.DataFrame(records)


def sweep_sequence_length(
    gpu: GPUConfig,
    model: ModelConfig,
    batch: int = 4,
    output_len: int = 128,
    prompt_lengths: Sequence[int] | None = None,
    parallel: ParallelConfig | None = None,
    weight_precision: Precision | str = Precision.FP16,
    kv_precision: Precision | str = Precision.FP16,
) -> pd.DataFrame:
    """Sweep input prompt lengths to characterize quadratic attention scaling and KV growth.

    Args:
        gpu: Base GPU configuration.
        model: Model architecture configuration.
        batch: Batch size.
        output_len: Output generation length in tokens.
        prompt_lengths: Sequence of prompt lengths in tokens.
        parallel: Parallelism strategy configuration.
        weight_precision: Weight data precision.
        kv_precision: KV cache precision.

    Returns:
        DataFrame containing performance metrics across prompt sequence lengths.
    """
    if prompt_lengths is None:
        prompt_lengths = [128, 256, 512, 1024, 2048, 4096, 8192]

    records = []
    for s in prompt_lengths:
        res = simulate(
            gpu=gpu,
            model=model,
            batch=batch,
            prompt_len=s,
            output_len=output_len,
            parallel=parallel,
            weight_precision=weight_precision,
            kv_precision=kv_precision,
        )
        records.append({
            "prompt_len": s,
            "total_context": s + output_len,
            "ttft_ms": res.ttft * 1e3,
            "tpot_ms": res.tpot * 1e3,
            "e2e_latency_s": res.e2e_latency,
            "tokens_per_sec": res.tokens_per_sec,
            "memory_per_gpu_gb": res.memory_per_gpu_gb,
            "bound_prefill": res.bound_prefill,
            "bound_decode": res.bound_decode,
            "is_oom": res.is_oom,
        })
    return pd.DataFrame(records)


def sweep_quantization(
    gpu: GPUConfig,
    model: ModelConfig,
    batch: int = 4,
    prompt_len: int = 1024,
    output_len: int = 128,
    precisions: Sequence[Precision | str] | None = None,
    parallel: ParallelConfig | None = None,
) -> pd.DataFrame:
    """Sweep precision formats (FP16, INT8, INT4) for weights and KV cache.

    Args:
        gpu: Base GPU configuration.
        model: Model architecture configuration.
        batch: Batch size.
        prompt_len: Input prompt length in tokens.
        output_len: Output generation length in tokens.
        precisions: Sequence of precisions to evaluate.
        parallel: Parallelism strategy configuration.

    Returns:
        DataFrame comparing memory footprints and latencies across precision levels.
    """
    if precisions is None:
        precisions = [Precision.FP16, Precision.INT8, Precision.INT4]

    records = []
    for p in precisions:
        prec = Precision.from_str(p)
        res = simulate(
            gpu=gpu,
            model=model,
            batch=batch,
            prompt_len=prompt_len,
            output_len=output_len,
            parallel=parallel,
            weight_precision=prec,
            kv_precision=prec,
        )
        records.append({
            "precision": prec.value,
            "bytes_per_param": prec.bytes_per_param,
            "ttft_ms": res.ttft * 1e3,
            "tpot_ms": res.tpot * 1e3,
            "e2e_latency_s": res.e2e_latency,
            "tokens_per_sec": res.tokens_per_sec,
            "memory_per_gpu_gb": res.memory_per_gpu_gb,
            "bound_prefill": res.bound_prefill,
            "bound_decode": res.bound_decode,
            "is_oom": res.is_oom,
        })
    return pd.DataFrame(records)


def sweep_tp_interconnect(
    gpu: GPUConfig,
    model: ModelConfig,
    batch: int = 8,
    prompt_len: int = 1024,
    output_len: int = 128,
    tp_degrees: Sequence[int] = (1, 2, 4, 8),
    interconnect_bandwidths: Sequence[float] | None = None,
    weight_precision: Precision | str = Precision.FP16,
    kv_precision: Precision | str = Precision.FP16,
) -> pd.DataFrame:
    """Cross Tensor Parallel degree with interconnect link bandwidths.

    Args:
        gpu: Base GPU configuration.
        model: Model architecture configuration.
        batch: Batch size.
        prompt_len: Input prompt length in tokens.
        output_len: Output generation length in tokens.
        tp_degrees: Sequence of tensor parallel degrees (e.g. 1, 2, 4, 8).
        interconnect_bandwidths: Sequence of interconnect bandwidths in GB/s.
        weight_precision: Weight data precision.
        kv_precision: KV cache precision.

    Returns:
        DataFrame with Cartesian product of TP degree and interconnect link bandwidths.
    """
    if interconnect_bandwidths is None:
        interconnect_bandwidths = [100.0, 300.0, 600.0, 900.0]

    records = []
    for tp in tp_degrees:
        for ibw in interconnect_bandwidths:
            swept_gpu = replace(gpu, interconnect_bandwidth_gbps=float(ibw))
            p_cfg = ParallelConfig(tp=tp)
            res = simulate(
                gpu=swept_gpu,
                model=model,
                batch=batch,
                prompt_len=prompt_len,
                output_len=output_len,
                parallel=p_cfg,
                weight_precision=weight_precision,
                kv_precision=kv_precision,
            )
            records.append({
                "tp_degree": tp,
                "interconnect_bandwidth_gbps": ibw,
                "ttft_ms": res.ttft * 1e3,
                "tpot_ms": res.tpot * 1e3,
                "e2e_latency_s": res.e2e_latency,
                "tokens_per_sec": res.tokens_per_sec,
                "comm_share": res.comm_share,
                "memory_per_gpu_gb": res.memory_per_gpu_gb,
                "is_oom": res.is_oom,
            })
    return pd.DataFrame(records)
