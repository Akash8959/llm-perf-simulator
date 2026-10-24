"""End-to-end LLM inference simulation and aggregate performance metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from llmsim.estimator import decode_estimate, memory_fit_check, prefill_estimate
from llmsim.hardware import GPUConfig
from llmsim.model import ModelConfig
from llmsim.parallelism import ParallelConfig
from llmsim.quantization import Precision


@dataclass(frozen=True)
class SimulationResult:
    """End-to-end inference performance simulation result.

    Attributes:
        ttft: Time to First Token (prefill phase latency) in seconds.
        tpot: Time Per Output Token (decode phase step latency averaged over generation) in seconds.
        e2e_latency: Total end-to-end execution duration (TTFT + decode generation) in seconds.
        tokens_per_sec: Generation throughput in tokens per second across active workers.
        memory_per_gpu_gb: Peak memory required per GPU in gigabytes (weights + KV cache).
        bound_prefill: Bottleneck regime during prefill ('compute' or 'memory').
        bound_decode: Bottleneck regime during decode ('compute' or 'memory').
        comm_share: Proportion of total execution duration spent in communication.
        is_oom: Whether peak memory requirement exceeds physical GPU capacity.
        memory_utilization: Peak memory footprint relative to physical GPU capacity.
        batch: Batch size per data-parallel replica.
        prompt_len: Input prompt sequence length in tokens.
        output_len: Number of tokens generated per sequence.
        world_size: Total GPU count across all parallelism dimensions.
    """

    ttft: float
    tpot: float
    e2e_latency: float
    tokens_per_sec: float
    memory_per_gpu_gb: float
    bound_prefill: str
    bound_decode: str
    comm_share: float
    is_oom: bool
    memory_utilization: float
    batch: int
    prompt_len: int
    output_len: int
    world_size: int

    def to_dict(self) -> dict[str, Any]:
        """Convert simulation results into a serializable dictionary."""
        return asdict(self)


def simulate(
    gpu: GPUConfig,
    model: ModelConfig,
    batch: int,
    prompt_len: int,
    output_len: int,
    parallel: ParallelConfig | None = None,
    weight_precision: Precision | str = Precision.FP16,
    kv_precision: Precision | str = Precision.FP16,
    eff_compute: float = 0.6,
    eff_memory: float = 0.8,
) -> SimulationResult:
    """Simulate end-to-end LLM inference performance across prefill and decode phases.

    Args:
        gpu: Target GPU hardware specification.
        model: Model architecture configuration.
        batch: Batch size per data parallel instance (B).
        prompt_len: Input prompt length in tokens (S).
        output_len: Number of output tokens to generate per sequence (O).
        parallel: Parallelism strategy configuration (TP, PP, DP, EP).
        weight_precision: Precision format for model weights.
        kv_precision: Precision format for KV cache storage.
        eff_compute: Compute hardware efficiency factor (default 0.60).
        eff_memory: Memory bandwidth hardware efficiency factor (default 0.80).

    Returns:
        SimulationResult containing TTFT, TPOT, e2e latency, throughput, memory, and bounds.
    """
    if batch <= 0:
        raise ValueError(f"batch must be positive, got {batch}.")
    if prompt_len <= 0:
        raise ValueError(f"prompt_len must be positive, got {prompt_len}.")
    if output_len <= 0:
        raise ValueError(f"output_len must be positive, got {output_len}.")

    p_weight = Precision.from_str(weight_precision)
    p_kv = Precision.from_str(kv_precision)
    p_cfg = parallel if parallel is not None else ParallelConfig()

    # 1. Memory Check at peak context length: C_peak = prompt_len + output_len
    peak_context = prompt_len + output_len
    mem_result = memory_fit_check(
        model=model,
        gpu=gpu,
        batch=batch,
        context_len=peak_context,
        weight_precision=p_weight,
        kv_precision=p_kv,
        parallel=p_cfg,
    )
    is_oom = not mem_result.fits
    peak_memory_gb = mem_result.total_memory_required_bytes / (1024**3)

    # 2. Prefill Phase (TTFT)
    pref_est = prefill_estimate(
        model=model,
        gpu=gpu,
        batch=batch,
        prompt_len=prompt_len,
        weight_precision=p_weight,
        kv_precision=p_kv,
        eff_compute=eff_compute,
        eff_memory=eff_memory,
        parallel=p_cfg,
    )
    ttft = pref_est.latency

    # 3. Decode Phase: Step context C from prompt_len to prompt_len + output_len - 1
    total_decode_latency = 0.0
    total_decode_comm = 0.0
    decode_bounds: list[str] = []

    for step in range(output_len):
        current_context = prompt_len + step
        dec_est = decode_estimate(
            model=model,
            gpu=gpu,
            batch=batch,
            context_len=current_context,
            weight_precision=p_weight,
            kv_precision=p_kv,
            eff_compute=eff_compute,
            eff_memory=eff_memory,
            parallel=p_cfg,
        )
        total_decode_latency += dec_est.latency
        total_decode_comm += dec_est.comm_time
        decode_bounds.append(dec_est.bound)

    tpot = total_decode_latency / float(output_len)
    e2e_latency = ttft + total_decode_latency

    # Overall communication share
    total_comm = pref_est.comm_time + total_decode_comm
    comm_share = total_comm / e2e_latency if e2e_latency > 0 else 0.0

    # Aggregate cluster throughput (accounting for Data Parallelism)
    effective_batch = batch * p_cfg.dp
    total_tokens_generated = effective_batch * output_len
    tokens_per_sec = (
        float(total_tokens_generated) / e2e_latency if e2e_latency > 0 else 0.0
    )

    # Dominant decode bottleneck
    mem_bound_count = sum(1 for bnd in decode_bounds if bnd == "memory")
    dominant_decode_bound = "memory" if mem_bound_count >= len(decode_bounds) / 2 else "compute"

    return SimulationResult(
        ttft=ttft,
        tpot=tpot,
        e2e_latency=e2e_latency,
        tokens_per_sec=tokens_per_sec,
        memory_per_gpu_gb=peak_memory_gb,
        bound_prefill=pref_est.bound,
        bound_decode=dominant_decode_bound,
        comm_share=comm_share,
        is_oom=is_oom,
        memory_utilization=mem_result.memory_utilization,
        batch=batch,
        prompt_len=prompt_len,
        output_len=output_len,
        world_size=p_cfg.world_size,
    )
