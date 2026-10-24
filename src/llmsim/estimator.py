"""Analytical roofline performance estimation for prefill and decode phases."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from llmsim.parallelism import ParallelConfig, all_to_all_time, ring_allreduce_time
from llmsim.quantization import Precision

if TYPE_CHECKING:
    from llmsim.hardware import GPUConfig
    from llmsim.model import ModelConfig


@dataclass(frozen=True)
class PhaseEstimate:
    """Performance metrics and roofline classification for an execution phase.

    Attributes:
        flops: Floating-point operations required for the phase (total across all devices).
        bytes: High-bandwidth memory traffic in bytes (total across all devices).
        arithmetic_intensity: Operational intensity (FLOPs / byte).
        compute_time: Execution duration limited solely by compute throughput (seconds).
        memory_time: Execution duration limited solely by memory bandwidth (seconds).
        latency: Total execution latency including communication = max(compute, memory) + comm.
        bound: Limiting execution bottleneck ('compute' or 'memory').
        comm_time: Inter-device communication latency (seconds).
        comm_share: Proportion of total latency spent in communication (0.0 to 1.0).
    """

    flops: float
    bytes: float
    arithmetic_intensity: float
    compute_time: float
    memory_time: float
    latency: float
    bound: str
    comm_time: float = 0.0
    comm_share: float = 0.0


@dataclass(frozen=True)
class MemoryFitResult:
    """Memory requirement analysis and capacity headroom per GPU.

    Attributes:
        fits: Whether weights and KV cache fit within GPU memory capacity.
        total_memory_required_bytes: Total required bytes per GPU.
        weight_memory_bytes: Parameter weight footprint per GPU in bytes.
        kv_cache_bytes: Key-Value cache footprint per GPU in bytes.
        gpu_capacity_bytes: Total physical GPU memory capacity in bytes.
        memory_utilization: Ratio of required memory to physical GPU capacity.
        max_batch_size: Maximum batch size supportable before OOM at current context length.
        max_context_length: Maximum context length supportable before OOM at current batch size.
    """

    fits: bool
    total_memory_required_bytes: int
    weight_memory_bytes: int
    kv_cache_bytes: int
    gpu_capacity_bytes: int
    memory_utilization: float
    max_batch_size: int
    max_context_length: int


def kv_cache_bytes(
    model: ModelConfig,
    batch: int,
    context_len: int,
    kv_precision: Precision = Precision.FP16,
) -> int:
    """Calculate the memory footprint in bytes for the KV cache.

    Formula:
        KV cache bytes = 2 * L * n_kv * d * C * B * b_kv

    Args:
        model: Model architecture configuration.
        batch: Number of parallel sequences (B).
        context_len: Sequence context length in tokens (C).
        kv_precision: Precision format for KV cache elements.

    Returns:
        Total KV cache size in bytes.
    """
    if batch <= 0:
        raise ValueError(f"batch must be positive, got {batch}.")
    if context_len <= 0:
        raise ValueError(f"context_len must be positive, got {context_len}.")

    layers = model.layers
    n_kv = model.kv_heads
    d = model.head_dim
    b_kv = kv_precision.bytes_per_param
    return int(2 * layers * n_kv * d * context_len * batch * b_kv)


def ridge_point(
    gpu: GPUConfig,
    precision: Precision = Precision.FP16,
    eff_compute: float | None = None,
    eff_memory: float | None = None,
) -> float:
    """Calculate the roofline ridge point (FLOPs / byte).

    Formula:
        Raw ridge point: peak_FLOPs / HBM_BW
        Effective ridge point: (peak_FLOPs * eff_compute) / (HBM_BW * eff_memory)

    Args:
        gpu: Target GPU configuration.
        precision: Arithmetic precision for peak compute calculation.
        eff_compute: Optional hardware compute efficiency factor (0.0 to 1.0).
        eff_memory: Optional hardware memory bandwidth efficiency factor (0.0 to 1.0).

    Returns:
        Operational intensity threshold in FLOPs/byte.
    """
    peak_flops = precision.get_peak_tflops(gpu) * 1e12
    hbm_bw = gpu.hbm_bandwidth_bytes_per_sec
    if eff_compute is not None and eff_memory is not None:
        return float((peak_flops * eff_compute) / (hbm_bw * eff_memory))
    return float(peak_flops / hbm_bw)


def prefill_estimate(
    model: ModelConfig,
    gpu: GPUConfig,
    batch: int,
    prompt_len: int,
    weight_precision: Precision = Precision.FP16,
    kv_precision: Precision = Precision.FP16,
    eff_compute: float = 0.6,
    eff_memory: float = 0.8,
    parallel: ParallelConfig | None = None,
) -> PhaseEstimate:
    """Estimate compute, memory traffic, latency, and roofline bound for the prefill phase.

    Formulas:
        Prefill FLOPs = 2 * active_params * B * S + 4 * L * B * S^2 * h
        Bytes = weights read once + activations + KV write
        Ring all-reduce comm = 2 * ((N-1)/N) * message_bytes / link_BW + (N-1) * link_latency

    Args:
        model: Model architecture configuration.
        gpu: Target GPU hardware configuration.
        batch: Batch size (B).
        prompt_len: Input prompt length in tokens (S).
        weight_precision: Precision format of model weights.
        kv_precision: Precision format for KV cache storage.
        eff_compute: Compute efficiency factor (default 0.60).
        eff_memory: Memory bandwidth efficiency factor (default 0.80).
        parallel: Parallelism strategy configuration.

    Returns:
        PhaseEstimate containing FLOPs, bytes, latency, communication time, and bound.
    """
    if batch <= 0:
        raise ValueError(f"batch must be positive, got {batch}.")
    if prompt_len <= 0:
        raise ValueError(f"prompt_len must be positive, got {prompt_len}.")
    if not (0.0 < eff_compute <= 1.0):
        raise ValueError(f"eff_compute must be in (0, 1], got {eff_compute}.")
    if not (0.0 < eff_memory <= 1.0):
        raise ValueError(f"eff_memory must be in (0, 1], got {eff_memory}.")

    cfg = parallel if parallel is not None else ParallelConfig()
    b = batch
    s = prompt_len
    l_layers = model.layers
    h = model.hidden
    b_w = weight_precision.bytes_per_param
    b_act = 2.0  # 16-bit activation tensors

    # 1. Total FLOPs across cluster
    gemm_flops = 2.0 * float(model.active_params) * float(b * s)
    attn_flops = 4.0 * float(l_layers) * float(b) * float(s**2) * float(h)
    flops = gemm_flops + attn_flops

    # 2. Total Bytes across cluster
    if model.num_experts <= 1:
        weights_bytes = float(model.active_params) * b_w
    else:
        num_tokens = b * s
        num_e = model.num_experts
        top_k = model.top_k
        prob_not_chosen = 1.0 - (top_k / num_e)
        e_hit = num_e * (1.0 - (prob_not_chosen**num_tokens))
        expert_ffn = 3 * h * model.ffn_dim
        ffn_params = (
            h * num_e
            + int(round(e_hit * expert_ffn))
            + model.shared_experts * expert_ffn
        )
        layer_params = model.attention_params_per_layer + ffn_params
        total_weights_read = model.embedding_params + l_layers * layer_params
        weights_bytes = float(total_weights_read) * b_w

    activations_bytes = 2.0 * float(l_layers) * float(b * s * h) * b_act
    kv_write_bytes = float(
        kv_cache_bytes(model, batch=b, context_len=s, kv_precision=kv_precision)
    )
    total_bytes = weights_bytes + activations_bytes + kv_write_bytes

    # 3. Per-device execution time
    # Sharding across Tensor Parallel (tp) and Pipeline Parallel (pp)
    tp_degree = float(cfg.tp)
    pp_degree = float(cfg.pp)
    shard_factor = tp_degree * pp_degree

    flops_per_gpu = flops / shard_factor
    bytes_per_gpu = total_bytes / shard_factor

    peak_flops = weight_precision.get_peak_tflops(gpu) * 1e12
    hbm_bw = gpu.hbm_bandwidth_bytes_per_sec

    compute_time = flops_per_gpu / (peak_flops * eff_compute)
    memory_time = bytes_per_gpu / (hbm_bw * eff_memory)

    # Pipeline bubble factor (default 4 microbatches per stage)
    if cfg.pp > 1:
        num_microbatches = 4 * cfg.pp
        bubble_fraction = (cfg.pp - 1) / (cfg.pp - 1 + num_microbatches)
        pipeline_mult = cfg.pp * (1.0 + bubble_fraction)
        compute_time *= pipeline_mult
        memory_time *= pipeline_mult

    exec_time = max(compute_time, memory_time)

    # 4. Communication Latency
    link_bw = float(gpu.interconnect_bandwidth_gbps * 1e9)
    link_lat = float(gpu.interconnect_latency_us * 1e-6)

    # TP: 2 All-Reduces per layer (attention and FFN)
    # Message bytes: B * S * h * b_act
    tp_msg_bytes = float(b * s * h * b_act)
    t_allreduce = ring_allreduce_time(tp_msg_bytes, cfg.tp, link_bw, link_lat)
    layers_per_stage = l_layers / pp_degree
    tp_comm_time = 2.0 * layers_per_stage * t_allreduce

    # EP: All-to-all for MoE dispatch and combine
    ep_comm_time = 0.0
    if model.is_moe and cfg.ep > 1:
        ep_msg_bytes = float(b * s * model.top_k * h * b_act)
        ep_comm_time = layers_per_stage * all_to_all_time(
            ep_msg_bytes, cfg.ep, link_bw, link_lat
        )

    comm_time = tp_comm_time + ep_comm_time
    total_latency = exec_time + comm_time
    comm_share = comm_time / total_latency if total_latency > 0 else 0.0

    arithmetic_intensity = flops / total_bytes if total_bytes > 0 else 0.0
    bound = "compute" if compute_time >= memory_time else "memory"

    return PhaseEstimate(
        flops=flops,
        bytes=total_bytes,
        arithmetic_intensity=arithmetic_intensity,
        compute_time=compute_time,
        memory_time=memory_time,
        latency=total_latency,
        bound=bound,
        comm_time=comm_time,
        comm_share=comm_share,
    )


def decode_estimate(
    model: ModelConfig,
    gpu: GPUConfig,
    batch: int,
    context_len: int,
    weight_precision: Precision = Precision.FP16,
    kv_precision: Precision = Precision.FP16,
    eff_compute: float = 0.6,
    eff_memory: float = 0.8,
    parallel: ParallelConfig | None = None,
) -> PhaseEstimate:
    """Estimate compute, memory traffic, latency, and roofline bound for one decode token step.

    Formulas:
        Decode FLOPs = 2 * active_params * B + 4 * L * B * C * h
        Bytes = weights read once per step + KV cache read
        MoE distinct experts hit = E * (1 - (1 - top_k / E)^B)

    Args:
        model: Model architecture configuration.
        gpu: Target GPU hardware configuration.
        batch: Batch size (B).
        context_len: Current sequence context length in tokens (C).
        weight_precision: Precision format of model weights.
        kv_precision: Precision format for KV cache storage.
        eff_compute: Compute efficiency factor (default 0.60).
        eff_memory: Memory bandwidth efficiency factor (default 0.80).
        parallel: Parallelism strategy configuration.

    Returns:
        PhaseEstimate containing single-step metrics, communication, and bound.
    """
    if batch <= 0:
        raise ValueError(f"batch must be positive, got {batch}.")
    if context_len <= 0:
        raise ValueError(f"context_len must be positive, got {context_len}.")
    if not (0.0 < eff_compute <= 1.0):
        raise ValueError(f"eff_compute must be in (0, 1], got {eff_compute}.")
    if not (0.0 < eff_memory <= 1.0):
        raise ValueError(f"eff_memory must be in (0, 1], got {eff_memory}.")

    cfg = parallel if parallel is not None else ParallelConfig()
    b = batch
    c = context_len
    l_layers = model.layers
    h = model.hidden
    b_w = weight_precision.bytes_per_param
    b_act = 2.0

    # 1. Total FLOPs across cluster
    gemm_flops = 2.0 * float(model.active_params) * float(b)
    attn_flops = 4.0 * float(l_layers) * float(b) * float(c) * float(h)
    flops = gemm_flops + attn_flops

    # 2. Total Bytes across cluster
    if model.num_experts <= 1:
        weights_bytes = float(model.active_params) * b_w
    else:
        num_e = model.num_experts
        top_k = model.top_k
        prob_not_chosen = 1.0 - (top_k / num_e)
        e_hit = num_e * (1.0 - (prob_not_chosen**b))
        expert_ffn = 3 * h * model.ffn_dim
        ffn_params = (
            h * num_e
            + (e_hit * expert_ffn)
            + model.shared_experts * expert_ffn
        )
        layer_params = model.attention_params_per_layer + ffn_params
        total_weights_read = model.embedding_params + l_layers * layer_params
        weights_bytes = float(total_weights_read) * b_w

    kv_read_bytes = float(
        kv_cache_bytes(model, batch=b, context_len=c, kv_precision=kv_precision)
    )
    total_bytes = weights_bytes + kv_read_bytes

    # 3. Per-device execution time
    tp_degree = float(cfg.tp)
    pp_degree = float(cfg.pp)
    shard_factor = tp_degree * pp_degree

    flops_per_gpu = flops / shard_factor
    bytes_per_gpu = total_bytes / shard_factor

    peak_flops = weight_precision.get_peak_tflops(gpu) * 1e12
    hbm_bw = gpu.hbm_bandwidth_bytes_per_sec

    compute_time = flops_per_gpu / (peak_flops * eff_compute)
    memory_time = bytes_per_gpu / (hbm_bw * eff_memory)

    if cfg.pp > 1:
        num_microbatches = 4 * cfg.pp
        bubble_fraction = (cfg.pp - 1) / (cfg.pp - 1 + num_microbatches)
        pipeline_mult = cfg.pp * (1.0 + bubble_fraction)
        compute_time *= pipeline_mult
        memory_time *= pipeline_mult

    exec_time = max(compute_time, memory_time)

    # 4. Communication Latency
    link_bw = float(gpu.interconnect_bandwidth_gbps * 1e9)
    link_lat = float(gpu.interconnect_latency_us * 1e-6)

    # TP: 2 All-Reduces per layer (message = B * 1 * h * b_act)
    tp_msg_bytes = float(b * 1 * h * b_act)
    t_allreduce = ring_allreduce_time(tp_msg_bytes, cfg.tp, link_bw, link_lat)
    layers_per_stage = l_layers / pp_degree
    tp_comm_time = 2.0 * layers_per_stage * t_allreduce

    # EP: All-to-all for MoE
    ep_comm_time = 0.0
    if model.is_moe and cfg.ep > 1:
        ep_msg_bytes = float(b * 1 * model.top_k * h * b_act)
        ep_comm_time = layers_per_stage * all_to_all_time(
            ep_msg_bytes, cfg.ep, link_bw, link_lat
        )

    comm_time = tp_comm_time + ep_comm_time
    total_latency = exec_time + comm_time
    comm_share = comm_time / total_latency if total_latency > 0 else 0.0

    arithmetic_intensity = flops / total_bytes if total_bytes > 0 else 0.0
    bound = "compute" if compute_time >= memory_time else "memory"

    return PhaseEstimate(
        flops=flops,
        bytes=total_bytes,
        arithmetic_intensity=arithmetic_intensity,
        compute_time=compute_time,
        memory_time=memory_time,
        latency=total_latency,
        bound=bound,
        comm_time=comm_time,
        comm_share=comm_share,
    )


def memory_fit_check(
    model: ModelConfig,
    gpu: GPUConfig,
    batch: int,
    context_len: int,
    weight_precision: Precision = Precision.FP16,
    kv_precision: Precision = Precision.FP16,
    tp: int = 1,
    parallel: ParallelConfig | None = None,
) -> MemoryFitResult:
    """Assess whether model weights and KV cache fit in GPU memory and calculate headroom limits.

    Args:
        model: Model architecture configuration.
        gpu: Target GPU hardware configuration.
        batch: Target batch size (B).
        context_len: Target context length in tokens (C).
        weight_precision: Precision format for weights.
        kv_precision: Precision format for KV cache.
        tp: Tensor parallel degree sharding memory (overridden if parallel specified).
        parallel: Optional ParallelConfig containing tp and pp degrees.

    Returns:
        MemoryFitResult containing fit flag, utilization, and max batch/context capacity.
    """
    tp_eff = parallel.tp if parallel is not None else tp
    pp_eff = parallel.pp if parallel is not None else 1

    if tp_eff <= 0:
        raise ValueError(f"tp must be positive, got {tp_eff}.")
    if pp_eff <= 0:
        raise ValueError(f"pp must be positive, got {pp_eff}.")

    shard_factor = tp_eff * pp_eff

    weight_bytes_total = int(model.total_params * weight_precision.bytes_per_param)
    weight_bytes_per_gpu = int(math.ceil(weight_bytes_total / shard_factor))

    kv_bytes_total = kv_cache_bytes(
        model, batch=batch, context_len=context_len, kv_precision=kv_precision
    )
    kv_bytes_per_gpu = int(math.ceil(kv_bytes_total / shard_factor))

    total_required = weight_bytes_per_gpu + kv_bytes_per_gpu
    capacity = gpu.memory_capacity_bytes
    fits = total_required <= capacity
    utilization = total_required / capacity if capacity > 0 else float("inf")

    # Available memory remaining for KV cache
    available_for_kv = max(0, capacity - weight_bytes_per_gpu)

    # Per-sequence KV cache bytes on this GPU
    per_seq_kv_gpu = (
        2
        * model.layers
        * model.kv_heads
        * model.head_dim
        * context_len
        * kv_precision.bytes_per_param
    ) / shard_factor
    max_batch = int(available_for_kv // per_seq_kv_gpu) if per_seq_kv_gpu > 0 else 0

    # Per-token KV cache bytes across entire batch on this GPU
    per_token_kv_gpu = (
        2
        * model.layers
        * model.kv_heads
        * model.head_dim
        * batch
        * kv_precision.bytes_per_param
    ) / shard_factor
    max_context = int(available_for_kv // per_token_kv_gpu) if per_token_kv_gpu > 0 else 0

    return MemoryFitResult(
        fits=fits,
        total_memory_required_bytes=total_required,
        weight_memory_bytes=weight_bytes_per_gpu,
        kv_cache_bytes=kv_bytes_per_gpu,
        gpu_capacity_bytes=capacity,
        memory_utilization=utilization,
        max_batch_size=max_batch,
        max_context_length=max_context,
    )
