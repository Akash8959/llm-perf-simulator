"""Fully worked analytical roofline example: LLaMA-3-8B on NVIDIA H100 SXM5.

Scenario:
- Model: LLaMA-3-8B (FP16 weights)
- Hardware: NVIDIA H100 SXM5 (80GB HBM3, 3350 GB/s, 989 TFLOPs FP16)
- Batch Size: 1
- Context Length: 2048
"""

from pathlib import Path

from llmsim.estimator import (
    decode_estimate,
    kv_cache_bytes,
    memory_fit_check,
    prefill_estimate,
    ridge_point,
)
from llmsim.hardware import load_gpu
from llmsim.model import load_model
from llmsim.quantization import Precision

ROOT = Path(__file__).resolve().parent.parent


def run_worked_example() -> None:
    """Execute and print detailed breakdown of the worked example."""
    gpu = load_gpu(ROOT / "configs" / "hardware" / "h100_sxm.yaml")
    model = load_model(ROOT / "configs" / "models" / "llama3_8b.yaml")

    batch = 1
    context_len = 2048
    eff_compute = 0.60
    eff_memory = 0.80

    print("=" * 80)
    print("WORKED EXAMPLE: LLaMA-3-8B on NVIDIA H100 SXM5")
    print("=" * 80)
    print(f"Model: {model.name}")
    print(f"  Layers (L): {model.layers}")
    print(f"  Hidden size (h): {model.hidden}")
    print(f"  Query heads (n_h): {model.heads}")
    print(f"  KV heads (n_kv): {model.kv_heads}")
    print(f"  Head dimension (d = h / n_h): {model.head_dim}")
    print(f"  FFN dimension (d_ffn): {model.ffn_dim}")
    print(f"  Vocab size (V): {model.vocab}")
    print(f"  Attention params / layer: {model.attention_params_per_layer:,}")
    print(f"  FFN params / layer: {model.ffn_params_per_layer:,}")
    print(f"  Total params: {model.total_params:,} ({model.total_params / 1e9:.3f} B)")
    print(f"  Active params: {model.active_params:,} ({model.active_params / 1e9:.3f} B)")
    print()

    print(f"Hardware: {gpu.name}")
    print(f"  Peak FP16 TFLOPs: {gpu.peak_tflops_fp16} TFLOPs")
    print(f"  HBM Bandwidth: {gpu.hbm_bandwidth_gbps} GB/s")
    print(f"  Memory Capacity: {gpu.memory_capacity_gb} GB")
    print(f"  Operational Compute Efficiency: {eff_compute * 100:.1f}%")
    print(f"  Operational Memory Efficiency: {eff_memory * 100:.1f}%")
    raw_rp = ridge_point(gpu, Precision.FP16)
    eff_rp = ridge_point(gpu, Precision.FP16, eff_compute=eff_compute, eff_memory=eff_memory)
    print(f"  Raw Ridge Point: {raw_rp:.2f} FLOP/byte")
    print(f"  Effective Ridge Point: {eff_rp:.2f} FLOP/byte")
    print()

    print(f"Execution Parameters: Batch B = {batch}, Context C = {context_len}")
    print("-" * 80)

    # KV Cache
    kv_bytes = kv_cache_bytes(
        model, batch=batch, context_len=context_len, kv_precision=Precision.FP16
    )
    print("1. KV Cache Footprint:")
    print("   Formula: 2 * L * n_kv * d * C * B * b_kv")
    print(
        f"   Calculation: 2 * {model.layers} * {model.kv_heads} * {model.head_dim} "
        f"* {context_len} * {batch} * 2"
    )
    print(f"   KV Cache Bytes: {kv_bytes:,} bytes ({kv_bytes / (1024**2):.2f} MB)")
    print()

    # Memory Fit
    mem_fit = memory_fit_check(model, gpu, batch=batch, context_len=context_len, tp=1)
    print("2. Memory Fit Check (TP = 1):")
    weight_gb = mem_fit.weight_memory_bytes / (1024**3)
    kv_mb = mem_fit.kv_cache_bytes / (1024**2)
    tot_gb = mem_fit.total_memory_required_bytes / (1024**3)
    cap_gb = mem_fit.gpu_capacity_bytes / (1024**3)
    print(f"   Model Weights Memory: {mem_fit.weight_memory_bytes:,} bytes ({weight_gb:.2f} GB)")
    print(f"   KV Cache Memory: {mem_fit.kv_cache_bytes:,} bytes ({kv_mb:.2f} MB)")
    print(
        f"   Total Required Memory: {mem_fit.total_memory_required_bytes:,} bytes "
        f"({tot_gb:.2f} GB)"
    )
    print(f"   GPU Capacity: {mem_fit.gpu_capacity_bytes:,} bytes ({cap_gb:.2f} GB)")
    print(f"   Memory Utilization: {mem_fit.memory_utilization * 100:.2f}% (Fits: {mem_fit.fits})")
    print(f"   Max Batch Size at Context {context_len}: {mem_fit.max_batch_size}")
    print(f"   Max Context Length at Batch {batch}: {mem_fit.max_context_length}")
    print()

    # Prefill Phase (prompt_len = 2048)
    prefill = prefill_estimate(
        model,
        gpu,
        batch=batch,
        prompt_len=context_len,
        weight_precision=Precision.FP16,
        kv_precision=Precision.FP16,
        eff_compute=eff_compute,
        eff_memory=eff_memory,
    )
    print("3. Prefill Phase (Prompt Length S = 2048, Batch B = 1):")
    print(f"   FLOPs: {prefill.flops:,.0f} ({prefill.flops / 1e12:.3f} TFLOPs)")
    print(f"   Bytes Transferred: {prefill.bytes:,.0f} bytes ({prefill.bytes / (1024**3):.2f} GB)")
    print(f"   Arithmetic Intensity: {prefill.arithmetic_intensity:.2f} FLOP/byte")
    print(f"   Compute Time: {prefill.compute_time * 1e3:.3f} ms")
    print(f"   Memory Time: {prefill.memory_time * 1e3:.3f} ms")
    print(f"   Prefill Latency (TTFT): {prefill.latency * 1e3:.3f} ms")
    print(f"   Bottleneck Classification: {prefill.bound.upper()}-BOUND")
    print()

    # Decode Phase (at context_len = 2048)
    decode = decode_estimate(
        model,
        gpu,
        batch=batch,
        context_len=context_len,
        weight_precision=Precision.FP16,
        kv_precision=Precision.FP16,
        eff_compute=eff_compute,
        eff_memory=eff_memory,
    )
    print("4. Decode Step Phase (Generating 1 token at Context C = 2048, Batch B = 1):")
    print(f"   FLOPs: {decode.flops:,.0f} ({decode.flops / 1e9:.3f} GFLOPs)")
    print(f"   Bytes Transferred: {decode.bytes:,.0f} bytes ({decode.bytes / (1024**3):.2f} GB)")
    print(f"   Arithmetic Intensity: {decode.arithmetic_intensity:.2f} FLOP/byte")
    print(f"   Compute Time: {decode.compute_time * 1e3:.3f} ms")
    print(f"   Memory Time: {decode.memory_time * 1e3:.3f} ms")
    print(f"   Decode Step Latency (TPOT): {decode.latency * 1e3:.3f} ms")
    print(f"   Bottleneck Classification: {decode.bound.upper()}-BOUND")
    print("=" * 80)


if __name__ == "__main__":
    run_worked_example()
