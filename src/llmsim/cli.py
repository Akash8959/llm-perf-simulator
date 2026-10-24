"""Command-line interface for the LLM inference performance simulator."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from llmsim.hardware import load_gpu
from llmsim.model import load_model
from llmsim.parallelism import ParallelConfig
from llmsim.plots import (
    plot_latency_vs_batch,
    plot_roofline,
    plot_throughput_vs_tp,
    plot_tpot_vs_hbm,
)
from llmsim.quantization import Precision
from llmsim.recommend import generate_recommendation
from llmsim.simulator import simulate
from llmsim.sweep import (
    sweep_batch_size,
    sweep_hbm_bandwidth,
    sweep_quantization,
    sweep_sequence_length,
    sweep_sm_count,
    sweep_tp_interconnect,
)


def build_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="llmsim",
        description="Analytical, roofline-style LLM inference performance simulator.",
    )
    parser.add_argument(
        "--hardware",
        type=str,
        required=True,
        help="Path to hardware YAML configuration file (e.g. configs/hardware/h100_sxm.yaml).",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model YAML configuration file (e.g. configs/models/llama3_8b.yaml).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Batch size per replica (default: 1).",
    )
    parser.add_argument(
        "--prompt-len",
        type=int,
        default=2048,
        help="Prompt length in tokens (default: 2048).",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=128,
        help="Number of tokens to generate per sequence (default: 128).",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Tensor parallel degree (default: 1).",
    )
    parser.add_argument(
        "--pp",
        type=int,
        default=1,
        help="Pipeline parallel degree (default: 1).",
    )
    parser.add_argument(
        "--dp",
        type=int,
        default=1,
        help="Data parallel degree (default: 1).",
    )
    parser.add_argument(
        "--ep",
        type=int,
        default=1,
        help="Expert parallel degree (default: 1).",
    )
    parser.add_argument(
        "--quant",
        type=str,
        default="fp16",
        choices=["fp16", "bf16", "fp8", "int8", "int4"],
        help="Weight precision format (default: fp16).",
    )
    parser.add_argument(
        "--kv-quant",
        type=str,
        default="fp16",
        choices=["fp16", "bf16", "fp8", "int8", "int4"],
        help="KV cache precision format (default: fp16).",
    )
    parser.add_argument(
        "--sweep",
        type=str,
        choices=["hbm", "sm", "batch", "seq", "quant", "tp"],
        help="Run a parametric sweep across the designated dimension.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate and save visualization plots to output directory.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results",
        help="Output directory for generated CSVs and plots (default: results).",
    )
    return parser


def format_table(rows: list[tuple[str, str, str]]) -> str:
    """Format tabular rows into a structured ASCII table."""
    col1_w = max(len(r[0]) for r in rows) + 2
    col2_w = max(len(r[1]) for r in rows) + 2
    col3_w = max(len(r[2]) for r in rows) + 2

    header = f"{'Metric':<{col1_w}} | {'Value':<{col2_w}} | {'Unit / Note':<{col3_w}}"
    sep = f"{'-' * col1_w}-+-{'-' * col2_w}-+-{'-' * col3_w}"
    lines = [header, sep]
    for m, v, u in rows:
        lines.append(f"{m:<{col1_w}} | {v:<{col2_w}} | {u:<{col3_w}}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Execute main CLI application entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gpu = load_gpu(args.hardware)
    model = load_model(args.model)
    parallel = ParallelConfig(tp=args.tp, pp=args.pp, dp=args.dp, ep=args.ep)
    weight_precision = Precision.from_str(args.quant)
    kv_precision = Precision.from_str(args.kv_quant)

    print("=" * 78)
    print(f"LLM PERFORMANCE SIMULATOR: {model.name.upper()} ON {gpu.name.upper()}")
    print("=" * 78)

    if args.sweep:
        print(f"Running sweep: {args.sweep.upper()}...")
        if args.sweep == "hbm":
            df = sweep_hbm_bandwidth(
                gpu, model, batch=args.batch, prompt_len=args.prompt_len,
                output_len=args.output_len, parallel=parallel,
                weight_precision=weight_precision, kv_precision=kv_precision
            )
            df.to_csv(out_dir / "sweep_hbm.csv", index=False)
            if args.plot:
                p_path = plot_tpot_vs_hbm(df, out_path=out_dir / "tpot_vs_hbm.png")
                print(f"Plot saved: {p_path}")
        elif args.sweep == "sm":
            df = sweep_sm_count(
                gpu, model, batch=args.batch, prompt_len=args.prompt_len,
                output_len=args.output_len, parallel=parallel,
                weight_precision=weight_precision, kv_precision=kv_precision
            )
            df.to_csv(out_dir / "sweep_sm.csv", index=False)
        elif args.sweep == "batch":
            df = sweep_batch_size(
                gpu, model, prompt_len=args.prompt_len,
                output_len=args.output_len, parallel=parallel,
                weight_precision=weight_precision, kv_precision=kv_precision
            )
            df.to_csv(out_dir / "sweep_batch.csv", index=False)
            if args.plot:
                p_path = plot_latency_vs_batch(df, out_path=out_dir / "latency_vs_batch.png")
                print(f"Plot saved: {p_path}")
        elif args.sweep == "seq":
            df = sweep_sequence_length(
                gpu, model, batch=args.batch,
                output_len=args.output_len, parallel=parallel,
                weight_precision=weight_precision, kv_precision=kv_precision
            )
            df.to_csv(out_dir / "sweep_seq.csv", index=False)
        elif args.sweep == "quant":
            df = sweep_quantization(
                gpu, model, batch=args.batch, prompt_len=args.prompt_len,
                output_len=args.output_len, parallel=parallel
            )
            df.to_csv(out_dir / "sweep_quant.csv", index=False)
        elif args.sweep == "tp":
            df = sweep_tp_interconnect(
                gpu, model, batch=args.batch, prompt_len=args.prompt_len,
                output_len=args.output_len,
                weight_precision=weight_precision, kv_precision=kv_precision
            )
            df.to_csv(out_dir / "sweep_tp.csv", index=False)
            if args.plot:
                p_path = plot_throughput_vs_tp(df, out_path=out_dir / "throughput_vs_tp.png")
                print(f"Plot saved: {p_path}")

        print("\nSweep Results Table:")
        print(df.to_string(index=False))

        rec = generate_recommendation(df)
        print("\n" + "=" * 78)
        print("BOTTLENECK RECOMMENDATION:")
        print("=" * 78)
        print(rec)
        print("=" * 78)
        return 0

    # Single configuration simulation
    res = simulate(
        gpu=gpu,
        model=model,
        batch=args.batch,
        prompt_len=args.prompt_len,
        output_len=args.output_len,
        parallel=parallel,
        weight_precision=weight_precision,
        kv_precision=kv_precision,
    )

    rows = [
        ("Time to First Token (TTFT)", f"{res.ttft * 1e3:.2f}", "ms"),
        ("Time Per Output Token (TPOT)", f"{res.tpot * 1e3:.2f}", "ms / token"),
        ("End-to-End Latency", f"{res.e2e_latency:.3f}", "seconds"),
        ("Generation Throughput", f"{res.tokens_per_sec:.1f}", "tokens / sec"),
        ("Peak Memory Per GPU", f"{res.memory_per_gpu_gb:.2f}", "GB"),
        ("Prefill Bottleneck Regime", res.bound_prefill.upper(), "compute / memory"),
        ("Decode Bottleneck Regime", res.bound_decode.upper(), "compute / memory"),
        ("Inter-GPU Comm Overhead", f"{res.comm_share * 100:.1f}%", "of e2e latency"),
        ("Memory Headroom Utilization", f"{res.memory_utilization * 100:.1f}%", "of capacity"),
        ("Out-of-Memory (OOM) Status", "TRIGGERED" if res.is_oom else "FIT", "Status"),
    ]

    print(format_table(rows))

    if args.plot:
        r_path = plot_roofline(
            gpu, model, batch=args.batch, prompt_len=args.prompt_len,
            out_path=out_dir / "roofline_chart.png"
        )
        print(f"\nRoofline plot generated: {r_path}")

    # Generate recommendation
    df_sample = sweep_batch_size(
        gpu, model, prompt_len=args.prompt_len, output_len=args.output_len
    )
    rec = generate_recommendation({"batch": df_sample})
    print("\n" + "=" * 78)
    print("BOTTLENECK RECOMMENDATION:")
    print("=" * 78)
    print(rec)
    print("=" * 78)

    return 0


if __name__ == "__main__":
    sys.exit(main())
