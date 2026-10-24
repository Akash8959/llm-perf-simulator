"""Matplotlib visualization generators for LLM roofline and sweep analysis."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from llmsim.estimator import decode_estimate, prefill_estimate, ridge_point
from llmsim.quantization import Precision

matplotlib.use("Agg")

if TYPE_CHECKING:
    from llmsim.hardware import GPUConfig
    from llmsim.model import ModelConfig


def plot_roofline(
    gpu: GPUConfig,
    model: ModelConfig,
    batch: int = 1,
    prompt_len: int = 2048,
    out_path: str | Path = "results/roofline_chart.png",
    eff_compute: float = 0.6,
    eff_memory: float = 0.8,
) -> Path:
    """Generate a log-log Roofline chart showing prefill and decode operational operating points.

    Args:
        gpu: GPU hardware configuration.
        model: Model architecture configuration.
        batch: Batch size for evaluated points.
        prompt_len: Prompt length in tokens.
        out_path: Destination path for saved PNG image.
        eff_compute: Compute hardware efficiency factor.
        eff_memory: Memory bandwidth efficiency factor.

    Returns:
        Path to the saved PNG image.
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pref = prefill_estimate(
        model, gpu, batch=batch, prompt_len=prompt_len,
        eff_compute=eff_compute, eff_memory=eff_memory
    )
    dec = decode_estimate(
        model, gpu, batch=batch, context_len=prompt_len,
        eff_compute=eff_compute, eff_memory=eff_memory
    )

    peak_compute_tflops = gpu.peak_tflops_fp16 * eff_compute
    bw_tb_s = (gpu.hbm_bandwidth_gbps * eff_memory) / 1000.0  # TB/s
    rp = ridge_point(gpu, Precision.FP16, eff_compute=eff_compute, eff_memory=eff_memory)

    # Operational intensities for plotting roofline curve
    intensities = np.logspace(-1, 4, 500)
    # Attainable performance: min(Peak Compute, Intensity * Bandwidth)
    attainable = np.minimum(peak_compute_tflops, intensities * bw_tb_s)

    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    ax.plot(
        intensities, attainable, color="#1f77b4", linewidth=2.5,
        label="Effective Roofline Ceiling"
    )
    ax.axvline(
        x=rp, color="gray", linestyle="--", alpha=0.7,
        label=f"Ridge Point ({rp:.1f} FLOP/B)"
    )

    # Attained TFLOPs: FLOPs / latency / 1e12
    pref_attained = (pref.flops / pref.latency) / 1e12
    dec_attained = (dec.flops / dec.latency) / 1e12

    ax.scatter(
        [pref.arithmetic_intensity], [pref_attained],
        color="#d62728", s=120, zorder=5, label=f"Prefill (S={prompt_len}, B={batch})"
    )
    pref_annot = (
        f" Prefill ({pref.bound.upper()})\n"
        f" I={pref.arithmetic_intensity:.1f} FLOP/B\n"
        f" {pref_attained:.1f} TFLOPs"
    )
    ax.annotate(
        pref_annot,
        xy=(pref.arithmetic_intensity, pref_attained),
        xytext=(pref.arithmetic_intensity * 0.4, pref_attained * 0.5),
        arrowprops=dict(facecolor="#d62728", arrowstyle="->", lw=1.2),
        fontsize=9, weight="bold", color="#d62728"
    )

    ax.scatter(
        [dec.arithmetic_intensity], [dec_attained],
        color="#2ca02c", s=120, zorder=5, label=f"Decode (C={prompt_len}, B={batch})"
    )
    dec_annot = (
        f" Decode ({dec.bound.upper()})\n"
        f" I={dec.arithmetic_intensity:.2f} FLOP/B\n"
        f" {dec_attained:.2f} TFLOPs"
    )
    ax.annotate(
        dec_annot,
        xy=(dec.arithmetic_intensity, dec_attained),
        xytext=(dec.arithmetic_intensity * 1.5, dec_attained * 0.8),
        arrowprops=dict(facecolor="#2ca02c", arrowstyle="->", lw=1.2),
        fontsize=9, weight="bold", color="#2ca02c"
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.1, 10000)
    ax.set_ylim(0.01, peak_compute_tflops * 2.0)
    ax.set_xlabel("Arithmetic Intensity (FLOPs / Byte)", fontsize=11, weight="bold")
    ax.set_ylabel("Attainable Performance (TFLOPs)", fontsize=11, weight="bold")
    ax.set_title(f"LLM Inference Roofline: {model.name} on {gpu.name}", fontsize=13, weight="bold")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(loc="lower right", frameon=True, fontsize=9)

    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_latency_vs_batch(
    df_batch: pd.DataFrame,
    out_path: str | Path = "results/latency_vs_batch.png",
) -> Path:
    """Plot Time to First Token (TTFT) and Time Per Output Token (TPOT) vs batch size.

    Args:
        df_batch: DataFrame produced by sweep_batch_size.
        out_path: Destination path for saved PNG image.

    Returns:
        Path to the saved PNG image.
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=150)
    color1 = "#1f77b4"
    color2 = "#ff7f0e"

    ax1.set_xlabel("Batch Size", fontsize=11, weight="bold")
    ax1.set_ylabel("Prefill TTFT (ms)", color=color1, fontsize=11, weight="bold")
    line1 = ax1.plot(
        df_batch["batch_size"], df_batch["ttft_ms"],
        marker="o", color=color1, linewidth=2, label="TTFT (ms)"
    )
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Decode TPOT (ms)", color=color2, fontsize=11, weight="bold")
    line2 = ax2.plot(
        df_batch["batch_size"], df_batch["tpot_ms"],
        marker="s", color=color2, linewidth=2, linestyle="--", label="TPOT (ms)"
    )
    ax2.tick_params(axis="y", labelcolor=color2)

    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper left", frameon=True)
    plt.title("Inference Latency vs Batch Size", fontsize=13, weight="bold")

    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_throughput_vs_tp(
    df_tp: pd.DataFrame,
    out_path: str | Path = "results/throughput_vs_tp.png",
) -> Path:
    """Plot aggregate throughput vs Tensor Parallel degree across interconnect bandwidths.

    Args:
        df_tp: DataFrame produced by sweep_tp_interconnect.
        out_path: Destination path for saved PNG image.

    Returns:
        Path to the saved PNG image.
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    for bw in sorted(df_tp["interconnect_bandwidth_gbps"].unique()):
        subset = df_tp[df_tp["interconnect_bandwidth_gbps"] == bw].sort_values("tp_degree")
        ax.plot(
            subset["tp_degree"], subset["tokens_per_sec"],
            marker="o", linewidth=2, label=f"Link BW {bw:.0f} GB/s"
        )

    ax.set_xlabel("Tensor Parallel Degree (TP)", fontsize=11, weight="bold")
    ax.set_ylabel("Throughput (Tokens / Sec)", fontsize=11, weight="bold")
    ax.set_title("Throughput vs TP Degree & Interconnect Bandwidth", fontsize=13, weight="bold")
    ax.set_xticks([1, 2, 4, 8])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(title="Interconnect", frameon=True)

    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_tpot_vs_hbm(
    df_hbm: pd.DataFrame,
    out_path: str | Path = "results/tpot_vs_hbm.png",
) -> Path:
    """Plot Time Per Output Token (TPOT) vs HBM Bandwidth.

    Args:
        df_hbm: DataFrame produced by sweep_hbm_bandwidth.
        out_path: Destination path for saved PNG image.

    Returns:
        Path to the saved PNG image.
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    ax.plot(
        df_hbm["hbm_bandwidth_gbps"], df_hbm["tpot_ms"],
        marker="o", color="#2ca02c", linewidth=2.5, label="Decode TPOT (ms)"
    )

    ax.set_xlabel("HBM Bandwidth (GB/s)", fontsize=11, weight="bold")
    ax.set_ylabel("Time Per Output Token (ms)", fontsize=11, weight="bold")
    ax.set_title("Decode TPOT Sensitivity to HBM Bandwidth", fontsize=13, weight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_bottleneck_summary_table(
    summary_df: pd.DataFrame,
    out_path: str | Path = "results/bottleneck_summary.png",
) -> Path:
    """Render a styled tabular summary image of inference bottlenecks.

    Args:
        summary_df: Formatted summary DataFrame containing metrics to present.
        out_path: Destination path for saved PNG image.

    Returns:
        Path to the saved PNG image.
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, max(2.5, len(summary_df) * 0.45 + 1.2)), dpi=180)
    ax.axis("off")

    table_data = [summary_df.columns.tolist()] + summary_df.values.tolist()
    table = ax.table(
        cellText=table_data,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.4)

    # Style header row
    for col_idx in range(len(summary_df.columns)):
        cell = table[0, col_idx]
        cell.set_facecolor("#2b5b84")
        cell.set_text_props(color="white", weight="bold")

    # Alternate row colors
    for row_idx in range(1, len(table_data)):
        bg_color = "#f2f4f8" if row_idx % 2 == 0 else "#ffffff"
        for col_idx in range(len(summary_df.columns)):
            table[row_idx, col_idx].set_facecolor(bg_color)

    plt.title("LLM Inference Performance & Bottleneck Summary", fontsize=12, weight="bold", pad=12)
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
