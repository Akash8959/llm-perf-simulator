"""Run all parametric performance sweeps and export CSV data and plots for LLaMA-3-8B on H100."""

from pathlib import Path

import pandas as pd

from llmsim.hardware import load_gpu
from llmsim.model import load_model
from llmsim.plots import (
    plot_bottleneck_summary_table,
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

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"


def run_all() -> None:
    """Execute all sweeps, generate plots, and print recommendation."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    gpu = load_gpu(ROOT / "configs" / "hardware" / "h100_sxm.yaml")
    model = load_model(ROOT / "configs" / "models" / "llama3_8b.yaml")

    print(f"Executing complete sweep suite for {model.name} on {gpu.name}...")

    # 1. HBM Bandwidth Sweep
    print("-> Sweeping HBM Bandwidth...")
    df_hbm = sweep_hbm_bandwidth(gpu, model, batch=1, prompt_len=2048, output_len=128)
    df_hbm.to_csv(RESULTS_DIR / "sweep_hbm.csv", index=False)

    # 2. SM Count Sweep
    print("-> Sweeping SM Count & Peak TFLOPs...")
    df_sm = sweep_sm_count(gpu, model, batch=16, prompt_len=2048, output_len=128)
    df_sm.to_csv(RESULTS_DIR / "sweep_sm.csv", index=False)

    # 3. Batch Size Sweep
    print("-> Sweeping Batch Sizes...")
    df_batch = sweep_batch_size(gpu, model, prompt_len=1024, output_len=128)
    df_batch.to_csv(RESULTS_DIR / "sweep_batch.csv", index=False)

    # 4. Sequence Length Sweep
    print("-> Sweeping Sequence Lengths...")
    df_seq = sweep_sequence_length(gpu, model, batch=4, output_len=128)
    df_seq.to_csv(RESULTS_DIR / "sweep_seq.csv", index=False)

    # 5. Quantization Sweep
    print("-> Sweeping Quantization Precision...")
    df_quant = sweep_quantization(gpu, model, batch=4, prompt_len=1024, output_len=128)
    df_quant.to_csv(RESULTS_DIR / "sweep_quant.csv", index=False)

    # 6. Tensor Parallel x Interconnect Sweep
    print("-> Sweeping TP Degree x Interconnect Bandwidth...")
    df_tp = sweep_tp_interconnect(gpu, model, batch=8, prompt_len=1024, output_len=128)
    df_tp.to_csv(RESULTS_DIR / "sweep_tp.csv", index=False)

    # 7. Generate Visualizations
    print("-> Generating roofline and sweep plots in results/...")
    plot_roofline(
        gpu, model, batch=1, prompt_len=2048, out_path=RESULTS_DIR / "roofline_chart.png"
    )
    plot_latency_vs_batch(df_batch, out_path=RESULTS_DIR / "latency_vs_batch.png")
    plot_throughput_vs_tp(df_tp, out_path=RESULTS_DIR / "throughput_vs_tp.png")
    plot_tpot_vs_hbm(df_hbm, out_path=RESULTS_DIR / "tpot_vs_hbm.png")

    # Summary table graphic
    hbm_3350_row = df_hbm[df_hbm["hbm_bandwidth_gbps"] == 3350.0].iloc[0]
    int8_row = df_quant[df_quant["precision"] == "int8"].iloc[0]
    b64_row = df_batch[df_batch["batch_size"] == 64].iloc[0]

    summary_data = pd.DataFrame([
        {
            "Config / Phase": "Prefill (B=1, S=2048)",
            "Bound": "Compute",
            "Latency (ms)": f"{df_hbm.iloc[0]['ttft_ms']:.1f}",
            "Throughput": f"{1000 / df_hbm.iloc[0]['ttft_ms']:.1f} tok/s",
            "Key Bottleneck": "SM Tensor Core FLOPs",
        },
        {
            "Config / Phase": "Decode (B=1, FP16)",
            "Bound": "Memory",
            "Latency (ms)": f"{hbm_3350_row['tpot_ms']:.2f}",
            "Throughput": f"{1000 / hbm_3350_row['tpot_ms']:.1f} tok/s",
            "Key Bottleneck": "HBM Bandwidth (3.35 TB/s)",
        },
        {
            "Config / Phase": "Decode (B=1, INT8)",
            "Bound": "Memory",
            "Latency (ms)": f"{int8_row['tpot_ms']:.2f}",
            "Throughput": f"{4000 / int8_row['e2e_latency_s']:.1f} tok/s",
            "Key Bottleneck": "2x Weight Footprint Reduction",
        },
        {
            "Config / Phase": "Batch Saturation (B=64)",
            "Bound": "Compute",
            "Latency (ms)": f"{b64_row['tpot_ms']:.2f}",
            "Throughput": f"{b64_row['tokens_per_sec']:.1f} tok/s",
            "Key Bottleneck": "Saturated HBM; Compute Bound",
        },
    ])
    plot_bottleneck_summary_table(summary_data, out_path=RESULTS_DIR / "bottleneck_summary.png")

    # 8. Generate Recommendation
    sweep_dict = {
        "hbm": df_hbm,
        "batch": df_batch,
        "quant": df_quant,
        "tp": df_tp,
    }
    rec = generate_recommendation(sweep_dict)
    print("\n" + "=" * 80)
    print("EXECUTIVE BOTTLENECK RECOMMENDATION:")
    print("=" * 80)
    print(rec)
    print("=" * 80 + "\n")
    print("All sweeps completed successfully. Output files saved in results/")


if __name__ == "__main__":
    run_all()
