"""Tests for the llmsim command-line interface."""

from pathlib import Path

import pytest

from llmsim.cli import main

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
HARDWARE = str(CONFIGS_DIR / "hardware" / "h100_sxm.yaml")
MODEL = str(CONFIGS_DIR / "models" / "llama3_8b.yaml")


def test_cli_single_simulation(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify single-configuration simulation runs and outputs results table."""
    exit_code = main([
        "--hardware", HARDWARE,
        "--model", MODEL,
        "--batch", "16",
        "--prompt-len", "2048",
        "--output-len", "256",
        "--tp", "2",
        "--quant", "int8",
    ])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "LLM PERFORMANCE SIMULATOR" in captured.out
    assert "Time to First Token (TTFT)" in captured.out
    assert "BOTTLENECK RECOMMENDATION" in captured.out


def test_cli_sweep_and_plot(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify running a sweep with plot output through the CLI."""
    out_dir = tmp_path / "cli_results"
    exit_code = main([
        "--hardware", HARDWARE,
        "--model", MODEL,
        "--batch", "1",
        "--prompt-len", "512",
        "--output-len", "32",
        "--sweep", "hbm",
        "--plot",
        "--out-dir", str(out_dir),
    ])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Running sweep: HBM" in captured.out
    assert (out_dir / "sweep_hbm.csv").is_file()
    assert (out_dir / "tpot_vs_hbm.png").is_file()


def test_cli_missing_required_args() -> None:
    """Verify missing required arguments exits with error."""
    with pytest.raises(SystemExit):
        main(["--batch", "4"])
