"""Hardware configuration and loading for LLM inference simulation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class GPUConfig:
    """Hardware specifications for a GPU accelerator.

    Attributes:
        name: Name or model of the GPU accelerator.
        sm_count: Number of Streaming Multiprocessors (SMs).
        peak_tflops_fp16: Peak FP16 compute throughput in TFLOPs.
        peak_tflops_bf16: Peak BF16 compute throughput in TFLOPs.
        peak_tflops_fp8: Peak FP8 compute throughput in TFLOPs.
        peak_tflops_int8: Peak INT8 compute throughput in TOPS (TFLOPs equivalent).
        hbm_bandwidth_gbps: High Bandwidth Memory (HBM) bandwidth in GB/s.
        memory_capacity_gb: Total GPU onboard HBM memory capacity in GB.
        l2_cache_mb: On-chip L2 cache size in MB.
        interconnect_bandwidth_gbps: Inter-GPU interconnect bandwidth (e.g. NVLink) in GB/s.
        interconnect_latency_us: Inter-GPU interconnect latency in microseconds.
    """

    name: str
    sm_count: int
    peak_tflops_fp16: float
    peak_tflops_bf16: float
    peak_tflops_fp8: float
    peak_tflops_int8: float
    hbm_bandwidth_gbps: float
    memory_capacity_gb: float
    l2_cache_mb: float
    interconnect_bandwidth_gbps: float
    interconnect_latency_us: float

    def __post_init__(self) -> None:
        """Validate GPU configuration parameters."""
        if not self.name or not isinstance(self.name, str):
            raise ValueError("GPU name must be a non-empty string.")
        if self.sm_count <= 0:
            raise ValueError(f"sm_count must be positive, got {self.sm_count}.")
        if self.peak_tflops_fp16 <= 0:
            raise ValueError(f"peak_tflops_fp16 must be positive, got {self.peak_tflops_fp16}.")
        if self.peak_tflops_bf16 <= 0:
            raise ValueError(f"peak_tflops_bf16 must be positive, got {self.peak_tflops_bf16}.")
        if self.peak_tflops_fp8 < 0:
            raise ValueError(f"peak_tflops_fp8 must be non-negative, got {self.peak_tflops_fp8}.")
        if self.peak_tflops_int8 <= 0:
            raise ValueError(f"peak_tflops_int8 must be positive, got {self.peak_tflops_int8}.")
        if self.hbm_bandwidth_gbps <= 0:
            raise ValueError(f"hbm_bandwidth_gbps must be positive, got {self.hbm_bandwidth_gbps}.")
        if self.memory_capacity_gb <= 0:
            raise ValueError(f"memory_capacity_gb must be positive, got {self.memory_capacity_gb}.")
        if self.l2_cache_mb <= 0:
            raise ValueError(f"l2_cache_mb must be positive, got {self.l2_cache_mb}.")
        if self.interconnect_bandwidth_gbps <= 0:
            raise ValueError(
                "interconnect_bandwidth_gbps must be positive, "
                f"got {self.interconnect_bandwidth_gbps}."
            )
        if self.interconnect_latency_us < 0:
            raise ValueError(
                f"interconnect_latency_us must be non-negative, got {self.interconnect_latency_us}."
            )

    @property
    def memory_capacity_bytes(self) -> int:
        """Return memory capacity in bytes."""
        return int(self.memory_capacity_gb * (1024**3))

    @property
    def hbm_bandwidth_bytes_per_sec(self) -> float:
        """Return HBM bandwidth in bytes per second."""
        return float(self.hbm_bandwidth_gbps * 1e9)


def load_gpu(path: str | Path) -> GPUConfig:
    """Load and validate a GPU configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Validated GPUConfig instance.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If configuration values are invalid or required fields are missing.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Hardware configuration file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        data: Any = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML content in {file_path}; expected dictionary.")

    required_keys = {
        "name",
        "sm_count",
        "peak_tflops_fp16",
        "peak_tflops_bf16",
        "peak_tflops_fp8",
        "peak_tflops_int8",
        "hbm_bandwidth_gbps",
        "memory_capacity_gb",
        "l2_cache_mb",
        "interconnect_bandwidth_gbps",
        "interconnect_latency_us",
    }
    missing_keys = required_keys - set(data.keys())
    if missing_keys:
        raise ValueError(f"Missing required hardware configuration fields: {sorted(missing_keys)}")

    return GPUConfig(
        name=str(data["name"]),
        sm_count=int(data["sm_count"]),
        peak_tflops_fp16=float(data["peak_tflops_fp16"]),
        peak_tflops_bf16=float(data["peak_tflops_bf16"]),
        peak_tflops_fp8=float(data["peak_tflops_fp8"]),
        peak_tflops_int8=float(data["peak_tflops_int8"]),
        hbm_bandwidth_gbps=float(data["hbm_bandwidth_gbps"]),
        memory_capacity_gb=float(data["memory_capacity_gb"]),
        l2_cache_mb=float(data["l2_cache_mb"]),
        interconnect_bandwidth_gbps=float(data["interconnect_bandwidth_gbps"]),
        interconnect_latency_us=float(data["interconnect_latency_us"]),
    )
