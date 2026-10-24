"""Precision and quantization formats for weights, KV cache, and activations."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llmsim.hardware import GPUConfig


class Precision(str, Enum):
    """Supported precision and quantization data types.

    Values:
        FP16: Standard 16-bit floating point (2 bytes).
        BF16: Brain 16-bit floating point (2 bytes).
        FP8: 8-bit floating point (1 byte).
        INT8: 8-bit integer quantization (1 byte).
        INT4: 4-bit integer quantization (0.5 bytes).
    """

    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"
    INT8 = "int8"
    INT4 = "int4"

    @property
    def bytes_per_param(self) -> float:
        """Memory footprint in bytes per parameter/element."""
        lookup = {
            Precision.FP16: 2.0,
            Precision.BF16: 2.0,
            Precision.FP8: 1.0,
            Precision.INT8: 1.0,
            Precision.INT4: 0.5,
        }
        return lookup[self]

    def get_peak_tflops(self, gpu: GPUConfig) -> float:
        """Retrieve the relevant peak compute throughput (TFLOPs) for this precision on a GPU.

        Args:
            gpu: Target GPU configuration.

        Returns:
            Peak compute throughput in TFLOPs.

        Raises:
            ValueError: If the precision is unsupported on the target GPU (e.g. FP8 on A100).
        """
        if self == Precision.FP16:
            return gpu.peak_tflops_fp16
        if self == Precision.BF16:
            return gpu.peak_tflops_bf16
        if self == Precision.FP8:
            if gpu.peak_tflops_fp8 <= 0:
                raise ValueError(f"FP8 compute is not supported on hardware '{gpu.name}'.")
            return gpu.peak_tflops_fp8
        if self == Precision.INT8:
            return gpu.peak_tflops_int8
        if self == Precision.INT4:
            # INT4 tensor core throughput runs at 2x INT8 peak throughput on modern architectures
            return gpu.peak_tflops_int8 * 2.0
        raise ValueError(f"Unknown precision: {self}")

    @classmethod
    def from_str(cls, val: str | Precision) -> Precision:
        """Parse string or Precision instance into Precision enum."""
        if isinstance(val, Precision):
            return val
        clean = val.strip().lower()
        for member in cls:
            if member.value == clean or member.name.lower() == clean:
                return member
        raise ValueError(
            f"Invalid precision '{val}'. Valid options are: {[m.value for m in cls]}"
        )
