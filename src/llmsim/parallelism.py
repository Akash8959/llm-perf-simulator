"""Distributed parallelism configurations and communication latency models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParallelConfig:
    """Parallelism strategy dimensions.

    Attributes:
        tp: Tensor parallel degree (shards layers intra-node).
        pp: Pipeline parallel degree (shards layers inter-node).
        dp: Data parallel degree (replicates model instances across batches).
        ep: Expert parallel degree (shards MoE experts).
    """

    tp: int = 1
    pp: int = 1
    dp: int = 1
    ep: int = 1

    def __post_init__(self) -> None:
        """Validate parallelism configuration."""
        if self.tp <= 0:
            raise ValueError(f"tp must be positive, got {self.tp}.")
        if self.pp <= 0:
            raise ValueError(f"pp must be positive, got {self.pp}.")
        if self.dp <= 0:
            raise ValueError(f"dp must be positive, got {self.dp}.")
        if self.ep <= 0:
            raise ValueError(f"ep must be positive, got {self.ep}.")

    @property
    def world_size(self) -> int:
        """Total number of GPUs participating across all parallelism dimensions."""
        return self.tp * self.pp * self.dp * self.ep


def ring_allreduce_time(
    message_bytes: float,
    num_gpus: int,
    link_bw_bytes_sec: float,
    link_latency_sec: float,
) -> float:
    """Calculate the communication duration for a Ring All-Reduce collective operation.

    Formula:
        t_ring = 2 * ((N - 1) / N) * (message_bytes / link_BW) + (N - 1) * link_latency

    Args:
        message_bytes: Volume of data per GPU to all-reduce in bytes (B * tokens * h * b_act).
        num_gpus: Number of GPUs participating in the ring (N).
        link_bw_bytes_sec: Interconnect link bandwidth in bytes per second.
        link_latency_sec: Interconnect link latency in seconds.

    Returns:
        Communication latency in seconds. Returns 0.0 if num_gpus <= 1.
    """
    if num_gpus <= 1:
        return 0.0
    if link_bw_bytes_sec <= 0:
        raise ValueError(f"link_bw_bytes_sec must be positive, got {link_bw_bytes_sec}.")
    if link_latency_sec < 0:
        raise ValueError(f"link_latency_sec must be non-negative, got {link_latency_sec}.")
    if message_bytes <= 0:
        return 0.0

    n = float(num_gpus)
    transfer_time = 2.0 * ((n - 1.0) / n) * (message_bytes / link_bw_bytes_sec)
    latency_time = (n - 1.0) * link_latency_sec
    return transfer_time + latency_time


def all_to_all_time(
    message_bytes: float,
    num_gpus: int,
    link_bw_bytes_sec: float,
    link_latency_sec: float,
) -> float:
    """Calculate communication duration for MoE All-to-All dispatch and combine operations.

    Formula:
        t_all_to_all = 2 * (((N - 1) / N) * (message_bytes / link_BW) + (N - 1) * link_latency)

    Args:
        message_bytes: Transferred payload volume in bytes.
        num_gpus: Number of GPUs across which experts are partitioned (N_ep).
        link_bw_bytes_sec: Interconnect link bandwidth in bytes per second.
        link_latency_sec: Interconnect link latency in seconds.

    Returns:
        Communication latency in seconds. Returns 0.0 if num_gpus <= 1.
    """
    if num_gpus <= 1:
        return 0.0
    if link_bw_bytes_sec <= 0:
        raise ValueError(f"link_bw_bytes_sec must be positive, got {link_bw_bytes_sec}.")
    if link_latency_sec < 0:
        raise ValueError(f"link_latency_sec must be non-negative, got {link_latency_sec}.")
    if message_bytes <= 0:
        return 0.0

    n = float(num_gpus)
    # Factor of 2 covers both dispatch (tokens -> experts) and combine (experts -> tokens)
    transfer_time = 2.0 * ((n - 1.0) / n) * (message_bytes / link_bw_bytes_sec)
    latency_time = 2.0 * (n - 1.0) * link_latency_sec
    return transfer_time + latency_time
