#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Topology-aware profiling utilities for placement-aware USP.

Provides shared helpers for building communication groups with different
physical topologies (consecutive vs strided rank placement) and for
linear regression fitting of communication time data.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np


def build_group_ranks_list(
    world_size: int,
    group_size: int,
    topology: str = "consecutive",
) -> List[List[int]]:
    """Build rank lists for communication groups under a given topology.

    Args:
        world_size: Total number of ranks.
        group_size: Size of each communication group.
        topology:
            "consecutive" — ranks [0..G-1], [G..2G-1], … (current default)
            "strided"     — stride = world_size // group_size,
                            ranks [0, stride, 2*stride, …] for each offset

    Returns:
        List of rank-lists, one per group.  Every rank in [0, world_size)
        appears in exactly one group.
    """
    assert world_size % group_size == 0, (
        f"world_size ({world_size}) must be divisible by group_size ({group_size})"
    )

    if topology == "consecutive":
        num_groups = world_size // group_size
        return [
            list(range(g * group_size, (g + 1) * group_size))
            for g in range(num_groups)
        ]
    elif topology == "strided":
        stride = world_size // group_size
        return [
            [offset + i * stride for i in range(group_size)]
            for offset in range(stride)
        ]
    else:
        raise ValueError(f"Unknown topology: {topology!r}. Use 'consecutive' or 'strided'.")


def linear_fit(xs: List[float], ys: List[float]) -> Dict[str, float]:
    """Fit y = alpha * x + beta via least-squares.

    Returns:
        {"alpha": slope, "beta": intercept, "r_squared": R²}
    """
    xs_arr = np.array(xs, dtype=np.float64)
    ys_arr = np.array(ys, dtype=np.float64)

    n = len(xs_arr)
    if n < 2:
        return {"alpha": 0.0, "beta": float(ys_arr[0]) if n == 1 else 0.0, "r_squared": 0.0}

    sx = np.sum(xs_arr)
    sy = np.sum(ys_arr)
    sxy = np.sum(xs_arr * ys_arr)
    sx2 = np.sum(xs_arr ** 2)

    denom = n * sx2 - sx * sx
    if abs(denom) < 1e-15:
        alpha = 0.0
        beta = sy / n
    else:
        alpha = (n * sxy - sx * sy) / denom
        beta = (sy - alpha * sx) / n

    y_mean = sy / n
    ss_tot = np.sum((ys_arr - y_mean) ** 2)
    ss_res = np.sum((ys_arr - (alpha * xs_arr + beta)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0

    return {"alpha": float(alpha), "beta": float(beta), "r_squared": float(r_squared)}


def topo_key(group_size: int, topology: str) -> str:
    """Canonical string key for a (group_size, topology) pair.

    Examples: "gs8_consecutive", "gs16_strided"
    """
    return f"gs{group_size}_{topology}"


def parse_topo_key(key: str) -> Tuple[int, str]:
    """Inverse of topo_key.  Returns (group_size, topology)."""
    # "gs8_consecutive" -> ("8", "consecutive")
    rest = key[2:]  # strip "gs"
    parts = rest.split("_", 1)
    return int(parts[0]), parts[1]
