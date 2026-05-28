"""
Main simulation loop.

run_simulation() accepts pluggable selector, encoder, and hooks so all
experiment configurations share the same loop body.

Hook signatures
---------------
  init_hook(stale_memory, stale_table, rng)  → None   called once before loop
  retrain_hook(stale_memory, stale_table, rng) → None  called every RETRAIN_INTERVAL
  on_write_fn(stale_addr)                    → None   called after each write
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from config import RETRAIN_INTERVAL
from typing import Optional
from stale_table import StaleTable
from pcm_sim import (
    compute_wd_prone,
    simulate_wd_errors,
    compute_wd_prone_mlc,
    simulate_wd_errors_mlc,
    compute_write_latency,
    compute_write_energy,
    compute_write_cost,
)


# ------------------------------------------------------------------ #
# Result container                                                     #
# ------------------------------------------------------------------ #

@dataclass
class SimResult:
    wd_prone_counts:  list[int]   = field(default_factory=list)
    wd_error_counts:  list[int]   = field(default_factory=list)
    write_latencies:  list[float] = field(default_factory=list)
    write_energies:   list[float] = field(default_factory=list)
    write_costs:      list[int]   = field(default_factory=list)
    vnr_counts:       list[int]   = field(default_factory=list)
    retrain_events:   list[int]   = field(default_factory=list)

    def summary(self) -> dict:
        a = np.array
        return {
            "mean_wd_prone":   float(a(self.wd_prone_counts).mean()),
            "mean_wd_errors":  float(a(self.wd_error_counts).mean()),
            "total_wd_errors": int(sum(self.wd_error_counts)),
            "mean_latency_ns": float(a(self.write_latencies).mean()),
            "mean_energy_nJ":  float(a(self.write_energies).mean()),
            "mean_write_cost": float(a(self.write_costs).mean()),
            "total_vnr":       int(sum(self.vnr_counts)),
            "n_retrains":      len(self.retrain_events),
            "n_writes":        len(self.wd_error_counts),
        }


# ------------------------------------------------------------------ #
# Type aliases                                                         #
# ------------------------------------------------------------------ #

SelectorFn  = Callable[..., int]
EncoderFn   = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
HookFn      = Callable[..., None]


def _noop(*_args, **_kwargs) -> None:
    pass


# ------------------------------------------------------------------ #
# Core loop                                                            #
# ------------------------------------------------------------------ #

def run_simulation(
    stale_blocks:     np.ndarray,
    write_requests:   np.ndarray,
    selector:         SelectorFn,
    encoder:          EncoderFn,
    rng:              np.random.Generator,
    init_hook:        HookFn = _noop,
    retrain_hook:     HookFn = _noop,
    on_write_fn:      Optional[Callable[[int], None]] = None,
    progress_every:   int = 5_000,
    ecc_level:        int = -1,
    retrain_interval: Optional[int] = None,
    cell_type:        str = "slc",   # "slc" | "mlc"
) -> SimResult:
    """
    Execute the full write workload and collect per-write metrics.
    stale_blocks is copied internally; the caller's array is not mutated.

    ecc_level : -1 = no ECC; 0+ = ECC-i corrects up to i errors, VnR if wd_errors > i
    retrain_interval : overrides RETRAIN_INTERVAL from config if provided
    cell_type : "slc" uses per-bit WD model; "mlc" uses 2-bit-cell MLC WD model
    """
    _retrain_interval = retrain_interval if retrain_interval is not None else RETRAIN_INTERVAL

    stale_memory = stale_blocks.copy()

    stale_table = StaleTable()
    stale_table.bulk_insert(list(range(len(stale_blocks))))

    # Initial training (LearnWD) or no-op (RandSel)
    init_hook(stale_memory, stale_table, rng)

    result = SimResult()
    overwrite_counter = 0
    t0 = time.perf_counter()

    for write_idx, new_block in enumerate(write_requests):

        # 1. Select stale block
        stale_addr  = selector(new_block, stale_table, stale_memory, rng=rng)
        stale_block = stale_memory[stale_addr]

        # 2. Encode
        encoded_block, write_mask = encoder(new_block, stale_block)

        # 3. Compute metrics
        if cell_type == "mlc":
            _vp      = compute_wd_prone_mlc(encoded_block, stale_block)
            wd_prone  = len(_vp)
            wd_errors = simulate_wd_errors_mlc(_vp, rng)
        else:
            wd_prone  = compute_wd_prone(encoded_block, stale_block, write_mask)
            wd_errors = simulate_wd_errors(wd_prone, rng)

        vnr = 0
        if ecc_level >= 0 and wd_errors > ecc_level:
            vnr = 1

        latency   = compute_write_latency(encoded_block, stale_block, vnr_count=vnr)
        energy    = compute_write_energy(encoded_block, stale_block)
        cost      = compute_write_cost(encoded_block, stale_block)

        # 4. Record
        result.wd_prone_counts.append(wd_prone)
        result.wd_error_counts.append(wd_errors)
        result.write_latencies.append(latency)
        result.write_energies.append(energy)
        result.write_costs.append(cost)
        result.vnr_counts.append(vnr)

        # 5. Update stale pool + cluster membership
        stale_memory[stale_addr] = encoded_block
        stale_table.delete(stale_addr)
        stale_table.insert(stale_addr, cluster_id=-1)

        # Notify model cache (LearnWD only; no-op for RandSel)
        if on_write_fn is not None:
            on_write_fn(stale_addr)

        # 6. Retrain trigger
        overwrite_counter += 1
        if overwrite_counter >= _retrain_interval:
            retrain_hook(stale_memory, stale_table, rng)
            result.retrain_events.append(write_idx)
            overwrite_counter = 0

        # 7. Progress report
        if progress_every and (write_idx + 1) % progress_every == 0:
            elapsed = time.perf_counter() - t0
            print(
                f"  [{write_idx + 1:>6} / {len(write_requests)}] "
                f"{elapsed:.1f}s  "
                f"mean WD errors: {np.mean(result.wd_error_counts):.4f}"
            )

    return result
