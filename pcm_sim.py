"""
PCM write simulation: WD-prone count, error sampling, latency/energy/cost.

All blocks are numpy arrays of shape (BLOCK_BITS,) with dtype uint8, values 0/1.
"""

import numpy as np
from config import (
    SLC_WD_WORDLINE, SLC_WD_BITLINE,
    SLC_RESET_LATENCY, SLC_SET_LATENCY, VNR_VERIFY_LATENCY,
    SLC_RESET_ENERGY, SLC_SET_ENERGY,
    RESET_COST_WEIGHT, SET_COST_WEIGHT,
    MLC_WD_00, MLC_WD_01, MLC_WD_11,
)

# MLC victim WD rate table indexed by 2-bit pattern integer:
#   0='00' (full amorphous, already RESET)
#   1='01'
#   2='10'  ← no disturbance
#   3='11'  (most crystalline, highest WD risk)
_MLC_VICTIM_RATES = np.array([MLC_WD_00, MLC_WD_01, 0.0, MLC_WD_11], dtype=np.float64)


def compute_wd_prone(new_block: np.ndarray,
                     stale_block: np.ndarray,
                     write_mask: np.ndarray) -> int:
    """
    Count WD-prone neighbor cells for a single 512-bit block write.

    A RESET at position i (write_mask[i]=1, new[i]=0) disturbs neighbor j
    if stale[j]=0 AND write_mask[j]=0 (idle zero cell).
    """
    B = len(new_block)

    # positions being RESET: written and going to 0
    resets = (write_mask == 1) & (new_block == 0)

    idle_zero = (stale_block == 0) & (write_mask == 0)

    # left neighbor contribution: reset at i → disturbs i-1
    left_contrib = np.zeros(B, dtype=np.int32)
    left_contrib[1:] = idle_zero[:-1].astype(np.int32)

    # right neighbor contribution: reset at i → disturbs i+1
    right_contrib = np.zeros(B, dtype=np.int32)
    right_contrib[:-1] = idle_zero[1:].astype(np.int32)

    return int(np.sum(resets.astype(np.int32) * (left_contrib + right_contrib)))


def simulate_wd_errors(wd_prone_count: int, rng: np.random.Generator) -> int:
    """
    Sample actual WD errors from binomial model (wordline + bitline).
    """
    if wd_prone_count == 0:
        return 0
    wordline_errors = rng.binomial(wd_prone_count, SLC_WD_WORDLINE)
    bitline_errors  = rng.binomial(wd_prone_count, SLC_WD_BITLINE)
    return int(wordline_errors + bitline_errors)


def _reset_set_counts(new_block: np.ndarray,
                      stale_block: np.ndarray) -> tuple[int, int]:
    reset_count = int(np.sum((stale_block == 1) & (new_block == 0)))
    set_count   = int(np.sum((stale_block == 0) & (new_block == 1)))
    return reset_count, set_count


def compute_write_latency(new_block: np.ndarray,
                          stale_block: np.ndarray,
                          vnr_count: int = 0) -> float:
    """Write latency in ns."""
    reset_count, set_count = _reset_set_counts(new_block, stale_block)
    base    = reset_count * SLC_RESET_LATENCY + set_count * SLC_SET_LATENCY
    vnr_lat = vnr_count * (SLC_RESET_LATENCY + VNR_VERIFY_LATENCY)
    return float(base + vnr_lat)


def compute_write_energy(new_block: np.ndarray,
                         stale_block: np.ndarray) -> float:
    """Write energy in nJ."""
    reset_count, set_count = _reset_set_counts(new_block, stale_block)
    return float(reset_count * SLC_RESET_ENERGY + set_count * SLC_SET_ENERGY)


def compute_write_cost(new_block: np.ndarray,
                       stale_block: np.ndarray) -> int:
    """Asymmetric write cost (RESET costs 2×, SET costs 1×)."""
    reset_count, set_count = _reset_set_counts(new_block, stale_block)
    return reset_count * RESET_COST_WEIGHT + set_count * SET_COST_WEIGHT


# ────────────────────────────────────────────────────────────────────── #
# MLC PCM (2 bits / cell)                                                #
# ────────────────────────────────────────────────────────────────────── #

def compute_wd_prone_mlc(
    new_block:   np.ndarray,   # (B,) uint8  — encoded block
    stale_block: np.ndarray,   # (B,) uint8
) -> np.ndarray:
    """
    Identify WD-prone MLC cell neighbors.

    Interprets a B-bit block as C = B//2 cells of 2 bits each (MSB first).
    An aggressor is an MLC cell being written (old cell ≠ new cell).
    A victim is an adjacent idle cell (not being written).

    Returns the stale pattern integers of every victim cell:
        0 = '00'  (amorphous — already RESET, WD rate MLC_WD_00)
        1 = '01'  (WD rate MLC_WD_01)
        2 = '10'  (WD rate 0)
        3 = '11'  (most crystalline, WD rate MLC_WD_11)

    len(returned array) == WD-prone cell count.
    """
    C = len(new_block) // 2                                  # 256 for 512-bit block

    new_cells   = new_block.reshape(C, 2)
    stale_cells = stale_block.reshape(C, 2)

    written = np.any(new_cells != stale_cells, axis=1)       # (C,) bool — aggressors
    idle    = ~written

    stale_pat = (stale_cells[:, 0] * 2 + stale_cells[:, 1]).astype(np.int32)  # (C,)

    # victim at i-1 if aggressor at i AND idle at i-1
    left_victim  = np.zeros(C, dtype=bool)
    # victim at i+1 if aggressor at i AND idle at i+1
    right_victim = np.zeros(C, dtype=bool)

    left_victim[:-1]  = written[1:]  & idle[:-1]
    right_victim[1:]  = written[:-1] & idle[1:]

    victim_mask = left_victim | right_victim
    return stale_pat[victim_mask]                            # variable-length int32 array


def simulate_wd_errors_mlc(
    victim_patterns: np.ndarray,          # returned by compute_wd_prone_mlc
    rng:             np.random.Generator,
) -> int:
    """
    Sample actual WD errors from per-pattern binomial draws.
    Each victim cell independently succeeds or fails according to its MLC
    resistance-state's error rate (_MLC_VICTIM_RATES).
    """
    if len(victim_patterns) == 0:
        return 0
    total = 0
    for pat_int in range(4):
        rate = _MLC_VICTIM_RATES[pat_int]
        if rate == 0.0:
            continue
        count = int(np.sum(victim_patterns == pat_int))
        if count > 0:
            total += int(rng.binomial(count, rate))
    return total
