"""
Sensitivity experiments: Exp#6-12.

Each function returns a list of result dicts for tabular printing.
"""

from __future__ import annotations
import time
from typing import Optional   # noqa: F401  (used in Exp#5 and Exp#6)

import numpy as np

from config      import BLOCK_BITS, STALE_POOL_SIZE, RETRAIN_INTERVAL
from data_loader import load_dataset
from encoding    import dcw_encode
from selector    import randsel, make_learnwd_selector
from learnwd     import LearnWDModel
from simulation  import run_simulation, SimResult
from stale_table import StaleTable
from pcm_sim     import (
    compute_wd_prone, simulate_wd_errors,
    compute_write_latency, compute_write_energy, compute_write_cost,
)


# ------------------------------------------------------------------ #
# Shared helpers                                                       #
# ------------------------------------------------------------------ #

def _run_randsel(stale_blocks, write_requests, seed, ecc_level=-1,
                 retrain_interval=None, cell_type="slc"):
    rng = np.random.default_rng(seed)
    return run_simulation(
        stale_blocks=stale_blocks,
        write_requests=write_requests,
        selector=randsel,
        encoder=dcw_encode,
        rng=rng,
        ecc_level=ecc_level,
        retrain_interval=retrain_interval,
        cell_type=cell_type,
        progress_every=0,
    )


def _run_learnwd(stale_blocks, write_requests, seed, ecc_level=-1,
                 retrain_interval=None, k=None, h=None,
                 block_bits=None, cluster_algo=None, cell_type="slc"):
    model_kwargs = {}
    if k is not None:
        model_kwargs["k"] = k
    if h is not None:
        model_kwargs["h"] = h
    if block_bits is not None:
        model_kwargs["block_bits"] = block_bits
    if cluster_algo is not None:
        model_kwargs["cluster_algo"] = cluster_algo
    model_kwargs["cell_type"] = cell_type

    model    = LearnWDModel(**model_kwargs)
    selector = make_learnwd_selector(model)

    def _train(stale_memory, stale_table, rng):
        model.train(stale_memory, stale_table)

    rng = np.random.default_rng(seed)
    result = run_simulation(
        stale_blocks=stale_blocks,
        write_requests=write_requests,
        selector=selector,
        encoder=dcw_encode,
        rng=rng,
        init_hook=_train,
        retrain_hook=_train,
        on_write_fn=model.invalidate,
        ecc_level=ecc_level,
        retrain_interval=retrain_interval,
        cell_type=cell_type,
        progress_every=0,
    )
    return result, model


# ------------------------------------------------------------------ #
# Exp#5 — MLC PCM cell type                                           #
# ------------------------------------------------------------------ #

def run_exp5_mlc(stale_blocks: np.ndarray, write_requests: np.ndarray,
                 seed: int = 0) -> list[dict]:
    """
    Exp#5: Compare SLC vs MLC cell model for RandSel and LearnWD (both with DCW).

    SLC  — 1 bit/cell, wordline WD 9.9%, bitline 11.5%.
    MLC  — 2 bits/cell (256 cells × 2 bits per 512-bit block).
           WD rates depend on victim cell's resistance state:
           '00'→12.3%  '01'→15.2%  '10'→0%  '11'→27.6%.
           LearnWD uses MLC-aware disturbance and aggressor vectors.

    Metric: mean WD errors per write, normalised to RandSel+SLC.
    Returns list of dicts with keys: selector, cell_type, mean_wd_errors.
    """
    print("\n=== Exp#5: MLC PCM cell type ===\n")
    rows: list[dict] = []
    baseline_wd: Optional[float] = None

    for cell_t in ("slc", "mlc"):
        for sel_name in ("RandSel", "LearnWD"):
            t0 = time.perf_counter()
            if sel_name == "RandSel":
                r = _run_randsel(stale_blocks, write_requests, seed,
                                 cell_type=cell_t)
            else:
                r, _ = _run_learnwd(stale_blocks, write_requests, seed,
                                    cell_type=cell_t)
            elapsed = time.perf_counter() - t0
            s = r.summary()

            if baseline_wd is None:
                baseline_wd = s["mean_wd_errors"]
            norm = s["mean_wd_errors"] / baseline_wd if baseline_wd else float("nan")

            rows.append({
                "selector":       sel_name,
                "cell_type":      cell_t,
                "mean_wd_errors": s["mean_wd_errors"],
                "mean_wd_prone":  s["mean_wd_prone"],
                "norm_wd":        norm,
            })
            print(
                f"  {sel_name} + {cell_t.upper()}: "
                f"WD errors={s['mean_wd_errors']:.4f}  "
                f"WD prone={s['mean_wd_prone']:.4f}  "
                f"norm={norm:.4f}  {elapsed:.1f}s"
            )

    return rows


# ------------------------------------------------------------------ #
# Exp#6 — ECC / VnR                                                   #
# ------------------------------------------------------------------ #

ECC_LEVELS = [0, 1, 2, 4, 8]


def run_exp6(stale_blocks: np.ndarray, write_requests: np.ndarray,
             seed: int = 0) -> list[dict]:
    """
    Sweep ECC levels 0/1/2/4/8 for RandSel and LearnWD (both with DCW).
    Metric: total VnR operations normalized to RandSel+ECC-0.
    Returns list of dicts with keys: selector, ecc, total_vnr, norm_vnr.
    """
    print("\n=== Exp#6: ECC / VnR ===\n")
    rows: list[dict] = []
    baseline_vnr: Optional[int] = None

    for ecc in ECC_LEVELS:
        # RandSel
        t0 = time.perf_counter()
        r = _run_randsel(stale_blocks, write_requests, seed, ecc_level=ecc)
        s = r.summary()
        elapsed = time.perf_counter() - t0
        vnr = s["total_vnr"]
        if baseline_vnr is None:
            baseline_vnr = vnr
        norm = vnr / baseline_vnr if baseline_vnr else float("nan")
        rows.append({"selector": "RandSel", "ecc": ecc,
                     "total_vnr": vnr, "norm_vnr": norm})
        print(f"  RandSel + ECC-{ecc}: VnR={vnr:>6}  norm={norm:.4f}  {elapsed:.1f}s")

        # LearnWD
        t0 = time.perf_counter()
        r, _ = _run_learnwd(stale_blocks, write_requests, seed, ecc_level=ecc)
        s = r.summary()
        elapsed = time.perf_counter() - t0
        vnr = s["total_vnr"]
        norm = vnr / baseline_vnr if baseline_vnr else float("nan")
        rows.append({"selector": "LearnWD", "ecc": ecc,
                     "total_vnr": vnr, "norm_vnr": norm})
        print(f"  LearnWD + ECC-{ecc}: VnR={vnr:>6}  norm={norm:.4f}  {elapsed:.1f}s")

    return rows


# ------------------------------------------------------------------ #
# Exp#7 — k sensitivity                                               #
# ------------------------------------------------------------------ #

K_VALUES = [2, 4, 8, 16, 32]


def run_exp7_k(stale_blocks: np.ndarray, write_requests: np.ndarray,
               seed: int = 0) -> list[dict]:
    """
    Sweep k ∈ {2,4,8,16,32}. Record mean WD errors and mean train time.
    Returns list of dicts with keys: k, mean_wd_errors, mean_train_time_s.
    """
    print("\n=== Exp#7: k sensitivity ===\n")
    rows: list[dict] = []

    for k in K_VALUES:
        t0 = time.perf_counter()
        r, model = _run_learnwd(stale_blocks, write_requests, seed, k=k)
        elapsed = time.perf_counter() - t0
        s = r.summary()
        mean_train = float(np.mean(model.train_times)) if model.train_times else 0.0
        rows.append({
            "k": k,
            "mean_wd_errors": s["mean_wd_errors"],
            "mean_train_time_s": mean_train,
        })
        print(
            f"  k={k:>2}: WD errors={s['mean_wd_errors']:.4f}  "
            f"train time={mean_train:.3f}s  total={elapsed:.1f}s"
        )

    return rows


# ------------------------------------------------------------------ #
# Exp#8 — h sensitivity                                               #
# ------------------------------------------------------------------ #

H_VALUES = [0, 1, 2, 4, 8, 16]


def run_exp8_h(stale_blocks: np.ndarray, write_requests: np.ndarray,
               seed: int = 0) -> list[dict]:
    """
    Sweep h ∈ {0,1,2,4,8,16}. Record mean WD errors.
    h=0 means no MinHash similarity — just pick first valid block in cluster.
    Returns list of dicts with keys: h, mean_wd_errors.
    """
    print("\n=== Exp#8: h sensitivity ===\n")
    rows: list[dict] = []

    for h in H_VALUES:
        t0 = time.perf_counter()
        r, _ = _run_learnwd(stale_blocks, write_requests, seed, h=h)
        elapsed = time.perf_counter() - t0
        s = r.summary()
        rows.append({"h": h, "mean_wd_errors": s["mean_wd_errors"]})
        print(f"  h={h:>2}: WD errors={s['mean_wd_errors']:.4f}  {elapsed:.1f}s")

    return rows


# ------------------------------------------------------------------ #
# Exp#9 — retrain frequency sensitivity                               #
# ------------------------------------------------------------------ #

RETRAIN_INTERVALS = [5_000, 10_000, 20_000, 30_000, 50_000]


def run_exp9_retrain(stale_blocks: np.ndarray, write_requests: np.ndarray,
                     seed: int = 0) -> list[dict]:
    """
    Sweep retrain interval ∈ {5k,10k,20k,30k,50k}. Record mean WD errors.
    Returns list of dicts with keys: interval, mean_wd_errors, n_retrains.
    """
    print("\n=== Exp#9: retrain interval sensitivity ===\n")
    rows: list[dict] = []

    for interval in RETRAIN_INTERVALS:
        t0 = time.perf_counter()
        r, _ = _run_learnwd(stale_blocks, write_requests, seed,
                             retrain_interval=interval)
        elapsed = time.perf_counter() - t0
        s = r.summary()
        rows.append({
            "interval": interval,
            "mean_wd_errors": s["mean_wd_errors"],
            "n_retrains": s["n_retrains"],
        })
        print(
            f"  interval={interval:>6}: WD errors={s['mean_wd_errors']:.4f}  "
            f"retrains={s['n_retrains']}  {elapsed:.1f}s"
        )

    return rows


# ------------------------------------------------------------------ #
# Exp#10 — block size sensitivity                                      #
# ------------------------------------------------------------------ #

# (label, bits)
BLOCK_SIZES = [
    ("64B",  64   * 8),
    ("256B", 256  * 8),
    ("1KB",  1024 * 8),
    ("4KB",  4096 * 8),
]


def run_exp10_blocksize(dataset: str = "mnist", seed: int = 0) -> list[dict]:
    """
    Sweep block sizes 64B / 256B / 1KB / 4KB.
    MNIST bits are tiled to fill 50k stale + 50k write blocks at each size.
    Returns list of dicts: size_label, block_bits, selector, mean_wd_errors.
    """
    print("\n=== Exp#10: block size sensitivity ===\n")
    rows: list[dict] = []

    for label, bb in BLOCK_SIZES:
        print(f"── {label} ({bb} bits) ──")
        stale, writes = load_dataset(dataset, block_bits=bb)

        for sel_name in ("RandSel", "LearnWD"):
            t0 = time.perf_counter()
            if sel_name == "RandSel":
                r = _run_randsel(stale, writes, seed)
            else:
                r, _ = _run_learnwd(stale, writes, seed, block_bits=bb)
            elapsed = time.perf_counter() - t0
            s = r.summary()
            rows.append({
                "size_label":     label,
                "block_bits":     bb,
                "selector":       sel_name,
                "mean_wd_errors": s["mean_wd_errors"],
            })
            print(f"  {sel_name}: WD errors={s['mean_wd_errors']:.4f}  {elapsed:.1f}s")

    return rows


# ------------------------------------------------------------------ #
# Exp#11 — clustering algorithm comparison                             #
# ------------------------------------------------------------------ #

CLUSTER_ALGOS = ["kmeans", "gmm", "birch"]


def run_exp11_algo(stale_blocks: np.ndarray, write_requests: np.ndarray,
                   seed: int = 0) -> list[dict]:
    """
    Compare k-means / GMM / BIRCH clustering for LearnWD+DCW on MNIST.
    Returns list of dicts: algo, mean_wd_errors, mean_train_time_s.
    """
    print("\n=== Exp#11: clustering algorithm comparison ===\n")
    rows: list[dict] = []

    for algo in CLUSTER_ALGOS:
        t0 = time.perf_counter()
        r, model = _run_learnwd(stale_blocks, write_requests, seed,
                                cluster_algo=algo)
        elapsed = time.perf_counter() - t0
        s = r.summary()
        mean_train = float(np.mean(model.train_times)) if model.train_times else 0.0
        rows.append({
            "algo":              algo,
            "mean_wd_errors":    s["mean_wd_errors"],
            "mean_train_time_s": mean_train,
        })
        print(
            f"  {algo:>6}: WD errors={s['mean_wd_errors']:.4f}  "
            f"train={mean_train:.3f}s  total={elapsed:.1f}s"
        )

    return rows


# ------------------------------------------------------------------ #
# Exp#12 — dataset switching                                           #
# ------------------------------------------------------------------ #

# 6 segments approximating SPEC's INB→… chain with locally-available datasets.
# Covers the full range of bit-density / correlation profiles.
SWITCH_DATASETS = [
    "mnist",             # seg 0 — sparse images   (~12 % ones)
    "synthetic_random",  # seg 1 — 50 % random
    "synthetic_alt",     # seg 2 — alternating 0101 (structured)
    "synthetic_dense",   # seg 3 — 90 % ones
    "synthetic_corr",    # seg 4 — locally-correlated runs
    "synthetic_sparse",  # seg 5 — 5 % ones (very sparse)
]


def _run_multiseg(
    stale_init:       np.ndarray,
    write_segs:       list[np.ndarray],
    selector_type:    str,
    seed:             int = 0,
    retrain_interval: int = RETRAIN_INTERVAL,
) -> list[SimResult]:
    """
    Keep stale_memory and model alive across all segments; reset only the
    per-segment SimResult bucket.  Retraining fires every retrain_interval
    writes regardless of which segment boundary we cross.
    """
    stale_memory = stale_init.copy()
    stale_table  = StaleTable()
    stale_table.bulk_insert(list(range(len(stale_memory))))

    if selector_type == "learnwd":
        model    = LearnWDModel(block_bits=stale_init.shape[1])
        selector = make_learnwd_selector(model)
        model.train(stale_memory, stale_table)
    else:
        model    = None
        selector = randsel

    rng = np.random.default_rng(seed)
    overwrite_counter = 0
    seg_results: list[SimResult] = []

    for seg_idx, seg_writes in enumerate(write_segs):
        seg_result = SimResult()

        for new_block in seg_writes:
            stale_addr  = selector(new_block, stale_table, stale_memory, rng=rng)
            stale_block = stale_memory[stale_addr]

            encoded_block, write_mask = dcw_encode(new_block, stale_block)
            wd_prone  = compute_wd_prone(encoded_block, stale_block, write_mask)
            wd_errors = simulate_wd_errors(wd_prone, rng)
            latency   = compute_write_latency(encoded_block, stale_block)
            energy    = compute_write_energy(encoded_block, stale_block)
            cost      = compute_write_cost(encoded_block, stale_block)

            seg_result.wd_prone_counts.append(wd_prone)
            seg_result.wd_error_counts.append(wd_errors)
            seg_result.write_latencies.append(latency)
            seg_result.write_energies.append(energy)
            seg_result.write_costs.append(cost)
            seg_result.vnr_counts.append(0)

            stale_memory[stale_addr] = encoded_block
            stale_table.delete(stale_addr)
            stale_table.insert(stale_addr, cluster_id=-1)
            if model is not None:
                model.invalidate(stale_addr)

            overwrite_counter += 1
            if overwrite_counter >= retrain_interval:
                if model is not None:
                    model.train(stale_memory, stale_table)
                    seg_result.retrain_events.append(
                        len(seg_result.wd_error_counts))
                overwrite_counter = 0

        seg_results.append(seg_result)
        s = seg_result.summary()
        print(
            f"    seg {seg_idx} ({SWITCH_DATASETS[seg_idx]}): "
            f"WD errors={s['mean_wd_errors']:.4f}  retrains={s['n_retrains']}"
        )

    return seg_results


def run_exp12_switch(seed: int = 0) -> list[dict]:
    """
    Dataset switching: 6 segments × 50k writes each; stale pool evolves
    continuously across switches.  Metric: per-segment mean WD errors.

    Note: SPEC references Kaggle datasets (INB etc.) not locally available.
    We approximate with MNIST + 5 synthetic variants spanning the same
    bit-density and correlation range.
    """
    print("\n=== Exp#12: dataset switching ===\n")

    seg_data: list[tuple[np.ndarray, np.ndarray]] = []
    for ds in SWITCH_DATASETS:
        stale, writes = load_dataset(ds)
        seg_data.append((stale, writes))
        print(f"  Loaded {ds}: stale {stale.shape}  writes {writes.shape}")

    stale_init = seg_data[0][0]
    write_segs = [sd[1] for sd in seg_data]

    rows: list[dict] = []
    for sel_type in ("randsel", "learnwd"):
        print(f"\n── {sel_type.upper()} ──")
        seg_results = _run_multiseg(stale_init, write_segs, sel_type, seed)
        for seg_idx, r in enumerate(seg_results):
            s = r.summary()
            rows.append({
                "selector":       sel_type,
                "segment":        seg_idx,
                "dataset":        SWITCH_DATASETS[seg_idx],
                "mean_wd_errors": s["mean_wd_errors"],
                "n_retrains":     s["n_retrains"],
            })

    return rows
