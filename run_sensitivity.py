"""
Sensitivity experiment runner — Exp#6-12.

Usage:
    python3 run_sensitivity.py [dataset] [seed] [experiments]

    dataset     : mnist | synthetic | raw:<path>   (default: mnist)
    seed        : integer RNG seed                  (default: 0)
    experiments : comma-separated subset of 5,6,7,8,9,10,11,12,all  (default: all)

Examples:
    python3 run_sensitivity.py mnist 0 all
    python3 run_sensitivity.py mnist 0 5
    python3 run_sensitivity.py mnist 0 10,11,12
"""

import sys
import time

import numpy as np

from data_loader import load_dataset
from experiments import (
    run_exp5_mlc,
    run_exp6,  run_exp7_k, run_exp8_h, run_exp9_retrain,
    run_exp10_blocksize, run_exp11_algo, run_exp12_switch,
    ECC_LEVELS, K_VALUES, H_VALUES, RETRAIN_INTERVALS,
    BLOCK_SIZES, CLUSTER_ALGOS, SWITCH_DATASETS,
)


# ------------------------------------------------------------------ #
# Printers                                                             #
# ------------------------------------------------------------------ #

def _print_exp5(rows: list[dict]) -> None:
    baseline = next((r["mean_wd_errors"] for r in rows
                     if r["selector"] == "RandSel" and r["cell_type"] == "slc"), 1.0)
    print("\n── Exp#5 Results (SLC vs MLC, WD normalised to RandSel+SLC) ──")
    print(f"  {'Selector':<10}  {'Cell':>4}  {'WD errors':>12}  "
          f"{'WD prone':>10}  {'Norm':>8}")
    print("  " + "─" * 52)
    for r in rows:
        print(
            f"  {r['selector']:<10}  {r['cell_type'].upper():>4}  "
            f"{r['mean_wd_errors']:>12.4f}  "
            f"{r['mean_wd_prone']:>10.4f}  "
            f"{r['norm_wd']:>8.4f}"
        )
    print(f"  (baseline RandSel+SLC WD errors = {baseline:.4f})")


def _print_exp6(rows: list[dict]) -> None:
    baseline = next((r["total_vnr"] for r in rows
                     if r["selector"] == "RandSel" and r["ecc"] == 0), 1)
    print("\n── Exp#6 Results (VnR normalised to RandSel+ECC-0) ──")
    print(f"  {'Selector':<10}  {'ECC':>4}  {'Total VnR':>10}  {'Norm':>8}")
    print("  " + "─" * 38)
    for r in rows:
        print(f"  {r['selector']:<10}  ECC-{r['ecc']:<2}  "
              f"{r['total_vnr']:>10}  {r['norm_vnr']:>8.4f}")
    print(f"  (baseline RandSel+ECC-0 = {baseline} VnR operations)")


def _print_exp7(rows: list[dict]) -> None:
    print("\n── Exp#7 Results (k sensitivity) ──")
    print(f"  {'k':>4}  {'WD errors':>12}  {'Mean train(s)':>14}")
    print("  " + "─" * 34)
    for r in rows:
        print(f"  {r['k']:>4}  {r['mean_wd_errors']:>12.4f}  "
              f"{r['mean_train_time_s']:>14.3f}")


def _print_exp8(rows: list[dict]) -> None:
    print("\n── Exp#8 Results (h sensitivity) ──")
    print(f"  {'h':>4}  {'WD errors':>12}")
    print("  " + "─" * 20)
    for r in rows:
        print(f"  {r['h']:>4}  {r['mean_wd_errors']:>12.4f}")


def _print_exp9(rows: list[dict]) -> None:
    print("\n── Exp#9 Results (retrain interval sensitivity) ──")
    print(f"  {'Interval':>10}  {'WD errors':>12}  {'Retrains':>10}")
    print("  " + "─" * 36)
    for r in rows:
        print(f"  {r['interval']:>10}  {r['mean_wd_errors']:>12.4f}  "
              f"{r['n_retrains']:>10}")


def _print_exp10(rows: list[dict]) -> None:
    print("\n── Exp#10 Results (block size sensitivity) ──")
    print(f"  {'Size':<6}  {'bits':>6}  {'Selector':<10}  {'WD errors':>12}")
    print("  " + "─" * 40)
    for r in rows:
        print(f"  {r['size_label']:<6}  {r['block_bits']:>6}  "
              f"{r['selector']:<10}  {r['mean_wd_errors']:>12.4f}")


def _print_exp11(rows: list[dict]) -> None:
    print("\n── Exp#11 Results (clustering algorithm) ──")
    print(f"  {'Algorithm':<8}  {'WD errors':>12}  {'Mean train(s)':>14}")
    print("  " + "─" * 38)
    for r in rows:
        print(f"  {r['algo']:<8}  {r['mean_wd_errors']:>12.4f}  "
              f"{r['mean_train_time_s']:>14.3f}")


def _print_exp12(rows: list[dict]) -> None:
    print("\n── Exp#12 Results (dataset switching) ──")
    print(f"  {'Selector':<10}  {'Seg':>4}  {'Dataset':<20}  "
          f"{'WD errors':>12}  {'Retrains':>10}")
    print("  " + "─" * 62)
    for r in rows:
        print(f"  {r['selector']:<10}  {r['segment']:>4}  "
              f"{r['dataset']:<20}  {r['mean_wd_errors']:>12.4f}  "
              f"{r['n_retrains']:>10}")


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

_ALL_EXPS = {"5", "6", "7", "8", "9", "10", "11", "12"}

_RUNNERS_NEED_DATA = {"5", "6", "7", "8", "9", "11"}   # need (stale, writes) from dataset arg
_RUNNERS_NO_DATA   = {"10", "12"}                       # load their own data internally


def main() -> None:
    dataset  = sys.argv[1] if len(sys.argv) > 1 else "mnist"
    seed     = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    exp_arg  = sys.argv[3] if len(sys.argv) > 3 else "all"

    if exp_arg == "all":
        chosen = sorted(_ALL_EXPS, key=int)
    else:
        chosen = [e.strip() for e in exp_arg.split(",")]
        for e in chosen:
            if e not in _ALL_EXPS:
                raise ValueError(
                    f"Unknown experiment '{e}'. Choose from {sorted(_ALL_EXPS)}")

    print(f"\n=== LearnWD Sensitivity  dataset={dataset}  seed={seed}"
          f"  experiments={chosen} ===\n")

    # Load shared data (used by Exp#6-9, #11)
    stale_blocks = write_requests = None
    if any(e in _RUNNERS_NEED_DATA for e in chosen):
        t0 = time.perf_counter()
        stale_blocks, write_requests = load_dataset(dataset)
        print(f"Data ready in {time.perf_counter()-t0:.2f}s"
              f"  stale {stale_blocks.shape}  writes {write_requests.shape}\n")

    def _need(r, w):
        return r, w

    printers = {
        "5":  _print_exp5,
        "6":  _print_exp6,
        "7":  _print_exp7,
        "8":  _print_exp8,
        "9":  _print_exp9,
        "10": _print_exp10,
        "11": _print_exp11,
        "12": _print_exp12,
    }

    for exp_id in chosen:
        if exp_id == "5":
            rows = run_exp5_mlc(stale_blocks, write_requests, seed)
        elif exp_id == "6":
            rows = run_exp6(stale_blocks, write_requests, seed)
        elif exp_id == "7":
            rows = run_exp7_k(stale_blocks, write_requests, seed)
        elif exp_id == "8":
            rows = run_exp8_h(stale_blocks, write_requests, seed)
        elif exp_id == "9":
            rows = run_exp9_retrain(stale_blocks, write_requests, seed)
        elif exp_id == "10":
            rows = run_exp10_blocksize(dataset, seed)
        elif exp_id == "11":
            rows = run_exp11_algo(stale_blocks, write_requests, seed)
        elif exp_id == "12":
            rows = run_exp12_switch(seed)
        else:
            raise ValueError(exp_id)

        printers[exp_id](rows)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
