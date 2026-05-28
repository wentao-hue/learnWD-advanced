"""
Experiment runner — Steps 1-4.

Usage:
    python3 run.py [dataset] [seed] [encoders]

    dataset  : mnist | synthetic | raw:<path>     (default: mnist)
    seed     : integer RNG seed                    (default: 0)
    encoders : comma-separated subset of dcw,dmpart,minwd,all  (default: all)

Example:
    python3 run.py mnist 0 all
    python3 run.py mnist 0 dcw,minwd

Output: per-metric table normalised to RandSel+DCW baseline.
"""

import sys
import time
from typing import Callable

import numpy as np

from data_loader import load_dataset
from encoding    import dcw_encode, dmpart_encode, minwd_encode, din_encode
from selector    import randsel, make_learnwd_selector
from learnwd     import LearnWDModel
from simulation  import run_simulation, SimResult


# ------------------------------------------------------------------ #
# Configuration registry                                              #
# ------------------------------------------------------------------ #

EncoderFn  = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]

ENCODERS: dict[str, EncoderFn] = {
    "dcw":    dcw_encode,
    "dmpart": dmpart_encode,
    "minwd":  minwd_encode,
    "din":    din_encode,
}


def _run_randsel(
    encoder_name:   str,
    encoder:        EncoderFn,
    stale_blocks:   np.ndarray,
    write_requests: np.ndarray,
    seed:           int,
) -> SimResult:
    rng = np.random.default_rng(seed)
    return run_simulation(
        stale_blocks   = stale_blocks,
        write_requests = write_requests,
        selector       = randsel,
        encoder        = encoder,
        rng            = rng,
        progress_every = 0,
    )


def _run_learnwd(
    encoder_name:   str,
    encoder:        EncoderFn,
    stale_blocks:   np.ndarray,
    write_requests: np.ndarray,
    seed:           int,
) -> SimResult:
    model    = LearnWDModel()
    selector = make_learnwd_selector(model)

    def _train(stale_memory, stale_table, rng):
        model.train(stale_memory, stale_table)

    rng = np.random.default_rng(seed)
    return run_simulation(
        stale_blocks   = stale_blocks,
        write_requests = write_requests,
        selector       = selector,
        encoder        = encoder,
        rng            = rng,
        init_hook      = _train,
        retrain_hook   = _train,
        on_write_fn    = model.invalidate,
        progress_every = 0,
    )


# ------------------------------------------------------------------ #
# Reporting                                                           #
# ------------------------------------------------------------------ #

METRICS = [
    ("WD errors",    "mean_wd_errors",  "{:8.4f}"),
    ("WD prone",     "mean_wd_prone",   "{:8.4f}"),
    ("Write cost",   "mean_write_cost", "{:8.2f}"),
    ("Energy (nJ)",  "mean_energy_nJ",  "{:10.6f}"),
    ("Latency (ns)", "mean_latency_ns", "{:10.2f}"),
]


def _print_table(
    results: dict[str, dict],   # label → summary dict
    baseline_label: str,
) -> None:
    labels   = list(results.keys())
    baseline = results[baseline_label]

    col_w = max(len(L) for L in labels) + 2

    # Header
    header_fields = ["Encoder+Selector".ljust(col_w)]
    for name, _, _ in METRICS:
        header_fields.append(f"{name:>14}")
    print("\n" + "  ".join(header_fields))
    print("─" * (col_w + len(METRICS) * 16))

    for label, s in results.items():
        row = [label.ljust(col_w)]
        for name, key, fmt in METRICS:
            norm = s[key] / baseline[key] if baseline[key] else float("nan")
            row.append(f"{norm:>14.4f}")
        print("  ".join(row))

    print("─" * (col_w + len(METRICS) * 16))
    print(f"  (normalised to '{baseline_label}' = 1.000  |  < 1 is better)\n")


# ------------------------------------------------------------------ #
# Main                                                                #
# ------------------------------------------------------------------ #

def main() -> None:
    dataset  = sys.argv[1] if len(sys.argv) > 1 else "mnist"
    seed     = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    enc_arg  = sys.argv[3] if len(sys.argv) > 3 else "all"

    if enc_arg == "all":
        chosen = list(ENCODERS.keys())
    else:
        chosen = [e.strip().lower() for e in enc_arg.split(",")]
        for e in chosen:
            if e not in ENCODERS:
                raise ValueError(f"Unknown encoder '{e}'. Choose from {list(ENCODERS)}")

    print(f"\n=== LearnWD Exp  dataset={dataset}  seed={seed}"
          f"  encoders={chosen} ===\n")

    # ---------------------------------------------------------------- #
    # Load data                                                         #
    # ---------------------------------------------------------------- #
    t0 = time.perf_counter()
    stale_blocks, write_requests = load_dataset(dataset)
    print(f"Data ready in {time.perf_counter()-t0:.2f}s"
          f"  stale {stale_blocks.shape}  writes {write_requests.shape}\n")

    # ---------------------------------------------------------------- #
    # Run all requested configurations                                  #
    # ---------------------------------------------------------------- #
    results: dict[str, dict] = {}
    BASELINE = "DCW + RandSel"

    configs = []
    for enc_name in chosen:
        configs.append(("RandSel", enc_name, _run_randsel))
        configs.append(("LearnWD", enc_name, _run_learnwd))

    for selector_name, enc_name, run_fn in configs:
        label = f"{enc_name.upper()} + {selector_name}"
        print(f"── {label} ──")
        t1 = time.perf_counter()
        result = run_fn(enc_name, ENCODERS[enc_name],
                        stale_blocks, write_requests, seed)
        elapsed = time.perf_counter() - t1
        s = result.summary()
        results[label] = s
        print(
            f"   {elapsed:.1f}s  |  WD errors {s['mean_wd_errors']:.4f}  "
            f"cost {s['mean_write_cost']:.2f}  "
            f"energy {s['mean_energy_nJ']:.4f} nJ  "
            f"retrains {s['n_retrains']}\n"
        )

    # ---------------------------------------------------------------- #
    # Table (normalised to RandSel+DCW)                                 #
    # ---------------------------------------------------------------- #
    if BASELINE not in results:
        # ensure baseline exists for normalisation even if not in chosen
        print(f"Running baseline {BASELINE} for normalisation …")
        t_b = time.perf_counter()
        r_b = _run_randsel("dcw", dcw_encode, stale_blocks, write_requests, seed)
        results[BASELINE] = r_b.summary()
        print(f"   {time.perf_counter()-t_b:.1f}s\n")

    # Reorder: baseline first
    ordered = {BASELINE: results[BASELINE]}
    ordered.update({k: v for k, v in results.items() if k != BASELINE})

    print("=" * 70)
    print("RESULTS  (normalised to RandSel+DCW)")
    print("=" * 70)
    _print_table(ordered, BASELINE)


if __name__ == "__main__":
    main()
