"""
Full 8-configuration experiment over all 15 datasets.

Configurations: RandSel / LearnWD  ×  DCW / DIN / DMPart / MinWD
Metrics (per dataset, normalised to RandSel+DCW baseline):
  Fig 13 — mean WD errors
  Fig 14 — mean write cost
  Fig 15 — mean write energy (nJ)
  Fig 16 — mean write latency (ns)

Usage:
    python3 run_all.py [seed]          # default seed = 0
    python3 run_all.py 0 --download    # (re)download datasets first
    python3 run_all.py 0 --fast        # 10 k writes (quick smoke-test)
"""

from __future__ import annotations
import json, sys, time
from pathlib import Path

import numpy as np

from config      import BLOCK_BITS, STALE_POOL_SIZE, WRITE_REQ_SIZE
from encoding    import dcw_encode, din_encode, dmpart_encode, minwd_encode
from selector    import randsel, make_learnwd_selector
from learnwd     import LearnWDModel
from simulation  import run_simulation

DATASETS_DIR = Path("datasets")
MANIFEST     = DATASETS_DIR / "manifest.json"

ENCODERS = {
    "DCW":    dcw_encode,
    "DIN":    din_encode,
    "DMPart": dmpart_encode,
    "MinWD":  minwd_encode,
}
DS_ORDER = ["AMZ", "BKS", "CES", "SPM", "WKM",
            "FMN", "MNI", "FRT", "RIE",
            "BEP", "TSC", "HNT", "IMD", "RTC", "INB"]

METRICS = [
    ("WD errors",   "mean_wd_errors",  "Fig 13"),
    ("Write cost",  "mean_write_cost", "Fig 14"),
    ("Energy (nJ)", "mean_energy_nJ",  "Fig 15"),
    ("Latency(ns)", "mean_latency_ns", "Fig 16"),
]


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _load_ds(code: str, n_writes: int) -> tuple[np.ndarray, np.ndarray]:
    arr = np.load(DATASETS_DIR / f"{code}.npy")        # (100_000, 512)
    stale  = arr[:STALE_POOL_SIZE]
    writes = arr[STALE_POOL_SIZE : STALE_POOL_SIZE + n_writes]
    return stale, writes


def _run(stale, writes, selector_fn, encoder, rng_seed, model=None):
    def _train(sm, st, _rng):
        if model: model.train(sm, st)
    rng = np.random.default_rng(rng_seed)
    return run_simulation(
        stale_blocks=stale,
        write_requests=writes,
        selector=selector_fn,
        encoder=encoder,
        rng=rng,
        init_hook=_train,
        retrain_hook=_train,
        on_write_fn=model.invalidate if model else None,
        progress_every=0,
    )


# ------------------------------------------------------------------ #
# Per-dataset 8-config run                                             #
# ------------------------------------------------------------------ #

def run_dataset(code: str, n_writes: int, seed: int) -> dict:
    """
    Run 8 configurations on one dataset.
    Returns dict keyed by config label → summary dict.
    """
    stale, writes = _load_ds(code, n_writes)
    results = {}

    for enc_name, encoder in ENCODERS.items():
        # ── RandSel ──────────────────────────────────────────────
        label = f"RandSel+{enc_name}"
        r = _run(stale, writes, randsel, encoder, seed)
        results[label] = r.summary()

        # ── LearnWD ──────────────────────────────────────────────
        label = f"LearnWD+{enc_name}"
        model    = LearnWDModel()
        selector = make_learnwd_selector(model)
        r = _run(stale, writes, selector, encoder, seed, model=model)
        results[label] = r.summary()

    return results


# ------------------------------------------------------------------ #
# Table printers                                                        #
# ------------------------------------------------------------------ #

CONFIG_LABELS = [
    "RandSel+DCW", "LearnWD+DCW",
    "RandSel+DIN", "LearnWD+DIN",
    "RandSel+DMPart", "LearnWD+DMPart",
    "RandSel+MinWD", "LearnWD+MinWD",
]

SHORT = {                        # short column headers
    "RandSel+DCW":    "RS+DCW",
    "LearnWD+DCW":    "LW+DCW",
    "RandSel+DIN":    "RS+DIN",
    "LearnWD+DIN":    "LW+DIN",
    "RandSel+DMPart": "RS+DMP",
    "LearnWD+DMPart": "LW+DMP",
    "RandSel+MinWD":  "RS+MWD",
    "LearnWD+MinWD":  "LW+MWD",
}


def _print_metric_table(
    all_results: dict,     # code → {label → summary}
    manifest:    dict,
    metric_key:  str,
    metric_name: str,
    figure_ref:  str,
) -> None:
    ds_codes = [c for c in DS_ORDER if c in all_results]
    col_w    = 8
    ds_col   = 5

    header = f"  {'DS':<{ds_col}}  {'Src':<4}  {'Type':<11}"
    for lbl in CONFIG_LABELS:
        header += f"  {SHORT[lbl]:>{col_w}}"
    print(f"\n{'─'*4} {figure_ref}: {metric_name} (norm to RandSel+DCW) {'─'*4}")
    print(header)
    print("  " + "─" * (len(header) - 2))

    for code in ds_codes:
        res    = all_results[code]
        base   = res["RandSel+DCW"][metric_key]
        m      = manifest.get(code, {})
        src    = "real" if m.get("source") == "real" else "~prx"
        ds_typ = m.get("type", "")[:11]

        row = f"  {code:<{ds_col}}  {src:<4}  {ds_typ:<11}"
        for lbl in CONFIG_LABELS:
            val  = res[lbl][metric_key]
            norm = val / base if base else float("nan")
            row += f"  {norm:>{col_w}.4f}"
        print(row)

    print("  " + "─" * (len(header) - 2))
    print(f"  (< 1 is better  |  RandSel+DCW = 1.000 per dataset)")


def print_all_tables(all_results: dict, manifest: dict) -> None:
    for metric_name, metric_key, fig_ref in METRICS:   # name, key, fig
        _print_metric_table(all_results, manifest, metric_key, metric_name, fig_ref)


# ------------------------------------------------------------------ #
# Main                                                                  #
# ------------------------------------------------------------------ #

def main() -> None:
    seed    = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    fast    = "--fast"     in sys.argv
    dl_flag = "--download" in sys.argv

    if dl_flag or not MANIFEST.exists():
        from download_datasets import download_all
        download_all()

    if not MANIFEST.exists():
        raise FileNotFoundError("Run download_datasets.py first.")

    manifest = json.loads(MANIFEST.read_text())
    available = [c for c in DS_ORDER if (DATASETS_DIR / f"{c}.npy").exists()]

    n_writes = 10_000 if fast else WRITE_REQ_SIZE
    print(f"\n=== LearnWD Full Experiment"
          f"  datasets={len(available)}  writes={n_writes}  seed={seed} ===\n")

    all_results: dict[str, dict] = {}
    t_total = time.perf_counter()

    for i, code in enumerate(available):
        m    = manifest.get(code, {})
        src  = "real" if m.get("source") == "real" else "proxy"
        print(f"[{i+1:02d}/{len(available)}] {code} — {m.get('name',code)}"
              f"  ({src}  density={m.get('bit_density',0):.3f})")
        t0 = time.perf_counter()
        all_results[code] = run_dataset(code, n_writes, seed)
        elapsed = time.perf_counter() - t0

        # Quick per-dataset summary line
        base_wd = all_results[code]["RandSel+DCW"]["mean_wd_errors"]
        lw_wd   = all_results[code]["LearnWD+DCW"]["mean_wd_errors"]
        norm    = lw_wd / base_wd if base_wd else float("nan")
        print(f"   {elapsed:.0f}s  RandSel+DCW WD={base_wd:.3f}  "
              f"LearnWD+DCW WD={lw_wd:.3f}  (norm {norm:.3f})\n")

        # Incremental save after every dataset (crash-safe)
        Path("results_all.json").write_text(
            json.dumps(all_results, indent=2))

    total_elapsed = time.perf_counter() - t_total
    print(f"Total time: {total_elapsed/60:.1f} min\n")
    print("=" * 90)
    print_all_tables(all_results, manifest)
    print("\nRaw results saved to results_all.json")


if __name__ == "__main__":
    main()
