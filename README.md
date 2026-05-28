# LearnWD-Advanced

An extended PCM (Phase Change Memory) Write Disturbance simulation framework, building upon the [LearnWD](https://github.com/lrh2000/learnwd) paper with significant engineering improvements and experimental coverage.

> **Status**: Active development — Reinforcement Learning enhanced selector coming soon.

---

## Overview

Write Disturbance (WD) is a key reliability challenge in PCM/NVM: when a RESET operation (1→0) is performed on a cell, the thermal pulse disturbs adjacent idle-zero cells, potentially flipping them. LearnWD addresses this by intelligently selecting *stale blocks* whose bit patterns minimise WD interactions with the incoming data.

This framework reproduces and extends the LearnWD paper with:

- **16 real-world datasets** spanning numerical, multimedia, and textual workloads
- **8 encoder × selector configurations** for comprehensive comparison
- **MLC (Multi-Level Cell) support** — 2 bits/cell with pattern-aware WD simulation
- **12 sensitivity experiments** covering ECC, clustering parameters, block size, retrain frequency, and dataset switching
- **Fully vectorised numpy implementation** (~100× faster than the original prototype)
- **Reproducible experiments** with fixed RNG seeds

---

## Results Summary

All metrics normalised to `RandSel + DCW` baseline (lower is better for WD errors):

| Configuration | WD Errors | Write Cost | Energy | Latency |
|--------------|-----------|------------|--------|---------|
| RandSel + DCW (baseline) | 1.000 | 1.000 | 1.000 | 1.000 |
| LearnWD + DCW | 0.543 | 0.697 | 0.835 | 0.837 |
| RandSel + MinWD | 0.053 | 1.982 | 3.072 | 3.088 |
| **LearnWD + MinWD** | **0.036** | **1.877** | **2.693** | **2.706** |

Full results across all 16 datasets and 12 sensitivity experiments: see [`RESULTS.md`](RESULTS.md).

---

## Project Structure

```
learnWD-advanced/
├── config.py            # PCM physical parameters (Table II from paper)
├── pcm_sim.py           # WD-prone counting, error sampling, latency/energy (SLC + MLC)
├── learnwd.py           # LearnWD four-component model
├── stale_table.py       # Stale pool data structure with O(1) cluster index
├── encoding.py          # DCW / DIN / DMPart / MinWD encoders
├── selector.py          # RandSel and LearnWD selector interfaces
├── simulation.py        # Main simulation loop (pluggable selector + encoder)
├── data_loader.py       # Dataset loading and block packing
├── experiments.py       # Sensitivity experiments Exp#5–12
├── run.py               # 8-config comparison on a single dataset
├── run_all.py           # Full 16-dataset × 8-config experiment
├── run_sensitivity.py   # Sensitivity experiment runner (Exp#5–12)
├── download_datasets.py # Dataset download and .npy cache generation
├── RESULTS.md           # Full experimental results
├── CODE_GUIDE.md        # Detailed code documentation
└── SPEC.md              # Implementation specification
```

---

## Datasets

16 real-world datasets across three categories (all pre-converted to 512-bit blocks):

| Code | Dataset | Type | Bit Density |
|------|---------|------|-------------|
| AMZ | Amazon Access Samples | Numerical | 0.298 |
| BKS | Bike Sharing | Numerical | 0.233 |
| CES | Census Income | Numerical | 0.109 |
| SPM | Spambase | Numerical | 0.102 |
| WKM | Wikipedia Math | Numerical | 0.108 |
| FMN | Fashion-MNIST | Multimedia | 0.094 |
| MNI | MNIST | Multimedia | 0.041 |
| FRT | Fruit Images | Multimedia | 0.487 |
| RIE | Rice Images | Multimedia | 0.427 |
| BEP | Brazilian E-Commerce | Textual | 0.412 |
| TSC | Twitter Sentiment | Textual | 0.446 |
| HNT | Health News | Textual | 0.453 |
| IMD | IMDb Reviews | Textual | 0.452 |
| RTC | Reuters | Textual | 0.453 |
| INB | Simple Wikipedia | Textual | 0.447 |
| KNS | Kensho Wikipedia | Textual | 0.453 |

Download and preprocess all datasets:
```bash
python3 download_datasets.py
```

---

## Installation

```bash
git clone https://github.com/wentao-hue/learnWD-advanced.git
cd learnWD-advanced
pip install numpy scikit-learn
```

---

## Usage

### Single dataset — 8 configurations

```bash
python3 run.py MNI        # MNIST, all 8 encoder×selector configs
python3 run.py INB 0 dcw  # Infobox, seed=0, DCW only
```

### Full 16-dataset experiment

```bash
python3 run_all.py
```

### Sensitivity experiments (Exp#5–12)

```bash
python3 run_sensitivity.py MNI 0 all   # all experiments on MNIST
python3 run_sensitivity.py MNI 0 5     # Exp#5: MLC vs SLC
python3 run_sensitivity.py MNI 0 6     # Exp#6: ECC / VnR
python3 run_sensitivity.py MNI 0 7     # Exp#7: k sensitivity
python3 run_sensitivity.py MNI 0 10    # Exp#10: block size sensitivity
```

---

## LearnWD Four-Component Model

```
new_block
    │
    ▼
① Aggressor Extractor       marks RESET-aggressor positions
    │
    ▼
② Cluster Selector          argmin dot(aggressor, centroid_k)
    │
    ▼
③ Similarity Estimator      MinHash nearest-neighbour in cluster
    │
    ▼
stale_addr  ──►  write new_block over stale_memory[stale_addr]
```

The model retrains every 20,000 writes using MiniBatchKMeans on disturbance vectors of current stale blocks. MLC mode uses pattern-weighted disturbance vectors (256-dim) and `patternAgg`-based aggressor extraction.

---

## Experiments

| Exp | Description | Key Finding |
|-----|------------|-------------|
| #1–4 | 8-config comparison (SLC) | LearnWD+MinWD: −96.4% WD errors |
| #5 | MLC vs SLC | LearnWD effective on MLC; −32% WD errors |
| #6 | ECC / VnR levels | LearnWD+ECC-8: −75.1% VnR operations |
| #7 | Cluster count k | k=8 optimal for MNIST; k=16 most robust |
| #8 | MinHash functions h | h=8 sweet spot; h=0 (no MinHash) −33% worse |
| #9 | Retrain interval | 20k writes optimal; 50k shows degradation |
| #10 | Block size | LearnWD generalises from 64B to 4KB |
| #11 | Clustering algorithm | k-means ≈ GMM > BIRCH |
| #12 | Dataset switching | LearnWD adapts across workload changes |

---

## Differences from Original LearnWD

| Aspect | Original | This Work |
|--------|---------|-----------|
| Dataset | Small hex sample files | 16 real datasets (100k blocks each) |
| Clustering | Batch KMeans | MiniBatchKMeans + GMM + BIRCH |
| MLC support | Feature extractors only | Full simulation + LearnWD features |
| MinHash | Pure Python loops | Vectorised numpy (~100× faster) |
| Stale pool | O(N) array deletion | O(1) bool-mask invalidation |
| Retrain | Online centroid update | Periodic full retrain every 20k writes |
| Reproducibility | Global numpy RNG | Seeded `np.random.default_rng` |

---

## Reference

This project extends the following work:

```
@inproceedings{learnwd,
  title     = {LearnWD: Mitigating Write Disturbance in PCM via Machine Learning},
  booktitle = {IEEE/ACM International Symposium on Microarchitecture (MICRO)},
  year      = {2023}
}
```

---

## License

MIT
