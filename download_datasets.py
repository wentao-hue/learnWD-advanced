"""
Download and pack all 15 LearnWD datasets into (100_000, 512) uint8 bit arrays.

Fallback chain per dataset:
  primary loader → synthetic proxy (type-matched + distinct bit density)

Real sources status (auto-detected at runtime):
  MNI  MNIST              sklearn openml 554
  FMN  Fashion-MNIST      sklearn openml 40996
  SPM  Spambase           ucimlrepo 94          (all-numeric)
  WKM  → Covertype        sklearn openml 150     (Kaggle-only original → sub)
  AMZ  Amazon Access      UCI zip→tgz→csv        (mixed binary)
  BKS  Bike Sharing       ucimlrepo 275 numeric  (drop date cols)
  CES  Census Income      ucimlrepo 20 numeric   (label-encode cats)
  IMD  → stanfordnlp/imdb HuggingFace Parquet
  TSC  → cardiffnlp/tweet_eval HuggingFace
  HNT  → ag_news          HuggingFace (health-news-like)
  RTC  → SetFit/ag_news   HuggingFace Parquet
  Kaggle-only (FRT,RIE,BEP,INB): typed synthetic proxy
"""

from __future__ import annotations
import io, json, ssl, tarfile, time, traceback, zipfile
import urllib.request
from pathlib import Path

import numpy as np

BLOCK_BITS = 512
N_TOTAL    = 100_000
OUT_DIR    = Path("datasets")
MANIFEST   = OUT_DIR / "manifest.json"

OUT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────── #
# Core helpers                                                          #
# ─────────────────────────────────────────────────────────────────── #

def _tile_to_blocks(bits: np.ndarray, n: int = N_TOTAL,
                    bb: int = BLOCK_BITS) -> np.ndarray:
    needed = n * bb
    if len(bits) < needed:
        bits = np.tile(bits, -(-needed // len(bits)))
    return bits[:needed].reshape(n, bb).astype(np.uint8)


def _bytes_to_blocks(raw: bytes | bytearray) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))
    return _tile_to_blocks(bits)


def _numeric_to_blocks(X) -> np.ndarray:
    import pandas as pd
    if hasattr(X, 'to_numpy'):
        X = X.to_numpy()
    arr = np.asarray(X, dtype=np.float32)
    return _bytes_to_blocks(arr.ravel().tobytes())


def _df_numeric_only(df) -> np.ndarray:
    """Select only numeric columns, label-encode object columns."""
    import pandas as pd
    out = df.copy()
    for col in out.select_dtypes(include='object').columns:
        out[col] = out[col].astype('category').cat.codes.astype(np.float32)
    numeric = out.select_dtypes(include='number')
    return _numeric_to_blocks(numeric)


def _text_to_blocks(texts) -> np.ndarray:
    raw = "\n".join(str(t) for t in texts).encode("utf-8", errors="replace")
    return _bytes_to_blocks(raw)


# ─────────────────────────────────────────────────────────────────── #
# Typed synthetic proxies — each with distinct bit-density profile     #
# ─────────────────────────────────────────────────────────────────── #

_PROXY_PARAMS = {
    # code: (density, run_len)  — None run_len = IID
    "AMZ": (0.15,  None),   # sparse binary access control
    "BKS": (0.46,  None),   # floating-point time series
    "CES": (0.43,  None),   # mixed numerical/categorical
    "FRT": (0.48,   20),    # dense colour images
    "RIE": (0.44,   24),    # grayscale rice images
    "BEP": (0.55,    8),    # e-commerce text / CSV
    "TSC": (0.57,    6),    # tweets
    "HNT": (0.54,    7),    # health news
    "IMD": (0.58,    5),    # movie reviews
    "RTC": (0.53,    9),    # Reuters wire text
    "INB": (0.52,   10),    # Wikipedia infobox text
}


def _proxy(code: str, seed: int = 99) -> np.ndarray:
    """
    Fast vectorised synthetic proxy using geometric run-length encoding.
    Produces correlated bits (if run is set) with the given bit density.
    """
    density, run = _PROXY_PARAMS.get(code, (0.48, None))
    rng   = np.random.default_rng(seed)
    total = N_TOTAL * BLOCK_BITS

    if run is None:
        bits = (rng.random(total) < density).astype(np.uint8)
        return bits.reshape(N_TOTAL, BLOCK_BITS)

    # Markov chain: alternating runs of 0s and 1s
    # steady-state density d = p01/(p01+p10)
    # mean run of 1s = 1/p10 = run  →  p10 = 1/run
    # → p01 = density * p10 / (1 - density)
    p10 = 1.0 / run
    p01 = min(density * p10 / max(1.0 - density, 1e-9), 1.0)

    # Estimate number of alternating runs needed (3× for safety)
    mean_run = 0.5 * (1.0 / p01 + 1.0 / p10)
    n_runs   = max(int(total / mean_run * 3), 2000)

    # Geometric run lengths for state-0 and state-1 alternately
    r0 = rng.geometric(p01, n_runs)  # runs of 0s
    r1 = rng.geometric(p10, n_runs)  # runs of 1s

    interleaved = np.empty(2 * n_runs, dtype=np.int64)
    interleaved[0::2] = r0
    interleaved[1::2] = r1
    states = np.empty(2 * n_runs, dtype=np.uint8)
    states[0::2] = 0
    states[1::2] = 1

    # Clip to required total
    cum = np.cumsum(interleaved)
    needed = np.searchsorted(cum, total, side='left') + 1
    lengths = interleaved[:needed].copy()
    if cum[needed - 1] > total:
        lengths[-1] -= int(cum[needed - 1] - total)

    bits = np.repeat(states[:needed], lengths)[:total]
    return bits.astype(np.uint8).reshape(N_TOTAL, BLOCK_BITS)


# ─────────────────────────────────────────────────────────────────── #
# Individual real loaders                                              #
# ─────────────────────────────────────────────────────────────────── #

def _load_MNI():
    from sklearn.datasets import fetch_openml
    print("    openml mnist_784 …")
    d = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    return _numeric_to_blocks(d.data)


def _load_FMN():
    from sklearn.datasets import fetch_openml
    print("    openml Fashion-MNIST id=40996 …")
    d = fetch_openml(data_id=40996, as_frame=False, parser="auto")
    return _numeric_to_blocks(d.data)


def _load_SPM():
    from ucimlrepo import fetch_ucirepo
    print("    ucimlrepo Spambase id=94 …")
    d = fetch_ucirepo(id=94)
    return _numeric_to_blocks(d.data.features.values)


def _load_WKM():
    """Covertype (openml 150) as numerical substitute for Kaggle WKM."""
    from sklearn.datasets import fetch_openml
    print("    openml Covertype id=150 (WKM sub) …")
    d = fetch_openml(data_id=150, as_frame=False, parser="auto")
    return _numeric_to_blocks(d.data)


def _load_AMZ():
    """UCI Amazon Access Samples: zip → tgz → csv."""
    url = ("https://archive.ics.uci.edu/static/public/216/"
           "amazon+access+samples.zip")
    print(f"    UCI direct {url} …")
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(url, timeout=120, context=ctx) as r:
        zip_data = r.read()
    with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
        tgz_names = [n for n in zf.namelist() if n.endswith(".tgz")]
        if not tgz_names:
            raise FileNotFoundError(f"No .tgz in zip: {zf.namelist()}")
        tgz_bytes = zf.read(tgz_names[0])
    with tarfile.open(fileobj=io.BytesIO(tgz_bytes), mode="r:gz") as tf:
        csv_members = [m for m in tf.getmembers()
                       if m.name.lower().endswith((".csv", ".data"))]
        if not csv_members:
            raise FileNotFoundError(f"No csv/data in tgz: {[m.name for m in tf.getmembers()]}")
        raw = tf.extractfile(csv_members[0]).read()
    return _bytes_to_blocks(raw)


def _load_BKS():
    from ucimlrepo import fetch_ucirepo
    print("    ucimlrepo Bike Sharing id=275 (numeric cols only) …")
    d = fetch_ucirepo(id=275)
    return _df_numeric_only(d.data.features)


def _load_CES():
    from ucimlrepo import fetch_ucirepo
    print("    ucimlrepo Census Income id=20 (label-encode cats) …")
    d = fetch_ucirepo(id=20)
    return _df_numeric_only(d.data.features)


def _load_IMD():
    from datasets import load_dataset
    print("    HuggingFace stanfordnlp/imdb …")
    ds = load_dataset("stanfordnlp/imdb", split="train+test")
    return _text_to_blocks(ds["text"])


def _load_TSC():
    from datasets import load_dataset
    print("    HuggingFace cardiffnlp/tweet_eval/sentiment …")
    ds = load_dataset("cardiffnlp/tweet_eval", "sentiment",
                      split="train+validation+test")
    return _text_to_blocks(ds["text"])


def _load_HNT():
    """ag_news as health-news-like substitute."""
    from datasets import load_dataset
    print("    HuggingFace fancyzhx/ag_news (HNT sub) …")
    ds = load_dataset("fancyzhx/ag_news", split="train+test")
    return _text_to_blocks(ds["text"])


def _load_RTC():
    from datasets import load_dataset
    print("    HuggingFace SetFit/ag_news (RTC sub) …")
    ds = load_dataset("SetFit/ag_news", split="train+test")
    return _text_to_blocks(ds["text"])


def _load_FRT():
    """Fruits 360 image dataset via kagglehub (moltean/fruits)."""
    import kagglehub
    print("    kagglehub moltean/fruits …")
    path = Path(kagglehub.dataset_download("moltean/fruits"))
    files = sorted(
        f for f in path.rglob("*")
        if f.suffix.lower() in (".jpg", ".jpeg", ".png") and f.is_file()
    )
    if not files:
        raise FileNotFoundError(f"No image files found under {path}")
    # Sample up to 3000 images to keep memory reasonable (~30 MB raw JPEG)
    rng_s = np.random.default_rng(17)
    if len(files) > 3000:
        idx   = rng_s.choice(len(files), 3000, replace=False)
        files = [files[i] for i in sorted(idx)]
    print(f"    reading {len(files)} image files …")
    raw = b"".join(f.read_bytes() for f in files)
    return _bytes_to_blocks(raw)


def _load_RIE():
    """Rice Image Dataset via kagglehub (muratkokludataset/rice-image-dataset)."""
    import kagglehub
    print("    kagglehub muratkokludataset/rice-image-dataset …")
    path = Path(kagglehub.dataset_download("muratkokludataset/rice-image-dataset"))
    files = sorted(
        f for f in path.rglob("*")
        if f.suffix.lower() in (".jpg", ".jpeg", ".png") and f.is_file()
    )
    if not files:
        raise FileNotFoundError(f"No image files found under {path}")
    rng_s = np.random.default_rng(18)
    if len(files) > 3000:
        idx   = rng_s.choice(len(files), 3000, replace=False)
        files = [files[i] for i in sorted(idx)]
    print(f"    reading {len(files)} image files …")
    raw = b"".join(f.read_bytes() for f in files)
    return _bytes_to_blocks(raw)


def _load_BEP():
    """Brazilian E-Commerce (Olist) via kagglehub (olistbr/brazilian-ecommerce)."""
    import kagglehub
    import pandas as pd
    print("    kagglehub olistbr/brazilian-ecommerce …")
    path = Path(kagglehub.dataset_download("olistbr/brazilian-ecommerce"))
    csvs = sorted(path.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found under {path}")
    # Prefer the reviews CSV (richest text); fall back to all CSVs
    review_csvs = [c for c in csvs if "review" in c.name.lower()]
    chosen = review_csvs if review_csvs else csvs
    frames = []
    for csv in chosen:
        try:
            frames.append(pd.read_csv(csv))
        except Exception:
            pass
    df = pd.concat(frames, ignore_index=True)
    # Extract text columns first; fall back to numeric
    text_cols = df.select_dtypes(include="object").columns.tolist()
    if text_cols:
        texts = df[text_cols].fillna("").astype(str).apply(
            lambda row: " ".join(row), axis=1
        ).tolist()
        return _text_to_blocks(texts)
    return _df_numeric_only(df)


def _load_INB():
    """Simple English Wikipedia via HuggingFace — infobox/article text substitute."""
    from datasets import load_dataset
    print("    HuggingFace wikimedia/wikipedia 20231101.simple (INB sub) …")
    ds = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train")
    return _text_to_blocks(ds["text"])


def _load_KNS():
    """Kensho-derived Wikimedia data — read directly from local datasets/kensho.zip.

    Streams link_annotated_text.jsonl line-by-line (no full extraction needed).
    Stops once 5 000 non-stub text sections have been collected.
    """
    import json as _json
    zip_path = OUT_DIR / "kensho.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"kensho.zip not found at {zip_path}")
    print(f"    streaming {zip_path} → link_annotated_text.jsonl …")
    texts: list[str] = []
    with zipfile.ZipFile(zip_path) as z:
        with z.open("link_annotated_text.jsonl") as f:
            for raw_line in f:
                try:
                    obj = _json.loads(raw_line)
                except Exception:
                    continue
                for sec in obj.get("sections", []):
                    t = sec.get("text", "").strip()
                    if len(t) > 50:          # skip one-liner stub sections
                        texts.append(t)
                if len(texts) >= 5000:
                    break
    if not texts:
        raise ValueError("No text sections found in kensho.zip")
    print(f"    collected {len(texts)} sections → converting to blocks …")
    return _text_to_blocks(texts)


# ─────────────────────────────────────────────────────────────────── #
# Registry                                                              #
# ─────────────────────────────────────────────────────────────────── #

REGISTRY = {
    "AMZ": {"name": "Amazon Access",     "type": "numerical",  "loader": _load_AMZ},
    "BKS": {"name": "Bike Sharing",      "type": "numerical",  "loader": _load_BKS},
    "CES": {"name": "Census Income",     "type": "numerical",  "loader": _load_CES},
    "SPM": {"name": "Spambase",          "type": "numerical",  "loader": _load_SPM},
    "WKM": {"name": "Wikipedia Math",    "type": "numerical",  "loader": _load_WKM},  # sub
    "FMN": {"name": "Fashion-MNIST",     "type": "multimedia", "loader": _load_FMN},
    "MNI": {"name": "MNIST",             "type": "multimedia", "loader": _load_MNI},
    "FRT": {"name": "Fruit Images",      "type": "multimedia", "loader": _load_FRT},
    "RIE": {"name": "Rice Images",       "type": "multimedia", "loader": _load_RIE},
    "BEP": {"name": "Brazilian E-Comm.", "type": "textual",    "loader": _load_BEP},
    "TSC": {"name": "Twitter Sentiment", "type": "textual",    "loader": _load_TSC},
    "HNT": {"name": "Health News",       "type": "textual",    "loader": _load_HNT},
    "IMD": {"name": "IMDb",              "type": "textual",    "loader": _load_IMD},
    "RTC": {"name": "Reuters",           "type": "textual",    "loader": _load_RTC},
    "INB": {"name": "Infobox",           "type": "textual",    "loader": _load_INB},
    "KNS": {"name": "Kensho Wikipedia",  "type": "textual",    "loader": _load_KNS},
}

ORDER = ["AMZ", "BKS", "CES", "SPM", "WKM",
         "FMN", "MNI", "FRT", "RIE",
         "BEP", "TSC", "HNT", "IMD", "RTC", "INB", "KNS"]


# ─────────────────────────────────────────────────────────────────── #
# Main                                                                  #
# ─────────────────────────────────────────────────────────────────── #

def download_all(force: bool = False) -> dict:
    manifest: dict = {}
    if MANIFEST.exists() and not force:
        manifest = json.loads(MANIFEST.read_text())

    seeds = {c: 10 + i for i, c in enumerate(ORDER)}

    for code in ORDER:
        npy_path = OUT_DIR / f"{code}.npy"
        if npy_path.exists() and code in manifest and not force:
            m = manifest[code]
            print(f"  [{code}] cached  source={m['source']}  "
                  f"density={m['bit_density']:.3f}")
            continue

        info = REGISTRY[code]
        print(f"  [{code}] {info['name']} ({info['type']}) …")
        t0     = time.perf_counter()
        blocks = None
        src    = "proxy"

        loader = info.get("loader")
        if loader is not None:
            try:
                blocks = loader()
                src = "real"
            except Exception as exc:
                print(f"    ⚠ {exc.__class__.__name__}: {exc}")
                print("    → synthetic proxy")

        if blocks is None:
            blocks = _proxy(code, seed=seeds[code])
            src    = "proxy"

        density = float(blocks.mean())
        elapsed = time.perf_counter() - t0
        np.save(npy_path, blocks)
        manifest[code] = {
            "name":        info["name"],
            "type":        info["type"],
            "source":      src,
            "bit_density": density,
        }
        MANIFEST.write_text(json.dumps(manifest, indent=2))
        flag = "✓" if src == "real" else "~"
        print(f"    {flag} {src:<6}  density={density:.3f}  {elapsed:.1f}s")

    # Summary
    print("\n── Dataset manifest ──────────────────────────────────────────")
    print(f"  {'':1}{'Code':<5} {'Name':<24} {'Type':<11} {'Src':<6} {'Density':>8}")
    print("  " + "─" * 58)
    for code in ORDER:
        m    = manifest[code]
        flag = "✓" if m["source"] == "real" else "~"
        print(f"  {flag}{code:<5} {m['name']:<24} {m['type']:<11} "
              f"{m['source']:<6} {m['bit_density']:>8.3f}")
    n_real  = sum(1 for m in manifest.values() if m["source"] == "real")
    n_proxy = len(manifest) - n_real
    print(f"\n  ✓ real/substituted ({n_real})   ~ synthetic proxy ({n_proxy})\n")
    return manifest


if __name__ == "__main__":
    import sys
    download_all(force="--force" in sys.argv)
