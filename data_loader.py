"""
Dataset loading and block packing.

Pipeline for every dataset:
  1. Load raw data as bytes
  2. numpy.unpackbits → bit stream, tiled if needed
  3. Slice into block_bits-wide blocks, discard remainder
  4. Return first (n_stale + n_writes) blocks

Supported sources
-----------------
  "mnist"              – MNIST via sklearn (downloads on first call)
  "synthetic"          – 50 % random bits (reproducible, seed 42)
  "synthetic_sparse"   – 5 % ones (simulates sparse/compressed data)
  "synthetic_dense"    – 90 % ones (simulates dense data)
  "synthetic_random"   – alias for "synthetic"
  "synthetic_alt"      – strict 0101… alternating pattern
  "synthetic_corr"     – locally-correlated blocks (runs of 0s and 1s)
  "raw:<path>"         – load any local file as raw bytes
"""

from __future__ import annotations
import numpy as np
from pathlib import Path
from config import BLOCK_BITS, STALE_POOL_SIZE, WRITE_REQ_SIZE


# ------------------------------------------------------------------ #
# Core helpers                                                         #
# ------------------------------------------------------------------ #

def _tile_to(bits: np.ndarray, n: int) -> np.ndarray:
    """Return a length-n view/copy of bits, tiling if bits is shorter."""
    if len(bits) >= n:
        return bits[:n]
    reps = -(-n // len(bits))          # ceil division
    return np.tile(bits, reps)[:n]


def bytes_to_blocks(
    data: bytes | np.ndarray,
    block_bits: int = BLOCK_BITS,
    n_blocks: int | None = None,
) -> np.ndarray:
    """
    Convert a raw byte sequence to (N, block_bits) uint8 blocks (values 0/1).
    If n_blocks is given, tile the bit stream to produce exactly that many blocks.
    Trailing bits that don't fill a full block are dropped otherwise.
    """
    if isinstance(data, np.ndarray):
        data = data.astype(np.uint8).tobytes()
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))

    if n_blocks is not None:
        bits = _tile_to(bits, n_blocks * block_bits)
        return bits.reshape(n_blocks, block_bits)

    n_full = len(bits) // block_bits
    return bits[: n_full * block_bits].reshape(n_full, block_bits)


# ------------------------------------------------------------------ #
# Loaders                                                              #
# ------------------------------------------------------------------ #

def _load_mnist(block_bits: int, n_blocks: int) -> np.ndarray:
    try:
        from sklearn.datasets import fetch_openml
        print("Loading MNIST via sklearn …")
        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        raw = mnist.data.astype(np.uint8)          # (70000, 784)
        return bytes_to_blocks(raw.tobytes(), block_bits=block_bits,
                                n_blocks=n_blocks)
    except Exception as e:
        raise RuntimeError(f"Could not load MNIST: {e}") from e


def _load_synthetic(
    density: float,
    block_bits: int,
    n_blocks: int,
    seed: int = 42,
) -> np.ndarray:
    """Generate n_blocks random blocks with the given ones density."""
    rng = np.random.default_rng(seed)
    bits = rng.random(n_blocks * block_bits) < density
    return bits.reshape(n_blocks, block_bits).astype(np.uint8)


def _load_synthetic_alt(block_bits: int, n_blocks: int) -> np.ndarray:
    """Strict 0101… alternating pattern, same for every block."""
    pattern = np.arange(block_bits, dtype=np.uint8) % 2           # 0,1,0,1,...
    return np.tile(pattern, (n_blocks, 1))


def _load_synthetic_corr(
    block_bits: int,
    n_blocks: int,
    run_len: int = 32,
    seed: int = 42,
) -> np.ndarray:
    """
    Locally-correlated: sequence of random runs of 0s and 1s with mean
    run length ~run_len.  Built with vectorised cumulative-XOR approach.
    """
    rng = np.random.default_rng(seed)
    total_bits = n_blocks * block_bits
    # Each position is a run-start with probability 1/run_len;
    # cumulative parity gives the current bit value.
    starts = (rng.random(total_bits) < (1.0 / run_len)).astype(np.uint8)
    bits   = np.cumsum(starts, dtype=np.int32) % 2
    return bits.reshape(n_blocks, block_bits).astype(np.uint8)


def _load_raw_file(path: str, block_bits: int, n_blocks: int) -> np.ndarray:
    data = Path(path).read_bytes()
    return bytes_to_blocks(data, block_bits=block_bits, n_blocks=n_blocks)


# ------------------------------------------------------------------ #
# Public API                                                           #
# ------------------------------------------------------------------ #

def load_dataset(
    name: str,
    block_bits: int = BLOCK_BITS,
    n_stale: int = STALE_POOL_SIZE,
    n_writes: int = WRITE_REQ_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a dataset and return (stale_blocks, write_requests).

    stale_blocks   : shape (n_stale, block_bits)  uint8
    write_requests : shape (n_writes, block_bits) uint8

    The raw bit stream is tiled if the source doesn't have enough data.
    """
    name    = name.strip()
    n_total = n_stale + n_writes

    if name == "mnist":
        blocks = _load_mnist(block_bits, n_total)

    elif name in ("synthetic", "synthetic_random"):
        blocks = _load_synthetic(0.50, block_bits, n_total, seed=42)

    elif name == "synthetic_sparse":
        blocks = _load_synthetic(0.05, block_bits, n_total, seed=43)

    elif name == "synthetic_dense":
        blocks = _load_synthetic(0.90, block_bits, n_total, seed=44)

    elif name == "synthetic_alt":
        blocks = _load_synthetic_alt(block_bits, n_total)

    elif name == "synthetic_corr":
        blocks = _load_synthetic_corr(block_bits, n_total)

    elif name.startswith("raw:"):
        blocks = _load_raw_file(name[4:], block_bits, n_total)

    else:
        # Try loading from datasets/<NAME>.npy (e.g. "MNI", "INB", "KNS" …)
        npy_path = Path("datasets") / f"{name}.npy"
        if npy_path.exists():
            arr = np.load(npy_path)             # (N, 512) uint8
            if arr.shape[1] != block_bits:
                raise ValueError(
                    f"Cached .npy for '{name}' has {arr.shape[1]} bits/block; "
                    f"requested {block_bits}."
                )
            bits_flat = arr.reshape(-1)
            bits = _tile_to(bits_flat, n_total * block_bits)
            blocks = bits[:n_total * block_bits].reshape(n_total, block_bits)
        else:
            raise ValueError(f"Unknown dataset: '{name}'")

    return blocks[:n_stale], blocks[n_stale:n_total]
