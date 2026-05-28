"""
Write encoding schemes.

Each encoder satisfies the interface:
    encoder(new_block, stale_block) -> (encoded_block, write_mask)

  encoded_block : (BLOCK_BITS,) uint8 — data actually stored in PCM
  write_mask    : (BLOCK_BITS,) uint8 — bits physically written
                  Always encoded_block XOR stale_block (differential write).

Implemented
-----------
  dcw_encode    — Data Comparison Write              (Step 1)
  dmpart_encode — Data Manipulation by Partitioning  (Step 4)
  minwd_encode  — Minimum WD-prone (4 transforms)    (Step 4)
  din_encode    — Data-Informed Narrow (FPC+BCH)      (Step 5)
"""

from __future__ import annotations
import numpy as np
from config import BLOCK_BITS

# ─────────────────────────────────────────────────────────────────── #
# DCW                                                                  #
# ─────────────────────────────────────────────────────────────────── #

def dcw_encode(
    new_block:   np.ndarray,
    stale_block: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Differential write: only write bits that changed."""
    write_mask = (new_block ^ stale_block).astype(np.uint8)
    return new_block, write_mask


# ─────────────────────────────────────────────────────────────────── #
# DMPart                                                               #
# ─────────────────────────────────────────────────────────────────── #

def dmpart_encode(
    new_block:   np.ndarray,
    stale_block: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Data Manipulation by Partitioning.
    Splits into 256 × 2-bit partitions, XORs all with the least-frequent
    2-bit pattern so the rarest pattern maps to 00.
    2-bit mask stored as external side-channel metadata (excluded from block).
    """
    parts   = new_block.reshape(256, 2)
    pat_idx = (parts[:, 0] << 1) | parts[:, 1]          # (256,) in {0,1,2,3}
    mask_int  = int(np.argmin(np.bincount(pat_idx, minlength=4)))
    mask_bits = np.array([mask_int >> 1, mask_int & 1], dtype=np.uint8)
    encoded_block = (parts ^ mask_bits).reshape(BLOCK_BITS).astype(np.uint8)
    write_mask    = (encoded_block ^ stale_block).astype(np.uint8)
    return encoded_block, write_mask


# ─────────────────────────────────────────────────────────────────── #
# MinWD helpers                                                        #
# ─────────────────────────────────────────────────────────────────── #

def _wd_prone_batch(candidates: np.ndarray,
                    stale_block: np.ndarray) -> np.ndarray:
    """
    Vectorised WD-prone count for C candidates vs one stale block.
    candidates  : (C, BLOCK_BITS) uint8
    stale_block : (BLOCK_BITS,)   uint8
    returns     : (C,) int32
    """
    stale = stale_block[np.newaxis, :]
    wm    = (candidates ^ stale).astype(np.uint8)
    resets = (wm == 1) & (candidates == 0)
    idle_z = (stale_block == 0)[np.newaxis, :] & (wm == 0)
    left   = np.zeros_like(idle_z, dtype=np.int32)
    right  = np.zeros_like(idle_z, dtype=np.int32)
    left[:,  1:]  = idle_z[:, :-1]
    right[:, :-1] = idle_z[:,  1:]
    return np.sum(resets * (left + right), axis=1).astype(np.int32)


_ONES = np.ones(BLOCK_BITS, dtype=np.uint8)
_HALF = BLOCK_BITS // 2


# ─────────────────────────────────────────────────────────────────── #
# MinWD                                                                #
# ─────────────────────────────────────────────────────────────────── #

def minwd_encode(
    new_block:   np.ndarray,
    stale_block: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Minimum WD-prone encoding: evaluate 4 transforms of new_block
    (original / inverted / halves-swapped / swapped+inverted),
    choose the one producing fewest WD-prone cells against stale_block.
    2-bit transform index stored as side-channel metadata.
    """
    swapped    = np.concatenate([new_block[_HALF:], new_block[:_HALF]])
    candidates = np.stack([
        new_block,
        _ONES ^ new_block,
        swapped,
        _ONES ^ swapped,
    ], dtype=np.uint8)                                      # (4, B)
    best          = int(np.argmin(_wd_prone_batch(candidates, stale_block)))
    encoded_block = candidates[best]
    write_mask    = (encoded_block ^ stale_block).astype(np.uint8)
    return encoded_block, write_mask


# ─────────────────────────────────────────────────────────────────── #
# DIN — FPC internals                                                  #
# ─────────────────────────────────────────────────────────────────── #

def _fpc_encode_word(w: int) -> tuple[int, int, int]:
    """
    Encode one 32-bit word under FPC priority order.
    Returns (pattern_code 0-7, data_value, data_bit_count).

    Priority (first match wins):
      000  all-zero word               → 0 data bits
      001  4-bit sign-extended         → 4 data bits
      010  1-byte sign-extended        → 8 data bits
      011  halfword zero-extended      → 16 data bits
      100  halfword sign-extended      → 16 data bits
      101  two equal halfwords         → 16 data bits
      110  single byte repeated ×4    → 8 data bits
      111  uncompressed               → 32 data bits
    """
    w  = int(w) & 0xFFFF_FFFF
    sw = w if w < 0x8000_0000 else w - 0x1_0000_0000   # signed view

    # 000 — zero
    if w == 0:
        return 0, 0, 0

    # 001 — 4-bit sign-extend: value in [-8, 7]
    if -8 <= sw <= 7:
        return 1, w & 0xF, 4

    # 010 — byte sign-extend: value in [-128, 127]
    if -128 <= sw <= 127:
        return 2, w & 0xFF, 8

    # 011 — halfword zero-extend: upper 16 bits == 0
    if (w >> 16) == 0:
        return 3, w & 0xFFFF, 16

    # 100 — halfword sign-extend: upper 16 bits == 0xFFFF and bit15 set
    hw = w & 0xFFFF
    if (hw & 0x8000) and (w == (0xFFFF_0000 | hw)):
        return 4, hw, 16

    # 101 — two equal halfwords
    if (w >> 16) == hw:
        return 5, hw, 16

    # 110 — byte repeated ×4
    # Note: any byte-repeated word (b,b,b,b) also satisfies "two equal halfwords"
    # (pattern 101) since lower_hw == upper_hw == 0xBBBB.  Pattern 101 always
    # fires first, making 110 unreachable in practice under this priority order.
    # Included here for completeness / decoder symmetry.
    b = w & 0xFF
    if w == b * 0x0101_0101:
        return 6, b, 8

    # 111 — uncompressed
    return 7, w, 32


def _bits_msb(val: int, n: int) -> list[int]:
    """n-bit MSB-first representation of val."""
    return [(val >> (n - 1 - i)) & 1 for i in range(n)]


def _fpc_compress(block: np.ndarray) -> np.ndarray:
    """
    FPC-compress a 512-bit block.

    Packs bits into 16 big-endian 32-bit words, encodes each with FPC,
    and concatenates (header, data) bit pairs MSB-first.
    Returns a variable-length uint8 bit array.
    """
    # 512 bits → 64 bytes → 16 big-endian uint32 words
    raw_bytes = np.packbits(block)                              # (64,) uint8
    words     = np.frombuffer(raw_bytes.tobytes(), dtype='>u4') # (16,) uint32

    out: list[int] = []
    for word in words:
        code, data_val, n_data = _fpc_encode_word(int(word))
        out.extend(_bits_msb(code, 3))
        if n_data:
            out.extend(_bits_msb(data_val, n_data))

    return np.array(out, dtype=np.uint8)


def _bch20(data_bits: np.ndarray) -> np.ndarray:
    """
    20-bit simplified parity word (BCH placeholder).
    Interleaves data_bits into 20 parity lanes and XORs each lane.
    Sufficient for Exp#6 overhead simulation; not a formal BCH code.
    """
    parity = np.zeros(20, dtype=np.uint8)
    n = len(data_bits)
    if n == 0:
        return parity
    # Vectorised: reshape to (ceil(n/20), 20) and XOR reduce
    pad = (-n % 20)
    padded = np.pad(data_bits, (0, pad))                  # length multiple of 20
    parity = padded.reshape(-1, 20).astype(np.uint8)
    return np.bitwise_xor.reduce(parity, axis=0)          # (20,)


# ─────────────────────────────────────────────────────────────────── #
# DIN                                                                  #
# ─────────────────────────────────────────────────────────────────── #

_DIN_MAX_DATA = BLOCK_BITS - 20   # 492 bits: FPC payload + zero-pad region
_DIN_BCH_OFF  = BLOCK_BITS - 20   # 492: where BCH starts


def din_encode(
    new_block:   np.ndarray,
    stale_block: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Data-Informed Narrow encoding.

    1. FPC-compress new_block.
    2. If compressed stream ≤ 492 bits:
         encoded_block = [compressed | zero-pad (to 492) | BCH-20]
       The zero-padding region is all-zero → no RESET writes there if
       stale is already 0, dramatically cutting WD-prone counts.
    3. If compressed stream > 492 bits: fall back to DCW.
    """
    compressed = _fpc_compress(new_block)
    n = len(compressed)

    if n > _DIN_MAX_DATA:
        return dcw_encode(new_block, stale_block)

    encoded = np.zeros(BLOCK_BITS, dtype=np.uint8)
    encoded[:n]              = compressed           # FPC bits
    # encoded[n:492]          = 0                   # zero-padding (already 0)
    encoded[_DIN_BCH_OFF:]   = _bch20(compressed)  # 20-bit BCH

    write_mask = (encoded ^ stale_block).astype(np.uint8)
    return encoded, write_mask
