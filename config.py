"""
PCM parameters and global constants (SPEC Table II).
"""

BLOCK_BITS = 512          # bits per block
STALE_POOL_SIZE = 50_000  # initial stale blocks
WRITE_REQ_SIZE  = 50_000  # write request count

# SLC PCM error rates
SLC_WD_WORDLINE = 0.099
SLC_WD_BITLINE  = 0.115

# SLC PCM timing (ns)
SLC_READ_LATENCY  = 100   # ns / cacheline
SLC_RESET_LATENCY = 100   # ns / bit (1→0)
SLC_SET_LATENCY   = 200   # ns / bit (0→1)
VNR_VERIFY_LATENCY = 100  # ns per VnR round (read + verify)

# SLC PCM energy (nJ)
SLC_READ_ENERGY  = 1.075   # nJ / cacheline
SLC_RESET_ENERGY = 0.0137  # nJ / bit
SLC_SET_ENERGY   = 0.0268  # nJ / bit

# Write cost weights (asymmetric)
RESET_COST_WEIGHT = 2
SET_COST_WEIGHT   = 1

# MLC PCM error rates (Exp#5)
MLC_WD_00 = 0.123
MLC_WD_01 = 0.152
MLC_WD_11 = 0.276

# LearnWD model parameters
DEFAULT_K       = 16   # number of clusters
DEFAULT_H       = 8    # number of MinHash functions
MINHASH_BITS    = 3    # bits kept per hash value

# Retrain trigger
RETRAIN_INTERVAL = 20_000  # overwrites between retrains
