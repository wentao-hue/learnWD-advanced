"""
LearnWD four-component model.

Components
----------
① Model Trainer   — disturbance-vector k-means + MinHash → populates StaleTable
② Aggressor Extractor — identify RESET-aggressor positions in new_block
③ Cluster Selector    — pick cluster with min dot(aggressor, centroid)
④ Similarity Estimator — MinHash nearest-neighbour inside chosen cluster

Performance design
------------------
After train(), per-cluster data is cached as numpy arrays plus a boolean
validity mask.  A single dict lookup + one in-place bool assignment
invalidates an address in O(1).  Similarity queries use vectorised numpy
over the valid rows, avoiding Python loops over cluster members.
"""

from __future__ import annotations
import time
from typing import Optional

import numpy as np
from config import BLOCK_BITS, DEFAULT_K, DEFAULT_H, MINHASH_BITS
from stale_table import StaleTable


class LearnWDModel:
    def __init__(
        self,
        k: int = DEFAULT_K,
        h: int = DEFAULT_H,
        minhash_bits: int = MINHASH_BITS,
        perm_seed: int = 42,
        block_bits: int = BLOCK_BITS,
        cluster_algo: str = "kmeans",   # "kmeans" | "gmm" | "birch"
        cell_type: str = "slc",         # "slc" | "mlc"
    ) -> None:
        self.k = k
        self.h = h
        self._block_bits  = block_bits
        self._cluster_algo = cluster_algo
        self.cell_type = cell_type
        self._mhash_mask = (1 << minhash_bits) - 1   # e.g. 0b111 = 7

        # Fixed random permutations (shared across all train / infer calls)
        rng = np.random.default_rng(perm_seed)
        if h > 0:
            self._perms: np.ndarray = np.stack(
                [rng.permutation(block_bits).astype(np.int32) for _ in range(h)]
            )  # (h, block_bits)
        else:
            self._perms = np.empty((0, block_bits), dtype=np.int32)

        # Populated by train()
        self.centroids: Optional[np.ndarray] = None  # (k, BLOCK_BITS) float32
        self.train_times: list[float] = []            # wall-clock seconds per retrain

        # Per-cluster cache (rebuilt on each retrain)
        self._c_addrs:  list[np.ndarray] = []   # k × (M_c,)   int32
        self._c_hashes: list[np.ndarray] = []   # k × (M_c, h) uint8
        self._c_valid:  list[np.ndarray] = []   # k × (M_c,)   bool
        self._addr_pos: dict[int, tuple[int, int]] = {}  # addr → (cluster, row)

    # ================================================================== #
    # ① Model Trainer                                                     #
    # ================================================================== #

    def train(self, stale_memory: np.ndarray, stale_table: StaleTable) -> None:
        """
        Run k-means on disturbance vectors, assign clusters, compute MinHash
        fingerprints, update StaleTable entries, rebuild similarity cache.
        """
        t0 = time.perf_counter()
        addresses = stale_table.all_addresses()
        n = len(addresses)
        addr_arr = np.array(addresses, dtype=np.int32)
        blocks = stale_memory[addr_arr]              # (N, BLOCK_BITS)

        # --- disturbance vectors ----------------------------------------
        dist_vecs = self._disturbance_batch(blocks)  # (N, BLOCK_BITS) float32

        # --- clustering -------------------------------------------------
        cluster_ids, centroids = self._cluster(dist_vecs, n)
        cluster_ids = cluster_ids.astype(np.int32)
        self.centroids = centroids.astype(np.float32)

        # --- MinHash fingerprints (batched) ------------------------------
        minhashes = self._minhash_batch(blocks)  # (N, h)

        # --- update StaleTable entries -----------------------------------
        for i, addr in enumerate(addresses):
            stale_table.update_cluster(
                int(addr_arr[i]), int(cluster_ids[i]), minhashes[i]
            )

        # --- rebuild per-cluster numpy cache ----------------------------
        self._rebuild_cache(addr_arr, cluster_ids, minhashes)

        elapsed = time.perf_counter() - t0
        self.train_times.append(elapsed)
        sizes = [len(self._c_addrs[c]) for c in range(self.k)]
        print(
            f"  [retrain] {n} blocks → k={self.k} clusters  "
            f"(min/avg/max={min(sizes)}/{n//self.k}/{max(sizes)})  "
            f"{elapsed:.2f}s"
        )

    def _cluster(
        self, dist_vecs: np.ndarray, n: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run the configured clustering algorithm; return (cluster_ids, centroids)."""
        algo = self._cluster_algo

        if algo == "kmeans":
            from sklearn.cluster import MiniBatchKMeans
            km = MiniBatchKMeans(
                n_clusters=self.k,
                random_state=0,
                n_init=3,
                batch_size=min(n, 4096),
                max_iter=100,
            )
            ids = km.fit_predict(dist_vecs)
            return ids, km.cluster_centers_

        elif algo == "gmm":
            from sklearn.mixture import GaussianMixture
            gm = GaussianMixture(
                n_components=self.k,
                covariance_type="diag",  # full is singular on high-dim sparse vecs
                reg_covar=1e-3,
                random_state=0,
                max_iter=100,
                n_init=1,
            )
            ids = gm.fit_predict(dist_vecs)
            return ids, gm.means_

        elif algo == "birch":
            from sklearn.cluster import Birch
            brc = Birch(n_clusters=self.k, threshold=0.5)
            ids = brc.fit_predict(dist_vecs)
            # Compute per-cluster means as centroids
            B = dist_vecs.shape[1]
            centroids = np.zeros((self.k, B), dtype=np.float64)
            for c in range(self.k):
                mask = ids == c
                if mask.any():
                    centroids[c] = dist_vecs[mask].mean(axis=0)
            return ids, centroids

        else:
            raise ValueError(f"Unknown cluster_algo: '{algo}'")

    def _rebuild_cache(
        self,
        addr_arr:    np.ndarray,   # (N,) int32
        cluster_ids: np.ndarray,   # (N,) int32
        minhashes:   np.ndarray,   # (N, h) uint8
    ) -> None:
        self._c_addrs  = []
        self._c_hashes = []
        self._c_valid  = []
        self._addr_pos = {}

        for c in range(self.k):
            mask = cluster_ids == c
            c_addrs  = addr_arr[mask]                    # (M_c,)
            c_hashes = minhashes[mask]                   # (M_c, h)
            c_valid  = np.ones(len(c_addrs), dtype=bool)
            self._c_addrs.append(c_addrs)
            self._c_hashes.append(c_hashes)
            self._c_valid.append(c_valid)
            for row, addr in enumerate(c_addrs):
                self._addr_pos[int(addr)] = (c, row)

    # ================================================================== #
    # ② Aggressor Extractor                                               #
    # ================================================================== #

    @staticmethod
    def extract_aggressor(new_block: np.ndarray) -> np.ndarray:
        """
        Mark positions where new_block=0 AND at least one neighbour is also 0.
        These are potential RESET aggressors if the stale block has 1s there.
        """
        B = len(new_block)
        zeros = new_block == 0
        left_zero  = np.empty(B, dtype=bool)
        right_zero = np.empty(B, dtype=bool)
        left_zero[0]   = False
        left_zero[1:]  = zeros[:-1]
        right_zero[-1] = False
        right_zero[:-1] = zeros[1:]
        return (zeros & (left_zero | right_zero)).astype(np.uint8)

    @staticmethod
    def extract_aggressor_mlc(new_block: np.ndarray) -> np.ndarray:
        """
        MLC aggressor vector: (C,) float32  where C = B//2.

        A cell written to '00' (= 0b00, pattern int 0) is an aggressor.
        It is suppressed to 0 only if BOTH adjacent neighbours are '10' (= 2).

        Logic mirrors featureExtractor.patternAgg / pickMLCAGG:
            patternAgg(cell, left_nbr) OR patternAgg(cell, right_nbr)
            where patternAgg = 1 if cell=='00' AND nbr!='10', else 0.
        """
        B = len(new_block)
        C = B // 2
        cells    = new_block.reshape(C, 2)
        cell_pat = (cells[:, 0] * 2 + cells[:, 1]).astype(np.int32)  # (C,)

        is_agg    = (cell_pat == 0).astype(np.float32)     # pattern '00' → aggressor candidate

        left_pat  = np.zeros(C, dtype=np.int32)
        right_pat = np.zeros(C, dtype=np.int32)
        left_pat[1:]  = cell_pat[:-1]
        right_pat[:-1] = cell_pat[1:]

        # Not suppressed: at least one neighbour is NOT '10'
        not_suppressed = ((left_pat != 2) | (right_pat != 2)).astype(np.float32)

        return is_agg * not_suppressed                      # (C,) float32

    # ================================================================== #
    # ③ Cluster Selector                                                  #
    # ================================================================== #

    def select_cluster(self, aggressor: np.ndarray) -> int:
        """
        Choose the cluster whose centroid minimises dot(aggressor, centroid).
        Low score → the cluster's stale blocks are least aligned with the
        new block's RESET positions → fewest WD interactions.
        """
        scores = self.centroids @ aggressor.astype(np.float32)  # (k,)
        return int(np.argmin(scores))

    # ================================================================== #
    # ④ Similarity Estimator                                              #
    # ================================================================== #

    def estimate_similarity(
        self,
        new_block:  np.ndarray,
        cluster_id: int,
        stale_table: StaleTable,
        rng: np.random.Generator,
    ) -> int:
        """
        Among stale blocks in cluster_id (still valid in cache), return the
        address whose MinHash fingerprint most closely matches new_block's.
        Falls back to uniform random from whole pool if cluster is empty.
        """
        valid = self._c_valid[cluster_id]

        if not np.any(valid):
            # entire cluster has been overwritten since last retrain
            all_addrs = stale_table.all_addresses()
            return int(rng.choice(all_addrs))

        c_addrs  = self._c_addrs[cluster_id][valid]    # (M',)   int32

        if self.h == 0:
            return int(c_addrs[0])

        c_hashes = self._c_hashes[cluster_id][valid]   # (M', h) uint8
        new_hash = self._minhash_single(new_block)     # (h,)    uint8
        sims = np.sum(c_hashes == new_hash, axis=1)    # (M',)   int
        best_row = int(np.argmax(sims))
        return int(c_addrs[best_row])

    # ================================================================== #
    # Cache invalidation                                                  #
    # ================================================================== #

    def invalidate(self, address: int) -> None:
        """Mark address as written (invalid in cluster cache)."""
        info = self._addr_pos.get(address)
        if info is not None:
            c, row = info
            self._c_valid[c][row] = False

    # ================================================================== #
    # Full select pipeline (used as selector in simulation)               #
    # ================================================================== #

    def select(
        self,
        new_block:   np.ndarray,
        stale_table: StaleTable,
        stale_memory: np.ndarray,
        rng: np.random.Generator,
    ) -> int:
        if self.centroids is None:
            # not yet trained — fall back to random
            addrs = stale_table.all_addresses()
            return int(rng.choice(addrs))
        if self.cell_type == "mlc":
            aggressor = self.extract_aggressor_mlc(new_block)
        else:
            aggressor = self.extract_aggressor(new_block)
        cluster_id = self.select_cluster(aggressor)
        return self.estimate_similarity(new_block, cluster_id, stale_table, rng)

    # ================================================================== #
    # Internal helpers                                                    #
    # ================================================================== #

    def _disturbance_batch(self, blocks: np.ndarray) -> np.ndarray:
        """Dispatch to SLC or MLC disturbance vector computation."""
        if self.cell_type == "mlc":
            return self._disturbance_batch_mlc(blocks)
        return self._disturbance_batch_slc(blocks)

    def _disturbance_batch_slc(self, blocks: np.ndarray) -> np.ndarray:
        """
        SLC vectorised disturbance vector: (N, B) float32.
        d[i] = 0 if block[i]=0;
               = #{adjacent 0-cells} if block[i]=1  → values in {0,1,2}
        """
        ones = (blocks == 1).astype(np.float32)            # (N, B)
        left_zero  = np.zeros_like(ones)
        right_zero = np.zeros_like(ones)
        left_zero[:,  1:]  = (blocks[:, :-1] == 0)
        right_zero[:, :-1] = (blocks[:,  1:] == 0)
        return ones * (left_zero + right_zero)              # (N, B) float32

    def _disturbance_batch_mlc(self, blocks: np.ndarray) -> np.ndarray:
        """
        MLC vectorised disturbance vector: (N, C) float32  where C = B//2.

        Interprets each 2-bit MLC cell pattern to compute WD susceptibility:
          - Cell not in '00' (victim candidate) × sum of aggressor-pattern
            probabilities from left and right neighbours.

        Aggressor-pattern probability lookup (pattern_value → probability):
          '00' (0) → 0.246  '01' (1) → 0.312  '10' (2) → 0.0  '11' (3) → 0.552

        Mirrors featureExtractor.patternProbability from the original paper.
        """
        N, B = blocks.shape
        C = B // 2                                          # 256 for 512-bit blocks

        cells    = blocks.reshape(N, C, 2)                 # (N, C, 2)
        cell_pat = (cells[:, :, 0] * 2 + cells[:, :, 1]).astype(np.int32)  # (N, C)

        # Per-pattern aggressor probability (indexed by 2-bit integer 0-3)
        _AGG_PROBS = np.array([0.246, 0.312, 0.0, 0.552], dtype=np.float32)

        left_pat  = np.zeros((N, C), dtype=np.int32)
        right_pat = np.zeros((N, C), dtype=np.int32)
        left_pat[:,  1:]  = cell_pat[:, :-1]
        right_pat[:, :-1] = cell_pat[:,  1:]

        victim_mask = (cell_pat != 0).astype(np.float32)   # '00' cells not disturbable
        prob_left   = _AGG_PROBS[left_pat]                  # (N, C)
        prob_right  = _AGG_PROBS[right_pat]                 # (N, C)

        return victim_mask * (prob_left + prob_right)       # (N, C) float32

    def _minhash_single(self, block: np.ndarray) -> np.ndarray:
        """MinHash fingerprint for a single block.  Returns (h,) uint8."""
        out = np.empty(self.h, dtype=np.uint8)
        for i in range(self.h):
            permuted = block[self._perms[i]]               # (B,)
            ones_idx = np.nonzero(permuted)[0]
            out[i] = (ones_idx[0] if len(ones_idx) else 0) & self._mhash_mask
        return out

    def _minhash_batch(self, blocks: np.ndarray) -> np.ndarray:
        """Vectorised MinHash for (N, B) block matrix.  Returns (N, h) uint8."""
        N = len(blocks)
        out = np.zeros((N, self.h), dtype=np.uint8)
        for i in range(self.h):
            permuted = blocks[:, self._perms[i]]           # (N, B)
            # argmax on binary array gives first '1'; all-zero rows → 0
            first_one = np.argmax(permuted, axis=1)        # (N,)
            out[:, i] = (first_one & self._mhash_mask).astype(np.uint8)
        return out
