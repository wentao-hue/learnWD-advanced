"""
Stale Table: tracks physical addresses of stale blocks along with
their cluster assignment and MinHash fingerprint.

Maintains a _cluster_idx inverted index so query_by_cluster() and
update_cluster() are both O(1) rather than O(N).
"""

from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class StaleEntry:
    physical_address: int
    cluster_id: int = -1
    minhash_values: np.ndarray = field(
        default_factory=lambda: np.zeros(8, dtype=np.uint8)
    )


class StaleTable:
    def __init__(self) -> None:
        self._entries: dict[int, StaleEntry] = {}
        # inverted index: cluster_id → set of physical addresses
        self._cluster_idx: defaultdict[int, set[int]] = defaultdict(set)

    # ------------------------------------------------------------------ #
    # Basic CRUD                                                           #
    # ------------------------------------------------------------------ #

    def insert(
        self,
        address: int,
        cluster_id: int = -1,
        minhash_values: Optional[np.ndarray] = None,
    ) -> None:
        mv = minhash_values if minhash_values is not None else np.zeros(8, dtype=np.uint8)
        self._entries[address] = StaleEntry(address, cluster_id, mv)
        self._cluster_idx[cluster_id].add(address)

    def delete(self, address: int) -> None:
        entry = self._entries.pop(address, None)
        if entry is not None:
            self._cluster_idx[entry.cluster_id].discard(address)

    def __contains__(self, address: int) -> bool:
        return address in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------ #
    # Queries                                                              #
    # ------------------------------------------------------------------ #

    def all_addresses(self) -> list[int]:
        return list(self._entries.keys())

    def query_by_cluster(self, cluster_id: int) -> list[StaleEntry]:
        return [self._entries[a] for a in self._cluster_idx.get(cluster_id, ())]

    def cluster_size(self, cluster_id: int) -> int:
        return len(self._cluster_idx.get(cluster_id, ()))

    # ------------------------------------------------------------------ #
    # Mutations                                                            #
    # ------------------------------------------------------------------ #

    def update_cluster(
        self,
        address: int,
        cluster_id: int,
        minhash_values: Optional[np.ndarray] = None,
    ) -> None:
        entry = self._entries.get(address)
        if entry is None:
            return
        self._cluster_idx[entry.cluster_id].discard(address)
        entry.cluster_id = cluster_id
        if minhash_values is not None:
            entry.minhash_values = minhash_values
        self._cluster_idx[cluster_id].add(address)

    # ------------------------------------------------------------------ #
    # Bulk helpers                                                         #
    # ------------------------------------------------------------------ #

    def bulk_insert(self, addresses: list[int]) -> None:
        for addr in addresses:
            self.insert(addr)
