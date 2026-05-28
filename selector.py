"""
Stale block selection strategies.

Interface contract:
    select(new_block, stale_table, stale_memory, rng, **kwargs) → physical_address

Factories return closures that satisfy this contract.
"""

from __future__ import annotations
import numpy as np
from stale_table import StaleTable


# ------------------------------------------------------------------ #
# RandSel                                                             #
# ------------------------------------------------------------------ #

def randsel(
    new_block:    np.ndarray,
    stale_table:  StaleTable,
    stale_memory: np.ndarray,
    rng:          np.random.Generator,
    **_kwargs,
) -> int:
    """Uniformly random selection from the stale pool."""
    addresses = stale_table.all_addresses()
    return addresses[rng.integers(0, len(addresses))]


# ------------------------------------------------------------------ #
# LearnWD                                                             #
# ------------------------------------------------------------------ #

def make_learnwd_selector(model: "LearnWDModel"):   # type: ignore[name-defined]
    """
    Return a selector function that wraps model.select().
    The returned function matches the standard selector interface.
    """
    def _select(
        new_block:    np.ndarray,
        stale_table:  StaleTable,
        stale_memory: np.ndarray,
        rng:          np.random.Generator,
        **_kwargs,
    ) -> int:
        return model.select(new_block, stale_table, stale_memory, rng)

    return _select
