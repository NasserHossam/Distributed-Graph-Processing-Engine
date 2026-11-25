"""1D block partition: rank r owns the contiguous vertex range [lo, hi).

The n vertices are split into `size` blocks whose lengths differ by at most 1
(the first n % size ranks get one extra vertex). Because blocks are contiguous,
"which rank owns vertex v?" is arithmetic -- no lookup table to share.
"""
import numpy as np


def block_range(n, size, rank):
    """Return (lo, hi) of the vertices owned by `rank`."""
    base, extra = divmod(n, size)
    lo = rank * base + min(rank, extra)
    hi = lo + base + (1 if rank < extra else 0)
    return lo, hi


def block_starts(n, size):
    """Array of length `size`: first vertex owned by each rank."""
    return np.array([block_range(n, size, r)[0] for r in range(size)], dtype=np.int64)