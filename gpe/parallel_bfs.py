"""Distributed level-synchronous BFS with mpi4py.

Partition: rank r owns vertices [lo, hi) and stores only their out-edges
(rows lo..hi of the adjacency matrix) plus the distances of its own vertices.

Each level (one BFS hop):
  1. every rank expands its local frontier -> candidate neighbour vertices
  2. candidates are grouped by owning rank and sent with one alltoall
  3. each owner keeps the ones it has not visited yet -> its next frontier
  4. an allreduce tells everyone whether any frontier is still non-empty
"""
import numpy as np
from mpi4py import MPI

from gpe.partition import block_range, block_starts

UNREACHED = -1


def bfs_mpi(A, source, comm=MPI.COMM_WORLD):
    """BFS from `source`. Returns this rank's slice of the distance array.

    `A` is the full CSR matrix (every rank has it); only rows [lo, hi) are used.
    """
    rank, size = comm.Get_rank(), comm.Get_size()
    n = A.shape[0]
    lo, hi = block_range(n, size, rank)
    starts = block_starts(n, size)
    A_local = A[lo:hi]                                    # my vertices' out-edges

    dist = np.full(hi - lo, UNREACHED, dtype=np.int64)    # distances of my vertices
    frontier = np.empty(0, dtype=np.int64)                # my frontier (global ids)
    if lo <= source < hi:
        dist[source - lo] = 0
        frontier = np.array([source], dtype=np.int64)

    level = 0
    while comm.allreduce(len(frontier), op=MPI.SUM) > 0:  # any work left anywhere?
        level += 1
        # 1. expand: out-neighbours (global ids) of my frontier, sorted and unique
        nbrs = np.unique(A_local[frontier - lo].indices).astype(np.int64)
        # 2. split by owner. nbrs is sorted and blocks are contiguous, so the
        #    pieces for ranks 0..size-1 are consecutive slices.
        outgoing = np.split(nbrs, np.searchsorted(nbrs, starts[1:]))
        incoming = comm.alltoall(outgoing)                # one array from every rank
        # 3. keep vertices I own that are still unvisited
        cand = np.unique(np.concatenate(incoming))
        new = cand[dist[cand - lo] == UNREACHED]
        dist[new - lo] = level
        frontier = new
    return dist


def gather_distances(local_dist, comm=MPI.COMM_WORLD):
    """Collect the full distance array on rank 0 (None on other ranks)."""
    parts = comm.gather(local_dist, root=0)               # blocks arrive in rank order
    return np.concatenate(parts) if comm.Get_rank() == 0 else None