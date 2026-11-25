"""Distributed PageRank with mpi4py (same maths as gpe.sequential.pagerank_seq).

Partition: rank r owns vertices [lo, hi). It stores
  * the in-edges of its own vertices (a (hi-lo) x n sparse matrix), and
  * the current rank values of its own vertices only.

Each iteration:
  1. every rank computes contrib[u] = rank[u] / outdeg[u] for its own vertices
  2. allreduce: total rank mass sitting on dangling vertices (one number)
  3. Allgatherv: every rank receives the full contrib vector (n numbers)
  4. each rank computes the new ranks of its own vertices from its in-edges
  5. allreduce: global L1 change, so every rank stops on the same iteration
"""
import numpy as np
from mpi4py import MPI

from gpe.partition import block_range


def pagerank_mpi(A, damping=0.85, tol=1e-10, max_iter=100, comm=MPI.COMM_WORLD):
    """Returns (this rank's slice of the rank vector, iterations)."""
    rank, size = comm.Get_rank(), comm.Get_size()
    n = A.shape[0]
    lo, hi = block_range(n, size, rank)
    counts = [block_range(n, size, r)[1] - block_range(n, size, r)[0] for r in range(size)]
    displs = [block_range(n, size, r)[0] for r in range(size)]

    # Local setup: out-degrees of MY vertices and the in-edges of MY vertices.
    out_deg = np.diff(A.indptr[lo:hi + 1]).astype(np.float64)
    dangling = out_deg == 0
    inv_deg = np.zeros(hi - lo)
    inv_deg[~dangling] = 1.0 / out_deg[~dangling]
    AT_local = A[:, lo:hi].T.tocsr()          # row i = in-neighbours of vertex lo+i

    r = np.full(hi - lo, 1.0 / n)             # my slice of the rank vector
    contrib = np.empty(n)                     # full contribution vector (filled by Allgatherv)
    for it in range(1, max_iter + 1):
        contrib_local = r * inv_deg
        dangling_mass = comm.allreduce(r[dangling].sum(), op=MPI.SUM)
        comm.Allgatherv(contrib_local, [contrib, counts, displs, MPI.DOUBLE])
        r_new = (1 - damping) / n + damping * dangling_mass / n + damping * (AT_local @ contrib)
        diff = comm.allreduce(np.abs(r_new - r).sum(), op=MPI.SUM)   # global L1 change
        r = r_new
        if diff < tol:
            break
    return r, it


def gather_ranks(local_rank, comm=MPI.COMM_WORLD):
    """Collect the full rank vector on rank 0 (None on other ranks)."""
    parts = comm.gather(local_rank, root=0)
    return np.concatenate(parts) if comm.Get_rank() == 0 else None