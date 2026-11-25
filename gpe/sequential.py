"""Sequential BFS and PageRank: the correctness reference for the MPI versions.

Both take the CSR adjacency matrix A (A[u, v] = 1 for an edge u -> v).
"""
import numpy as np

UNREACHED = -1


def bfs_seq(A, source):
    """Level-synchronous BFS. Returns hop distances, -1 for unreachable vertices.

    Each loop iteration expands the whole current frontier by one hop. The MPI
    version keeps exactly this structure, so results are directly comparable.
    """
    n = A.shape[0]
    dist = np.full(n, UNREACHED, dtype=np.int64)
    dist[source] = 0
    frontier = np.array([source], dtype=np.int64)
    level = 0
    while len(frontier) > 0:
        level += 1
        # A[frontier] is the sub-matrix of the frontier's rows; its column
        # indices are all out-neighbours of the frontier vertices.
        neighbours = np.unique(A[frontier].indices)
        new = neighbours[dist[neighbours] == UNREACHED]   # not visited yet
        dist[new] = level
        frontier = new
    return dist


def pagerank_seq(A, damping=0.85, tol=1e-10, max_iter=100):
    """Power-iteration PageRank. Returns (ranks, iterations); ranks sum to 1.

        r_new[v] = (1-d)/n  +  d * dangling_mass/n  +  d * sum_{u->v} r[u]/outdeg[u]

    "Dangling" vertices (no out-edges) would leak probability, so their mass is
    spread evenly over all vertices. Stops when the L1 change < tol.
    """
    n = A.shape[0]
    out_deg = np.diff(A.indptr).astype(np.float64)
    dangling = out_deg == 0
    inv_deg = np.zeros(n)
    inv_deg[~dangling] = 1.0 / out_deg[~dangling]
    AT = A.T.tocsr()                      # row v of AT = in-neighbours of v

    r = np.full(n, 1.0 / n)
    for it in range(1, max_iter + 1):
        contrib = r * inv_deg                         # what each vertex sends per out-edge
        dangling_mass = r[dangling].sum()
        r_new = (1 - damping) / n + damping * dangling_mass / n + damping * (AT @ contrib)
        diff = np.abs(r_new - r).sum()                # L1 change
        r = r_new
        if diff < tol:
            break
    return r, it