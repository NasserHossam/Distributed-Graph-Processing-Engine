"""Reading edge-list files into a SciPy sparse adjacency matrix.

File format: one edge per line, "u v" (whitespace separated), vertex ids are
integers 0..n-1. Lines starting with '#' are comments.

The graph is stored as a CSR matrix A where A[u, v] = 1 means a directed edge
u -> v. CSR makes "all out-neighbours of u" a cheap slice, which is what BFS
needs, and A.T (in-edges) is what PageRank needs.
"""
import sys

import numpy as np
import scipy.sparse as sp


def edges_to_csr(edges, n=None, undirected=False):
    """Build the adjacency matrix from an (m, 2) integer array of edges.

    Self-loops and duplicate edges are removed so every implementation sees the
    same simple graph. If n is None it is taken as (largest id + 1).
    """
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    u, v = edges[:, 0], edges[:, 1]
    if n is None:
        n = int(edges.max()) + 1 if len(edges) else 0
    if undirected:                       # add the reverse of every edge
        u, v = np.concatenate([u, v]), np.concatenate([v, u])
    keep = u != v                        # drop self-loops
    u, v = u[keep], v[keep]
    A = sp.csr_matrix((np.ones(len(u)), (u, v)), shape=(n, n))
    A.data[:] = 1.0                      # duplicates were summed; reset to 1
    return A


def read_edge_list(path, undirected=False):
    """Read an edge-list text file and return the CSR adjacency matrix."""
    edges = np.loadtxt(path, dtype=np.int64, comments="#", ndmin=2)
    return edges_to_csr(edges, undirected=undirected)


if __name__ == "__main__":
    # Quick inspection:  python -m gpe.graph_io data/small.txt
    A = read_edge_list(sys.argv[1])
    out_deg = np.diff(A.indptr)          # CSR row lengths = out-degrees
    print(f"vertices={A.shape[0]} edges={A.nnz} "
          f"max_out_degree={out_deg.max()} avg_out_degree={A.nnz / A.shape[0]:.2f} "
          f"dangling(no out-edges)={(out_deg == 0).sum()}")
    if A.shape[0] <= 10:
        print("adjacency (row = source, column = target):")
        print(A.toarray().astype(int))