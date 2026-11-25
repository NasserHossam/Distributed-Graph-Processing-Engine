"""Tests for the sequential baseline: known answers + independent references.

Run from the project root:   python -m pytest -v
"""
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import shortest_path

from gpe.generate import random_edges
from gpe.graph_io import edges_to_csr, read_edge_list
from gpe.sequential import bfs_seq, pagerank_seq

DATA = Path(__file__).resolve().parent.parent / "data"


def test_bfs_small_known_answer():
    A = read_edge_list(DATA / "small.txt")
    # Worked out by hand in data/small.txt (-1 = unreachable).
    assert bfs_seq(A, 0).tolist() == [0, 1, 1, 2, 3, -1]


def test_pagerank_two_node_closed_form():
    A = read_edge_list(DATA / "two_node.txt")
    r, _ = pagerank_seq(A, damping=0.85, tol=1e-14, max_iter=1000)
    np.testing.assert_allclose(r, [1 / 2.85, 1.85 / 2.85], atol=1e-10)


def test_pagerank_small_properties():
    A = read_edge_list(DATA / "small.txt")
    r, _ = pagerank_seq(A)
    assert abs(r.sum() - 1.0) < 1e-12        # ranks form a probability distribution
    assert abs(r[1] - r[2]) < 1e-12          # vertices 1 and 2 are symmetric


def test_bfs_matches_scipy_on_random_graphs():
    for seed in range(5):
        A = edges_to_csr(random_edges(300, 3, seed), n=300)
        ref = shortest_path(A, unweighted=True, indices=0)   # independent implementation
        ref = np.where(np.isinf(ref), -1, ref).astype(np.int64)
        assert bfs_seq(A, 0).tolist() == ref.tolist()


def test_pagerank_matches_dense_solve_on_random_graphs():
    d = 0.85
    for seed in range(3):
        n = 200
        A = edges_to_csr(random_edges(n, 2, seed), n=n)       # sparse: has dangling vertices
        # Independent reference: solve (I - d*M) r = (1-d)/n directly, where M is
        # the column-stochastic matrix with dangling columns replaced by 1/n.
        M = np.zeros((n, n))
        deg = np.diff(A.indptr)
        for u in range(n):
            if deg[u] == 0:
                M[:, u] = 1.0 / n
            else:
                M[A.indices[A.indptr[u]:A.indptr[u + 1]], u] = 1.0 / deg[u]
        ref = np.linalg.solve(np.eye(n) - d * M, np.full(n, (1 - d) / n))
        r, _ = pagerank_seq(A, damping=d, tol=1e-13, max_iter=1000)
        assert np.abs(r - ref).sum() < 1e-9