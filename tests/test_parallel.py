"""Parallel-vs-sequential correctness tests. Each case launches a real
`mpiexec -n N python run_parallel.py ...` and compares the saved result.

Run from the project root:   python -m pytest -v
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from gpe.generate import random_edges
from gpe.graph_io import edges_to_csr, read_edge_list
from gpe.sequential import bfs_seq, pagerank_seq

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def run_mpi(nprocs, args):
    cmd = ["mpiexec", "-n", str(nprocs), sys.executable, "run_parallel.py", *map(str, args)]
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert res.returncode == 0, f"mpiexec failed:\n{res.stdout}\n{res.stderr}"


def make_random_graph(tmp_path, n, avg_degree, seed):
    path = tmp_path / f"rand_{n}_{seed}.txt"
    np.savetxt(path, random_edges(n, avg_degree, seed), fmt="%d")
    return path


# --------------------------------------------------------------------- BFS ---
@pytest.mark.parametrize("nprocs", [1, 2, 3, 4, 8])   # 8 > 6 vertices: some ranks own nothing
def test_bfs_small_graph(nprocs, tmp_path):
    out = tmp_path / "dist.npy"
    run_mpi(nprocs, ["--algo", "bfs", "--graph", DATA / "small.txt", "--source", 0, "--out", out])
    assert np.load(out).tolist() == [0, 1, 1, 2, 3, -1]   # hand-computed answer


@pytest.mark.parametrize("nprocs", [1, 2, 3, 4])
@pytest.mark.parametrize("n, avg_degree, seed, source", [(500, 3, 1, 0), (1000, 2, 2, 777), (2000, 5, 3, 123)])
def test_bfs_random_graphs(nprocs, n, avg_degree, seed, source, tmp_path):
    graph = make_random_graph(tmp_path, n, avg_degree, seed)
    out = tmp_path / "dist.npy"
    run_mpi(nprocs, ["--algo", "bfs", "--graph", graph, "--source", source, "--out", out])
    expected = bfs_seq(read_edge_list(graph), source)
    assert np.array_equal(np.load(out), expected)         # exact match required


# ---------------------------------------------------------------- PageRank ---
PR_TOL = 1e-8   # L1 distance allowed between parallel and sequential ranks.
                # They differ only by floating-point summation order (~1e-15),
                # plus at most one iteration if the stopping test flips (~1e-10).


def l1(a, b):
    return float(np.abs(a - b).sum())


@pytest.mark.parametrize("nprocs", [1, 2, 3])            # 3 > 2 vertices: one rank owns nothing
def test_pagerank_two_node_closed_form(nprocs, tmp_path):
    out = tmp_path / "pr.npy"
    run_mpi(nprocs, ["--algo", "pr", "--graph", DATA / "two_node.txt", "--tol", 1e-14,
                     "--max-iter", 1000, "--out", out])
    assert l1(np.load(out), np.array([1 / 2.85, 1.85 / 2.85])) < 1e-9   # hand-derived answer


@pytest.mark.parametrize("nprocs", [1, 2, 3, 4, 8])
def test_pagerank_small_graph(nprocs, tmp_path):
    out = tmp_path / "pr.npy"
    run_mpi(nprocs, ["--algo", "pr", "--graph", DATA / "small.txt", "--out", out])
    expected, _ = pagerank_seq(read_edge_list(DATA / "small.txt"))
    assert l1(np.load(out), expected) < PR_TOL
    assert abs(np.load(out).sum() - 1.0) < 1e-9


@pytest.mark.parametrize("nprocs", [1, 2, 3, 4])
@pytest.mark.parametrize("n, avg_degree, seed", [(500, 3, 1), (1000, 2, 2), (2000, 5, 3)])
def test_pagerank_random_graphs(nprocs, n, avg_degree, seed, tmp_path):
    graph = make_random_graph(tmp_path, n, avg_degree, seed)   # avg_degree 2-3 => dangling vertices
    out = tmp_path / "pr.npy"
    run_mpi(nprocs, ["--algo", "pr", "--graph", graph, "--out", out])
    expected, _ = pagerank_seq(read_edge_list(graph))
    assert l1(np.load(out), expected) < PR_TOL