"""Benchmark the sequential and MPI versions and write results/results.csv.

    python benchmarks/run_benchmarks.py --procs 1 2 4 8 --sizes 10000 100000 300000

What is timed: the algorithm call only (graph file loading is excluded).
  * sequential : bfs_seq / pagerank_seq, in this process
  * parallel   : bfs_mpi / pagerank_mpi under mpiexec, including each rank's local
                 setup and the communication; time = slowest rank (Barrier at the end)
Every number is the median of --repeats runs.
  speedup    = T(1 process) / T(p processes)       (same MPI code, 1 vs p ranks)
  efficiency = speedup / p
"""
import argparse
import csv
import re
import subprocess
import sys
import time
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))                       # so `import gpe` works from here

import numpy as np                                  # noqa: E402

from gpe.generate import random_edges               # noqa: E402
from gpe.graph_io import read_edge_list             # noqa: E402
from gpe.sequential import bfs_seq, pagerank_seq    # noqa: E402


def ensure_graph(n, avg_degree, seed=1):
    """Generate data/random_<n>.txt once; reuse it on later runs."""
    path = ROOT / "data" / f"random_{n}.txt"
    if not path.exists():
        np.savetxt(path, random_edges(n, avg_degree, seed), fmt="%d",
                   header=f"random graph n={n} avg_degree={avg_degree} seed={seed}")
    return path


def time_sequential(algo, A, repeats):
    fn = (lambda: bfs_seq(A, 0)) if algo == "bfs" else (lambda: pagerank_seq(A))
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return median(times)


def time_mpi(algo, graph, procs, repeats):
    cmd = ["mpiexec", "-n", str(procs), sys.executable, "run_parallel.py",
           "--algo", algo, "--graph", str(graph), "--repeats", str(repeats)]
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"mpiexec failed:\n{res.stdout}\n{res.stderr}")
    line = [l for l in res.stdout.splitlines() if l.startswith("RESULT")][-1]
    return float(re.search(r"median_s=([0-9.eE+-]+)", line).group(1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--procs", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--sizes", type=int, nargs="+", default=[10_000, 100_000, 300_000],
                   help="number of vertices of each random graph")
    p.add_argument("--avg-degree", type=float, default=10)
    p.add_argument("--algos", nargs="+", default=["bfs", "pr"], choices=["bfs", "pr"])
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--out", default=str(ROOT / "results" / "results.csv"))
    a = p.parse_args()
    if 1 not in a.procs:
        sys.exit("--procs must include 1 (needed as the speedup baseline)")

    (ROOT / "data").mkdir(exist_ok=True)
    Path(a.out).parent.mkdir(exist_ok=True)
    rows = []
    for n in a.sizes:
        graph = ensure_graph(n, a.avg_degree)
        A = read_edge_list(graph)
        for algo in a.algos:
            seq = time_sequential(algo, A, a.repeats)
            print(f"[{algo} n={n} m={A.nnz}] sequential: {seq:.4f}s", flush=True)
            times = {}
            for procs in sorted(a.procs):
                times[procs] = time_mpi(algo, graph, procs, a.repeats)
                print(f"[{algo} n={n}]   {procs} procs: {times[procs]:.4f}s", flush=True)
            for procs, t in times.items():
                speedup = times[1] / t
                rows.append(dict(algo=algo, n=n, m=A.nnz, procs=procs,
                                 median_s=f"{t:.6f}", seq_median_s=f"{seq:.6f}",
                                 speedup=f"{speedup:.3f}", efficiency=f"{speedup / procs:.3f}",
                                 speedup_vs_seq=f"{seq / t:.3f}"))

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()