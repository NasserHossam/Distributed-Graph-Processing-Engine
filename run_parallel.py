"""Run a parallel algorithm under MPI, time it, optionally save the result.

    mpiexec -n 4 python run_parallel.py --algo bfs --graph data/small.txt --source 0
    mpiexec -n 4 python run_parallel.py --algo pr  --graph data/random_10k.txt
"""
import argparse
from statistics import median

import numpy as np
from mpi4py import MPI

from gpe.graph_io import read_edge_list
from gpe.parallel_bfs import bfs_mpi, gather_distances
from gpe.parallel_pagerank import gather_ranks, pagerank_mpi


def timed(fn, comm, repeats):
    """Run fn() `repeats` times; return (last result, median seconds).

    Barriers before and after make every rank start together and wait for the
    slowest one, so the time is that of the whole parallel run.
    """
    times = []
    for _ in range(repeats):
        comm.Barrier()
        t0 = MPI.Wtime()
        result = fn()
        comm.Barrier()
        times.append(MPI.Wtime() - t0)
    return result, median(times)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["bfs", "pr"], required=True)
    p.add_argument("--graph", required=True)
    p.add_argument("--source", type=int, default=0, help="BFS source vertex")
    p.add_argument("--damping", type=float, default=0.85, help="PageRank damping")
    p.add_argument("--tol", type=float, default=1e-10, help="PageRank L1 tolerance")
    p.add_argument("--max-iter", type=int, default=100, help="PageRank iteration cap")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--out", help="rank 0 saves the full result here (.npy)")
    a = p.parse_args()

    comm = MPI.COMM_WORLD
    A = read_edge_list(a.graph)          # every rank reads the whole graph (simple; see README)

    if a.algo == "bfs":
        local, t = timed(lambda: bfs_mpi(A, a.source, comm), comm, a.repeats)
        full = gather_distances(local, comm)
        extra = f"max_dist={full.max()}" if comm.Get_rank() == 0 else ""
    else:
        (local, iters), t = timed(
            lambda: pagerank_mpi(A, a.damping, a.tol, a.max_iter, comm), comm, a.repeats)
        full = gather_ranks(local, comm)
        extra = f"iters={iters}"

    if comm.Get_rank() == 0:
        if a.out:
            np.save(a.out, full)
        print(f"RESULT algo={a.algo} procs={comm.Get_size()} n={A.shape[0]} "
              f"m={A.nnz} median_s={t:.6f} {extra}")


if __name__ == "__main__":
    main()