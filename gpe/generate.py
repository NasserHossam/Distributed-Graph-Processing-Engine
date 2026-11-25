"""Random directed graph generator.

    python -m gpe.generate --n 10000 --avg-degree 10 --seed 1 --out data/random_10k.txt

Each of the n * avg_degree edges picks its source and target uniformly at
random (duplicates/self-loops are removed when the file is loaded). Uniform
random graphs have fairly even degrees, so a simple block partition of the
vertices is well balanced. Real-world graphs are more skewed; see README.
"""
import argparse

import numpy as np


def random_edges(n, avg_degree, seed=0):
    """Return an (m, 2) array of random directed edges, m = n * avg_degree."""
    rng = np.random.default_rng(seed)
    m = int(n * avg_degree)
    return rng.integers(0, n, size=(m, 2), dtype=np.int64)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n", type=int, required=True, help="number of vertices")
    p.add_argument("--avg-degree", type=float, default=10, help="edges per vertex")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="output edge-list file")
    a = p.parse_args()

    edges = random_edges(a.n, a.avg_degree, a.seed)
    np.savetxt(a.out, edges, fmt="%d",
               header=f"random graph n={a.n} avg_degree={a.avg_degree} seed={a.seed}")
    print(f"wrote {len(edges)} edges over {a.n} vertices to {a.out}")


if __name__ == "__main__":
    main()