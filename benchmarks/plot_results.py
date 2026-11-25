"""Plot speedup and efficiency from results/results.csv and print a Markdown table.

    python benchmarks/plot_results.py
Writes results/bfs_scaling.png, results/pr_scaling.png, results/results_table.md
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                               # write PNG files, no window
import matplotlib.pyplot as plt                     # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
NAMES = {"bfs": "BFS", "pr": "PageRank"}


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def plot_algo(algo, rows):
    sizes = sorted({int(r["n"]) for r in rows})
    fig, (ax_s, ax_e) = plt.subplots(1, 2, figsize=(11, 4))
    procs_all = sorted({int(r["procs"]) for r in rows})
    for n in sizes:
        pts = sorted((int(r["procs"]), float(r["speedup"]), float(r["efficiency"]))
                     for r in rows if int(r["n"]) == n)
        label = f"n = {n:,}"
        ax_s.plot([p for p, _, _ in pts], [s for _, s, _ in pts], "o-", label=label)
        ax_e.plot([p for p, _, _ in pts], [e for _, _, e in pts], "o-", label=label)
    ax_s.plot(procs_all, procs_all, "k--", alpha=0.4, label="ideal")
    ax_e.axhline(1.0, color="k", linestyle="--", alpha=0.4, label="ideal")
    ax_s.set(title=f"{NAMES[algo]}: speedup  (T1 / Tp)", xlabel="MPI processes", ylabel="speedup")
    ax_e.set(title=f"{NAMES[algo]}: parallel efficiency  (speedup / p)", xlabel="MPI processes",
             ylabel="efficiency")
    for ax in (ax_s, ax_e):
        ax.set_xticks(procs_all)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    out = RESULTS / f"{algo}_scaling.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def markdown_table(rows):
    lines = ["| Algorithm | Vertices | Edges | Procs | Median time (ms) | Speedup | Efficiency | Sequential (ms) |",
             "|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["algo"], int(r["n"]), int(r["procs"]))):
        lines.append(f"| {NAMES[r['algo']]} | {int(r['n']):,} | {int(r['m']):,} | {r['procs']} | "
                     f"{float(r['median_s']) * 1000:.1f} | {r['speedup']} | {r['efficiency']} | "
                     f"{float(r['seq_median_s']) * 1000:.1f} |")
    return "\n".join(lines)


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS / "results.csv"
    rows = load(path)
    by_algo = defaultdict(list)
    for r in rows:
        by_algo[r["algo"]].append(r)
    for algo, rs in by_algo.items():
        plot_algo(algo, rs)
    table = markdown_table(rows)
    (RESULTS / "results_table.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()