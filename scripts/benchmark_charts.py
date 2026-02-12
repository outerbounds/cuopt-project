"""Generate benchmark charts from the latest LPBenchmarkFlow run.

Pulls results from Metaflow artifacts and produces two matplotlib figures:
  1. Log-log solve time comparison (4 solvers x 5 sizes)
  2. GPU speedup vs each CPU solver

Usage:
    METAFLOW_PROFILE=yellow python scripts/benchmark_charts.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from metaflow import Flow

# ---------------------------------------------------------------------------
# Pull data from latest run
# ---------------------------------------------------------------------------
run = Flow("LPBenchmarkFlow").latest_run
compare = run["compare"].task

series = {
    "cuOpt PDLP (GPU)": compare["cuopt_results"].data,
    "OR-Tools PDLP (CPU)": compare["ortools_results"].data,
    "HiGHS Simplex": compare["simplex_results"].data,
    "HiGHS IPM": compare["ipm_results"].data,
}

print(f"Run {run.id} ({run.created_at})")
for name, results in series.items():
    times = [r["solve_time"] for r in results]
    print(f"  {name:>22}: {' | '.join(f'{t:.3f}s' for t in times)}")

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
colors = {
    "cuOpt PDLP (GPU)": "#E6786C",
    "OR-Tools PDLP (CPU)": "#D4A03C",
    "HiGHS Simplex": "#4C9878",
    "HiGHS IPM": "#5B8DBE",
}
markers = {
    "cuOpt PDLP (GPU)": "o",
    "OR-Tools PDLP (CPU)": "s",
    "HiGHS Simplex": "^",
    "HiGHS IPM": "D",
}


def regions_formatter(x, _):
    return f"{x/1000:.0f}K" if x >= 1000 else str(int(x))


out_dir = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Figure 1: Solve time comparison (log-log)
# ---------------------------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(8, 5))

for name, results in series.items():
    regions = [r["n_regions"] for r in results]
    times = [r["solve_time"] for r in results]
    ax1.plot(
        regions,
        times,
        marker=markers[name],
        color=colors[name],
        linewidth=2,
        markersize=7,
        label=name,
    )

    # Label the last point with solve time
    ax1.annotate(
        f"{times[-1]:.1f}s",
        (regions[-1], times[-1]),
        textcoords="offset points",
        xytext=(8, 0),
        fontsize=8,
        color=colors[name],
        fontweight="bold",
    )

ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.xaxis.set_major_formatter(FuncFormatter(regions_formatter))
ax1.set_xlabel("Problem Size (regions)")
ax1.set_ylabel("Solve Time (seconds)")
ax1.set_title("LP Solver Benchmark — Solve Time vs Problem Size", fontweight="bold")
ax1.legend(loc="upper left", fontsize=9)
ax1.grid(True, alpha=0.3, which="both")

fig1.tight_layout()
fig1.savefig(out_dir / "benchmark_solve_times.png", dpi=150, bbox_inches="tight")
print(f"\nSaved {out_dir / 'benchmark_solve_times.png'}")

# ---------------------------------------------------------------------------
# Figure 2: GPU speedup vs CPU solvers
# ---------------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(8, 5))

cuopt = series["cuOpt PDLP (GPU)"]
cpu_solvers = {k: v for k, v in series.items() if k != "cuOpt PDLP (GPU)"}

for name, results in cpu_solvers.items():
    regions = [r["n_regions"] for r in results]
    speedups = [
        cpu["solve_time"] / gpu["solve_time"] for cpu, gpu in zip(results, cuopt)
    ]
    ax2.plot(
        regions,
        speedups,
        marker=markers[name],
        color=colors[name],
        linewidth=2,
        markersize=7,
        label=f"vs {name}",
    )

    # Label last point
    ax2.annotate(
        f"{speedups[-1]:.0f}x",
        (regions[-1], speedups[-1]),
        textcoords="offset points",
        xytext=(8, 0),
        fontsize=9,
        color=colors[name],
        fontweight="bold",
    )

ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.axhline(y=1, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax2.xaxis.set_major_formatter(FuncFormatter(regions_formatter))
ax2.set_xlabel("Problem Size (regions)")
ax2.set_ylabel("Speedup (x faster than CPU)")
ax2.set_title("cuOpt GPU Speedup vs CPU Solvers", fontweight="bold")
ax2.legend(loc="upper left", fontsize=9)
ax2.grid(True, alpha=0.3, which="both")

fig2.tight_layout()
fig2.savefig(out_dir / "benchmark_gpu_speedup.png", dpi=150, bbox_inches="tight")
print(f"Saved {out_dir / 'benchmark_gpu_speedup.png'}")
