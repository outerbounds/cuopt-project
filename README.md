# GPU Optimization with NVIDIA cuOpt on Outerbounds

Five optimization problems solved with NVIDIA cuOpt on GPU, orchestrated by Metaflow on Outerbounds. Each flow is a real-world optimization problem expressed as a word problem.

## The Problems

### 1. Farm Planning (LP)

> An agricultural company manages 10,000 farm regions. Each region can grow wheat, corn, soybeans, or cotton. Every region has its own land and water limits. Five shared labor pools serve all regions, and global market caps limit how much of each crop can be sold. **Maximize total profit by deciding how many acres of each crop to plant in each region.** Then sweep crop prices to test sensitivity.

**Flow:** `flows/lp-minimal/flow.py` | **Type:** Linear Program (40K variables, 20K constraints)

### 2. Portfolio Optimization (MAD LP)

> An investor has 200 stocks with a year of daily returns. For a given target return, find the portfolio that **minimizes risk**, measured by Mean Absolute Deviation (an LP alternative to Markowitz variance). Sweep across 20 return targets to trace the efficient frontier.

**Flow:** `flows/qp-minimal/flow.py` | **Type:** Linear Program (452 variables per frontier point, 506 constraints)

### 3. Warehouse Location (MILP)

> A logistics company has 100 candidate warehouse sites and 500 customers. Each warehouse has a fixed opening cost. Shipping cost depends on distance and demand. **Decide which warehouses to open (yes/no) and how to assign customers to minimize total cost.** Compare GPU-only heuristics (fast, approximate) against full branch-and-bound (exact, slower).

**Flow:** `flows/milp-minimal/flow.py` | **Type:** Mixed-Integer LP (100 binary + 50K continuous variables)

### 4. Delivery Routing (VRP)

> A delivery company has a depot and customers scattered across a city in clusters (like real neighborhoods). Trucks have limited capacity. **Find routes for the trucks that visit every customer, respect capacity, and minimize total distance.** Run at 50, 200, 500, and 1,000 customers in parallel to see how GPU routing scales.

**Flow:** `flows/vrp-minimal/flow.py` | **Type:** Capacitated Vehicle Routing (50-1000 customers, foreach fanout)

### 5. LP Solver Benchmark

> Same farm planning problem as #1, run at 1K, 5K, 10K, 50K, and 100K regions with four solvers: cuOpt PDLP (GPU), OR-Tools PDLP (CPU), HiGHS Simplex, and HiGHS IPM. All solvers receive identical scipy CSR sparse input. **Isolates GPU vs CPU for the same algorithm (PDLP) and compares first-order methods against classical solvers.**

**Flow:** `flows/lp-benchmark/flow.py` | **Type:** LP scaling benchmark (4 solvers, 5 sizes, 20 tasks)

## Deployment Patterns

Two ways to run cuOpt on Outerbounds — **per-step GPU** (each solve step gets its own GPU pod) and **persistent GPU server** (shared cuOpt REST API). All five flows above use per-step GPU. The `deployments/cuopt-server/` directory shows the server pattern. See [docs/deployment-patterns.md](docs/deployment-patterns.md) for details and when to use each.

## Why Metaflow + Outerbounds

Each flow demonstrates practical patterns for running optimization on managed infrastructure:

- **`@kubernetes(gpu=1, compute_pool='gpu-multi-training')`** — GPU provisioning. Steps that need a GPU get one; steps that don't run on CPU pools. No cluster management.
- **`@conda(packages={...})`** — Dependency isolation. cuOpt + RAPIDS on GPU nodes, scipy on CPU nodes. Each step gets exactly the packages it needs.
- **`foreach`** — Parallel fan-out. The farm LP sweeps 5 price scenarios simultaneously. The VRP solves 4 instance sizes in parallel. The benchmark tests 5 sizes per solver concurrently.
- **Branching DAG** — The MILP flow runs heuristic and exact solvers in parallel branches. The benchmark flow runs 4 solver branches in parallel, each fanning out across 5 sizes.
- **`@card`** — Every flow produces a visual result card (charts, tables) viewable in the Outerbounds UI.
- **`@gpu_profile`** — GPU utilization tracking on every GPU step.
- **`outerbounds app deploy`** — Persistent GPU services. cuOpt server runs as a long-lived deployment; flows call it via REST.

The key point: **Metaflow handles the infrastructure** (GPU scheduling, dependency packaging, parallel execution, artifact passing) so you can focus on the optimization problem itself. Each flow is a single Python file.

## Quick Start

```bash
cd cuopt-project
./run_all.sh           # all flows
./run_all.sh lp        # just farm planning
./run_all.sh vrp       # just delivery routing
./run_all.sh bench     # just LP solver benchmark
```

Or run individually:

```bash
PYTHONPATH="$PWD" METAFLOW_PROFILE=yellow \
  python flows/lp-minimal/flow.py --environment=fast-bakery run
```

## Architecture

```
cuopt-project/
├── obproject.toml              # Outerbounds project config
├── run_all.sh                  # Runner script
├── src/
│   ├── __init__.py             # METAFLOW_PACKAGE_POLICY = 'include'
│   └── problems/
│       ├── farm_lp.py          # Farm LP builder (sparse scipy CSR)
│       └── cvrp.py             # CVRP instance generator (clustered)
├── deployments/
│   └── cuopt-server/
│       └── config.yml          # Persistent GPU server (Outerbounds Deployment)
├── figures/
│   └── scaling_comparison.py   # Matplotlib figure from Metaflow artifacts
└── flows/
    ├── lp-minimal/flow.py      # Farm planning LP
    ├── qp-minimal/flow.py      # Portfolio MAD LP
    ├── milp-minimal/flow.py    # Warehouse location MILP
    ├── vrp-minimal/flow.py     # Delivery routing VRP (multi-scale)
    └── lp-benchmark/flow.py    # LP solver benchmark (4 solvers)
```

## Compute

- **GPU steps:** `gpu-multi-training` pool (A10G, 24GB VRAM)
- **CPU steps:** `c5-2x-task` pool
- **Conda:** cuOpt 25.08 from `rapidsai` and `nvidia` channels
