# Deployment Patterns: Per-Step GPU vs Persistent Server

Two ways to run GPU-accelerated optimization on Outerbounds. Which one to use depends on your workload.

## Pattern A: Per-Step GPU (Metaflow Workflow)

Each optimization step provisions its own GPU pod, imports cuOpt as a library, solves the problem, and releases the GPU. This is the pattern used in all five demo flows.

```
┌───────┐     ┌──────────────┐     ┌──────────────┐     ┌─────────┐
│ start │────▶│ solve (GPU)  │────▶│ solve (GPU)  │────▶│ analyze │
│ (CPU) │     │ scenario 1   │     │ scenario 2   │     │  (CPU)  │
└───────┘     └──────────────┘     └──────────────┘     └─────────┘
                    each step gets its own GPU pod
```

**How it works:**
- `@kubernetes(gpu=1)` provisions a GPU pod for each solve step
- `@conda(packages={...})` installs cuOpt + RAPIDS into the step's environment
- cuOpt runs as a Python library — `from cuopt.linear_programming.problem import Problem`
- GPU is released when the step finishes

**When to use it:**
- **Batch workflows** — daily/weekly runs where you want the GPU only when solving
- **Parallel fan-out** — `foreach` launches N independent GPU pods; Metaflow handles scheduling
- **Heterogeneous steps** — GPU for cuOpt, CPU for data prep/charting
- **Infrequent runs** — pay for GPU only during the solve

**Trade-off:** Each GPU pod pays startup overhead (~30-60s). For a flow with 5 foreach tasks, that's 5x the startup cost.

## Pattern B: Persistent GPU Server (Outerbounds Deployment)

cuOpt runs as a long-lived REST API on a dedicated GPU. Flows submit problems via HTTP and poll for solutions. The GPU stays warm between requests.

```
┌──────────────────────────────────────────────────────────┐
│  Outerbounds Deployment: cuopt-server (GPU, always-on)   │
│  POST /cuopt/request  →  GET /cuopt/solution/{id}        │
└──────────────────────────────────────────────────────────┘
         ▲                    ▲                    ▲
         │ HTTP               │ HTTP               │ HTTP
    ┌────┴────┐          ┌────┴────┐          ┌────┴────┐
    │ solve 1 │          │ solve 2 │          │ solve 3 │
    │  (CPU)  │          │  (CPU)  │          │  (CPU)  │
    └─────────┘          └─────────┘          └─────────┘
              flow steps are CPU-only
```

**How it works:**
- cuOpt server runs as `outerbounds app deploy` with `resources.gpu: "1"`
- Flows are CPU-only — they POST scipy CSR data to the server and poll for solutions
- The server accepts LP, MILP, and VRP problems via REST (`/cuopt/request`)
- Multiple flows share the same GPU

**When to use it:**
- **High-frequency solving** — many small problems per hour; no pod startup per solve
- **Shared GPU across teams** — one GPU serves multiple flows and users
- **Interactive applications** — sub-second optimization responses for dashboards
- **Cost control** — one always-on GPU vs spinning up N GPU pods

**Trade-off:** REST API only accepts CSR matrix format for LP/MILP — not the algebraic `Problem.addVariable()` API. Always-on GPU costs money even when idle (mitigated by autoscaling to zero).

## When to Use Which

| Factor | Per-Step GPU | Persistent Server |
|--------|-------------|-------------------|
| Run frequency | Daily/weekly batch | Continuous/hourly |
| Problem construction | Algebraic API or CSR | CSR matrices only |
| Concurrent solves | foreach fan-out (N GPUs) | Sequential on 1 GPU |
| Startup latency | ~30-60s per pod | ~10ms per request |
| Cost model | Pay per solve-minute | Pay for always-on GPU |
| Best for | Batch pipelines, benchmarks | APIs, dashboards, microservices |

In practice, many teams use both: **per-step for batch pipelines** (backtesting, what-if analysis) and **server for real-time applications** (route optimization API, portfolio rebalancing).

## Server Deployment

The deployment config lives at `deployments/cuopt-server/config.yml`. Deploy with:

```bash
outerbounds app deploy --config-file deployments/cuopt-server/config.yml
```

See `src/clients/cuopt_server.py` for the client that flows use to call the server with Outerbounds auth.
