# Portfolio Optimization with NVIDIA cuOpt

GPU-accelerated CVaR portfolio optimization using NVIDIA cuOpt on H200 GPUs.

## Quick Start

```bash
python flows/portfolio/flow.py --no-pylint --environment=fast-bakery run --with kubernetes
```

## What It Does

Solves CVaR (Conditional Value-at-Risk) portfolio optimization on GPU:
- Fetches historical price data (or generates synthetic data)
- Generates Monte Carlo scenarios for risk estimation
- Solves LP optimization using cuOpt's GPU-accelerated solver
- Sweeps across risk levels to build an efficient frontier
- Produces a Metaflow card with visualizations

## Project Structure

```
optimization-project/
├── flows/portfolio/flow.py   # Main optimization flow
├── src/__init__.py           # Package policy for Metaflow
└── README.md
```

## Parameters

- `--n_assets`: Number of assets (default: 100)
- `--n_scenarios`: Monte Carlo scenarios (default: 1000)
- `--confidence`: CVaR confidence level (default: 0.95)
