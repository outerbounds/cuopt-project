# Portfolio Optimization with NVIDIA cuOpt

GPU-accelerated portfolio optimization using NVIDIA cuOpt on Outerbounds.

## What This Project Does

Solves large-scale portfolio optimization problems (1000+ assets, 2000+ scenarios) in seconds using cuOpt's GPU-accelerated linear programming solver on H100 GPUs. The example implements **CVaR (Conditional Value-at-Risk) optimization** - a risk-aware approach that optimizes for tail risk rather than just variance.

## Quick Start

```bash
# Run the portfolio optimization flow
python flows/portfolio/flow.py run
```

The flow will:
1. Fetch S&P 500 historical price data
2. Generate Monte Carlo scenarios for future returns
3. Solve CVaR optimization on GPU using cuOpt
4. Produce a Metaflow card with efficient frontier, portfolio weights, and risk metrics

## Why GPU Optimization?

Traditional portfolio optimization with 500 assets and 2000 scenarios creates an LP with ~1M constraints. CPU solvers take minutes; cuOpt on H100 solves in seconds. This enables:

- **Real-time rebalancing** as market conditions change
- **Scenario sweeps** across risk tolerance levels
- **Monte Carlo at scale** with thousands of simulations

## Project Structure

```
optimization-project/
├── flows/
│   └── portfolio/
│       └── flow.py          # Main optimization flow
├── src/
│   ├── optimization.py      # cuOpt problem formulation
│   ├── data.py              # Market data fetching
│   └── cards.py             # Visualization components
└── README.md
```

## The Optimization Problem

We solve the CVaR portfolio optimization:

```
maximize:  μᵀw - λ·CVaR_α(w)
subject to: Σwᵢ = 1  (fully invested)
            wᵢ ≥ 0   (long only)
```

Where:
- `w` = portfolio weights
- `μ` = expected returns
- `λ` = risk aversion parameter
- `CVaR_α` = expected loss in the worst α% of scenarios

---

## Future Vision

This project demonstrates a single powerful flow. The architecture could expand to a full **Portfolio Strategy Registry**:

```
IngestMarketData → BuildScenarios → OptimizePortfolio → BacktestStrategy → PromoteStrategy
       ↓                ↓                  ↓                   ↓
   market_data     scenarios        portfolio_weights     backtest_results
   (data asset)   (data asset)       (model asset)        (evaluation)
```

### Potential Extensions

**Multi-Flow Pipeline**
- Scheduled market data ingestion
- Triggered re-optimization when data updates
- Backtesting with quality gates (max drawdown, Sharpe thresholds)
- Champion/challenger pattern for production portfolios

**Enterprise Use Cases**
- **Asset-backed lending**: Optimal loan allocation across warehouse facilities
- **Private credit**: Portfolio construction with covenant constraints
- **Treasury management**: Cash flow optimization across funding sources

### Collaboration Opportunities

This project explores patterns relevant to quantitative finance workflows:
- NVIDIA cuOpt for GPU-accelerated optimization
- Outerbounds for reproducible, versioned optimization runs
- Rich visualization for portfolio analytics

---

## References

- [NVIDIA cuOpt Examples](https://github.com/NVIDIA/cuopt-examples)
- [CVaR Portfolio Optimization](https://github.com/NVIDIA/cuopt-examples/tree/main/portfolio_optimization)
- [Outerbounds Documentation](https://docs.outerbounds.com)
