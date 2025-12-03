"""
Portfolio Optimization Flow using NVIDIA cuOpt.

Solves large-scale CVaR (Conditional Value-at-Risk) portfolio optimization
on GPU, demonstrating the power of H100 for financial optimization.

The flow:
1. Fetches S&P 500 historical prices
2. Computes returns and generates Monte Carlo scenarios
3. Solves CVaR optimization using cuOpt's GPU-accelerated LP solver
4. Sweeps across risk levels to build efficient frontier
5. Produces a rich Metaflow card with visualizations
"""

from metaflow import step, card, Parameter, kubernetes, conda, current
from metaflow.profilers import gpu_profile
from metaflow.cards import Markdown, Table, VegaChart
from obproject import ProjectFlow
import src


class PortfolioOptimizationFlow(ProjectFlow):
    """
    GPU-accelerated portfolio optimization with CVaR risk measure.

    Demonstrates solving a large LP (500 assets x 2000 scenarios = 1M constraints)
    in seconds on H100 GPU using NVIDIA cuOpt.
    """

    n_assets = Parameter(
        "n_assets",
        help="Number of assets to include (max ~500 for S&P 500)",
        default=100,
    )

    n_scenarios = Parameter(
        "n_scenarios",
        help="Number of Monte Carlo scenarios for CVaR estimation",
        default=1000,
    )

    confidence_level = Parameter(
        "confidence",
        help="CVaR confidence level (e.g., 0.95 = worst 5% of scenarios)",
        default=0.95,
    )

    @conda(packages={'pandas': '>=2.0', 'numpy': '>=1.24'})
    @step
    def start(self):
        """Fetch S&P 500 price data and compute returns."""
        import pandas as pd
        import numpy as np

        print(f"Fetching S&P 500 data for top {self.n_assets} assets...")

        # Fetch S&P 500 constituent prices
        # Using a public dataset for reproducibility
        url = "https://raw.githubusercontent.com/NVIDIA/cuopt-examples/main/portfolio_optimization/data/sp500_closefull.csv"

        try:
            prices = pd.read_csv(url, index_col=0, parse_dates=True)
            print(f"Loaded {len(prices.columns)} assets, {len(prices)} trading days")
        except Exception as e:
            print(f"Failed to fetch data: {e}")
            print("Using synthetic data instead...")
            # Generate synthetic price data if fetch fails
            np.random.seed(42)
            n_days = 252 * 2  # 2 years
            tickers = [f"ASSET_{i:03d}" for i in range(self.n_assets)]
            returns = np.random.randn(n_days, self.n_assets) * 0.02 + 0.0003
            prices = pd.DataFrame(
                100 * np.exp(np.cumsum(returns, axis=0)),
                columns=tickers,
                index=pd.date_range(end=pd.Timestamp.now(), periods=n_days, freq='B')
            )

        # Select top N assets by available data
        prices = prices.dropna(axis=1, how='any')
        self.tickers = list(prices.columns[:self.n_assets])
        prices = prices[self.tickers]

        # Compute log returns
        self.returns = np.log(prices / prices.shift(1)).dropna().values
        self.n_assets_actual = len(self.tickers)
        self.n_observations = len(self.returns)

        print(f"Selected {self.n_assets_actual} assets with {self.n_observations} return observations")

        # Compute expected returns and covariance (annualized)
        self.expected_returns = np.mean(self.returns, axis=0) * 252
        self.covariance = np.cov(self.returns.T) * 252

        self.next(self.generate_scenarios)

    @conda(packages={'numpy': '>=1.24'})
    @step
    def generate_scenarios(self):
        """Generate Monte Carlo scenarios for CVaR estimation."""
        import numpy as np

        print(f"Generating {self.n_scenarios} Monte Carlo scenarios...")

        np.random.seed(42)

        # Use historical returns + Monte Carlo simulation
        # Combine historical scenarios with simulated ones
        n_historical = min(len(self.returns), self.n_scenarios // 2)
        historical_scenarios = self.returns[-n_historical:]

        # Generate additional scenarios from multivariate normal
        n_simulated = self.n_scenarios - n_historical
        mean = np.mean(self.returns, axis=0)
        cov = np.cov(self.returns.T)

        simulated_scenarios = np.random.multivariate_normal(
            mean, cov, size=n_simulated
        )

        self.scenarios = np.vstack([historical_scenarios, simulated_scenarios])
        print(f"Generated {len(self.scenarios)} total scenarios")
        print(f"  - {n_historical} historical")
        print(f"  - {n_simulated} simulated")

        self.next(self.optimize)

    @conda(packages={
        'rapidsai::cudf': '25.08.*',
        'nvidia::cuopt-server': '25.08.*',
        'nvidia::cuopt-sh-client': '25.08.*',
    })
    @gpu_profile(interval=1)
    @kubernetes(gpu=1, compute_pool='obp-nebius-h200-1gpu')
    @step
    def optimize(self):
        """Solve CVaR portfolio optimization on GPU using cuOpt."""
        import numpy as np
        import time

        from cuopt.linear_programming.problem import Problem, CONTINUOUS, MAXIMIZE
        from cuopt.linear_programming.solver_settings import SolverSettings

        print(f"\n{'='*60}")
        print("NVIDIA cuOpt Portfolio Optimization")
        print(f"{'='*60}")
        print(f"Assets: {self.n_assets_actual}")
        print(f"Scenarios: {len(self.scenarios)}")
        print(f"Confidence level: {self.confidence_level}")
        print(f"Problem size: ~{self.n_assets_actual * len(self.scenarios):,} constraints")

        # Sweep across risk aversion levels to build efficient frontier
        risk_aversions = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
        self.frontier_results = []

        alpha = 1 - self.confidence_level  # tail probability
        n_scenarios = len(self.scenarios)

        for lambda_risk in risk_aversions:
            start_time = time.time()

            # Build CVaR optimization problem
            # Variables: w (weights), t (VaR threshold), u (CVaR auxiliary)
            problem = Problem("CVaR_Portfolio")

            # Portfolio weights: w_i in [0, 1]
            w_vars = []
            for i in range(self.n_assets_actual):
                w_vars.append(problem.addVariable(
                    lb=0.0, ub=1.0, vtype=CONTINUOUS, name=f"w_{i}"
                ))

            # VaR threshold: t (large bounds for practical unboundedness)
            t_var = problem.addVariable(lb=-1e6, ub=1e6, vtype=CONTINUOUS, name="t")

            # CVaR auxiliary variables: u_s >= 0 for each scenario
            u_vars = []
            for s in range(n_scenarios):
                u_vars.append(problem.addVariable(
                    lb=0.0, ub=1e6, vtype=CONTINUOUS, name=f"u_{s}"
                ))

            # Constraint 1: sum(w) = 1 (fully invested)
            budget_expr = sum(w_vars)
            problem.addConstraint(budget_expr == 1.0, name="budget")

            # Constraint 2: u_s >= -r_s^T w - t for each scenario
            # Rearranged: u_s + t + r_s^T w >= 0
            for s in range(n_scenarios):
                scenario_expr = u_vars[s] + t_var
                for i in range(self.n_assets_actual):
                    scenario_expr = scenario_expr + self.scenarios[s, i] * w_vars[i]
                problem.addConstraint(scenario_expr >= 0.0, name=f"cvar_{s}")

            # Objective: maximize expected return - lambda * CVaR
            # CVaR = t + (1/alpha) * E[u]
            obj_expr = sum(self.expected_returns[i] * w_vars[i] for i in range(self.n_assets_actual))
            obj_expr = obj_expr - lambda_risk * t_var
            obj_expr = obj_expr - lambda_risk / (n_scenarios * alpha) * sum(u_vars)

            problem.setObjective(obj_expr, sense=MAXIMIZE)

            # Solve
            settings = SolverSettings()
            result = problem.solve(settings)

            solve_time = time.time() - start_time

            # Extract solution
            weights = np.array([w.Value for w in w_vars])
            var_threshold = t_var.Value

            # Compute portfolio metrics
            portfolio_return = np.dot(weights, self.expected_returns)
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(self.covariance, weights)))

            # Compute actual CVaR from scenarios
            scenario_returns = self.scenarios @ weights
            var_pct = np.percentile(scenario_returns, alpha * 100)
            cvar = -np.mean(scenario_returns[scenario_returns <= var_pct])

            self.frontier_results.append({
                'lambda': lambda_risk,
                'return': float(portfolio_return),
                'volatility': float(portfolio_vol),
                'cvar': float(cvar),
                'var': float(-var_pct),
                'sharpe': float(portfolio_return / portfolio_vol) if portfolio_vol > 0 else 0,
                'weights': weights.tolist(),
                'solve_time': solve_time,
                'n_nonzero': int(np.sum(weights > 0.001)),
            })

            print(f"  λ={lambda_risk:5.1f}: Return={portfolio_return:6.2%}, "
                  f"Vol={portfolio_vol:6.2%}, CVaR={cvar:6.2%}, "
                  f"Time={solve_time:.2f}s, Holdings={np.sum(weights > 0.001)}")

        # Select the portfolio with best Sharpe ratio as "optimal"
        best_idx = np.argmax([r['sharpe'] for r in self.frontier_results])
        self.optimal_portfolio = self.frontier_results[best_idx]
        self.optimal_weights = np.array(self.optimal_portfolio['weights'])

        print(f"\nOptimal portfolio (max Sharpe):")
        print(f"  Expected Return: {self.optimal_portfolio['return']:.2%}")
        print(f"  Volatility: {self.optimal_portfolio['volatility']:.2%}")
        print(f"  Sharpe Ratio: {self.optimal_portfolio['sharpe']:.2f}")
        print(f"  CVaR (95%): {self.optimal_portfolio['cvar']:.2%}")

        self.next(self.build_card)

    @conda(packages={'numpy': '>=1.24'})
    @card(type='blank')
    @step
    def build_card(self):
        """Build rich visualization card."""
        import numpy as np

        # === Header ===
        current.card.append(Markdown("# Portfolio Optimization Results"))
        current.card.append(Markdown(
            f"**{self.n_assets_actual} assets** | "
            f"**{len(self.scenarios):,} scenarios** | "
            f"**{self.confidence_level:.0%} CVaR confidence**"
        ))

        # === What is this? ===
        current.card.append(Markdown("## What This Solves"))
        current.card.append(Markdown(
            "This flow solves **CVaR (Conditional Value-at-Risk) portfolio optimization** — "
            "a risk-aware approach that optimizes for tail risk rather than just variance.\n\n"
            "**The optimization problem:**\n"
            "- **Objective:** Maximize expected return minus a risk penalty\n"
            "- **Risk measure:** CVaR captures the expected loss in the worst 5% of scenarios\n"
            "- **Constraints:** Weights must sum to 1 (fully invested), no short selling\n\n"
            f"**Problem scale:** {self.n_assets_actual} assets × {len(self.scenarios):,} scenarios = "
            f"**{self.n_assets_actual * len(self.scenarios):,} constraints** in the LP formulation. "
            "Each scenario adds a constraint to linearize the CVaR risk measure."
        ))

        # === Optimal Portfolio Summary ===
        current.card.append(Markdown("## Optimal Portfolio (Max Sharpe)"))

        summary_rows = [
            ["Expected Annual Return", f"{self.optimal_portfolio['return']:.2%}"],
            ["Annual Volatility", f"{self.optimal_portfolio['volatility']:.2%}"],
            ["Sharpe Ratio", f"{self.optimal_portfolio['sharpe']:.2f}"],
            ["CVaR (95%)", f"{self.optimal_portfolio['cvar']:.2%}"],
            ["VaR (95%)", f"{self.optimal_portfolio['var']:.2%}"],
            ["Number of Holdings", f"{self.optimal_portfolio['n_nonzero']}"],
            ["Solve Time", f"{self.optimal_portfolio['solve_time']:.2f}s"],
        ]
        current.card.append(Table(summary_rows, headers=["Metric", "Value"]))

        # === Efficient Frontier Chart ===
        current.card.append(Markdown("## Efficient Frontier"))
        current.card.append(Markdown(
            "*Each point represents an optimal portfolio at a different risk aversion level. "
            "The highlighted point is the maximum Sharpe ratio portfolio.*"
        ))

        frontier_data = []
        best_idx = int(np.argmax([x['sharpe'] for x in self.frontier_results]))
        for i, r in enumerate(self.frontier_results):
            frontier_data.append({
                "volatility": r['volatility'] * 100,
                "return": r['return'] * 100,
                "sharpe": r['sharpe'],
                "lambda": r['lambda'],
                "optimal": bool(i == best_idx)  # Convert to Python bool for JSON
            })

        frontier_spec = {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "width": 500,
            "height": 300,
            "data": {"values": frontier_data},
            "layer": [
                {
                    "mark": {"type": "line", "color": "#4C9878"},
                    "encoding": {
                        "x": {"field": "volatility", "type": "quantitative", "title": "Volatility (%)"},
                        "y": {"field": "return", "type": "quantitative", "title": "Expected Return (%)"},
                    }
                },
                {
                    "mark": {"type": "point", "size": 100},
                    "encoding": {
                        "x": {"field": "volatility", "type": "quantitative"},
                        "y": {"field": "return", "type": "quantitative"},
                        "color": {
                            "condition": {"test": "datum.optimal", "value": "#E6786C"},
                            "value": "#4C9878"
                        },
                        "tooltip": [
                            {"field": "return", "title": "Return (%)", "format": ".2f"},
                            {"field": "volatility", "title": "Volatility (%)", "format": ".2f"},
                            {"field": "sharpe", "title": "Sharpe", "format": ".2f"},
                            {"field": "lambda", "title": "Risk Aversion"}
                        ]
                    }
                }
            ]
        }
        current.card.append(VegaChart(frontier_spec))

        # === Top Holdings ===
        current.card.append(Markdown("## Top 10 Holdings"))

        top_indices = np.argsort(self.optimal_weights)[-10:][::-1]
        holdings_rows = []
        for idx in top_indices:
            weight = self.optimal_weights[idx]
            if weight > 0.001:
                holdings_rows.append([
                    self.tickers[idx],
                    f"{weight:.2%}",
                    f"{self.expected_returns[idx]:.2%}",
                ])

        if holdings_rows:
            current.card.append(Table(
                holdings_rows,
                headers=["Ticker", "Weight", "Expected Return"]
            ))

        # === Holdings Distribution Chart ===
        current.card.append(Markdown("## Portfolio Composition"))

        holdings_data = []
        for idx in top_indices[:10]:
            if self.optimal_weights[idx] > 0.001:
                holdings_data.append({
                    "ticker": self.tickers[idx],
                    "weight": self.optimal_weights[idx] * 100
                })

        if holdings_data:
            holdings_spec = {
                "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                "width": 400,
                "height": 250,
                "data": {"values": holdings_data},
                "mark": "bar",
                "encoding": {
                    "x": {"field": "weight", "type": "quantitative", "title": "Weight (%)"},
                    "y": {"field": "ticker", "type": "nominal", "title": "Asset", "sort": "-x"},
                    "color": {"value": "#4C9878"},
                    "tooltip": [
                        {"field": "ticker", "title": "Ticker"},
                        {"field": "weight", "title": "Weight (%)", "format": ".2f"}
                    ]
                }
            }
            current.card.append(VegaChart(holdings_spec))

        # === Performance Comparison ===
        current.card.append(Markdown("## Risk-Return Tradeoff Across Portfolios"))

        tradeoff_rows = []
        for r in self.frontier_results:
            tradeoff_rows.append([
                f"{r['lambda']:.1f}",
                f"{r['return']:.2%}",
                f"{r['volatility']:.2%}",
                f"{r['cvar']:.2%}",
                f"{r['sharpe']:.2f}",
                f"{r['n_nonzero']}",
                f"{r['solve_time']:.2f}s",
            ])

        current.card.append(Table(
            tradeoff_rows,
            headers=["Risk Aversion", "Return", "Volatility", "CVaR", "Sharpe", "Holdings", "Solve Time"]
        ))

        # === GPU Performance ===
        current.card.append(Markdown("## Solver Performance"))
        total_time = sum(r['solve_time'] for r in self.frontier_results)
        avg_time = total_time / len(self.frontier_results)
        current.card.append(Markdown(
            f"- **Total optimization time:** {total_time:.2f}s\n"
            f"- **Average per portfolio:** {avg_time:.2f}s\n"
            f"- **Problem size:** {self.n_assets_actual} assets x {len(self.scenarios):,} scenarios\n"
            f"- **Approximate constraints:** {self.n_assets_actual * len(self.scenarios):,}"
        ))

        self.next(self.end)

    @step
    def end(self):
        """Done."""
        print(f"\n{'='*60}")
        print("Portfolio Optimization Complete")
        print(f"{'='*60}")
        print(f"Optimal Sharpe Ratio: {self.optimal_portfolio['sharpe']:.2f}")
        print(f"Expected Return: {self.optimal_portfolio['return']:.2%}")
        print(f"View the Metaflow card for detailed visualizations.")


if __name__ == "__main__":
    PortfolioOptimizationFlow()
