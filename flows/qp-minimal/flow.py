"""
Portfolio Optimization (LP) using NVIDIA cuOpt on GPU.

Mean-Absolute-Deviation portfolio optimization — an LP alternative to
Markowitz mean-variance that replaces quadratic variance with linear MAD risk.

Demonstrates:
    - cuOpt LP for a finance problem (portfolio optimization)
    - Efficient frontier construction by sweeping target returns
    - GPU-accelerated linear programming
    - MAD risk: a practical LP-based alternative to quadratic Markowitz
"""

from metaflow import step, card, Parameter, kubernetes, conda, current
from metaflow.profilers import gpu_profile
from metaflow.cards import Markdown, Table, VegaChart
from obproject import ProjectFlow
import src


CUOPT_CONDA = {
    "rapidsai::cudf": "25.08.*",
    "nvidia::cuopt-server": "25.08.*",
    "nvidia::cuopt-sh-client": "25.08.*",
}


class CuOptPortfolioFlow(ProjectFlow):
    """Portfolio optimization via MAD LP — minimize risk on GPU.

    Given N assets with historical returns, find the minimum-risk portfolio
    achieving a target return using Mean Absolute Deviation (MAD) as the
    risk measure. MAD is an LP-formable alternative to variance.

    Sweeps across return targets to build the efficient frontier.
    """

    n_assets = Parameter(
        "n_assets",
        help="Number of assets in the portfolio",
        default=200,
    )

    n_frontier_points = Parameter(
        "n_frontier_points",
        help="Number of points on the efficient frontier",
        default=20,
    )

    @conda(disabled=True)
    @step
    def start(self):
        """Generate synthetic asset data and target return levels."""
        import numpy as np

        np.random.seed(42)
        n = int(self.n_assets)
        T = 252  # trading days

        # Generate synthetic daily returns
        mu_annual = np.random.uniform(0.05, 0.25, n)
        vols = np.random.uniform(0.10, 0.40, n)
        daily_mu = mu_annual / T
        daily_vol = vols / np.sqrt(T)

        # Correlated returns via single-market-factor model
        # r_it = daily_mu_i + daily_vol_i * (sqrt(rho) * z_market + sqrt(1-rho) * z_idio)
        # All assets positively correlated — can't diversify away market risk
        rho = 0.4  # pairwise correlation (~typical equity market)
        z_market = np.random.randn(T)
        R = np.zeros((T, n))
        for i in range(n):
            z_idio = np.random.randn(T)
            R[:, i] = daily_mu[i] + daily_vol[i] * (
                np.sqrt(rho) * z_market + np.sqrt(1 - rho) * z_idio
            )
        self.returns_matrix = R.tolist()

        self.expected_returns = mu_annual.tolist()
        self.volatilities = vols.tolist()
        self.asset_names = [f"Asset_{i:02d}" for i in range(n)]
        self.n_obs = T

        # Sweep return targets across feasible range
        # Start high enough that the return constraint is binding at every point
        # (otherwise low-target points all collapse to the same min-risk portfolio)
        min_ret = mu_annual.min()
        max_ret = mu_annual.max()
        mean_ret = mu_annual.mean()
        self.return_targets = np.linspace(
            mean_ret * 0.6, max_ret * 0.95, int(self.n_frontier_points)
        ).tolist()

        print(
            f"Assets: {n}, Observations: {T}, Frontier points: {len(self.return_targets)}"
        )
        print(f"Return range: [{min_ret:.2%}, {max_ret:.2%}]")

        self.next(self.solve_frontier)

    @conda(packages=CUOPT_CONDA)
    @gpu_profile(interval=1)
    @kubernetes(gpu=1, compute_pool="gpu-multi-training")
    @step
    def solve_frontier(self):
        """Solve MAD LP for each target return to build the efficient frontier.

        MAD formulation:
            min  (1/T) * sum_t(d_t)
            s.t. d_t >= sum_i(w_i * (r_it - mu_i))    for t = 1..T
                 d_t >= -sum_i(w_i * (r_it - mu_i))   for t = 1..T
                 sum_i(w_i * mu_i) >= target_return
                 sum_i(w_i) = 1
                 w_i >= 0, d_t >= 0
        """
        import numpy as np
        import time
        from cuopt.linear_programming.problem import Problem, CONTINUOUS, MINIMIZE
        from cuopt.linear_programming.solver_settings import SolverSettings

        R = np.array(self.returns_matrix)  # T x n
        mu = np.array(self.expected_returns)
        n = len(mu)
        T = R.shape[0]

        # Compute deviation matrix: r_it - mu_i (daily returns minus mean)
        daily_mu = mu / T  # approximate daily expected return
        dev = R - R.mean(axis=0)  # deviations from sample mean

        self.frontier = []

        for target_return in self.return_targets:
            t0 = time.time()

            problem = Problem(f"MAD_{target_return:.4f}")

            # Portfolio weight variables: w_i in [0, 1]
            w = []
            for i in range(n):
                w.append(
                    problem.addVariable(
                        lb=0.0,
                        ub=1.0,
                        vtype=CONTINUOUS,
                        name=f"w_{i}",
                    )
                )

            # Auxiliary variables for absolute deviations: d_t >= 0
            d = []
            for t in range(T):
                d.append(
                    problem.addVariable(
                        lb=0.0,
                        ub=1e6,
                        vtype=CONTINUOUS,
                        name=f"d_{t}",
                    )
                )

            # Budget constraint: weights sum to 1
            problem.addConstraint(sum(w) == 1.0, name="budget")

            # Return target: sum(w_i * mu_i) >= target
            ret_expr = sum(float(mu[i]) * w[i] for i in range(n))
            problem.addConstraint(ret_expr >= target_return, name="target")

            # MAD constraints: d_t >= |sum_i(w_i * dev_it)|
            # Split into: d_t >= sum(...) and d_t + sum(...) >= 0
            for t in range(T):
                pos_dev = sum(float(dev[t, i]) * w[i] for i in range(n))
                neg_dev = sum(float(-dev[t, i]) * w[i] for i in range(n))
                problem.addConstraint(d[t] >= pos_dev, name=f"mad_pos_{t}")
                problem.addConstraint(d[t] >= neg_dev, name=f"mad_neg_{t}")

            # Objective: minimize mean absolute deviation
            obj = sum(d[t] for t in range(T)) * (1.0 / T)
            problem.setObjective(obj, sense=MINIMIZE)

            settings = SolverSettings()
            settings.set_parameter("time_limit", 60)
            problem.solve(settings)

            solve_time = time.time() - t0
            status = problem.Status.name

            if status == "Optimal":
                weights = np.array([v.Value for v in w])
                port_return = float(np.dot(weights, mu))
                mad_daily = problem.ObjValue
                # Annualize: MAD scales linearly, vol scales with sqrt(T)
                mad_annual = float(mad_daily * T)
                approx_vol_annual = float(mad_daily * np.sqrt(np.pi / 2) * np.sqrt(T))
                risk_free = 0.05
                sharpe = (
                    (port_return - risk_free) / approx_vol_annual
                    if approx_vol_annual > 0
                    else 0
                )

                self.frontier.append(
                    {
                        "target_return": target_return,
                        "actual_return": port_return,
                        "mad_risk": mad_annual,
                        "approx_vol": approx_vol_annual,
                        "sharpe": sharpe,
                        "weights": weights.tolist(),
                        "n_holdings": int(np.sum(weights > 0.001)),
                        "solve_time": solve_time,
                        "status": status,
                        "n_vars": n + T,
                        "n_constraints": 1 + 1 + 2 * T,
                    }
                )
                print(
                    f"  Target {target_return:.2%}: vol={approx_vol_annual:.2%}, "
                    f"sharpe={sharpe:.2f}, holdings={np.sum(weights > 0.001)}, "
                    f"time={solve_time:.4f}s"
                )
            else:
                self.frontier.append(
                    {
                        "target_return": target_return,
                        "status": status,
                        "solve_time": solve_time,
                    }
                )
                print(f"  Target {target_return:.2%}: {status}")

        self.next(self.build_card)

    @conda(disabled=True)
    @card(type="blank")
    @step
    def build_card(self):
        """Build visualization card."""
        import numpy as np

        feasible = [p for p in self.frontier if p["status"] == "Optimal"]

        current.card.append(Markdown("# Portfolio Optimization (MAD LP)"))
        current.card.append(
            Markdown(
                f"**{len(self.expected_returns)} assets** | "
                f"**{self.n_obs} observations** | "
                f"**{len(self.frontier)} frontier points** | "
                f"**Solver:** cuOpt LP (GPU)"
            )
        )

        if not feasible:
            current.card.append(Markdown("No feasible solutions found."))
            self.next(self.end)
            return

        # Efficient frontier chart
        current.card.append(Markdown("## Efficient Frontier"))

        best_idx = int(np.argmax([p["sharpe"] for p in feasible]))
        chart_data = []
        for i, p in enumerate(feasible):
            chart_data.append(
                {
                    "risk": p["approx_vol"] * 100,
                    "return": p["actual_return"] * 100,
                    "sharpe": p["sharpe"],
                    "holdings": p["n_holdings"],
                    "optimal": bool(i == best_idx),
                }
            )

        spec = {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "width": 500,
            "height": 300,
            "data": {"values": chart_data},
            "layer": [
                {
                    "mark": {"type": "line", "color": "#4C9878"},
                    "encoding": {
                        "x": {
                            "field": "risk",
                            "type": "quantitative",
                            "title": "Annualized Volatility (%)",
                        },
                        "y": {
                            "field": "return",
                            "type": "quantitative",
                            "title": "Expected Return (%)",
                        },
                    },
                },
                {
                    "mark": {"type": "point", "size": 80},
                    "encoding": {
                        "x": {"field": "risk", "type": "quantitative"},
                        "y": {"field": "return", "type": "quantitative"},
                        "color": {
                            "condition": {"test": "datum.optimal", "value": "#E6786C"},
                            "value": "#4C9878",
                        },
                        "tooltip": [
                            {"field": "return", "title": "Return (%)", "format": ".2f"},
                            {
                                "field": "risk",
                                "title": "Volatility (%)",
                                "format": ".2f",
                            },
                            {"field": "sharpe", "title": "Sharpe", "format": ".2f"},
                            {"field": "holdings", "title": "Holdings"},
                        ],
                    },
                },
            ],
        }
        current.card.append(VegaChart(spec))

        # Best portfolio
        best = feasible[best_idx]
        current.card.append(Markdown("## Best Portfolio (Max Sharpe)"))
        summary = [
            ["Expected Return", f"{best['actual_return']:.2%}"],
            ["Annualized Volatility", f"{best['approx_vol']:.2%}"],
            ["Sharpe Ratio", f"{best['sharpe']:.2f}"],
            ["Holdings", f"{best['n_holdings']}"],
            ["Solve Time", f"{best['solve_time']:.4f}s"],
        ]
        current.card.append(Table(summary, headers=["Metric", "Value"]))

        # Top holdings
        weights = np.array(best["weights"])
        top_idx = np.argsort(weights)[-10:][::-1]
        holdings = [
            [
                self.asset_names[i],
                f"{weights[i]:.2%}",
                f"{self.expected_returns[i]:.2%}",
            ]
            for i in top_idx
            if weights[i] > 0.001
        ]
        if holdings:
            current.card.append(Markdown("## Top Holdings"))
            current.card.append(
                Table(holdings, headers=["Asset", "Weight", "Exp Return"])
            )

        # Problem size info
        current.card.append(Markdown("## Problem Structure"))
        current.card.append(
            Markdown(
                f"MAD portfolio LP with {len(self.expected_returns)} assets "
                f"and {self.n_obs} return observations:\n"
                f"- **Variables:** {best.get('n_vars', 'N/A')} "
                f"({len(self.expected_returns)} weights + {self.n_obs} deviation vars)\n"
                f"- **Constraints:** {best.get('n_constraints', 'N/A')} "
                f"(budget + return target + 2x{self.n_obs} MAD)\n\n"
                f"*Note: MAD (Mean Absolute Deviation) is an LP-formable alternative "
                f"to Markowitz variance. cuOpt QP support available in 25.12+.*"
            )
        )

        # Frontier table
        current.card.append(Markdown("## Full Frontier"))
        rows = [
            [
                f"{p['actual_return']:.2%}",
                f"{p['approx_vol']:.2%}",
                f"{p['sharpe']:.2f}",
                f"{p['n_holdings']}",
                f"{p['solve_time']:.4f}s",
            ]
            for p in feasible
        ]
        current.card.append(
            Table(
                rows,
                headers=["Return", "Volatility", "Sharpe", "Holdings", "Solve Time"],
            )
        )

        # Solver performance
        total_time = sum(p["solve_time"] for p in feasible)
        current.card.append(Markdown("## Solver Performance"))
        current.card.append(
            Markdown(
                f"- **Total time:** {total_time:.2f}s for {len(feasible)} LP solves\n"
                f"- **Average per solve:** {total_time / len(feasible):.4f}s"
            )
        )

        self.next(self.end)

    @conda(disabled=True)
    @step
    def end(self):
        feasible = [p for p in self.frontier if p["status"] == "Optimal"]
        if feasible:
            best = max(feasible, key=lambda p: p["sharpe"])
            print(
                f"Best Sharpe: {best['sharpe']:.2f} at {best['actual_return']:.2%} return"
            )


if __name__ == "__main__":
    CuOptPortfolioFlow()
