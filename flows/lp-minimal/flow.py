"""
LP: Farm Planning Optimization using NVIDIA cuOpt on GPU.

Large-scale linear program — maximize profit from planting crops across
thousands of farm fields, subject to local land/water limits and shared
resource pool constraints that couple all regions together.

Demonstrates:
    - cuOpt LP problem construction at scale (40K+ variables)
    - Dense coupling constraints (shared labor pools across all regions)
    - Solver settings (method selection, time limits)
    - Solution extraction (values, status, solve time)
    - Sensitivity analysis via cost-multiplier scenario sweep
"""

from metaflow import step, card, Parameter, kubernetes, conda, current
from metaflow.profilers import gpu_profile
from metaflow.cards import Markdown, Table, VegaChart
from obproject import ProjectFlow
import src


CUOPT_CONDA = {
    'rapidsai::cudf': '25.08.*',
    'nvidia::cuopt-server': '25.08.*',
    'nvidia::cuopt-sh-client': '25.08.*',
}


class CuOptLPFlow(ProjectFlow):
    """LP: Multi-region farm planning — maximize profit from crop allocation on GPU.

    An agricultural company manages thousands of farm fields. Each field has
    its own land and water limits. Five shared labor pools and per-crop market
    capacity constraints couple all fields together, creating a large dense LP.

    Variables: n_regions x 4 crops.
    Constraints: n_regions x 2 (local) + 4 (market caps) + 5 (shared labor pools).
    The scenario sweep tests sensitivity to crop price changes.
    """

    n_regions = Parameter(
        "n_regions",
        help="Number of farm regions (scales LP size: vars = n_regions x 4 crops)",
        default=10000,
    )

    cost_multipliers = Parameter(
        "cost_multipliers",
        help="Comma-separated price multipliers for scenario sweep (e.g., '0.8,1.0,1.2')",
        default="0.7,0.85,1.0,1.15,1.3",
    )

    @conda(disabled=True)
    @step
    def start(self):
        """Set up scenarios and fan out."""
        self.scenarios = [float(x) for x in self.cost_multipliers.split(",")]
        n_reg = int(self.n_regions)
        n_vars = n_reg * 4
        n_cons = n_reg * 2 + 4 + 5  # per-region (land + water) + market caps + labor pools
        print(f"Regions: {n_reg}, Variables: {n_vars}, Constraints: {n_cons}")
        print(f"Price scenarios: {self.scenarios}")
        self.next(self.solve_scenario, foreach="scenarios")

    @conda(packages=CUOPT_CONDA)
    @gpu_profile(interval=1)
    @kubernetes(gpu=1, compute_pool='gpu-multi-training')
    @step
    def solve_scenario(self):
        """Solve one price scenario on GPU with cuOpt."""
        import numpy as np
        import time
        from cuopt.linear_programming.problem import Problem, CONTINUOUS, MAXIMIZE
        from cuopt.linear_programming.solver_settings import SolverSettings

        price_mult = self.input
        n_reg = int(self.n_regions)

        # --- Problem data ---
        crops = ["Wheat", "Corn", "Soybeans", "Cotton"]
        n_crops = len(crops)

        np.random.seed(42)

        # Per-region variation: base profit with regional multipliers
        base_profit = np.array([180.0, 220.0, 200.0, 160.0])
        region_mult = 0.8 + 0.4 * np.random.rand(n_reg)
        profit = np.outer(region_mult, base_profit) * price_mult

        # Resource usage per acre
        labor_per_acre = np.array([10.0, 14.0, 8.0, 12.0])
        water_per_acre = np.array([2.5, 3.5, 2.0, 4.0])

        # Per-region limits
        land_limit = 500.0 + 200.0 * np.random.rand(n_reg)
        water_limit = 1500.0 + 500.0 * np.random.rand(n_reg)

        # Shared labor pools: 5 skill types used across all regions
        n_pools = 5
        saved_state = np.random.get_state()
        np.random.seed(99)
        pool_usage = np.random.uniform(0.5, 2.0, (n_pools, n_crops))
        np.random.set_state(saved_state)
        pool_capacity = np.array([
            float(pool_usage[k].sum()) * 150.0 * n_reg for k in range(n_pools)
        ])

        # Global market caps per crop (limits total supply across all regions)
        market_cap = 200.0 * n_reg

        min_acres, max_acres = 10.0, 400.0

        # --- Build cuOpt problem ---
        t0 = time.time()
        problem = Problem("MultiRegionFarm")

        # Decision variables: x[r][c] = acres of crop c in region r
        x = []
        for r in range(n_reg):
            row = []
            for c in range(n_crops):
                row.append(problem.addVariable(
                    lb=min_acres, ub=max_acres,
                    vtype=CONTINUOUS, name=f"x_{r}_{c}",
                ))
            x.append(row)

        # Objective: maximize total profit across all regions
        obj = sum(
            float(profit[r, c]) * x[r][c]
            for r in range(n_reg) for c in range(n_crops)
        )
        problem.setObjective(obj, sense=MAXIMIZE)

        # Per-region land constraints
        for r in range(n_reg):
            expr = sum(x[r][c] for c in range(n_crops))
            problem.addConstraint(expr <= float(land_limit[r]), name=f"land_{r}")

        # Per-region water constraints
        for r in range(n_reg):
            expr = sum(float(water_per_acre[c]) * x[r][c] for c in range(n_crops))
            problem.addConstraint(expr <= float(water_limit[r]), name=f"water_{r}")

        # Global market cap per crop (couples all regions — dense constraints)
        for c in range(n_crops):
            cap_expr = sum(x[r][c] for r in range(n_reg))
            problem.addConstraint(cap_expr <= market_cap, name=f"market_{crops[c]}")

        # Shared labor pools (each pool uses all regions x crops — dense coupling)
        for k in range(n_pools):
            pool_expr = sum(
                float(pool_usage[k, c]) * x[r][c]
                for r in range(n_reg) for c in range(n_crops)
            )
            problem.addConstraint(pool_expr <= float(pool_capacity[k]),
                                  name=f"labor_pool_{k}")

        build_time = time.time() - t0
        n_vars = n_reg * n_crops
        n_cons = n_reg * 2 + n_crops + n_pools

        # --- Solve ---
        settings = SolverSettings()
        settings.set_parameter("time_limit", 120)

        t1 = time.time()
        problem.solve(settings)
        solve_time = time.time() - t1

        status = problem.Status.name

        if status == "Optimal":
            total_profit = problem.ObjValue

            # Extract all variable values for efficient aggregation
            vals = np.array([
                [x[r][c].Value for c in range(n_crops)] for r in range(n_reg)
            ])

            crop_totals = {crops[c]: float(vals[:, c].sum()) for c in range(n_crops)}
            total_land = float(vals.sum())
            total_labor = float((vals * labor_per_acre).sum())

            pool_utils = []
            for k in range(n_pools):
                used = float((vals * pool_usage[k]).sum())
                pool_utils.append(used / float(pool_capacity[k]))

            self.result = {
                "scenario": price_mult,
                "status": status,
                "total_profit": float(total_profit),
                "crop_totals": crop_totals,
                "total_land": total_land,
                "total_labor": total_labor,
                "pool_utilizations": pool_utils,
                "avg_pool_util": float(np.mean(pool_utils)),
                "solve_time": solve_time,
                "build_time": build_time,
                "n_vars": n_vars,
                "n_constraints": n_cons,
                "n_regions": n_reg,
            }
        else:
            self.result = {
                "scenario": price_mult,
                "status": status,
                "total_profit": None,
                "solve_time": solve_time,
                "build_time": build_time,
                "n_vars": n_vars,
                "n_constraints": n_cons,
                "n_regions": n_reg,
            }

        profit_val = self.result.get('total_profit') or 0
        print(f"Scenario {price_mult:.0%}: {status}, profit=${profit_val:,.0f}, "
              f"vars={n_vars}, constraints={n_cons}, "
              f"build={build_time:.2f}s, solve={solve_time:.4f}s")

        self.next(self.join)

    @conda(disabled=True)
    @card(type='blank')
    @step
    def join(self, inputs):
        """Aggregate results and build card."""
        self.results = sorted(
            [i.result for i in inputs], key=lambda r: r["scenario"]
        )
        feasible = [r for r in self.results if r.get("total_profit") is not None]
        baseline = next((r for r in feasible if r["scenario"] == 1.0), feasible[0] if feasible else None)

        # --- Card ---
        current.card.append(Markdown("# LP: Multi-Region Farm Planning"))

        if baseline:
            current.card.append(Markdown(
                f"**{baseline['n_regions']} regions** | "
                f"**{baseline['n_vars']} variables** | "
                f"**{baseline['n_constraints']} constraints** | "
                f"**Solver:** cuOpt LP (GPU)"
            ))

            # Baseline summary
            current.card.append(Markdown("## Baseline Solution (1.0x prices)"))
            rows = [[crop, f"{acres:,.0f}"] for crop, acres in baseline["crop_totals"].items()]
            current.card.append(Table(rows, headers=["Crop", "Total Acres (all regions)"]))

            current.card.append(Markdown(f"**Total Profit:** ${baseline['total_profit']:,.0f}"))
            current.card.append(Markdown(f"**Solve Time:** {baseline['solve_time']:.4f}s"))
            current.card.append(Markdown(f"**Build Time:** {baseline['build_time']:.2f}s"))

            # Shared resource pool utilization
            if baseline.get('pool_utilizations'):
                current.card.append(Markdown("## Shared Resource Pools"))
                pool_names = ["Manual Labor", "Skilled Labor", "Equipment", "Transport", "Storage"]
                pool_rows = [
                    [pool_names[i], f"{u:.1%}"]
                    for i, u in enumerate(baseline['pool_utilizations'])
                ]
                current.card.append(Table(pool_rows, headers=["Labor Pool", "Utilization"]))
                current.card.append(Markdown(f"**Avg Pool Utilization:** {baseline['avg_pool_util']:.1%}"))

        # Scenario sweep chart
        if len(feasible) > 1:
            current.card.append(Markdown("## Price Sensitivity"))
            chart_data = [
                {"price_mult": f"{r['scenario']:.0%}", "profit": r["total_profit"]}
                for r in feasible
            ]
            spec = {
                "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                "width": 400, "height": 250,
                "data": {"values": chart_data},
                "mark": {"type": "bar", "color": "#4C9878"},
                "encoding": {
                    "x": {"field": "price_mult", "type": "nominal", "title": "Price Multiplier", "sort": None},
                    "y": {"field": "profit", "type": "quantitative", "title": "Total Profit ($)"},
                    "tooltip": [
                        {"field": "price_mult", "title": "Price"},
                        {"field": "profit", "title": "Profit ($)", "format": ",.0f"},
                    ],
                },
            }
            current.card.append(VegaChart(spec))

            # Scenario table
            scenario_rows = []
            for r in feasible:
                delta = ((r["total_profit"] - baseline["total_profit"]) / baseline["total_profit"] * 100) if baseline else 0
                scenario_rows.append([
                    f"{r['scenario']:.0%}",
                    f"${r['total_profit']:,.0f}",
                    f"{delta:+.1f}%",
                    f"{r['solve_time']:.4f}s",
                ])
            current.card.append(Table(scenario_rows, headers=["Price", "Profit", "vs Baseline", "Solve Time"]))

        self.next(self.end)

    @conda(disabled=True)
    @step
    def end(self):
        baseline = next((r for r in self.results if r["scenario"] == 1.0), None)
        if baseline and baseline.get("total_profit"):
            print(f"Optimal profit at baseline prices: ${baseline['total_profit']:,.2f}")


if __name__ == "__main__":
    CuOptLPFlow()
