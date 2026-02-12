"""
MILP: Capacitated Facility Location using NVIDIA cuOpt on GPU.

Choose which facilities to open (binary) and assign customers respecting
capacity limits. Minimizes total fixed + transport cost.

Demonstrates:
    - cuOpt INTEGER variables (binary open/close decisions)
    - Capacity constraints that create genuine LP relaxation gap
    - GPU heuristics (fast, good) vs full branch-and-bound (optimal, slower)
    - Quality-vs-speed tradeoff in combinatorial optimization
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


class CuOptMILPFlow(ProjectFlow):
    """Capacitated facility location — open/close + assign on GPU.

    Given candidate facility sites with limited capacity and customer
    locations with demand, decide which facilities to open and how to
    assign customers to minimize total fixed + transport cost.
    Capacity constraints create a hard MILP with genuine optimality gap.
    """

    n_facilities = Parameter("n_facilities", default=100)
    n_customers = Parameter("n_customers", default=1000)

    def _build_problem(self, name):
        """Build the CFLP MILP model. Called inside conda-enabled steps."""
        from cuopt.linear_programming.problem import (
            Problem, CONTINUOUS, INTEGER, MINIMIZE,
        )

        nf = len(self.facility_x)
        nc = len(self.customer_x)

        problem = Problem(name)

        # Binary: open facility j?
        y = [problem.addVariable(lb=0, ub=1, vtype=INTEGER, name=f"y_{j}")
             for j in range(nf)]

        # Continuous: fraction of customer i served by facility j
        x = [[problem.addVariable(lb=0.0, ub=1.0, vtype=CONTINUOUS,
                                   name=f"x_{i}_{j}")
              for j in range(nf)] for i in range(nc)]

        # Objective: minimize fixed + transport cost
        obj = sum(float(self.fixed_cost[j]) * y[j] for j in range(nf))
        obj += sum(
            float(self.transport_cost[i][j]) * x[i][j]
            for i in range(nc) for j in range(nf)
        )
        problem.setObjective(obj, sense=MINIMIZE)

        # Each customer must be fully assigned
        for i in range(nc):
            problem.addConstraint(
                sum(x[i][j] for j in range(nf)) == 1.0,
                name=f"assign_{i}",
            )

        # Capacity: total demand at facility j <= capacity_j * y_j
        for j in range(nf):
            served = sum(float(self.demand[i]) * x[i][j] for i in range(nc))
            problem.addConstraint(
                served <= float(self.capacity[j]) * y[j],
                name=f"cap_{j}",
            )

        return problem, y, x

    @conda(disabled=True)
    @step
    def start(self):
        """Generate capacitated facility location data."""
        import random
        import math

        random.seed(42)
        nf = int(self.n_facilities)
        nc = int(self.n_customers)

        # Facility sites: location, fixed cost, capacity
        self.facility_x = [random.uniform(0, 100) for _ in range(nf)]
        self.facility_y = [random.uniform(0, 100) for _ in range(nf)]
        self.fixed_cost = [random.uniform(2000, 8000) for _ in range(nf)]

        # Customer locations and demand
        self.customer_x = [random.uniform(0, 100) for _ in range(nc)]
        self.customer_y = [random.uniform(0, 100) for _ in range(nc)]
        self.demand = [random.uniform(5, 20) for _ in range(nc)]

        # Tight capacities: total capacity ≈ 1.2x total demand
        # Forces ~83% of facilities open → hard combinatorial decisions
        total_demand = sum(self.demand)
        avg_cap = (total_demand * 1.2) / nf
        self.capacity = [random.uniform(avg_cap * 0.5, avg_cap * 1.5)
                         for _ in range(nf)]

        # Transport cost = distance × demand × unit cost
        self.transport_cost = []
        for i in range(nc):
            row = []
            for j in range(nf):
                dist = math.sqrt(
                    (self.customer_x[i] - self.facility_x[j]) ** 2
                    + (self.customer_y[i] - self.facility_y[j]) ** 2
                )
                row.append(dist * self.demand[i] * 0.1)
            self.transport_cost.append(row)

        total_cap = sum(self.capacity)
        print(f"Capacitated Facility Location: {nf} facilities, {nc} customers")
        print(f"Total demand: {total_demand:,.0f}, "
              f"Total capacity: {total_cap:,.0f} "
              f"(slack: {total_cap / total_demand:.2f}x)")
        print(f"Variables: {nf} binary + {nf * nc:,} continuous = "
              f"{nf + nf * nc:,}")
        print(f"Constraints: {nc} assignment + {nf} capacity = {nc + nf}")

        self.next(self.solve_heuristic, self.solve_exact)

    @conda(packages=CUOPT_CONDA)
    @gpu_profile(interval=1)
    @kubernetes(gpu=1, compute_pool='gpu-multi-training')
    @step
    def solve_heuristic(self):
        """Solve with GPU-only heuristics (fast, no optimality proof)."""
        import time
        from cuopt.linear_programming.solver_settings import SolverSettings

        t0 = time.time()
        problem, y, _ = self._build_problem("CFLP_Heuristic")
        build_time = time.time() - t0

        settings = SolverSettings()
        settings.set_parameter("time_limit", 30)
        settings.set_parameter("mip_heuristics_only", True)

        t1 = time.time()
        problem.solve(settings)
        solve_time = time.time() - t1

        nf = len(self.facility_x)
        status = problem.Status.name

        # Try to extract solution (works for Optimal, Feasible,
        # FeasibleFound, and sometimes TimeLimit)
        try:
            total_cost = problem.ObjValue
            open_facs = [j for j in range(nf) if y[j].Value > 0.5]
            fixed_total = sum(self.fixed_cost[j] for j in open_facs)
            self.heuristic_result = {
                "status": "Feasible" if "Feasible" in status else status,
                "total_cost": float(total_cost),
                "fixed_cost": float(fixed_total),
                "transport_cost": float(total_cost - fixed_total),
                "n_open": len(open_facs),
                "open_facilities": open_facs,
                "solve_time": solve_time,
                "build_time": build_time,
            }
        except Exception:
            self.heuristic_result = {
                "status": status, "total_cost": None,
                "solve_time": solve_time, "build_time": build_time,
            }

        h = self.heuristic_result
        cost = h.get('total_cost') or 0
        print(f"Heuristic: {h['status']}, cost=${cost:,.0f}, "
              f"open={h.get('n_open', 0)}, "
              f"build={build_time:.1f}s, solve={solve_time:.2f}s")

        self.next(self.compare)

    @conda(packages=CUOPT_CONDA)
    @gpu_profile(interval=1)
    @kubernetes(gpu=1, compute_pool='gpu-multi-training')
    @step
    def solve_exact(self):
        """Solve with full branch-and-bound (GPU+CPU, optimality proof)."""
        import time
        from cuopt.linear_programming.solver_settings import SolverSettings

        t0 = time.time()
        problem, y, _ = self._build_problem("CFLP_Exact")
        build_time = time.time() - t0

        settings = SolverSettings()
        settings.set_parameter("time_limit", 600)
        settings.set_parameter("mip_heuristics_only", False)

        t1 = time.time()
        problem.solve(settings)
        solve_time = time.time() - t1

        nf = len(self.facility_x)
        status = problem.Status.name

        try:
            total_cost = problem.ObjValue
            open_facs = [j for j in range(nf) if y[j].Value > 0.5]
            fixed_total = sum(self.fixed_cost[j] for j in open_facs)
            self.exact_result = {
                "status": status,
                "total_cost": float(total_cost),
                "fixed_cost": float(fixed_total),
                "transport_cost": float(total_cost - fixed_total),
                "n_open": len(open_facs),
                "open_facilities": open_facs,
                "solve_time": solve_time,
                "build_time": build_time,
            }
        except Exception:
            self.exact_result = {
                "status": status, "total_cost": None,
                "solve_time": solve_time, "build_time": build_time,
            }

        e = self.exact_result
        cost = e.get('total_cost') or 0
        print(f"Exact: {e['status']}, cost=${cost:,.0f}, "
              f"open={e.get('n_open', 0)}, "
              f"build={build_time:.1f}s, solve={solve_time:.2f}s")

        self.next(self.compare)

    @conda(disabled=True)
    @card(type='blank')
    @step
    def compare(self, inputs):
        """Compare heuristic vs exact and build card."""
        self.merge_artifacts(inputs, include=[
            'heuristic_result', 'exact_result',
            'facility_x', 'facility_y', 'fixed_cost', 'capacity',
            'customer_x', 'customer_y', 'demand',
        ])

        nf = len(self.facility_x)
        nc = len(self.customer_x)
        h = self.heuristic_result
        e = self.exact_result

        current.card.append(Markdown("# MILP: Capacitated Facility Location"))
        current.card.append(Markdown(
            f"**{nf} facilities** | **{nc:,} customers** | "
            f"**{nf} binary + {nf * nc:,} continuous vars** | "
            f"**Solver:** cuOpt MILP (GPU)"
        ))

        # Comparison table
        current.card.append(Markdown("## GPU Heuristics vs Full Branch-and-Bound"))
        h_status = "Feasible" if h.get("total_cost") else h.get("status", "N/A")
        e_status = e.get("status", "N/A")
        rows = [
            ["Status", h_status, e_status],
            ["Total Cost",
             f"${h['total_cost']:,.0f}" if h.get("total_cost") else "N/A",
             f"${e['total_cost']:,.0f}" if e.get("total_cost") else "N/A"],
            ["Fixed Cost",
             f"${h['fixed_cost']:,.0f}" if h.get("fixed_cost") else "N/A",
             f"${e['fixed_cost']:,.0f}" if e.get("fixed_cost") else "N/A"],
            ["Transport Cost",
             f"${h['transport_cost']:,.0f}" if h.get("transport_cost") else "N/A",
             f"${e['transport_cost']:,.0f}" if e.get("transport_cost") else "N/A"],
            ["Facilities Opened",
             str(h.get("n_open", "N/A")), str(e.get("n_open", "N/A"))],
            ["Solve Time",
             f"{h['solve_time']:.2f}s", f"{e['solve_time']:.2f}s"],
        ]
        current.card.append(Table(rows, headers=["Metric", "GPU Heuristics", "Full B&B"]))

        # Tradeoff summary
        if h.get("total_cost") and e.get("total_cost"):
            gap = (h["total_cost"] - e["total_cost"]) / e["total_cost"] * 100
            if gap > 0.01:
                current.card.append(Markdown(
                    f"**Result:** Heuristic found a good solution "
                    f"(**{gap:.1f}% above optimal**). "
                    f"B&B proved optimality — capacity constraints create "
                    f"a genuine integrality gap that heuristics alone "
                    f"cannot close."
                ))
            else:
                current.card.append(Markdown(
                    f"**Result:** Both methods found the same optimum."
                ))

        # Facility map comparing both solutions
        h_open = set(h.get("open_facilities", []))
        e_open = set(e.get("open_facilities", []))
        both_open = h_open & e_open
        only_heuristic = h_open - e_open
        only_exact = e_open - h_open

        current.card.append(Markdown("## Facility Decisions"))
        if only_heuristic or only_exact:
            current.card.append(Markdown(
                f"Both methods agree on **{len(both_open)}** facilities. "
                f"**{len(only_heuristic)}** opened only by heuristic, "
                f"**{len(only_exact)}** only by B&B."
            ))

        map_data = []
        for j in range(nf):
            if j in both_open:
                ftype = "Both Open"
            elif j in only_heuristic:
                ftype = "Heuristic Only"
            elif j in only_exact:
                ftype = "B&B Only"
            else:
                ftype = "Closed"
            map_data.append({
                "x": self.facility_x[j], "y": self.facility_y[j],
                "type": ftype,
            })

        map_spec = {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "width": 400, "height": 400,
            "data": {"values": map_data},
            "mark": {"type": "point", "filled": True, "size": 200},
            "encoding": {
                "x": {"field": "x", "type": "quantitative", "title": "X"},
                "y": {"field": "y", "type": "quantitative", "title": "Y"},
                "color": {
                    "field": "type", "type": "nominal",
                    "title": "Decision",
                    "scale": {
                        "domain": ["Both Open", "Heuristic Only",
                                   "B&B Only", "Closed"],
                        "range": ["#4C9878", "#E6786C",
                                  "#5B8DBE", "#cccccc"],
                    },
                },
                "shape": {
                    "field": "type", "type": "nominal",
                    "scale": {
                        "domain": ["Both Open", "Heuristic Only",
                                   "B&B Only", "Closed"],
                        "range": ["diamond", "triangle-up",
                                  "triangle-down", "circle"],
                    },
                },
            },
        }
        current.card.append(VegaChart(map_spec))

        # Problem structure
        current.card.append(Markdown("## Problem Structure"))
        current.card.append(Markdown(
            f"Capacitated facility location with **{nf}** candidate sites "
            f"and **{nc:,}** customers:\n"
            f"- **Binary variables:** {nf} (open/close each facility)\n"
            f"- **Continuous variables:** {nf * nc:,} "
            f"(assignment fractions)\n"
            f"- **Constraints:** {nc:,} assignment + {nf} capacity = "
            f"{nc + nf:,}\n\n"
            f"*Capacity limits weaken the LP relaxation, creating genuine "
            f"integrality gap that requires branch-and-bound to close.*"
        ))

        self.next(self.end)

    @conda(disabled=True)
    @step
    def end(self):
        h = self.heuristic_result
        e = self.exact_result
        print(f"\nCapacitated Facility Location Results:")
        print(f"{'Method':<20} {'Cost':>12} {'Facilities':>12} {'Solve':>10}")
        hc = h.get('total_cost') or 0
        ec = e.get('total_cost') or 0
        print(f"{'Heuristic':<20} "
              f"${hc:>11,.0f} "
              f"{h.get('n_open', 0):>12} "
              f"{h['solve_time']:>9.2f}s")
        print(f"{'Branch-and-Bound':<20} "
              f"${ec:>11,.0f} "
              f"{e.get('n_open', 0):>12} "
              f"{e['solve_time']:>9.2f}s")


if __name__ == "__main__":
    CuOptMILPFlow()
