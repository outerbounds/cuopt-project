"""
LP Solver Benchmark: Four solvers, one problem, increasing scale.

Runs the same multi-region farm LP at increasing scale with four
solvers — cuOpt PDLP (GPU), OR-Tools PDLP (CPU), HiGHS simplex,
and HiGHS IPM — all receiving identical scipy CSR sparse input.

Isolates GPU vs CPU for the same algorithm (PDLP) and compares
first-order methods (PDLP) against classical solvers (simplex, IPM).
"""

from metaflow import step, card, Parameter, kubernetes, conda, pypi, current
from metaflow.profilers import gpu_profile
from metaflow.cards import Markdown, Table, VegaChart
from obproject import ProjectFlow
import src


CUOPT_CONDA = {
    'rapidsai::cudf': '25.08.*',
    'nvidia::cuopt-server': '25.08.*',
    'nvidia::cuopt-sh-client': '25.08.*',
    'scipy': '>=1.11',
    'numpy': '>=1.24',
}

ORTOOLS_PYPI = {
    'scipy': '>=1.11',
    'numpy': '>=1.24',
    'ortools': '>=9.8',
}

CPU_CONDA = {
    'scipy': '>=1.11',
    'numpy': '>=1.24',
}

# Size labels for human-readable display
SIZE_LABELS = {
    1_000: 'XS',
    5_000: 'S',
    10_000: 'M',
    50_000: 'L',
    100_000: 'XL',
}


class LPBenchmarkFlow(ProjectFlow):
    """LP Solver Benchmark across problem sizes.

    Compares four LP solvers on the same farm planning LP at increasing
    scale. All solvers receive identical scipy CSR sparse input.
    Measures build time (CSR → solver) and solve time separately.
    """

    sizes = Parameter(
        "sizes",
        help="Comma-separated region counts for scaling test",
        default="1000,5000,10000,50000,100000",
    )

    @conda(disabled=True)
    @step
    def start(self):
        """Parse sizes and fan out to four solver branches."""
        self.size_list = [int(x) for x in self.sizes.split(",")]
        print("LP Solver Benchmark: Farm Planning LP")
        print(f"{'Size':>6} {'Regions':>10} {'Variables':>12} "
              f"{'Constraints':>14} {'Nonzeros':>12}")
        labels = ['XS', 'S', 'M', 'L', 'XL']
        for i, s in enumerate(self.size_list):
            label = labels[i] if i < len(labels) else f'S{i}'
            print(f"{label:>6} {s:>10,} {s * 4:>12,} "
                  f"{2 * s + 9:>14,} {32 * s:>12,}")
        self.next(self.fan_cuopt, self.fan_ortools,
                  self.fan_simplex, self.fan_ipm)

    # --- cuOpt PDLP (GPU) ---

    @conda(disabled=True)
    @step
    def fan_cuopt(self):
        """Fan out cuOpt PDLP solves across sizes."""
        self.next(self.solve_cuopt, foreach='size_list')

    @conda(packages=CUOPT_CONDA)
    @gpu_profile(interval=1)
    @kubernetes(gpu=1, compute_pool='gpu-multi-training')
    @step
    def solve_cuopt(self):
        """Solve with cuOpt PDLP on GPU (sparse CSR input)."""
        import numpy as np
        import time
        from cuopt.linear_programming.solver_settings import SolverSettings
        from src.problems.farm_lp import generate_farm_data, build_sparse_arrays

        n_regions = self.input
        data = generate_farm_data(n_regions)

        t0 = time.time()
        A, b, c_obj, lb, ub = build_sparse_arrays(data)

        from cuopt.linear_programming.data_model import DataModel
        from cuopt.linear_programming.solver import Solve

        dm = DataModel()
        dm.set_csr_constraint_matrix(
            A.data.astype(np.float64),
            A.indices.astype(np.int32),
            A.indptr.astype(np.int32),
        )
        dm.set_objective_coefficients(c_obj.astype(np.float64))
        dm.set_constraint_lower_bounds(
            np.full(data['n_constraints'], -np.inf, dtype=np.float64),
        )
        dm.set_constraint_upper_bounds(b.astype(np.float64))
        dm.set_variable_lower_bounds(lb.astype(np.float64))
        dm.set_variable_upper_bounds(ub.astype(np.float64))
        dm.set_maximize(True)
        build_time = time.time() - t0

        settings = SolverSettings()
        settings.set_parameter("time_limit", 300)

        t1 = time.time()
        solution = Solve(dm, settings)
        solve_time = time.time() - t1

        status = str(solution.get_termination_reason())
        is_optimal = "optimal" in status.lower()
        total_profit = float(solution.get_primal_objective()) if is_optimal else None

        self.result = {
            'n_regions': n_regions,
            'n_vars': data['n_vars'],
            'n_constraints': data['n_constraints'],
            'nnz': data['nnz'],
            'solver': 'cuOpt PDLP (GPU)',
            'status': 'Optimal' if is_optimal else status,
            'total_profit': total_profit,
            'build_time': build_time,
            'solve_time': solve_time,
            'total_time': build_time + solve_time,
        }

        print(f"cuOpt {n_regions:,} regions: {self.result['status']}, "
              f"profit=${total_profit or 0:,.0f}, "
              f"build={build_time:.4f}s, solve={solve_time:.4f}s")

        self.next(self.join_cuopt)

    @conda(disabled=True)
    @step
    def join_cuopt(self, inputs):
        """Collect cuOpt PDLP results."""
        self.cuopt_results = sorted(
            [i.result for i in inputs], key=lambda r: r['n_regions']
        )
        self.next(self.compare)

    # --- OR-Tools PDLP (CPU) ---

    @conda(disabled=True)
    @step
    def fan_ortools(self):
        """Fan out OR-Tools PDLP solves across sizes."""
        self.next(self.solve_ortools, foreach='size_list')

    @pypi(packages=ORTOOLS_PYPI)
    @kubernetes(compute_pool='c5-2x-task')
    @step
    def solve_ortools(self):
        """Solve with OR-Tools PDLP on CPU (sparse CSR input)."""
        from src.problems.farm_lp import (
            generate_farm_data, solve_with_ortools_pdlp,
        )

        n_regions = self.input
        data = generate_farm_data(n_regions)
        result = solve_with_ortools_pdlp(data)

        self.result = {
            'n_regions': n_regions,
            'n_vars': data['n_vars'],
            'n_constraints': data['n_constraints'],
            'nnz': data['nnz'],
            'solver': 'OR-Tools PDLP (CPU)',
            **result,
        }

        print(f"OR-Tools {n_regions:,} regions: {result['status']}, "
              f"profit=${result.get('total_profit') or 0:,.0f}, "
              f"build={result['build_time']:.4f}s, solve={result['solve_time']:.4f}s")

        self.next(self.join_ortools)

    @conda(disabled=True)
    @step
    def join_ortools(self, inputs):
        """Collect OR-Tools PDLP results."""
        self.ortools_results = sorted(
            [i.result for i in inputs], key=lambda r: r['n_regions']
        )
        self.next(self.compare)

    # --- HiGHS Simplex (CPU) ---

    @conda(disabled=True)
    @step
    def fan_simplex(self):
        """Fan out HiGHS simplex solves across sizes."""
        self.next(self.solve_simplex, foreach='size_list')

    @conda(packages=CPU_CONDA)
    @kubernetes(compute_pool='c5-2x-task')
    @step
    def solve_simplex(self):
        """Solve with HiGHS dual simplex on CPU."""
        from src.problems.farm_lp import generate_farm_data, solve_with_scipy

        n_regions = self.input
        data = generate_farm_data(n_regions)
        result = solve_with_scipy(data, method='highs-ds')

        self.result = {
            'n_regions': n_regions,
            'n_vars': data['n_vars'],
            'n_constraints': data['n_constraints'],
            'nnz': data['nnz'],
            'solver': 'HiGHS Simplex',
            **result,
        }

        print(f"Simplex {n_regions:,} regions: {result['status']}, "
              f"profit=${result.get('total_profit') or 0:,.0f}, "
              f"build={result['build_time']:.4f}s, solve={result['solve_time']:.4f}s")

        self.next(self.join_simplex)

    @conda(disabled=True)
    @step
    def join_simplex(self, inputs):
        """Collect HiGHS simplex results."""
        self.simplex_results = sorted(
            [i.result for i in inputs], key=lambda r: r['n_regions']
        )
        self.next(self.compare)

    # --- HiGHS IPM (CPU) ---

    @conda(disabled=True)
    @step
    def fan_ipm(self):
        """Fan out HiGHS IPM solves across sizes."""
        self.next(self.solve_ipm, foreach='size_list')

    @conda(packages=CPU_CONDA)
    @kubernetes(compute_pool='c5-2x-task')
    @step
    def solve_ipm(self):
        """Solve with HiGHS interior point method on CPU."""
        from src.problems.farm_lp import generate_farm_data, solve_with_scipy

        n_regions = self.input
        data = generate_farm_data(n_regions)
        result = solve_with_scipy(data, method='highs-ipm')

        self.result = {
            'n_regions': n_regions,
            'n_vars': data['n_vars'],
            'n_constraints': data['n_constraints'],
            'nnz': data['nnz'],
            'solver': 'HiGHS IPM',
            **result,
        }

        print(f"IPM {n_regions:,} regions: {result['status']}, "
              f"profit=${result.get('total_profit') or 0:,.0f}, "
              f"build={result['build_time']:.4f}s, solve={result['solve_time']:.4f}s")

        self.next(self.join_ipm)

    @conda(disabled=True)
    @step
    def join_ipm(self, inputs):
        """Collect HiGHS IPM results."""
        self.ipm_results = sorted(
            [i.result for i in inputs], key=lambda r: r['n_regions']
        )
        self.next(self.compare)

    # --- Analysis ---

    @conda(packages=CPU_CONDA)
    @kubernetes(compute_pool='c5-2x-task')
    @card(type='blank')
    @step
    def compare(self, inputs):
        """Build benchmark analysis card with all 4 solvers."""
        import numpy as np

        for inp in inputs:
            if hasattr(inp, 'cuopt_results'):
                self.cuopt_results = inp.cuopt_results
            if hasattr(inp, 'ortools_results'):
                self.ortools_results = inp.ortools_results
            if hasattr(inp, 'simplex_results'):
                self.simplex_results = inp.simplex_results
            if hasattr(inp, 'ipm_results'):
                self.ipm_results = inp.ipm_results

        all_series = [self.cuopt_results, self.ortools_results,
                      self.simplex_results, self.ipm_results]
        solver_names = [s[0]['solver'] for s in all_series]

        current.card.append(Markdown("# LP Solver Benchmark"))
        current.card.append(Markdown(
            "**Problem:** Multi-region farm planning LP "
            "(4n vars, 2n+9 constraints, 32n nonzeros)  \n"
            "**Input:** All solvers receive identical scipy CSR sparse "
            "matrix — no API overhead differences  \n"
            "**Solvers:** cuOpt PDLP (GPU A10G) · OR-Tools PDLP (CPU) · "
            "HiGHS Simplex (CPU) · HiGHS IPM (CPU)"
        ))

        # --- Results table ---
        current.card.append(Markdown("## Results"))
        rows = []
        for i in range(len(self.cuopt_results)):
            row = [self.cuopt_results, self.ortools_results,
                   self.simplex_results, self.ipm_results]
            n_reg = row[0][i]['n_regions']
            solve_times = [s[i]['solve_time'] for s in row]
            best = min(solve_times)
            best_idx = solve_times.index(best)

            rows.append([
                f"{n_reg:,}",
                f"{n_reg * 4:,}",
            ] + [
                f"**{t:.4f}s**" if j == best_idx else f"{t:.4f}s"
                for j, t in enumerate(solve_times)
            ])

        current.card.append(Table(rows, headers=[
            "Regions", "Variables",
        ] + solver_names))

        # --- Solve time chart (log-log, regions x-axis) ---
        chart_data = []
        colors = ["#E6786C", "#D4A03C", "#4C9878", "#5B8DBE"]
        for series, name in zip(all_series, solver_names):
            for r in series:
                chart_data.append({
                    "regions": r['n_regions'],
                    "time": r['solve_time'],
                    "solver": name,
                    "vars": r['n_vars'],
                })

        solve_spec = {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "width": 550, "height": 320,
            "data": {"values": chart_data},
            "params": [{"name": "hover", "select": {
                "type": "point", "on": "pointerover",
                "nearest": True, "fields": ["solver"],
            }}],
            "mark": {"type": "line", "point": True, "strokeWidth": 2},
            "encoding": {
                "x": {
                    "field": "regions", "type": "quantitative",
                    "title": "Regions",
                    "scale": {"type": "log"},
                    "axis": {
                        "values": [1000, 5000, 10000, 50000, 100000],
                        "labelExpr": "datum.value >= 1000 ? format(datum.value / 1000, ',') + 'K' : datum.value",
                    },
                },
                "y": {
                    "field": "time", "type": "quantitative",
                    "title": "Solve time (seconds)",
                    "scale": {"type": "log"},
                },
                "color": {
                    "field": "solver", "type": "nominal",
                    "title": "Solver",
                    "scale": {"range": colors},
                },
                "opacity": {
                    "condition": {"param": "hover", "value": 1},
                    "value": 0.3,
                },
                "tooltip": [
                    {"field": "solver", "title": "Solver"},
                    {"field": "regions", "title": "Regions", "format": ","},
                    {"field": "vars", "title": "Variables", "format": ","},
                    {"field": "time", "title": "Solve Time (s)",
                     "format": ".4f"},
                ],
            },
        }
        current.card.append(Markdown("## Solve Time by Problem Size"))
        current.card.append(VegaChart(solve_spec))

        # --- Speedup chart (GPU speedup over each CPU solver) ---
        speedup_data = []
        cpu_series = {
            'OR-Tools PDLP (CPU)': self.ortools_results,
            'HiGHS Simplex': self.simplex_results,
            'HiGHS IPM': self.ipm_results,
        }
        for cpu_name, cpu_results in cpu_series.items():
            for cu, cpu in zip(self.cuopt_results, cpu_results):
                speedup = cpu['solve_time'] / cu['solve_time']
                speedup_data.append({
                    "regions": cu['n_regions'],
                    "speedup": round(speedup, 1),
                    "vs": cpu_name,
                })

        speedup_spec = {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "width": 550, "height": 280,
            "data": {"values": speedup_data},
            "params": [{"name": "hover", "select": {
                "type": "point", "on": "pointerover",
                "nearest": True, "fields": ["vs"],
            }}],
            "mark": {"type": "line", "point": True, "strokeWidth": 2},
            "encoding": {
                "x": {
                    "field": "regions", "type": "quantitative",
                    "title": "Regions",
                    "scale": {"type": "log"},
                    "axis": {
                        "values": [1000, 5000, 10000, 50000, 100000],
                        "labelExpr": "datum.value >= 1000 ? format(datum.value / 1000, ',') + 'K' : datum.value",
                    },
                },
                "y": {
                    "field": "speedup", "type": "quantitative",
                    "title": "GPU Speedup (x faster)",
                    "scale": {"type": "log"},
                },
                "color": {
                    "field": "vs", "type": "nominal",
                    "title": "cuOpt GPU vs",
                    "scale": {"range": ["#D4A03C", "#4C9878", "#5B8DBE"]},
                },
                "opacity": {
                    "condition": {"param": "hover", "value": 1},
                    "value": 0.3,
                },
                "tooltip": [
                    {"field": "vs", "title": "cuOpt vs"},
                    {"field": "regions", "title": "Regions", "format": ","},
                    {"field": "speedup", "title": "Speedup",
                     "format": ".1f"},
                ],
            },
        }
        current.card.append(Markdown("## GPU Speedup vs CPU Solvers"))
        current.card.append(VegaChart(speedup_spec))

        # --- Key findings ---
        current.card.append(Markdown("## Findings"))
        findings = []

        # PDLP GPU vs PDLP CPU (isolates GPU effect)
        largest_cuopt = self.cuopt_results[-1]
        largest_ortools = self.ortools_results[-1]
        gpu_vs_cpu_pdlp = (largest_ortools['solve_time'] /
                           largest_cuopt['solve_time'])
        findings.append(
            f"- **Same algorithm, different hardware:** cuOpt PDLP (GPU) is "
            f"**{gpu_vs_cpu_pdlp:.0f}x** faster than OR-Tools PDLP (CPU) at "
            f"{largest_cuopt['n_regions']:,} regions "
            f"({largest_cuopt['n_vars']:,} variables)"
        )

        # PDLP GPU vs best classical (isolates algorithm + hardware)
        largest_simplex = self.simplex_results[-1]
        largest_ipm = self.ipm_results[-1]
        best_classical = min(largest_simplex['solve_time'],
                            largest_ipm['solve_time'])
        best_name = ("IPM" if largest_ipm['solve_time'] < largest_simplex['solve_time']
                     else "Simplex")
        gpu_vs_classical = best_classical / largest_cuopt['solve_time']
        findings.append(
            f"- **GPU PDLP vs best CPU classical:** "
            f"**{gpu_vs_classical:.0f}x** faster than {best_name}"
        )

        # IPM vs Simplex
        ipm_vs_simplex = (largest_simplex['solve_time'] /
                          largest_ipm['solve_time'])
        if ipm_vs_simplex > 1.1:
            findings.append(
                f"- IPM is **{ipm_vs_simplex:.1f}x** faster than simplex "
                f"at scale"
            )

        # OR-Tools PDLP vs classical (isolates algorithm on same hardware)
        ortools_vs_classical = best_classical / largest_ortools['solve_time']
        if ortools_vs_classical > 1.1:
            findings.append(
                f"- CPU PDLP is **{ortools_vs_classical:.1f}x** faster than "
                f"{best_name} — PDLP algorithm advantage even without GPU"
            )
        elif ortools_vs_classical < 0.9:
            findings.append(
                f"- {best_name} is **{1/ortools_vs_classical:.1f}x** faster "
                f"than CPU PDLP — classical solvers still competitive on CPU"
            )

        current.card.append(Markdown("\n".join(findings)))

        self.next(self.end)

    @conda(disabled=True)
    @step
    def end(self):
        """Print summary."""
        print("\nLP Solver Benchmark Complete")
        print(f"{'Regions':>10} {'Variables':>12} {'cuOpt PDLP':>14} "
              f"{'OR-Tools PDLP':>14} {'Simplex':>14} {'IPM':>14} "
              f"{'Winner':>10}")
        for cu, ort, sx, ipm in zip(self.cuopt_results, self.ortools_results,
                                     self.simplex_results, self.ipm_results):
            times = {
                'cuOpt': cu['solve_time'],
                'OR-Tools': ort['solve_time'],
                'Simplex': sx['solve_time'],
                'IPM': ipm['solve_time'],
            }
            winner = min(times, key=times.get)
            print(f"{cu['n_regions']:>10,} "
                  f"{cu['n_vars']:>12,} "
                  f"{cu['solve_time']:>14.4f}s "
                  f"{ort['solve_time']:>14.4f}s "
                  f"{sx['solve_time']:>14.4f}s "
                  f"{ipm['solve_time']:>14.4f}s "
                  f"{winner:>10}")


if __name__ == "__main__":
    LPBenchmarkFlow()
