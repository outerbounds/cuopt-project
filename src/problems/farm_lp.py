"""
Multi-Region Farm Planning LP builder.

Vectorized scipy sparse construction for the farm planning LP.
Same problem as cuopt-lp/flow.py but built as sparse matrices
instead of cuOpt's algebraic API.

Variables: n_regions x 4 crops
Constraints:
    - n_regions land   (sum of crops <= land_limit per region)
    - n_regions water  (weighted sum <= water_limit per region)
    - 4 market cap     (sum across regions per crop <= market_cap)
    - 5 labor pools    (weighted sum across all vars <= pool_capacity)
Total: 2n + 9 constraints, 4n variables, 32n nonzeros
"""

import numpy as np
import time


def generate_farm_data(n_regions, price_mult=1.0, seed=42):
    """Generate all data for the multi-region farm LP.

    Returns a dict of numpy arrays. Deterministic given n_regions + seed.
    Uses the same random generation as cuopt-lp/flow.py for consistency.
    """
    n_crops = 4
    crops = ["Wheat", "Corn", "Soybeans", "Cotton"]

    np.random.seed(seed)

    base_profit = np.array([180.0, 220.0, 200.0, 160.0])
    region_mult = 0.8 + 0.4 * np.random.rand(n_regions)
    profit = np.outer(region_mult, base_profit) * price_mult

    water_per_acre = np.array([2.5, 3.5, 2.0, 4.0])

    land_limit = 500.0 + 200.0 * np.random.rand(n_regions)
    water_limit = 1500.0 + 500.0 * np.random.rand(n_regions)

    saved_state = np.random.get_state()
    np.random.seed(99)
    pool_usage = np.random.uniform(0.5, 2.0, (5, n_crops))
    np.random.set_state(saved_state)
    pool_capacity = np.array([
        pool_usage[k].sum() * 150.0 * n_regions for k in range(5)
    ])

    market_cap = 200.0 * n_regions
    min_acres, max_acres = 10.0, 400.0

    return {
        'n_regions': n_regions,
        'n_crops': n_crops,
        'n_vars': n_regions * n_crops,
        'n_constraints': 2 * n_regions + n_crops + 5,
        'nnz': 32 * n_regions,
        'crops': crops,
        'profit': profit,
        'water_per_acre': water_per_acre,
        'land_limit': land_limit,
        'water_limit': water_limit,
        'pool_usage': pool_usage,
        'pool_capacity': pool_capacity,
        'market_cap': market_cap,
        'min_acres': min_acres,
        'max_acres': max_acres,
        'price_mult': price_mult,
    }


def build_sparse_arrays(data):
    """Build scipy sparse CSR constraint matrix and vectors.

    All constraints are <= inequalities.
    Returns (A_csr, b_ub, c_maximize, lb, ub).
    """
    from scipy import sparse

    n_reg = data['n_regions']
    n_crops = data['n_crops']
    n_vars = data['n_vars']
    water_per_acre = data['water_per_acre']
    pool_usage = data['pool_usage']

    row_idx = np.arange(n_reg)
    rows, cols, vals = [], [], []

    # 1. Land constraints: row r, all 4 crop vars = 1.0
    for c in range(n_crops):
        rows.append(row_idx)
        cols.append(row_idx * n_crops + c)
        vals.append(np.ones(n_reg))

    # 2. Water constraints: row n_reg + r
    offset = n_reg
    for c in range(n_crops):
        rows.append(offset + row_idx)
        cols.append(row_idx * n_crops + c)
        vals.append(np.full(n_reg, water_per_acre[c]))

    # 3. Market cap per crop: row 2*n_reg + c
    offset = 2 * n_reg
    for c in range(n_crops):
        rows.append(np.full(n_reg, offset + c))
        cols.append(row_idx * n_crops + c)
        vals.append(np.ones(n_reg))

    # 4. Labor pools: row 2*n_reg + n_crops + k
    offset = 2 * n_reg + n_crops
    for k in range(5):
        for c in range(n_crops):
            rows.append(np.full(n_reg, offset + k))
            cols.append(row_idx * n_crops + c)
            vals.append(np.full(n_reg, pool_usage[k, c]))

    n_cons = data['n_constraints']
    A = sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_cons, n_vars),
    )

    b = np.concatenate([
        data['land_limit'],
        data['water_limit'],
        np.full(n_crops, data['market_cap']),
        data['pool_capacity'],
    ])

    c_maximize = data['profit'].ravel()
    lb = np.full(n_vars, data['min_acres'])
    ub = np.full(n_vars, data['max_acres'])

    return A, b, c_maximize, lb, ub


def solve_with_ortools_pdlp(data):
    """Solve farm LP with OR-Tools PDLP (CPU). Sparse CSR input via C++ API.

    Uses fill_model_from_sparse_data() — single C++ call, no Python loops.
    Same PDLP algorithm as cuOpt but running on CPU, isolating the GPU effect.
    """
    from ortools.linear_solver.python import model_builder_helper as mbh

    t0 = time.time()
    A, b, c_max, lb, ub = build_sparse_arrays(data)

    helper = mbh.ModelBuilderHelper()
    # fill_model_from_sparse_data minimizes, so negate for maximization
    helper.fill_model_from_sparse_data(
        lb.astype(np.float64),
        ub.astype(np.float64),
        (-c_max).astype(np.float64),
        np.full(len(b), -np.inf, dtype=np.float64),
        b.astype(np.float64),
        A.astype(np.float64),
    )
    build_time = time.time() - t0

    solver = mbh.ModelSolverHelper('pdlp')
    t1 = time.time()
    solver.solve(helper)
    solve_time = time.time() - t1

    # OR-Tools status: 0=OPTIMAL, 1=FEASIBLE, 2=INFEASIBLE, etc.
    status_int = solver.status()
    status_str = solver.status_string()
    is_solved = status_int in (0, 1)  # OPTIMAL or FEASIBLE
    total_profit = -solver.objective_value() if is_solved else None

    if is_solved and not status_str:
        status_str = 'Optimal' if status_int == 0 else 'Feasible'

    return {
        'status': 'Optimal' if status_int == 0 else (status_str or f'status={status_int}'),
        'total_profit': total_profit,
        'build_time': build_time,
        'solve_time': solve_time,
        'total_time': build_time + solve_time,
    }


def solve_with_scipy(data, method='highs'):
    """Solve farm LP with scipy/HiGHS. Measures build and solve time separately.

    method: 'highs' (auto simplex/IPM), 'highs-ds' (dual simplex), 'highs-ipm'
    """
    from scipy.optimize import linprog

    t0 = time.time()
    A, b, c_max, lb, ub = build_sparse_arrays(data)
    build_time = time.time() - t0

    # scipy minimizes, so negate the profit vector
    bounds = list(zip(lb, ub))

    t1 = time.time()
    result = linprog(-c_max, A_ub=A, b_ub=b, bounds=bounds, method=method)
    solve_time = time.time() - t1

    n_crops = data['n_crops']
    if result.success:
        total_profit = -result.fun
        vals_2d = result.x.reshape(data['n_regions'], n_crops)
        crop_totals = {
            data['crops'][c]: float(vals_2d[:, c].sum()) for c in range(n_crops)
        }
    else:
        total_profit = None
        crop_totals = {}

    return {
        'status': 'Optimal' if result.success else result.message,
        'total_profit': total_profit,
        'crop_totals': crop_totals,
        'build_time': build_time,
        'solve_time': solve_time,
        'total_time': build_time + solve_time,
    }
