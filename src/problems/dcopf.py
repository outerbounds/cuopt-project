"""
DC Optimal Power Flow (DC-OPF) LP builder.

Builds a linear program from power grid data:
    min  sum(c_g * p_g)          -- minimize generation cost
    s.t. Bp @ theta = p_inj      -- DC power balance (B-matrix formulation)
         |f_l| <= f_max_l         -- line flow limits
         p_min_g <= p_g <= p_max_g -- generator limits

Variables: generator dispatch p_g, bus voltage angles theta_b (slack bus fixed)

Returns problem data as numpy arrays suitable for both cuOpt and scipy.
"""

import numpy as np


def build_ieee118_data():
    """Build a simplified IEEE 118-bus test case.

    Returns a dict with all grid parameters needed for DC-OPF.
    This uses representative parameters rather than loading from file,
    keeping the flow self-contained (no pandapower dependency on GPU nodes).
    """
    np.random.seed(118)

    n_bus = 118
    n_gen = 54
    n_branch = 186

    # Generator locations (representative buses) — ensure slack bus (0) has one
    other_buses = list(range(1, n_bus))
    gen_buses = sorted([0] + np.random.choice(other_buses, n_gen - 1, replace=False).tolist())

    # Generator cost coefficients ($/MWh) — typical range
    gen_cost = np.random.uniform(10, 80, n_gen)

    # Generator limits (MW)
    gen_pmin = np.random.uniform(0, 20, n_gen)
    gen_pmax = np.random.uniform(50, 400, n_gen)

    # Bus loads (MW) — 99 load buses, rest are zero
    load_buses = sorted(np.random.choice(n_bus, 99, replace=False).tolist())
    bus_load = np.zeros(n_bus)
    bus_load[load_buses] = np.random.uniform(10, 150, 99)
    total_load = bus_load.sum()

    # Scale generation capacity to exceed total load
    gen_pmax = gen_pmax * (total_load * 1.3 / gen_pmax.sum())

    # Branch data: from_bus, to_bus, reactance, flow_limit
    branches_from = []
    branches_to = []
    branch_reactance = []
    branch_flow_limit = []

    # Generate connected network topology
    # First, spanning tree to ensure connectivity
    connected = {0}
    unconnected = set(range(1, n_bus))
    while unconnected:
        f = np.random.choice(list(connected))
        t = np.random.choice(list(unconnected))
        branches_from.append(int(f))
        branches_to.append(int(t))
        branch_reactance.append(float(np.random.uniform(0.01, 0.15)))
        branch_flow_limit.append(float(np.random.uniform(500, 2000)))
        connected.add(t)
        unconnected.discard(t)

    # Add remaining branches for meshed network
    for _ in range(n_branch - (n_bus - 1)):
        f = np.random.randint(0, n_bus)
        t = np.random.randint(0, n_bus)
        while t == f:
            t = np.random.randint(0, n_bus)
        branches_from.append(int(f))
        branches_to.append(int(t))
        branch_reactance.append(float(np.random.uniform(0.01, 0.15)))
        branch_flow_limit.append(float(np.random.uniform(500, 2000)))

    return {
        "n_bus": n_bus,
        "n_gen": n_gen,
        "n_branch": len(branches_from),
        "gen_buses": gen_buses,
        "gen_cost": gen_cost.tolist(),
        "gen_pmin": gen_pmin.tolist(),
        "gen_pmax": gen_pmax.tolist(),
        "bus_load": bus_load.tolist(),
        "branches_from": branches_from,
        "branches_to": branches_to,
        "branch_reactance": branch_reactance,
        "branch_flow_limit": branch_flow_limit,
        "slack_bus": 0,
        "case_name": "IEEE 118-bus (synthetic)",
    }


def generate_demand_scenarios(grid_data, n_scenarios, seed=42):
    """Generate demand scenarios by scaling bus loads.

    Each scenario multiplies each bus load by a random factor
    drawn from Uniform(0.7, 1.3), producing independent LP instances.

    Returns list of demand vectors (one per scenario).
    """
    np.random.seed(seed)
    base_load = np.array(grid_data["bus_load"])
    scenarios = []
    for i in range(n_scenarios):
        multipliers = np.random.uniform(0.85, 1.15, len(base_load))
        # Only scale nonzero loads
        scaled = base_load * multipliers
        scaled[base_load == 0] = 0
        scenarios.append(scaled.tolist())
    return scenarios


def build_dcopf_lp_arrays(grid_data, demand_vector=None):
    """Build DC-OPF as standard LP arrays: min c'x s.t. Ax <= b, lb <= x <= ub.

    Returns dict with:
        c: cost vector
        A_ub: inequality constraint matrix (CSR components)
        b_ub: inequality RHS
        A_eq: equality constraint matrix (CSR components)
        b_eq: equality RHS
        lb, ub: variable bounds
        var_names: variable name list
        constraint_names: constraint name list
    """
    n_bus = grid_data["n_bus"]
    n_gen = grid_data["n_gen"]
    n_branch = grid_data["n_branch"]
    slack = grid_data["slack_bus"]

    gen_buses = grid_data["gen_buses"]
    gen_cost = np.array(grid_data["gen_cost"])
    gen_pmin = np.array(grid_data["gen_pmin"])
    gen_pmax = np.array(grid_data["gen_pmax"])
    bus_load = np.array(demand_vector if demand_vector is not None else grid_data["bus_load"])

    branches_from = grid_data["branches_from"]
    branches_to = grid_data["branches_to"]
    branch_x = np.array(grid_data["branch_reactance"])
    branch_fmax = np.array(grid_data["branch_flow_limit"])

    # Variables: p_g (n_gen) + theta_b (n_bus - 1, excluding slack)
    # Map non-slack buses to theta variable indices
    theta_buses = [b for b in range(n_bus) if b != slack]
    theta_idx = {b: i for i, b in enumerate(theta_buses)}
    n_theta = len(theta_buses)
    n_vars = n_gen + n_theta

    # Cost vector: gen costs for p_g, 0 for theta
    c = np.zeros(n_vars)
    c[:n_gen] = gen_cost

    # Variable bounds
    lb = np.zeros(n_vars)
    ub = np.full(n_vars, np.inf)
    lb[:n_gen] = gen_pmin
    ub[:n_gen] = gen_pmax
    # Theta bounds: wide enough for DC-OPF (angles can be large in radians)
    lb[n_gen:] = -100.0
    ub[n_gen:] = 100.0

    # Equality constraints: power balance at each bus
    # For each bus b: sum(p_g at b) - sum(B_bl * theta_l) = load_b
    # Where B_bl = 1/x for branch between b and l
    n_eq = n_bus
    A_eq = np.zeros((n_eq, n_vars))
    b_eq = bus_load.copy()

    # Generator injections
    for g in range(n_gen):
        bus = gen_buses[g]
        A_eq[bus, g] = 1.0

    # Branch susceptance contributions to power balance
    for br in range(n_branch):
        f = branches_from[br]
        t = branches_to[br]
        b_val = 1.0 / branch_x[br]  # susceptance

        # Flow from f to t: b * (theta_f - theta_t)
        # Power balance at f: -b * (theta_f - theta_t) added to injection
        # Wait — in DC-OPF, P_ft = b * (theta_f - theta_t)
        # Power balance: sum(P_gen) - P_load = sum(P_ft) over branches from bus
        # So: sum(P_gen_b) - load_b - sum_over_branches(B*(theta_b - theta_other)) = 0
        # Rearranged: sum(P_gen_b) - sum(B*(theta_b - theta_other)) = load_b

        if f != slack:
            A_eq[f, n_gen + theta_idx[f]] -= b_val
        if t != slack:
            A_eq[f, n_gen + theta_idx[t]] += b_val

        if t != slack:
            A_eq[t, n_gen + theta_idx[t]] -= b_val
        if f != slack:
            A_eq[t, n_gen + theta_idx[f]] += b_val

    # Inequality constraints: line flow limits
    # |B_l * (theta_f - theta_t)| <= f_max
    # Split into: B*(theta_f - theta_t) <= f_max AND -B*(theta_f - theta_t) <= f_max
    n_ineq = 2 * n_branch
    A_ub = np.zeros((n_ineq, n_vars))
    b_ub = np.zeros(n_ineq)

    for br in range(n_branch):
        f = branches_from[br]
        t = branches_to[br]
        b_val = 1.0 / branch_x[br]

        # Forward: B*(theta_f - theta_t) <= f_max
        if f != slack:
            A_ub[2 * br, n_gen + theta_idx[f]] = b_val
        if t != slack:
            A_ub[2 * br, n_gen + theta_idx[t]] = -b_val
        b_ub[2 * br] = branch_fmax[br]

        # Backward: -B*(theta_f - theta_t) <= f_max
        if f != slack:
            A_ub[2 * br + 1, n_gen + theta_idx[f]] = -b_val
        if t != slack:
            A_ub[2 * br + 1, n_gen + theta_idx[t]] = b_val
        b_ub[2 * br + 1] = branch_fmax[br]

    var_names = [f"pg_{g}" for g in range(n_gen)] + [f"theta_{b}" for b in theta_buses]

    return {
        "c": c,
        "A_eq": A_eq,
        "b_eq": b_eq,
        "A_ub": A_ub,
        "b_ub": b_ub,
        "lb": lb,
        "ub": ub,
        "var_names": var_names,
        "n_gen": n_gen,
        "n_bus": n_bus,
        "n_branch": n_branch,
        "n_vars": n_vars,
        "n_eq": n_eq,
        "n_ineq": n_ineq,
    }
