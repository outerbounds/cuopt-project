"""
CVRP instance generator for benchmarking cuOpt routing.

Generates Capacitated Vehicle Routing Problem instances with
clustered customer locations (mimics real delivery patterns).
Deterministic given n_customers + seed.

Each instance has:
    - 1 depot (center of region)
    - n_customers with demands and (x, y) coordinates
    - Euclidean cost/transit matrices
    - Vehicle count = ceil(total_demand / capacity)
"""

import numpy as np
import math


def generate_cvrp(n_customers, vehicle_capacity=100, seed=42):
    """Generate a CVRP instance with clustered customer locations.

    Customers are grouped into clusters (city districts) with
    some scatter. More realistic than uniform random placement.

    Returns dict with all problem data needed by cuOpt routing API.
    """
    rng = np.random.RandomState(seed)

    # --- Customer locations: clustered around district centers ---
    n_clusters = max(3, n_customers // 30)
    cluster_centers_x = rng.uniform(10, 90, n_clusters)
    cluster_centers_y = rng.uniform(10, 90, n_clusters)

    cust_x = np.empty(n_customers)
    cust_y = np.empty(n_customers)
    assignments = rng.randint(0, n_clusters, n_customers)

    for i in range(n_customers):
        cx = cluster_centers_x[assignments[i]]
        cy = cluster_centers_y[assignments[i]]
        cust_x[i] = cx + rng.normal(0, 5)
        cust_y[i] = cy + rng.normal(0, 5)

    # Clamp to [0, 100]
    cust_x = np.clip(cust_x, 0, 100)
    cust_y = np.clip(cust_y, 0, 100)

    # Depot at center
    depot_x, depot_y = 50.0, 50.0

    # All locations: depot (index 0) + customers (1..n)
    loc_x = np.concatenate([[depot_x], cust_x])
    loc_y = np.concatenate([[depot_y], cust_y])
    n_locations = len(loc_x)

    # --- Demands: 1-10 per customer, depot = 0 ---
    demands = np.concatenate([[0], rng.randint(1, 11, n_customers)])
    total_demand = int(demands.sum())

    # --- Vehicle count ---
    n_vehicles = math.ceil(total_demand / vehicle_capacity)
    # Add 20% spare capacity
    n_vehicles = max(n_vehicles, math.ceil(n_vehicles * 1.2))

    # --- Cost matrix (Euclidean) ---
    dx = loc_x[:, None] - loc_x[None, :]
    dy = loc_y[:, None] - loc_y[None, :]
    cost_matrix = np.sqrt(dx**2 + dy**2)

    return {
        'name': f'cluster-{n_customers}',
        'n_customers': n_customers,
        'n_locations': n_locations,
        'n_vehicles': n_vehicles,
        'vehicle_capacity': vehicle_capacity,
        'loc_x': loc_x,
        'loc_y': loc_y,
        'demands': demands,
        'total_demand': total_demand,
        'cost_matrix': cost_matrix,
    }
