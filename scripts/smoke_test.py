"""Smoke test: Energy Mix LP.

Finds the cheapest way to power a city using solar, wind, gas, and nuclear
given emission caps, budget limits, and a minimum renewable requirement.

Usage:
    METAFLOW_PROFILE=yellow python scripts/smoke_test.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy import sparse
from metaflow.apps import AppDeployer
from src.clients.cuopt_server import CuOptClient

# ---------------------------------------------------------------------------
# 1. Auto-discover server URL
# ---------------------------------------------------------------------------
apps = AppDeployer.list_deployments(name="cuopt-server")
if not apps:
    raise RuntimeError("cuopt-server not found in deployments")
url = apps[0].public_url
print(f"Server URL: {url}")

client = CuOptClient(server_url=url)

# ---------------------------------------------------------------------------
# 2. Health check
# ---------------------------------------------------------------------------
client.health_check(timeout=10)
print("Health check passed")

# ---------------------------------------------------------------------------
# 3. Energy Mix LP
#
# Variables (MW): Solar ($50), Wind ($40), Gas ($70), Nuclear ($90)
#
# Constraints:
#   1) Meet demand:          Solar + Wind + Gas + Nuclear >= 1000
#   2) Emissions cap:        0.5*Gas + 0.3*Nuclear       <=  200  tons/hr
#   3) Budget cap:           50*S + 40*W + 70*G + 90*N   <= 65000 $/hr
#   4) Renewable minimum:    Solar + Wind                 >=  400
#
# Variable bounds:  Solar<=500, Wind<=400, Gas<=600, Nuclear<=300
# Objective:        minimize 50*S + 40*W + 70*G + 90*N
# ---------------------------------------------------------------------------
sources = ["Solar", "Wind", "Gas", "Nuclear"]
cost_per_mwh = np.array([50.0, 40.0, 70.0, 90.0])

# Constraint matrix (4 constraints x 4 variables)
# Row 0: demand  — written as -S - W - G - N <= -1000 (>= flipped)
# Row 1: emissions
# Row 2: budget
# Row 3: renewables — written as -S - W <= -400 (>= flipped)
A = sparse.csr_matrix(
    [
        [-1.0, -1.0, -1.0, -1.0],  # demand (flipped >=)
        [0.0, 0.0, 0.5, 0.3],  # emissions
        [50.0, 40.0, 70.0, 90.0],  # budget
        [-1.0, -1.0, 0.0, 0.0],  # renewables (flipped >=)
    ]
)
b = np.array([-1000.0, 200.0, 65000.0, -400.0])

c = cost_per_mwh
lb = np.array([0.0, 0.0, 0.0, 0.0])
ub = np.array([500.0, 400.0, 600.0, 300.0])

# ---------------------------------------------------------------------------
# 4. Solve
# ---------------------------------------------------------------------------
result = client.solve_lp(A, b, c, lb, ub, maximize=False, time_limit=30)

print(f"\nStatus:    {result['status']}")
print(f"Objective: {result['objective_value']}")
print(f"Solution:  {result['solution']}")
print(f"Solve time: {result['solve_time']:.2f}s")

if result["solution"]:
    sol = np.array(result["solution"])
    print("\n--- Energy Mix ---")
    for name, mw, cost in zip(sources, sol, sol * cost_per_mwh):
        print(f"  {name:>8}: {mw:6.1f} MW  (${cost:,.0f}/hr)")
    print(
        f"  {'Total':>8}: {sol.sum():6.1f} MW  (${(sol * cost_per_mwh).sum():,.0f}/hr)"
    )

# ---------------------------------------------------------------------------
# 5. Visualization
# ---------------------------------------------------------------------------
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    sol = np.array(result["solution"])
    source_colors = {
        "Solar": "#f4a41a",
        "Wind": "#4aa3df",
        "Gas": "#7f8c8d",
        "Nuclear": "#9b59b6",
    }

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1, 1.2]}
    )

    # ---- Left panel: Allocation vs Capacity ----
    # For each variable, show allocated MW as a filled bar and upper bound as
    # the full capacity outline. This immediately shows which variables are
    # "at their bound" (the key LP insight).
    y_pos = np.arange(len(sources))
    caps = ub  # upper bounds = capacity
    alloc = sol

    # Capacity background (light)
    ax1.barh(
        y_pos,
        caps,
        height=0.5,
        color="#ecf0f1",
        edgecolor="#bdc3c7",
        linewidth=1,
        label="Capacity",
    )
    # Allocated (filled)
    bars = ax1.barh(
        y_pos,
        alloc,
        height=0.5,
        color=[source_colors[s] for s in sources],
        edgecolor="white",
    )

    for i, (a, c_val) in enumerate(zip(alloc, caps)):
        pct = a / c_val * 100
        tag = "AT BOUND" if pct > 99 else f"{pct:.0f}%"
        ax1.text(
            c_val + 8,
            i,
            f"{a:.0f}/{c_val:.0f} MW  ({tag})",
            va="center",
            fontsize=9,
            fontweight="bold" if pct > 99 else "normal",
            color="#c0392b" if pct > 99 else "#555",
        )

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(
        [f"{s}\n${c:.0f}/MWh" for s, c in zip(sources, cost_per_mwh)], fontsize=9
    )
    ax1.set_xlabel("MW")
    ax1.set_title("Variable Allocation vs Capacity", fontweight="bold")
    ax1.set_xlim(0, max(caps) * 1.55)
    ax1.legend(loc="lower right", fontsize=8)

    # ---- Right panel: Constraint Slack Analysis ----
    # Show each constraint as: actual value bar + limit marker.
    # "Binding" = no slack (actual == limit). "Slack" = room remaining.
    constraint_info = [
        ("Demand (>= 1000 MW)", sol.sum(), 1000, ">="),
        ("Emissions (<= 200 t)", 0.5 * sol[2] + 0.3 * sol[3], 200, "<="),
        ("Budget (<= $65K)", (sol * cost_per_mwh).sum() / 1000, 65, "<="),
        ("Renewables (>= 400 MW)", sol[0] + sol[1], 400, ">="),
    ]

    y_pos2 = np.arange(len(constraint_info))
    for i, (name, actual, limit, sense) in enumerate(constraint_info):
        is_binding = abs(actual - limit) < 1e-3 * max(abs(limit), 1)

        if sense == "<=":
            # Bar = actual, marker at limit. Slack = limit - actual.
            slack = limit - actual
            bar_color = "#e74c3c" if is_binding else "#2ecc71"
            ax2.barh(i, actual, height=0.45, color=bar_color, alpha=0.85)
            # Limit marker
            ax2.plot(limit, i, "k|", markersize=18, markeredgewidth=2)
            if is_binding:
                label = f"{actual:.1f}  — BINDING"
            else:
                label = f"{actual:.1f}  (slack: {slack:.1f})"
        else:
            # >= constraint: bar = actual, marker at requirement.
            surplus = actual - limit
            bar_color = "#e74c3c" if is_binding else "#3498db"
            ax2.barh(i, actual, height=0.45, color=bar_color, alpha=0.85)
            ax2.plot(limit, i, "k|", markersize=18, markeredgewidth=2)
            if is_binding:
                label = f"{actual:.1f}  — BINDING"
            else:
                label = f"{actual:.1f}  (surplus: {surplus:.1f})"

        max_val = max(actual, limit)
        ax2.text(
            max_val + max_val * 0.03,
            i,
            label,
            va="center",
            fontsize=8.5,
            fontweight="bold" if is_binding else "normal",
            color="#c0392b" if is_binding else "#555",
        )

    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels([c[0] for c in constraint_info], fontsize=9)
    ax2.set_xlabel("Value (black marker = limit)")
    ax2.set_title("Constraint Slack Analysis", fontweight="bold")
    # Pad x-axis for labels
    max_x = max(max(c[1], c[2]) for c in constraint_info)
    ax2.set_xlim(0, max_x * 1.65)

    fig.suptitle(
        f"Energy Mix Optimization — Optimal Cost: ${(sol * cost_per_mwh).sum():,.0f}/hr",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()

    out_path = Path(__file__).resolve().parent / "energy_mix_result.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to {out_path}")
except ImportError:
    print("\nmatplotlib not installed — skipping chart")
