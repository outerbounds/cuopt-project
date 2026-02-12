"""
VRP: Capacitated Vehicle Routing at multiple scales on GPU.

A delivery company has a depot and customers scattered across a city.
Each customer has a package demand. Trucks have limited capacity.
Find routes that visit every customer, respect capacity, and
minimize total distance.

Runs the same CVRP structure at 50, 200, 500, and 1000 customers
in parallel to show how cuOpt GPU routing scales.

Demonstrates:
    - cuOpt routing.DataModel API with cudf.DataFrame
    - foreach fanout: one GPU task per problem size
    - Cost matrix + capacity constraints
    - Solution extraction and comparison across scales
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


class CuOptVRPFlow(ProjectFlow):
    """CVRP: Route delivery trucks across multiple problem scales.

    Generates clustered customer locations at increasing scale,
    solves each on GPU in parallel, and compares results.
    """

    sizes = Parameter(
        "sizes",
        help="Comma-separated customer counts",
        default="50,200,500,1000",
    )
    vehicle_capacity = Parameter(
        "vehicle_capacity",
        help="Capacity per vehicle",
        default=100,
    )

    @conda(disabled=True)
    @step
    def start(self):
        """Generate CVRP instances at each scale."""
        from src.problems.cvrp import generate_cvrp

        self.size_list = [int(x) for x in self.sizes.split(",")]
        cap = int(self.vehicle_capacity)

        # Generate and store each instance (as serializable dicts)
        self.instances = []
        print("CVRP Instances:")
        print(f"{'Customers':>10} {'Vehicles':>10} {'Total Demand':>14} {'Matrix Size':>14}")
        for n in self.size_list:
            data = generate_cvrp(n, vehicle_capacity=cap)
            # Store as lists for Metaflow serialization (numpy → list)
            self.instances.append({
                'name': data['name'],
                'n_customers': data['n_customers'],
                'n_locations': data['n_locations'],
                'n_vehicles': data['n_vehicles'],
                'vehicle_capacity': data['vehicle_capacity'],
                'loc_x': data['loc_x'].tolist(),
                'loc_y': data['loc_y'].tolist(),
                'demands': data['demands'].tolist(),
                'total_demand': data['total_demand'],
                'cost_matrix': data['cost_matrix'].tolist(),
            })
            n_loc = data['n_locations']
            print(f"{n:>10,} {data['n_vehicles']:>10} "
                  f"{data['total_demand']:>14,} {n_loc:>10,}x{n_loc}")

        self.next(self.solve_vrp, foreach='instances')

    @conda(packages=CUOPT_CONDA)
    @gpu_profile(interval=1)
    @kubernetes(gpu=1, compute_pool='gpu-multi-training')
    @step
    def solve_vrp(self):
        """Solve one CVRP instance on GPU."""
        import numpy as np
        import time
        import cudf
        from cuopt import routing

        inst = self.input
        n_loc = inst['n_locations']
        n_cust = inst['n_customers']
        n_veh = inst['n_vehicles']
        cap = inst['vehicle_capacity']

        t0 = time.time()

        # Cost matrix as cudf DataFrame
        cost_df = cudf.DataFrame(inst['cost_matrix'])

        # Create routing data model
        dm = routing.DataModel(n_loc, n_veh, n_cust)
        dm.add_cost_matrix(cost_df)

        # Orders map to customer locations (1..n_loc-1)
        dm.set_order_locations(cudf.Series(list(range(1, n_loc))))

        # Capacity constraint
        order_demand = cudf.Series(inst['demands'][1:])  # customers only
        vehicle_cap = cudf.Series([cap] * n_veh)
        dm.add_capacity_dimension("demand", order_demand, vehicle_cap)

        # All vehicles start/end at depot (location 0)
        dm.set_vehicle_locations(
            cudf.Series([0] * n_veh),
            cudf.Series([0] * n_veh),
        )

        build_time = time.time() - t0

        # Solve
        settings = routing.SolverSettings()
        settings.set_time_limit(30.0)

        t1 = time.time()
        solution = routing.Solve(dm, settings)
        solve_time = time.time() - t1

        # Extract solution
        if solution.get_status() == 0:  # feasible
            route_df = solution.get_route()
            routes_by_vehicle = {}
            for _, row in route_df.to_pandas().iterrows():
                vid = int(row['truck_id'])
                loc = int(row['route'])
                if vid not in routes_by_vehicle:
                    routes_by_vehicle[vid] = []
                routes_by_vehicle[vid].append(loc)

            cost_matrix = inst['cost_matrix']
            demands = inst['demands']
            total_distance = 0
            vehicle_stats = []
            for vid, route in sorted(routes_by_vehicle.items()):
                dist = sum(
                    cost_matrix[route[i]][route[i+1]]
                    for i in range(len(route) - 1)
                )
                load = sum(demands[loc] for loc in route if loc != 0)
                total_distance += dist
                vehicle_stats.append({
                    "vehicle": vid,
                    "n_stops": len(route) - 2,
                    "distance": float(dist),
                    "load": int(load),
                    "route": route,
                })

            self.result = {
                "name": inst['name'],
                "n_customers": n_cust,
                "n_vehicles": n_veh,
                "vehicle_capacity": cap,
                "status": "Feasible",
                "total_distance": float(total_distance),
                "vehicles_used": len(routes_by_vehicle),
                "vehicle_stats": vehicle_stats,
                "build_time": build_time,
                "solve_time": solve_time,
            }
        else:
            self.result = {
                "name": inst['name'],
                "n_customers": n_cust,
                "n_vehicles": n_veh,
                "vehicle_capacity": cap,
                "status": f"Status {solution.get_status()}",
                "total_distance": 0,
                "vehicles_used": 0,
                "vehicle_stats": [],
                "build_time": build_time,
                "solve_time": solve_time,
            }

        # Store location data for card visualization
        self.loc_x = inst['loc_x']
        self.loc_y = inst['loc_y']
        self.demands = inst['demands']

        r = self.result
        print(f"CVRP {n_cust} customers: {r['status']}, "
              f"distance={r['total_distance']:.1f}, "
              f"vehicles={r['vehicles_used']}/{n_veh}, "
              f"solve={solve_time:.2f}s")

        self.next(self.join_results)

    @conda(disabled=True)
    @step
    def join_results(self, inputs):
        """Collect results across all sizes."""
        self.all_results = sorted(
            [i.result for i in inputs], key=lambda r: r['n_customers']
        )
        # Keep smallest instance for route map (few trucks = readable)
        smallest = min(inputs, key=lambda i: i.result['n_customers'])
        self.map_loc_x = smallest.loc_x
        self.map_loc_y = smallest.loc_y
        self.map_result = smallest.result
        self.next(self.build_card)

    @conda(disabled=True)
    @card(type='blank')
    @step
    def build_card(self):
        """Build comparison card across all scales."""
        cap = self.all_results[0].get('vehicle_capacity', 100)

        current.card.append(Markdown("# CVRP: Delivery Routing at Scale"))
        current.card.append(Markdown(
            "**Problem:** Route trucks from a depot to customers, "
            "minimize total distance  \n"
            f"**Constraint:** Each truck carries at most {cap} units  \n"
            "**Solver:** cuOpt Routing on GPU (A10G) | "
            "30s time limit per instance"
        ))

        # --- Scaling table ---
        current.card.append(Markdown("## Results by Scale"))
        rows = []
        for r in self.all_results:
            rows.append([
                f"{r['n_customers']:,}",
                f"{r['vehicles_used']}/{r['n_vehicles']}",
                f"{r['total_distance']:,.0f}",
                f"{r['total_distance'] / r['n_customers']:.1f}",
                f"{r['solve_time']:.2f}s",
            ])
        current.card.append(Table(rows, headers=[
            "Customers", "Trucks Used", "Total Distance",
            "Dist / Customer", "Solve Time",
        ]))

        # --- Distance per customer chart ---
        feasible = [r for r in self.all_results if r['status'] == 'Feasible']
        dist_data = [
            {
                "customers": r['n_customers'],
                "total_distance": r['total_distance'],
                "dist_per_customer": r['total_distance'] / r['n_customers'],
            }
            for r in feasible
        ]
        dist_spec = {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "width": 400, "height": 250,
            "data": {"values": dist_data},
            "mark": {"type": "bar", "color": "#E6786C"},
            "encoding": {
                "x": {"field": "customers", "type": "ordinal",
                       "title": "Customers"},
                "y": {"field": "dist_per_customer", "type": "quantitative",
                       "title": "Distance per Customer"},
                "tooltip": [
                    {"field": "customers", "title": "Customers"},
                    {"field": "total_distance", "title": "Total Distance",
                     "format": ",.0f"},
                    {"field": "dist_per_customer", "title": "Dist/Customer",
                     "format": ".1f"},
                ],
            },
        }
        current.card.append(Markdown("## Routing Efficiency by Scale"))
        current.card.append(VegaChart(dist_spec))

        # --- Route map for smallest instance (lines showing actual routes) ---
        r = self.map_result
        n_cust = r['n_customers']
        current.card.append(Markdown(
            f"## Route Map: {n_cust} Customers, "
            f"{r['vehicles_used']} Trucks"
        ))

        # Build route line data: (x, y, visit_order, truck) per stop
        route_lines = []
        for vs in r.get('vehicle_stats', []):
            vid = vs['vehicle']
            route = vs.get('route', [])
            for order, loc_idx in enumerate(route):
                route_lines.append({
                    "x": self.map_loc_x[loc_idx],
                    "y": self.map_loc_y[loc_idx],
                    "order": order,
                    "truck": f"Truck {vid}",
                })

        # Customer + depot point data
        point_data = [{
            "x": self.map_loc_x[0], "y": self.map_loc_y[0],
            "type": "Depot",
        }]
        for i in range(1, len(self.map_loc_x)):
            point_data.append({
                "x": self.map_loc_x[i], "y": self.map_loc_y[i],
                "type": "Customer",
            })

        map_spec = {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "width": 450, "height": 450,
            "layer": [
                {
                    "data": {"values": route_lines},
                    "mark": {"type": "line", "strokeWidth": 1.5,
                             "opacity": 0.7},
                    "encoding": {
                        "x": {"field": "x", "type": "quantitative",
                               "title": "X"},
                        "y": {"field": "y", "type": "quantitative",
                               "title": "Y"},
                        "order": {"field": "order"},
                        "color": {
                            "field": "truck", "type": "nominal",
                            "title": "Truck",
                        },
                    },
                },
                {
                    "data": {"values": point_data},
                    "mark": {"type": "point", "filled": True},
                    "encoding": {
                        "x": {"field": "x", "type": "quantitative"},
                        "y": {"field": "y", "type": "quantitative"},
                        "size": {
                            "condition": {
                                "test": "datum.type === 'Depot'",
                                "value": 300,
                            },
                            "value": 40,
                        },
                        "shape": {
                            "condition": {
                                "test": "datum.type === 'Depot'",
                                "value": "diamond",
                            },
                            "value": "circle",
                        },
                        "color": {
                            "condition": {
                                "test": "datum.type === 'Depot'",
                                "value": "#333",
                            },
                            "value": "#999",
                        },
                    },
                },
            ],
        }
        current.card.append(VegaChart(map_spec))

        self.next(self.end)

    @conda(disabled=True)
    @step
    def end(self):
        """Print summary."""
        print("\nCVRP Routing Results:")
        print(f"{'Customers':>10} {'Distance':>12} {'Vehicles':>10} {'Solve':>10}")
        for r in self.all_results:
            print(f"{r['n_customers']:>10,} "
                  f"{r['total_distance']:>12,.0f} "
                  f"{r['vehicles_used']:>5}/{r['n_vehicles']:<4} "
                  f"{r['solve_time']:>10.2f}s")


if __name__ == "__main__":
    CuOptVRPFlow()
