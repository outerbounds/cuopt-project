"""
Client for the persistent cuOpt server deployment.

Flows use this to submit LP/MILP/VRP problems to the shared GPU server
instead of provisioning per-step GPU pods.

Usage from a Metaflow step:
    from src.clients.cuopt_server import CuOptClient

    client = CuOptClient(server_url)
    result = client.solve_lp(A_csr, b_ub, c_obj, lb, ub, maximize=True)
"""

import os
import time
import numpy as np
import requests


def _get_auth_headers():
    """Get Outerbounds API auth headers for the cuOpt server."""
    # In a Kubernetes task, METAFLOW_SERVICE_AUTH_KEY is set automatically
    token = os.environ.get("METAFLOW_SERVICE_AUTH_KEY")
    if token:
        return {"x-api-key": token}
    # Fallback: try Metaflow config (works locally with `outerbounds configure`)
    try:
        from metaflow.metaflow_config import SERVICE_HEADERS

        if SERVICE_HEADERS and "x-api-key" in SERVICE_HEADERS:
            return SERVICE_HEADERS
    except ImportError:
        pass
    raise RuntimeError(
        "No auth token found. Set METAFLOW_SERVICE_AUTH_KEY or "
        "run `outerbounds configure` to set up local credentials."
    )


class CuOptClient:
    """REST client for a cuOpt server running as an Outerbounds deployment."""

    def __init__(self, server_url=None):
        self.server_url = (
            server_url
            or os.environ.get("CUOPT_SERVER_URL")
            or "http://cuopt-server:8000"
        )
        self.headers = {
            "Content-Type": "application/json",
            "CLIENT-VERSION": "custom",
            **_get_auth_headers(),
        }

    def health_check(self, timeout=60):
        """Wait for cuOpt server to be ready."""
        url = f"{self.server_url}/cuopt/health"
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                r = requests.get(url, headers=self.headers, timeout=5)
                if r.status_code == 200:
                    return True
            except requests.ConnectionError:
                pass
            time.sleep(2)
        raise RuntimeError(f"cuOpt server not ready at {self.server_url}")

    def _submit_and_poll(self, payload, time_limit=300):
        """Submit a problem and poll for the solution."""
        url = f"{self.server_url}/cuopt/request"
        r = requests.post(url, json=payload, headers=self.headers, timeout=30)
        r.raise_for_status()
        req_id = r.json()["reqId"]

        solution_url = f"{self.server_url}/cuopt/solution/{req_id}"
        t0 = time.time()
        while True:
            resp = requests.get(solution_url, headers=self.headers, timeout=30)
            if resp.status_code == 200:
                body = resp.json()
                if "response" in body:
                    return body
                # 200 but solver not done yet — keep polling
                time.sleep(0.5)
            elif resp.status_code == 201:
                time.sleep(0.5)
            else:
                resp.raise_for_status()
            if time.time() - t0 > time_limit + 30:
                raise TimeoutError(f"cuOpt solve timed out after {time_limit}s")

    def solve_lp(self, A_csr, b_ub, c_obj, lb, ub, maximize=True, time_limit=300):
        """Submit an LP/MILP to the cuOpt server.

        Args:
            A_csr: scipy CSR sparse constraint matrix
            b_ub: constraint upper bounds
            c_obj: objective coefficients
            lb: variable lower bounds
            ub: variable upper bounds
            maximize: True for maximization
            time_limit: solver time limit in seconds

        Returns:
            dict with status, objective_value, solution, solve_time
        """
        self.health_check()

        payload = {
            "csr_constraint_matrix": {
                "offsets": A_csr.indptr.astype(int).tolist(),
                "indices": A_csr.indices.astype(int).tolist(),
                "values": A_csr.data.astype(float).tolist(),
            },
            "constraint_bounds": {
                "upper_bounds": b_ub.astype(float).tolist(),
                "lower_bounds": ["ninf"] * len(b_ub),
            },
            "objective_data": {
                "coefficients": c_obj.astype(float).tolist(),
                "scalability_factor": 1.0,
                "offset": 0.0,
            },
            "variable_bounds": {
                "upper_bounds": ub.astype(float).tolist(),
                "lower_bounds": lb.astype(float).tolist(),
            },
            "maximize": maximize,
            "solver_config": {"time_limit": time_limit},
        }

        t0 = time.time()
        result = self._submit_and_poll(payload, time_limit)
        solve_time = time.time() - t0

        solver_resp = result.get("response", {}).get("solver_response", {})
        solution = solver_resp.get("solution", {})
        return {
            "status": solver_resp.get("status", "unknown"),
            "objective_value": solution.get("primal_objective"),
            "solution": solution.get("primal_solution"),
            "solve_time": solve_time,
        }
