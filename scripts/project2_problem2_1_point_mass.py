"""Solve Project 2.1: convex point-mass navigation."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plotting import plot_optimizer_convergence, plot_point_mass_trajectories
from project2 import run_problem2_1


def main() -> None:
    results = run_problem2_1(seed=0)
    out_dir = ROOT / "figures" / "project2"
    trajectory_output = out_dir / "problem2_1_trajectories.png"
    convergence_output = out_dir / "problem2_1_convergence.png"
    plot_point_mass_trajectories(results, trajectory_output)
    plot_optimizer_convergence(results, convergence_output)

    cem = results["cem"]
    gradient = results["gradient"]
    eigvals = results["hessian_eigenvalues"]
    config = results["config"]
    optimal_control = results["analytic_controls"][0]

    print(f"Wrote {trajectory_output}")
    print(f"Wrote {convergence_output}")
    print("Problem 2.1 hypothesis:")
    print("  Because the objective is convex quadratic in u, gradient descent should be far more sample-efficient than CEM.")
    print("Analytic optimum:")
    print(f"  constant u_t = {np.round(optimal_control, 6)}")
    print(f"  optimal cost = {results['analytic_cost']:.8f}")
    print("Optimizer comparison:")
    for result in (cem, gradient):
        converged = result.converged_iteration if result.converged_iteration is not None else "not reached"
        print(
            f"  {result.name}: final cost={result.final_cost:.8f}, "
            f"iterations={result.iterations}, convergence iter={converged}, "
            f"wall time={result.wall_time:.4f}s, evaluations={result.evaluations}"
        )
    print("Hessian analysis:")
    print(f"  shape = {results['hessian'].shape}")
    print(f"  min eigenvalue = {eigvals.min():.6f}")
    print(f"  max eigenvalue = {eigvals.max():.6f}")
    print("  The Hessian is positive definite because lambda > 0; the landscape has a single global optimum.")
    print(f"  Expected eigenvalues: 2*lambda={2 * config.control_weight:.6f} and 2*(lambda + T*dt^2)={2 * (config.control_weight + config.horizon * config.dt**2):.6f}.")


if __name__ == "__main__":
    main()
