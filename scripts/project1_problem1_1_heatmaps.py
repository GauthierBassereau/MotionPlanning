"""Generate Project 1.1 heatmaps and print analytical-gradient notes."""

from __future__ import annotations

from pathlib import Path
import sys

import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from costs import COST_FNS, grad_linear_analytic, grad_quadratic_analytic
from plotting import evaluate_grid, plot_cost_heatmaps
from project1 import TARGET_11


def main() -> None:
    grid_values = {name: evaluate_grid(cost_fn, TARGET_11, resolution=250) for name, cost_fn in COST_FNS.items()}
    output = ROOT / "figures" / "project1" / "problem1_1_heatmaps.png"
    plot_cost_heatmaps(grid_values, output)

    q_example = jnp.array([0.2, 0.5, -0.1])
    print(f"Wrote {output}")
    print("Analytical gradients:")
    print("  Quadratic: grad J = 2 * J_p(q)^T * (p(q) - p*)")
    print("  Linear:    grad J = J_p(q)^T * (p(q) - p*) / ||p(q) - p*||")
    print("  Linear cost is undefined at any exact IK solution where p(q) = p*.")
    print("  Near a solution, its gradient direction can change abruptly while its norm does not vanish.")
    print("  Sparse cost has zero gradient almost everywhere and gives naive small-population CEM little ranking signal.")
    print(f"Example grad_quadratic({q_example.tolist()}): {grad_quadratic_analytic(q_example, TARGET_11)}")
    print(f"Example grad_linear({q_example.tolist()}): {grad_linear_analytic(q_example, TARGET_11)}")


if __name__ == "__main__":
    main()
