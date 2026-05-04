"""Run Project 1.4 sensitivity experiments."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plotting import plot_covariance_convergence, plot_elite_fraction_convergence, plot_metric_sweep
from project1 import (
    covariance_strategy_sweep,
    dimensionality_sweep,
    elite_fraction_sweep,
)


def summarize_result(name: str, results: dict) -> None:
    print(name)
    for key, metrics in results.items():
        iterations = metrics["iterations"]
        final_costs = metrics["final_costs"]
        print(
            f"  {key}: iterations={iterations.mean():.2f} +/- {iterations.std():.2f}, "
            f"final J={final_costs.mean():.3e} +/- {final_costs.std():.3e}"
        )


def main() -> None:
    out_dir = ROOT / "figures" / "project1"

    elite_results = elite_fraction_sweep()
    elite_convergence_output = out_dir / "problem1_4_elite_fraction_convergence.png"
    plot_elite_fraction_convergence(elite_results, elite_convergence_output)
    print(f"Wrote {elite_convergence_output}")

    covariance_results = covariance_strategy_sweep()
    covariance_convergence_output = out_dir / "problem1_4_covariance_convergence.png"
    plot_covariance_convergence(covariance_results, covariance_convergence_output)
    print(f"Wrote {covariance_convergence_output}")
    summarize_result("Covariance strategy sweep", covariance_results)

    dimension_results = dimensionality_sweep()
    dimensions = np.asarray(list(dimension_results.keys()))
    dim_iter_mean = np.asarray([dimension_results[d]["iterations"].mean() for d in dimensions])
    dim_iter_std = np.asarray([dimension_results[d]["iterations"].std() for d in dimensions])
    dim_output = out_dir / "problem1_4_dimensionality_scaling.png"
    plot_metric_sweep(
        dimensions,
        dim_iter_mean,
        dim_iter_std,
        dim_output,
        "Number of links",
        "Dimensionality scaling",
    )
    print(f"Wrote {dim_output}")
    summarize_result("Dimensionality sweep", dimension_results)


if __name__ == "__main__":
    main()
