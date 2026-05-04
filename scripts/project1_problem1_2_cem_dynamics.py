"""Run Project 1.2 CEM experiments and generate figures."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plotting import (
    evaluate_grid,
    plot_cem_evolution,
    plot_convergence,
    plot_sparse_first_hit,
)
from project1 import (
    TARGET_11,
    convergence_statistics,
    project1_cost,
    run_project1_cem,
    sparse_first_hit_statistics,
)


def main() -> None:
    cost_names = ("quadratic", "linear", "sparse")
    for cost_name in cost_names:
        q1, q2, values = evaluate_grid(project1_cost(cost_name), resolution=250)
        history = run_project1_cem(cost_name, seed=0)
        output = ROOT / "figures" / "project1" / f"problem1_2_{cost_name}_evolution.png"
        plot_cem_evolution(q1, q2, values, history, output)
        print(f"Wrote {output}")
        if cost_name == "sparse":
            print(f"Sparse seed-0 first feasible iteration: {history.feasible_first_iteration}")

    summary, _ = convergence_statistics(cost_names=cost_names, seeds=range(10))
    convergence_output = ROOT / "figures" / "project1" / "problem1_2_convergence.png"
    plot_convergence(summary, convergence_output)
    print(f"Wrote {convergence_output}")

    sparse_hits = sparse_first_hit_statistics(population_sizes=(500, 2000, 5000), seeds=range(50), iterations=30)
    populations = np.asarray(list(sparse_hits.keys()))
    means = np.asarray([vals.mean() for vals in sparse_hits.values()])
    stds = np.asarray([vals.std() for vals in sparse_hits.values()])
    sparse_output = ROOT / "figures" / "project1" / "problem1_2_sparse_first_hit_vs_population.png"
    plot_sparse_first_hit(populations, means, stds, sparse_output)
    print(f"Wrote {sparse_output}")
    for population, vals in sparse_hits.items():
        misses = int(np.sum(vals > 30))
        print(f"N={population}: E[tau_first]={vals.mean():.2f}, std={vals.std():.2f}, misses={misses}/50")


if __name__ == "__main__":
    main()
