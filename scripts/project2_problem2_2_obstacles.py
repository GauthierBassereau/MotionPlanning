"""Solve Project 2.2: point-mass navigation with obstacles."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plotting import (
    plot_all_obstacle_trajectories,
    plot_worst_obstacle_trajectories,
    plot_waypoint_landscape,
)
from project2 import run_problem2_2, waypoint_landscape


def print_summary(results: dict[str, object]) -> None:
    def print_method(name: str, summary: dict[str, object]) -> None:
        costs = summary["costs"]
        margins = summary["margins"]
        wall_times = summary["wall_times"]
        evaluations = summary["evaluations"]
        print(f"  {name}:")
        print(f"    success rate = {summary['success_rate']:.1%}")
        print(f"    final cost = {costs.mean():.3f} +/- {costs.std():.3f}")
        print(f"    min collision margin = {margins.mean():.3f} +/- {margins.std():.3f}")
        print(f"    wall time = {wall_times.mean():.3f}s +/- {wall_times.std():.3f}s")
        print(f"    evaluations = {evaluations.mean():.0f} +/- {evaluations.std():.0f}")

    print("Problem 2.2 CEM vs. Adam comparison over 20 seeds:")
    for name in ("CEM", "Adam"):
        print_method(name, results["summaries"][name])
    print("Problem 2.2 warm-start result:")
    print_method("CEM10+Adam", results["summaries"]["CEM10+Adam"])


def main() -> None:
    out_dir = ROOT / "figures" / "project2"
    results = run_problem2_2(seeds=range(20))
    config = results["config"]

    wx, wy, costs = waypoint_landscape(config=config, resolution=200)
    landscape_output = out_dir / "problem2_2_waypoint_landscape.png"
    all_trajectories_output = out_dir / "problem2_2_all_final_trajectories.png"
    worst_trajectories_output = out_dir / "problem2_2_worst_final_trajectories.png"

    plot_waypoint_landscape(wx, wy, costs, config, landscape_output)
    plot_all_obstacle_trajectories(
        {"CEM": results["cem"], "Adam": results["gradient"]},
        config,
        all_trajectories_output,
    )
    plot_worst_obstacle_trajectories(
        {"CEM": results["cem"], "Adam": results["gradient"]},
        config,
        worst_trajectories_output,
    )

    print(f"Wrote {landscape_output}")
    print(f"Wrote {all_trajectories_output}")
    print(f"Wrote {worst_trajectories_output}")
    print_summary(results)
    print("Interpretation notes:")
    print("  The waypoint landscape is non-convex because circular obstacle penalties create separated low-cost basins.")
    print("  CEM explores basins globally; Adam refines locally and can stay in colliding basins when initialized poorly.")
    print("  The hybrid uses CEM for basin identification and Adam for local refinement.")


if __name__ == "__main__":
    main()
