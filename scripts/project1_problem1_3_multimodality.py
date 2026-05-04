"""Run Project 1.3 multimodality and mode-collapse experiments."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kinematics import forward_kinematics
from plotting import (
    plot_final_mode_scatter,
    plot_manipulator_configurations,
    plot_mode_distribution,
)
from project1 import (
    TARGET_13,
    manipulator_points,
    run_mode_collapse_experiment,
    two_branch_ik_target_15,
)


def main() -> None:
    out_dir = ROOT / "figures" / "project1"
    references = two_branch_ik_target_15()
    positions = np.asarray(forward_kinematics(references))
    errors = np.linalg.norm(positions - np.asarray(TARGET_13), axis=1)

    manipulator_output = out_dir / "problem1_3_ik_configurations.png"
    plot_manipulator_configurations(references, np.asarray(TARGET_13), manipulator_points, manipulator_output)
    print(f"Wrote {manipulator_output}")
    print("Analytical IK references for target (1.5, 0.0):")
    for idx, (q, error) in enumerate(zip(references, errors, strict=True)):
        print(f"  Mode {idx}: q={np.round(q, 6)}, end-effector error={error:.3e}")

    histories, final_samples, labels, cluster_centers, distances = run_mode_collapse_experiment(seeds=range(50))
    distribution_output = out_dir / "problem1_3_mode_distribution.png"
    scatter_output = out_dir / "problem1_3_final_solution_scatter.png"
    plot_mode_distribution(labels, distribution_output)
    plot_final_mode_scatter(final_samples, labels, cluster_centers, scatter_output)
    print(f"Wrote {distribution_output}")
    print(f"Wrote {scatter_output}")

    unique, counts = np.unique(labels, return_counts=True)
    print("Empirical mode distribution over 50 CEM seeds:")
    for label, count in zip(unique, counts, strict=True):
        print(f"  Mode {int(label)}: {int(count)} runs ({count / len(labels):.1%})")
    print("Joint-space k-means cluster centers:")
    for idx, center in enumerate(cluster_centers):
        print(f"  Cluster {idx}: q={np.round(center, 6)}")
    print(f"Mean nearest-cluster joint-space distance: {distances.mean():.3f} +/- {distances.std():.3f}")
    print(f"Mean final best cost: {np.mean([float(h.best_costs[-1]) for h in histories]):.3e}")


if __name__ == "__main__":
    main()
