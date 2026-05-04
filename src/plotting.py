"""Plotting helpers for assignment figures."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.patches import Circle
import numpy as np


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def joint_grid(resolution: int = 250, q_min: float = -np.pi, q_max: float = np.pi) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q1 = np.linspace(q_min, q_max, resolution)
    q2 = np.linspace(q_min, q_max, resolution)
    q1_grid, q2_grid = np.meshgrid(q1, q2, indexing="xy")
    q = np.stack([q1_grid, q2_grid, np.zeros_like(q1_grid)], axis=-1)
    return q1_grid, q2_grid, q


def evaluate_grid(cost_fn, target=None, resolution: int = 250) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q1_grid, q2_grid, q = joint_grid(resolution)
    if target is None:
        values = np.asarray(cost_fn(jnp.asarray(q)))
    else:
        values = np.asarray(cost_fn(jnp.asarray(q), jnp.asarray(target)))
    return q1_grid, q2_grid, values


def plot_cost_heatmaps(grid_values: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], output_path: str | Path) -> None:
    ensure_parent(output_path)
    fig, axes = plt.subplots(1, len(grid_values), figsize=(15, 4.5), constrained_layout=True)
    if len(grid_values) == 1:
        axes = [axes]
    for ax, (name, (q1, q2, values)) in zip(axes, grid_values.items(), strict=True):
        im = ax.pcolormesh(q1, q2, values, shading="auto", cmap="viridis")
        ax.set_title(name.capitalize())
        ax.set_xlabel("$q_1$")
        ax.set_ylabel("$q_2$")
        fig.colorbar(im, ax=ax)
    fig.suptitle("Cost landscapes with $q_3 = 0$")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def confidence_ellipse(mean: np.ndarray, cov: np.ndarray, ax, n_std: float = 2.0, **kwargs) -> Ellipse:
    cov2 = cov[:2, :2]
    vals, vecs = np.linalg.eigh(cov2)
    vals = np.maximum(vals, 0.0)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2.0 * n_std * np.sqrt(vals)
    ellipse = Ellipse(xy=mean[:2], width=width, height=height, angle=angle, fill=False, **kwargs)
    ax.add_patch(ellipse)
    return ellipse


def plot_cem_evolution(
    q1_grid: np.ndarray,
    q2_grid: np.ndarray,
    values: np.ndarray,
    history,
    output_path: str | Path,
    iterations: tuple[int, ...] = (1, 5, 10, 15, 20, 30),
) -> None:
    ensure_parent(output_path)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for ax, iteration in zip(axes.ravel(), iterations, strict=True):
        idx = iteration - 1
        im = ax.pcolormesh(q1_grid, q2_grid, values, shading="auto", cmap="viridis")
        confidence_ellipse(
            np.asarray(history.means[idx]),
            np.asarray(history.covariances[idx]),
            ax,
            edgecolor="white",
            linewidth=2.0,
        )
        elites = np.asarray(history.elite_sets[idx])
        ax.scatter(elites[:, 0], elites[:, 1], s=5, c="tab:red", alpha=0.35, label="elites")
        ax.scatter(history.means[idx, 0], history.means[idx, 1], s=35, c="white", edgecolor="black", zorder=4)
        ax.set_title(f"Iteration {iteration}")
        ax.set_xlabel("$q_1$")
        ax.set_ylabel("$q_2$")
        ax.set_xlim(q1_grid.min(), q1_grid.max())
        ax.set_ylim(q2_grid.min(), q2_grid.max())
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_convergence(summary: dict[str, tuple[np.ndarray, np.ndarray]], output_path: str | Path) -> None:
    ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for name, (mean, std) in summary.items():
        x = np.arange(1, len(mean) + 1)
        ax.plot(x, mean, label=name.capitalize())
        ax.fill_between(x, mean - std, mean + std, alpha=0.2)
    ax.set_xlabel("CEM iteration")
    ax.set_ylabel("Best cost")
    ax.set_title("CEM convergence over 10 seeds")
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.legend()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_sparse_first_hit(populations: np.ndarray, means: np.ndarray, stds: np.ndarray, output_path: str | Path) -> None:
    ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.errorbar(populations, means, yerr=stds, marker="o", capsize=4)
    ax.set_xlabel("Population size N")
    ax.set_ylabel(r"$E[\tau_{first}]$")
    ax.set_title("First feasible sparse-cost sample")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_manipulator_configurations(solutions: np.ndarray, target: np.ndarray, points_fn, output_path: str | Path) -> None:
    ensure_parent(output_path)
    fig, axes = plt.subplots(1, len(solutions), figsize=(10, 4.5), constrained_layout=True)
    if len(solutions) == 1:
        axes = [axes]
    for idx, (ax, q) in enumerate(zip(axes, solutions, strict=True)):
        points = points_fn(q)
        ax.plot(points[:, 0], points[:, 1], marker="o", linewidth=2.5)
        ax.scatter([target[0]], [target[1]], c="tab:red", marker="x", s=80, label="target")
        ax.set_title(f"Mode {idx}: q={np.round(q, 3)}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-1.5, 3.2)
        ax.set_ylim(-2.5, 2.5)
        ax.grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_mode_distribution(labels: np.ndarray, output_path: str | Path) -> None:
    ensure_parent(output_path)
    unique, counts = np.unique(labels, return_counts=True)
    fig, ax = plt.subplots(figsize=(5.5, 4), constrained_layout=True)
    ax.bar([f"Mode {int(label)}" for label in unique], counts, color=["tab:blue", "tab:orange"][: len(unique)])
    ax.set_ylabel("Runs")
    ax.set_title("CEM final mode over 50 seeds")
    for i, count in enumerate(counts):
        ax.text(i, count + 0.5, str(count), ha="center")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_final_mode_scatter(final_samples: np.ndarray, labels: np.ndarray, cluster_centers: np.ndarray, output_path: str | Path) -> None:
    ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    scatter = ax.scatter(final_samples[:, 0], final_samples[:, 1], c=labels, cmap="tab10", alpha=0.8)
    ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c="black", marker="x", s=100, label="cluster centers")
    ax.set_xlabel("$q_1$")
    ax.set_ylabel("$q_2$")
    ax.set_title("Final CEM solutions projected to $(q_1, q_2)$")
    ax.legend()
    fig.colorbar(scatter, ax=ax, ticks=np.unique(labels))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_metric_sweep(
    x_values: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    output_path: str | Path,
    xlabel: str,
    title: str,
) -> None:
    ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
    ax.errorbar(x_values, means, yerr=stds, marker="o", capsize=4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Iterations to J < 0.01")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_elite_fraction_convergence(results: dict, output_path: str | Path) -> None:
    ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for fraction, metrics in results.items():
        mean_costs = np.asarray(metrics["cost_histories"]).mean(axis=0)
        iterations = np.arange(1, len(mean_costs) + 1)
        ax.plot(iterations, mean_costs, marker="o", markersize=3, linewidth=1.8, label=f"{100.0 * fraction:g}%")
    ax.set_xlabel("CEM iteration")
    ax.set_ylabel("Mean best cost")
    ax.set_yscale("symlog", linthresh=1e-8)
    ax.set_title("Elite fraction convergence")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Elite fraction")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_covariance_convergence(results: dict, output_path: str | Path) -> None:
    ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    colors = {
        ("full", 0.5): "#a1d99b",
        ("full", 1.0): "#41ab5d",
        ("full", 1.5): "#238b45",
        ("full", 2.0): "#005a32",
        ("diagonal", 0.5): "#fcae91",
        ("diagonal", 1.0): "#fb6a4a",
        ("diagonal", 1.5): "#de2d26",
        ("diagonal", 2.0): "#cb181d",
    }
    for (strategy, sigma), metrics in sorted(results.items()):
        mean_costs = np.asarray(metrics["cost_histories"]).mean(axis=0)
        iterations = np.arange(1, len(mean_costs) + 1)
        color = colors.get((strategy, float(sigma)))
        ax.plot(
            iterations,
            mean_costs,
            marker="o",
            markersize=3,
            linewidth=1.8,
            color=color,
            label=rf"{strategy}, $\sigma_0={sigma:g}$",
        )
    ax.set_xlabel("CEM iteration")
    ax.set_ylabel("Mean best cost")
    ax.set_yscale("symlog", linthresh=1e-8)
    ax.set_title("Covariance strategy convergence")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_point_mass_trajectories(results: dict[str, object], output_path: str | Path) -> None:
    from project2 import point_mass_rollout

    ensure_parent(output_path)
    config = results["config"]
    cem = results["cem"]
    gradient = results["gradient"]
    analytic = results["analytic_controls"]
    analytic_traj = np.asarray(point_mass_rollout(jnp.asarray(analytic), config))

    fig, ax = plt.subplots(figsize=(6.5, 5), constrained_layout=True)
    ax.plot(analytic_traj[:, 0], analytic_traj[:, 1], color="black", linewidth=2.0, label="Analytic optimum")
    ax.plot(cem.trajectory[:, 0], cem.trajectory[:, 1], marker="o", markersize=3, label="CEM")
    ax.plot(gradient.trajectory[:, 0], gradient.trajectory[:, 1], marker="s", markersize=3, label="Adam")
    ax.scatter([config.start[0]], [config.start[1]], c="tab:green", s=60, label="Start")
    ax.scatter([config.goal[0]], [config.goal[1]], c="tab:red", marker="x", s=80, label="Goal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Problem 2.1 point-mass trajectories")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_optimizer_convergence(results: dict[str, object], output_path: str | Path) -> None:
    ensure_parent(output_path)
    analytic_cost = results["analytic_cost"]
    fig, ax = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
    for key in ("cem", "gradient"):
        result = results[key]
        x = np.arange(1, len(result.costs) + 1)
        ax.plot(x, result.costs - analytic_cost, label=result.name)
    ax.axhline(1e-4, color="black", linestyle="--", linewidth=1.0, label="1e-4 tolerance")
    ax.set_xlabel("Optimizer iteration")
    ax.set_ylabel("Cost gap to analytic optimum")
    ax.set_yscale("symlog", linthresh=1e-8)
    ax.set_title("Problem 2.1 convergence")
    ax.legend()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _draw_obstacles(ax, config) -> None:
    for center, radius in zip(config.centers, config.radii, strict=True):
        ax.add_patch(Circle(center, radius, color="tab:red", alpha=0.25))
        ax.add_patch(Circle(center, radius, fill=False, color="tab:red", linewidth=1.5))


def plot_waypoint_landscape(wx: np.ndarray, wy: np.ndarray, costs: np.ndarray, config, output_path: str | Path) -> None:
    ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    im = ax.pcolormesh(wx, wy, np.log10(costs + 1e-6), shading="auto", cmap="viridis")
    _draw_obstacles(ax, config)
    ax.scatter([config.point_mass.start[0]], [config.point_mass.start[1]], c="tab:green", s=70, label="Start")
    ax.scatter([config.point_mass.goal[0]], [config.point_mass.goal[1]], c="black", marker="x", s=90, label="Goal")
    ax.set_xlabel("$w_x$")
    ax.set_ylabel("$w_y$")
    ax.set_title("Problem 2.2 waypoint cost landscape")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$\log_{10}(J + 10^{-6})$")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_obstacle_trajectories(runs_by_name: dict[str, list], config, output_path: str | Path, max_per_method: int = 4) -> None:
    ensure_parent(output_path)
    fig, axes = plt.subplots(1, len(runs_by_name), figsize=(5.2 * len(runs_by_name), 4.8), constrained_layout=True)
    if len(runs_by_name) == 1:
        axes = [axes]
    for ax, (name, runs) in zip(axes, runs_by_name.items(), strict=True):
        _draw_obstacles(ax, config)
        ordered = sorted(runs, key=lambda r: r.final_cost)
        for run in ordered[:max_per_method]:
            ax.plot(run.trajectory[:, 0], run.trajectory[:, 1], marker="o", markersize=2.5, linewidth=1.6, alpha=0.8)
        ax.scatter([config.point_mass.start[0]], [config.point_mass.start[1]], c="tab:green", s=60)
        ax.scatter([config.point_mass.goal[0]], [config.point_mass.goal[1]], c="tab:red", marker="x", s=80)
        ax.set_title(name)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(-0.25, 3.25)
        ax.set_ylim(-0.45, 2.75)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
    fig.suptitle("Best obstacle-navigation trajectories")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_all_obstacle_trajectories(runs_by_name: dict[str, list], config, output_path: str | Path) -> None:
    ensure_parent(output_path)
    fig, axes = plt.subplots(1, len(runs_by_name), figsize=(5.2 * len(runs_by_name), 4.8), constrained_layout=True)
    if len(runs_by_name) == 1:
        axes = [axes]
    for ax, (name, runs) in zip(axes, runs_by_name.items(), strict=True):
        _draw_obstacles(ax, config)
        for run in runs:
            ax.plot(run.trajectory[:, 0], run.trajectory[:, 1], linewidth=1.2, alpha=0.45)
        ax.scatter([config.point_mass.start[0]], [config.point_mass.start[1]], c="tab:green", s=60, label="Start")
        ax.scatter([config.point_mass.goal[0]], [config.point_mass.goal[1]], c="tab:red", marker="x", s=80, label="Goal")
        ax.set_title(f"{name}: all final trajectories")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(-0.25, 3.25)
        ax.set_ylim(-0.45, 2.75)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")
    fig.suptitle("All final obstacle-navigation trajectories")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_worst_obstacle_trajectories(runs_by_name: dict[str, list], config, output_path: str | Path, max_per_method: int = 5) -> None:
    ensure_parent(output_path)
    fig, axes = plt.subplots(1, len(runs_by_name), figsize=(5.2 * len(runs_by_name), 4.8), constrained_layout=True)
    if len(runs_by_name) == 1:
        axes = [axes]
    for ax, (name, runs) in zip(axes, runs_by_name.items(), strict=True):
        _draw_obstacles(ax, config)
        ordered = sorted(runs, key=lambda r: r.final_cost, reverse=True)
        for run in ordered[:max_per_method]:
            ax.plot(
                run.trajectory[:, 0],
                run.trajectory[:, 1],
                marker="o",
                markersize=2.5,
                linewidth=1.6,
                alpha=0.8,
                label=f"J={run.final_cost:.2f}",
            )
        ax.scatter([config.point_mass.start[0]], [config.point_mass.start[1]], c="tab:green", s=60, label="Start")
        ax.scatter([config.point_mass.goal[0]], [config.point_mass.goal[1]], c="tab:red", marker="x", s=80, label="Goal")
        ax.set_title(f"{name}: worst {max_per_method} final costs")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(-0.25, 3.25)
        ax.set_ylim(-0.45, 2.75)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle("Worst final-cost obstacle-navigation trajectories")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_cartpole_convergence(results: dict[str, object], output_path: str | Path) -> None:
    ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for key in ("cem", "gradient", "cem_contact", "gradient_contact"):
        result = results[key]
        x = np.arange(1, len(result.costs) + 1)
        ax.plot(x, result.costs, label=result.name)
    ax.set_xlabel("Optimizer iteration")
    ax.set_ylabel("Best / current cost")
    ax.set_yscale("symlog", linthresh=1e-2)
    ax.set_title("Problem 2.3 cart-pole convergence")
    ax.legend()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_cartpole_learning_rate_sweep(results: dict[str, object], output_path: str | Path) -> None:
    ensure_parent(output_path)
    panels = [
        ("Adam", results["no_contact"]),
        ("Adam hard-stop", results["hard_stop"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, (title, runs) in zip(axes, panels, strict=True):
        for learning_rate, result in runs.items():
            x = np.arange(1, len(result.costs) + 1)
            ax.plot(x, result.costs, linewidth=1.8, label=rf"$\alpha={learning_rate:g}$")
        ax.set_xlabel("Adam iteration")
        ax.set_ylabel("Current cost")
        ax.set_yscale("symlog", linthresh=1e-2)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(title="Learning rate")
    fig.suptitle("Problem 2.3 Adam learning-rate sensitivity")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_cartpole_gradient_magnitude(gradients: dict[str, np.ndarray], output_path: str | Path) -> None:
    ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for name, grad in gradients.items():
        ax.plot(np.arange(len(grad)), np.abs(grad), label=name)
    ax.set_xlabel("Control time step t")
    ax.set_ylabel(r"$|\partial J / \partial u_t|$")
    ax.set_yscale("symlog", linthresh=1e-8)
    ax.set_title("Cart-pole control-gradient magnitude")
    ax.legend()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _draw_cartpole_frame(ax, state: np.ndarray, config, color: str, alpha: float = 1.0) -> None:
    x = state[0]
    theta = state[2]
    cart_w = 0.32
    cart_h = 0.16
    pivot = np.array([x, cart_h / 2.0])
    tip = pivot + config.pole_length * np.array([np.sin(theta), np.cos(theta)])
    ax.add_patch(plt.Rectangle((x - cart_w / 2.0, -cart_h / 2.0), cart_w, cart_h, color=color, alpha=0.25 * alpha))
    ax.plot([pivot[0], tip[0]], [pivot[1], tip[1]], color=color, linewidth=2.0, alpha=alpha)
    ax.scatter([tip[0]], [tip[1]], color=color, s=24, alpha=alpha)


def plot_cartpole_frame_sequence(results: dict[str, object], output_path: str | Path) -> None:
    ensure_parent(output_path)
    config = results["config"]
    methods = [("CEM", results["cem"], "tab:blue"), ("Adam", results["gradient"], "tab:orange")]
    frame_indices = np.linspace(0, config.horizon, 9, dtype=int)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, (name, result, color) in zip(axes, methods, strict=True):
        ax.axhline(-0.08, color="black", linewidth=1.0)
        for idx, frame in enumerate(frame_indices):
            _draw_cartpole_frame(ax, result.trajectory[frame], config, color, alpha=0.25 + 0.75 * idx / (len(frame_indices) - 1))
        ax.set_title(f"{name}: J={result.final_cost:.2f}")
        ax.set_xlabel("cart x")
        ax.set_ylabel("height")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2)
    fig.suptitle("Problem 2.3 frame sequence")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
