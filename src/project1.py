"""Experiment builders for Project 1."""

from __future__ import annotations

from functools import partial

import jax.numpy as jnp
import numpy as np

from cem import CEMConfig, run_cem
from costs import COST_FNS
from kinematics import forward_kinematics

TARGET_11 = jnp.array([2.0, 1.0], dtype=jnp.float32)
TARGET_13 = jnp.array([1.5, 0.0], dtype=jnp.float32)
DEFAULT_CEM = CEMConfig(population_size=500, elite_size=50, iterations=30, dim=3, initial_mean=0.0, initial_sigma=1.0)


def project1_cost(name: str, target=TARGET_11):
    if name not in COST_FNS:
        raise KeyError(f"Unknown cost {name!r}. Expected one of {sorted(COST_FNS)}")
    return partial(COST_FNS[name], target=target)


def run_project1_cem(cost_name: str, seed: int, config: CEMConfig = DEFAULT_CEM):
    feasible_threshold = 0.0 if cost_name == "sparse" else None
    return run_cem(project1_cost(cost_name), seed=seed, config=config, feasible_threshold=feasible_threshold)


def wrap_to_pi(q: np.ndarray) -> np.ndarray:
    """Wrap angles to [-pi, pi)."""
    return (np.asarray(q) + np.pi) % (2.0 * np.pi) - np.pi


def two_branch_ik_target_15() -> np.ndarray:
    """Return two exact IK branches for target (1.5, 0) with q3 fixed to zero.

    With q3 = 0, links 2 and 3 act as one length-2 link. The resulting two-link
    geometry gives elbow-up and elbow-down solutions.
    """
    target_x = 1.5
    l1 = 1.0
    l23 = 2.0
    cos_q2 = (target_x**2 - l1**2 - l23**2) / (2.0 * l1 * l23)
    q2_abs = np.arccos(cos_q2)
    solutions = []
    for q2 in (q2_abs, -q2_abs):
        q1 = -np.arctan2(l23 * np.sin(q2), l1 + l23 * np.cos(q2))
        solutions.append([q1, q2, 0.0])
    return wrap_to_pi(np.asarray(solutions))


def angle_embedding(qs: np.ndarray) -> np.ndarray:
    qs = np.asarray(qs)
    return np.concatenate([np.cos(qs), np.sin(qs)], axis=-1)


def circular_kmeans_modes(final_samples: np.ndarray, k: int = 2, iterations: int = 50) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cluster joint-angle samples with a sin/cos embedding."""
    samples = np.asarray(final_samples)
    embedded = angle_embedding(samples)
    centers = [embedded[0]]
    while len(centers) < k:
        distances = np.min(
            np.stack([np.linalg.norm(embedded - center, axis=1) for center in centers], axis=1),
            axis=1,
        )
        centers.append(embedded[int(np.argmax(distances))])
    centers = np.asarray(centers)

    labels = np.zeros(len(samples), dtype=int)
    for _ in range(iterations):
        distances = np.stack([np.linalg.norm(embedded - center, axis=1) for center in centers], axis=1)
        next_labels = np.argmin(distances, axis=1)
        next_centers = centers.copy()
        for idx in range(k):
            members = embedded[next_labels == idx]
            if len(members):
                next_centers[idx] = members.mean(axis=0)
        if np.array_equal(labels, next_labels):
            centers = next_centers
            break
        labels = next_labels
        centers = next_centers

    center_angles = np.arctan2(centers[:, samples.shape[1] :], centers[:, : samples.shape[1]])
    nearest_distances = np.asarray([np.linalg.norm(embedded[i] - centers[labels[i]]) for i in range(len(samples))])
    return labels, wrap_to_pi(center_angles), nearest_distances


def run_mode_collapse_experiment(seeds=range(50), target=TARGET_13, iterations: int = 30):
    config = CEMConfig(population_size=500, elite_size=50, iterations=iterations, dim=3, initial_mean=0.0, initial_sigma=1.0)
    histories = [run_cem(project1_cost("quadratic", target=target), seed=seed, config=config) for seed in seeds]
    final_samples = np.stack([np.asarray(history.best_samples[-1]) for history in histories], axis=0)
    labels, centers, distances = circular_kmeans_modes(wrap_to_pi(final_samples), k=2)
    return histories, wrap_to_pi(final_samples), labels, centers, distances


def convergence_statistics(cost_names=("quadratic", "linear", "sparse"), seeds=range(10), config: CEMConfig = DEFAULT_CEM):
    summary = {}
    histories = {}
    for cost_name in cost_names:
        runs = [run_project1_cem(cost_name, seed, config) for seed in seeds]
        histories[cost_name] = runs
        costs = np.stack([np.asarray(run.best_costs) for run in runs], axis=0)
        summary[cost_name] = (costs.mean(axis=0), costs.std(axis=0))
    return summary, histories


def sparse_first_hit_statistics(population_sizes=(500, 2000, 5000), seeds=range(50), iterations: int = 30):
    results = {}
    for population_size in population_sizes:
        config = CEMConfig(
            population_size=population_size,
            elite_size=max(1, population_size // 10),
            iterations=iterations,
            dim=3,
            initial_mean=0.0,
            initial_sigma=1.0,
        )
        hits = []
        for seed in seeds:
            history = run_project1_cem("sparse", seed, config)
            hits.append(history.feasible_first_iteration if history.feasible_first_iteration is not None else iterations + 1)
        results[population_size] = np.asarray(hits, dtype=float)
    return results


def iterations_to_threshold(best_costs: np.ndarray, threshold: float = 0.01) -> int:
    hits = np.flatnonzero(np.asarray(best_costs) < threshold)
    return int(hits[0] + 1) if len(hits) else len(best_costs) + 1


def elite_fraction_sweep(fractions=(0.01, 0.05, 0.10, 0.25, 0.50), seeds=range(100)):
    results = {}
    for fraction in fractions:
        population_size = 500
        config = CEMConfig(
            population_size=population_size,
            elite_size=max(1, int(round(population_size * fraction))),
            iterations=50,
            dim=3,
            initial_mean=0.0,
            initial_sigma=1.0,
        )
        histories = [run_project1_cem("quadratic", seed, config) for seed in seeds]
        results[fraction] = {
            "cost_histories": np.stack([np.asarray(h.best_costs) for h in histories], axis=0),
        }
    return results


def covariance_strategy_sweep(
    strategies=("full", "diagonal"),
    sigmas=(0.5, 1.0, 1.5, 2.0),
    seeds=range(100),
    threshold: float = 0.01,
):
    results = {}
    for strategy in strategies:
        for sigma in sigmas:
            config = CEMConfig(
                population_size=500,
                elite_size=50,
                iterations=30,
                dim=3,
                initial_mean=0.0,
                initial_sigma=sigma,
                covariance=strategy,
            )
            histories = [run_project1_cem("quadratic", seed, config) for seed in seeds]
            results[(strategy, sigma)] = {
                "iterations": np.asarray([iterations_to_threshold(h.best_costs, threshold) for h in histories], dtype=float),
                "final_costs": np.asarray([float(h.best_costs[-1]) for h in histories]),
                "cost_histories": np.stack([np.asarray(h.best_costs) for h in histories], axis=0),
            }
    return results


def dimensionality_sweep(dimensions=(3, 6, 10), seeds=range(100), threshold: float = 0.01):
    results = {}
    for dim in dimensions:
        config = CEMConfig(population_size=500, elite_size=50, iterations=30, dim=dim, initial_mean=0.0, initial_sigma=1.0)
        histories = [run_project1_cem("quadratic", seed, config) for seed in seeds]
        results[dim] = {
            "iterations": np.asarray([iterations_to_threshold(h.best_costs, threshold) for h in histories], dtype=float),
            "final_costs": np.asarray([float(h.best_costs[-1]) for h in histories]),
        }
    return results


def manipulator_points(q: np.ndarray, link_lengths: np.ndarray | None = None) -> np.ndarray:
    q = np.asarray(q)
    if link_lengths is None:
        link_lengths = np.ones(q.shape[-1])
    theta = np.cumsum(q)
    increments = np.stack([link_lengths * np.cos(theta), link_lengths * np.sin(theta)], axis=-1)
    return np.concatenate([np.zeros((1, 2)), np.cumsum(increments, axis=0)], axis=0)


def final_position_errors(qs: np.ndarray, target=TARGET_13) -> np.ndarray:
    positions = np.asarray(forward_kinematics(jnp.asarray(qs)))
    return np.linalg.norm(positions - np.asarray(target), axis=-1)
