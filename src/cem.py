"""Cross-Entropy Method optimizer with full history capture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class CEMConfig:
    population_size: int = 500
    elite_size: int = 50
    iterations: int = 30
    dim: int = 3
    initial_mean: float | jnp.ndarray = 0.0
    initial_sigma: float = 1.0
    min_variance: float = 1e-8
    covariance: str = "full"


@dataclass(frozen=True)
class CEMHistory:
    means: jnp.ndarray
    covariances: jnp.ndarray
    best_costs: jnp.ndarray
    best_samples: jnp.ndarray
    elite_sets: jnp.ndarray
    elite_costs: jnp.ndarray
    feasible_first_iteration: int | None


def _initial_mean(config: CEMConfig) -> jnp.ndarray:
    mean = jnp.asarray(config.initial_mean, dtype=jnp.float32)
    if mean.ndim == 0:
        return jnp.full((config.dim,), mean)
    return mean.astype(jnp.float32)


def _regularize_cov(cov: jnp.ndarray, min_variance: float) -> jnp.ndarray:
    cov = 0.5 * (cov + cov.T)
    eigenvalues, eigenvectors = jnp.linalg.eigh(cov)
    clipped = jnp.maximum(eigenvalues, min_variance)
    return (eigenvectors * clipped) @ eigenvectors.T


def _sample_gaussian(key: jax.Array, mean: jnp.ndarray, cov: jnp.ndarray, count: int, min_variance: float) -> jnp.ndarray:
    cov = 0.5 * (cov + cov.T)
    eigenvalues, eigenvectors = jnp.linalg.eigh(cov)
    clipped = jnp.maximum(eigenvalues, min_variance)
    factor = eigenvectors * jnp.sqrt(clipped)
    noise = jax.random.normal(key, shape=(count, mean.shape[0]), dtype=mean.dtype)
    return mean + noise @ factor.T


def run_cem(
    cost_fn: Callable[[jnp.ndarray], jnp.ndarray],
    seed: int,
    config: CEMConfig,
    feasible_threshold: float | None = None,
) -> CEMHistory:
    """Run CEM and record per-iteration optimizer state.

    Iteration indices in ``feasible_first_iteration`` are 1-based, matching the
    assignment statement.
    """
    key = jax.random.PRNGKey(seed)
    mean = _initial_mean(config)
    cov = (config.initial_sigma**2) * jnp.eye(config.dim, dtype=jnp.float32)

    means = []
    covariances = []
    best_costs = []
    best_samples = []
    elite_sets = []
    elite_costs = []
    feasible_first_iteration = None

    for iteration in range(1, config.iterations + 1):
        key, sample_key = jax.random.split(key)
        samples = _sample_gaussian(sample_key, mean, cov, config.population_size, config.min_variance)
        costs = cost_fn(samples)
        order = jnp.argsort(costs)
        elite_idx = order[: config.elite_size]
        elites = samples[elite_idx]
        elite_vals = costs[elite_idx]

        if feasible_threshold is not None and feasible_first_iteration is None:
            if bool(jnp.any(costs <= feasible_threshold)):
                feasible_first_iteration = iteration

        mean = jnp.mean(elites, axis=0)
        centered = elites - mean
        if config.covariance == "full":
            cov = centered.T @ centered / max(config.elite_size - 1, 1)
        elif config.covariance == "diagonal":
            cov = jnp.diag(jnp.var(elites, axis=0))
        else:
            raise ValueError(f"Unknown covariance strategy: {config.covariance}")
        cov = _regularize_cov(cov, config.min_variance)

        best_idx = order[0]
        means.append(mean)
        covariances.append(cov)
        best_costs.append(costs[best_idx])
        best_samples.append(samples[best_idx])
        elite_sets.append(elites)
        elite_costs.append(elite_vals)

    return CEMHistory(
        means=jnp.stack(means),
        covariances=jnp.stack(covariances),
        best_costs=jnp.stack(best_costs),
        best_samples=jnp.stack(best_samples),
        elite_sets=jnp.stack(elite_sets),
        elite_costs=jnp.stack(elite_costs),
        feasible_first_iteration=feasible_first_iteration,
    )
