"""Cost functions and analytical gradients for Project 1."""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp

from kinematics import forward_kinematics, planar_jacobian

CostFn = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]


def reaching_error(q: jnp.ndarray, target: jnp.ndarray, link_lengths: jnp.ndarray | None = None) -> jnp.ndarray:
    """Return ``p(q) - target``."""
    return forward_kinematics(q, link_lengths) - jnp.asarray(target, dtype=jnp.asarray(q).dtype)


def cost_quadratic(q: jnp.ndarray, target: jnp.ndarray, link_lengths: jnp.ndarray | None = None) -> jnp.ndarray:
    """Squared Euclidean reaching error."""
    err = reaching_error(q, target, link_lengths)
    return jnp.sum(err * err, axis=-1)


def cost_linear(q: jnp.ndarray, target: jnp.ndarray, link_lengths: jnp.ndarray | None = None) -> jnp.ndarray:
    """Euclidean reaching error."""
    err = reaching_error(q, target, link_lengths)
    return jnp.linalg.norm(err, axis=-1)


def cost_sparse(
    q: jnp.ndarray,
    target: jnp.ndarray,
    link_lengths: jnp.ndarray | None = None,
    epsilon: float = 0.1,
) -> jnp.ndarray:
    """Indicator cost: zero inside an epsilon ball around the target, one outside."""
    return jnp.where(cost_linear(q, target, link_lengths) < epsilon, 0.0, 1.0)


def grad_quadratic_analytic(
    q: jnp.ndarray,
    target: jnp.ndarray,
    link_lengths: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Analytical gradient of ``||p(q)-target||^2``.

    ``grad J = 2 J_p(q)^T (p(q)-target)``.
    """
    err = reaching_error(q, target, link_lengths)
    jac = planar_jacobian(q, link_lengths)
    return 2.0 * jnp.einsum("...ij,...i->...j", jac, err)


def grad_linear_analytic(
    q: jnp.ndarray,
    target: jnp.ndarray,
    link_lengths: jnp.ndarray | None = None,
    eps: float = 1e-12,
) -> jnp.ndarray:
    """Analytical gradient of ``||p(q)-target||`` away from zero error.

    The true gradient is undefined where ``p(q) == target``. This implementation
    returns NaNs at those points so scripts can expose the singularity explicitly.
    """
    err = reaching_error(q, target, link_lengths)
    norm = jnp.linalg.norm(err, axis=-1)
    jac = planar_jacobian(q, link_lengths)
    grad = jnp.einsum("...ij,...i->...j", jac, err) / jnp.expand_dims(jnp.maximum(norm, eps), -1)
    return jnp.where(jnp.expand_dims(norm > eps, -1), grad, jnp.nan)


COST_FNS: dict[str, CostFn] = {
    "quadratic": cost_quadratic,
    "linear": cost_linear,
    "sparse": cost_sparse,
}
