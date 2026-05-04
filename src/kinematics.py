"""Planar serial-chain kinematics in JAX."""

from __future__ import annotations

import jax.numpy as jnp


def forward_kinematics(q: jnp.ndarray, link_lengths: jnp.ndarray | None = None) -> jnp.ndarray:
    """Return end-effector positions for planar joint angles.

    Args:
        q: Array with shape ``(..., n)``.
        link_lengths: Optional array with shape ``(n,)``. Defaults to unit links.

    Returns:
        Array with shape ``(..., 2)`` containing ``(x, y)``.
    """
    q = jnp.asarray(q)
    if link_lengths is None:
        link_lengths = jnp.ones(q.shape[-1], dtype=q.dtype)
    else:
        link_lengths = jnp.asarray(link_lengths, dtype=q.dtype)

    theta = jnp.cumsum(q, axis=-1)
    x = jnp.sum(link_lengths * jnp.cos(theta), axis=-1)
    y = jnp.sum(link_lengths * jnp.sin(theta), axis=-1)
    return jnp.stack([x, y], axis=-1)


def planar_jacobian(q: jnp.ndarray, link_lengths: jnp.ndarray | None = None) -> jnp.ndarray:
    """Analytical Jacobian ``dp/dq`` for a planar serial chain.

    Supports batched inputs. Output shape is ``(..., 2, n)``.
    """
    q = jnp.asarray(q)
    if link_lengths is None:
        link_lengths = jnp.ones(q.shape[-1], dtype=q.dtype)
    else:
        link_lengths = jnp.asarray(link_lengths, dtype=q.dtype)

    theta = jnp.cumsum(q, axis=-1)
    sin_terms = link_lengths * jnp.sin(theta)
    cos_terms = link_lengths * jnp.cos(theta)
    # Joint i affects every downstream link j >= i.
    dx_dq = -jnp.flip(jnp.cumsum(jnp.flip(sin_terms, axis=-1), axis=-1), axis=-1)
    dy_dq = jnp.flip(jnp.cumsum(jnp.flip(cos_terms, axis=-1), axis=-1), axis=-1)
    return jnp.stack([dx_dq, dy_dq], axis=-2)
