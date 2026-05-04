"""Project 2 trajectory optimization utilities."""

from __future__ import annotations

from dataclasses import dataclass
import time

import jax
import jax.numpy as jnp
import numpy as np

from cem import CEMConfig, CEMHistory, run_cem


@dataclass(frozen=True)
class PointMassConfig:
    dt: float = 0.1
    horizon: int = 20
    control_dim: int = 2
    start: tuple[float, float] = (0.0, 0.0)
    goal: tuple[float, float] = (3.0, 2.0)
    control_weight: float = 0.01


@dataclass(frozen=True)
class ObstacleConfig:
    point_mass: PointMassConfig = PointMassConfig()
    centers: tuple[tuple[float, float], ...] = ((1.0, 1.0), (2.0, 0.5), (1.5, 2.0))
    radii: tuple[float, ...] = (0.4, 0.4, 0.4)
    obstacle_weight: float = 1000.0


@dataclass(frozen=True)
class CartPoleConfig:
    cart_mass: float = 1.0
    pole_mass: float = 0.1
    pole_length: float = 0.5
    gravity: float = 9.81
    dt: float = 0.02
    horizon: int = 100
    start: tuple[float, float, float, float] = (0.0, 0.0, float(np.pi), 0.0)
    control_weight: float = 0.01


@dataclass(frozen=True)
class OptimizerResult:
    name: str
    controls: np.ndarray
    trajectory: np.ndarray
    costs: np.ndarray
    final_cost: float
    iterations: int
    converged_iteration: int | None
    wall_time: float
    evaluations: int


def point_mass_rollout(controls: jnp.ndarray, config: PointMassConfig = PointMassConfig()) -> jnp.ndarray:
    """Roll out integrator dynamics for controls with shape ``(..., T, 2)``."""
    controls = jnp.asarray(controls)
    start = jnp.asarray(config.start, dtype=controls.dtype)
    increments = config.dt * controls
    positions = start + jnp.cumsum(increments, axis=-2)
    start_batch = jnp.broadcast_to(start, controls.shape[:-2] + (1, config.control_dim))
    return jnp.concatenate([start_batch, positions], axis=-2)


def point_mass_cost(controls: jnp.ndarray, config: PointMassConfig = PointMassConfig()) -> jnp.ndarray:
    controls = jnp.asarray(controls)
    trajectory = point_mass_rollout(controls, config)
    goal = jnp.asarray(config.goal, dtype=controls.dtype)
    terminal_error = trajectory[..., -1, :] - goal
    terminal_cost = jnp.sum(terminal_error * terminal_error, axis=-1)
    effort_cost = config.control_weight * jnp.sum(controls * controls, axis=(-2, -1))
    return terminal_cost + effort_cost


def flat_point_mass_cost(flat_controls: jnp.ndarray, config: PointMassConfig = PointMassConfig()) -> jnp.ndarray:
    controls = flat_controls.reshape(flat_controls.shape[:-1] + (config.horizon, config.control_dim))
    return point_mass_cost(controls, config)


def analytic_point_mass_solution(config: PointMassConfig = PointMassConfig()) -> np.ndarray:
    goal = np.asarray(config.goal, dtype=float) - np.asarray(config.start, dtype=float)
    denominator = config.horizon * config.dt**2 + config.control_weight
    constant_control = config.dt * goal / denominator
    return np.broadcast_to(constant_control, (config.horizon, config.control_dim)).copy()


def point_mass_hessian(config: PointMassConfig = PointMassConfig()) -> np.ndarray:
    """Analytical Hessian of the convex point-mass objective."""
    t = config.horizon
    d = config.control_dim
    h_time = 2.0 * (config.control_weight * np.eye(t) + config.dt**2 * np.ones((t, t)))
    return np.kron(h_time, np.eye(d))


def first_within(costs: np.ndarray, target: float, tolerance: float) -> int | None:
    hits = np.flatnonzero(np.asarray(costs) <= target + tolerance)
    return int(hits[0] + 1) if len(hits) else None


def run_point_mass_cem(
    seed: int = 0,
    config: PointMassConfig = PointMassConfig(),
    tolerance: float = 1e-4,
) -> tuple[OptimizerResult, CEMHistory]:
    dim = config.horizon * config.control_dim
    cem_config = CEMConfig(
        population_size=1000,
        elite_size=100,
        iterations=50,
        dim=dim,
        initial_mean=0.0,
        initial_sigma=1.0,
        covariance="diagonal",
    )
    cost_fn = lambda flat_u: flat_point_mass_cost(flat_u, config)
    optimal_cost = float(point_mass_cost(jnp.asarray(analytic_point_mass_solution(config)), config))

    start_time = time.perf_counter()
    history = run_cem(cost_fn, seed=seed, config=cem_config)
    wall_time = time.perf_counter() - start_time

    controls = np.asarray(history.best_samples[-1]).reshape(config.horizon, config.control_dim)
    trajectory = np.asarray(point_mass_rollout(jnp.asarray(controls), config))
    costs = np.asarray(history.best_costs)
    result = OptimizerResult(
        name="CEM",
        controls=controls,
        trajectory=trajectory,
        costs=costs,
        final_cost=float(costs[-1]),
        iterations=cem_config.iterations,
        converged_iteration=first_within(costs, optimal_cost, tolerance),
        wall_time=wall_time,
        evaluations=cem_config.population_size * cem_config.iterations,
    )
    return result, history


def run_point_mass_gradient(
    config: PointMassConfig = PointMassConfig(),
    learning_rate: float = 0.1,
    max_iterations: int = 500,
    tolerance: float = 1e-4,
) -> OptimizerResult:
    cost_and_grad = jax.jit(jax.value_and_grad(lambda flat_u: flat_point_mass_cost(flat_u, config)))
    flat_u = jnp.zeros((config.horizon * config.control_dim,), dtype=jnp.float32)
    m = jnp.zeros_like(flat_u)
    v = jnp.zeros_like(flat_u)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    optimal_cost = float(point_mass_cost(jnp.asarray(analytic_point_mass_solution(config)), config))

    # Compile outside timing.
    cost_and_grad(flat_u)

    costs = []
    converged_iteration = None
    start_time = time.perf_counter()
    for iteration in range(1, max_iterations + 1):
        cost, grad = cost_and_grad(flat_u)
        costs.append(float(cost))
        if converged_iteration is None and float(cost) <= optimal_cost + tolerance:
            converged_iteration = iteration
            break
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad * grad)
        m_hat = m / (1.0 - beta1**iteration)
        v_hat = v / (1.0 - beta2**iteration)
        flat_u = flat_u - learning_rate * m_hat / (jnp.sqrt(v_hat) + eps)
    wall_time = time.perf_counter() - start_time

    controls = np.asarray(flat_u).reshape(config.horizon, config.control_dim)
    trajectory = np.asarray(point_mass_rollout(jnp.asarray(controls), config))
    final_cost = float(point_mass_cost(jnp.asarray(controls), config))
    if not costs or costs[-1] != final_cost:
        costs.append(final_cost)
    return OptimizerResult(
        name="Adam",
        controls=controls,
        trajectory=trajectory,
        costs=np.asarray(costs),
        final_cost=final_cost,
        iterations=len(costs),
        converged_iteration=converged_iteration,
        wall_time=wall_time,
        evaluations=2 * len(costs),
    )


def run_problem2_1(seed: int = 0) -> dict[str, object]:
    config = PointMassConfig()
    analytic_controls = analytic_point_mass_solution(config)
    analytic_cost = float(point_mass_cost(jnp.asarray(analytic_controls), config))
    cem_result, cem_history = run_point_mass_cem(seed=seed, config=config)
    grad_result = run_point_mass_gradient(config=config)
    hessian = point_mass_hessian(config)
    eigvals = np.linalg.eigvalsh(hessian)
    return {
        "config": config,
        "analytic_controls": analytic_controls,
        "analytic_cost": analytic_cost,
        "cem": cem_result,
        "cem_history": cem_history,
        "gradient": grad_result,
        "hessian": hessian,
        "hessian_eigenvalues": eigvals,
    }


def obstacle_arrays(config: ObstacleConfig = ObstacleConfig()) -> tuple[jnp.ndarray, jnp.ndarray]:
    return (
        jnp.asarray(config.centers, dtype=jnp.float32),
        jnp.asarray(config.radii, dtype=jnp.float32),
    )


def obstacle_penalty(trajectory: jnp.ndarray, config: ObstacleConfig = ObstacleConfig()) -> jnp.ndarray:
    centers, radii = obstacle_arrays(config)
    distances = jnp.linalg.norm(trajectory[..., :, None, :] - centers, axis=-1)
    penetration = jnp.maximum(0.0, radii - distances)
    return config.obstacle_weight * jnp.sum(penetration * penetration, axis=(-2, -1))


def point_mass_obstacle_cost(flat_controls: jnp.ndarray, config: ObstacleConfig = ObstacleConfig()) -> jnp.ndarray:
    pm = config.point_mass
    controls = flat_controls.reshape(flat_controls.shape[:-1] + (pm.horizon, pm.control_dim))
    trajectory = point_mass_rollout(controls, pm)
    base_cost = point_mass_cost(controls, pm)
    return base_cost + obstacle_penalty(trajectory, config)


def trajectory_collision_margin(trajectory: np.ndarray, config: ObstacleConfig = ObstacleConfig()) -> float:
    centers = np.asarray(config.centers)
    radii = np.asarray(config.radii)
    distances = np.linalg.norm(np.asarray(trajectory)[:, None, :] - centers[None, :, :], axis=-1)
    return float(np.min(distances - radii))


def is_collision_free(trajectory: np.ndarray, config: ObstacleConfig = ObstacleConfig(), tol: float = 1e-6) -> bool:
    return trajectory_collision_margin(trajectory, config) >= -tol


def controls_from_waypoint(waypoint: np.ndarray, config: PointMassConfig = PointMassConfig()) -> np.ndarray:
    waypoint = np.asarray(waypoint, dtype=float)
    start = np.asarray(config.start, dtype=float)
    goal = np.asarray(config.goal, dtype=float)
    half = config.horizon // 2
    first = np.linspace(start, waypoint, half + 1)
    second = np.linspace(waypoint, goal, config.horizon - half + 1)[1:]
    positions = np.concatenate([first, second], axis=0)
    return np.diff(positions, axis=0) / config.dt


def waypoint_landscape(
    config: ObstacleConfig = ObstacleConfig(),
    resolution: int = 180,
    xlim: tuple[float, float] = (-0.25, 3.25),
    ylim: tuple[float, float] = (-0.5, 2.75),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wx = np.linspace(xlim[0], xlim[1], resolution)
    wy = np.linspace(ylim[0], ylim[1], resolution)
    wx_grid, wy_grid = np.meshgrid(wx, wy, indexing="xy")
    costs = np.zeros_like(wx_grid)
    for index in np.ndindex(wx_grid.shape):
        controls = controls_from_waypoint(np.array([wx_grid[index], wy_grid[index]]), config.point_mass)
        costs[index] = float(point_mass_obstacle_cost(jnp.asarray(controls.reshape(-1)), config))
    return wx_grid, wy_grid, costs


def run_obstacle_cem(
    seed: int,
    config: ObstacleConfig = ObstacleConfig(),
    iterations: int = 50,
    initial_sigma: float = 2.0,
) -> tuple[OptimizerResult, CEMHistory]:
    pm = config.point_mass
    cem_config = CEMConfig(
        population_size=1000,
        elite_size=100,
        iterations=iterations,
        dim=pm.horizon * pm.control_dim,
        initial_mean=0.0,
        initial_sigma=initial_sigma,
        covariance="diagonal",
    )
    cost_fn = lambda flat_u: point_mass_obstacle_cost(flat_u, config)
    start_time = time.perf_counter()
    history = run_cem(cost_fn, seed=seed, config=cem_config)
    wall_time = time.perf_counter() - start_time
    controls = np.asarray(history.best_samples[-1]).reshape(pm.horizon, pm.control_dim)
    trajectory = np.asarray(point_mass_rollout(jnp.asarray(controls), pm))
    result = OptimizerResult(
        name="CEM",
        controls=controls,
        trajectory=trajectory,
        costs=np.asarray(history.best_costs),
        final_cost=float(history.best_costs[-1]),
        iterations=iterations,
        converged_iteration=None,
        wall_time=wall_time,
        evaluations=cem_config.population_size * iterations,
    )
    return result, history


def run_obstacle_gradient(
    seed: int,
    config: ObstacleConfig = ObstacleConfig(),
    initial_controls: np.ndarray | None = None,
    initial_noise: float = 0.5,
    learning_rate: float = 0.05,
    max_iterations: int = 500,
    name: str = "Adam",
) -> OptimizerResult:
    pm = config.point_mass
    cost_and_grad = jax.jit(jax.value_and_grad(lambda flat_u: point_mass_obstacle_cost(flat_u, config)))
    if initial_controls is None:
        key = jax.random.PRNGKey(seed)
        flat_u = initial_noise * jax.random.normal(key, (pm.horizon * pm.control_dim,), dtype=jnp.float32)
    else:
        flat_u = jnp.asarray(initial_controls.reshape(-1), dtype=jnp.float32)

    m = jnp.zeros_like(flat_u)
    v = jnp.zeros_like(flat_u)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    cost_and_grad(flat_u)

    costs = []
    start_time = time.perf_counter()
    for iteration in range(1, max_iterations + 1):
        cost, grad = cost_and_grad(flat_u)
        costs.append(float(cost))
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad * grad)
        m_hat = m / (1.0 - beta1**iteration)
        v_hat = v / (1.0 - beta2**iteration)
        flat_u = flat_u - learning_rate * m_hat / (jnp.sqrt(v_hat) + eps)
    wall_time = time.perf_counter() - start_time

    controls = np.asarray(flat_u).reshape(pm.horizon, pm.control_dim)
    trajectory = np.asarray(point_mass_rollout(jnp.asarray(controls), pm))
    final_cost = float(point_mass_obstacle_cost(jnp.asarray(flat_u), config))
    costs.append(final_cost)
    return OptimizerResult(
        name=name,
        controls=controls,
        trajectory=trajectory,
        costs=np.asarray(costs),
        final_cost=final_cost,
        iterations=max_iterations,
        converged_iteration=None,
        wall_time=wall_time,
        evaluations=2 * max_iterations,
    )


def run_obstacle_hybrid(seed: int, config: ObstacleConfig = ObstacleConfig()) -> tuple[OptimizerResult, CEMHistory]:
    cem_result, history = run_obstacle_cem(seed=seed, config=config, iterations=10)
    gradient = run_obstacle_gradient(
        seed=seed,
        config=config,
        initial_controls=cem_result.controls,
        initial_noise=0.0,
        learning_rate=0.03,
        max_iterations=500,
        name="CEM10+Adam",
    )
    result = OptimizerResult(
        name="CEM10+Adam",
        controls=gradient.controls,
        trajectory=gradient.trajectory,
        costs=np.concatenate([cem_result.costs, gradient.costs]),
        final_cost=gradient.final_cost,
        iterations=10 + gradient.iterations,
        converged_iteration=None,
        wall_time=cem_result.wall_time + gradient.wall_time,
        evaluations=cem_result.evaluations + gradient.evaluations,
    )
    return result, history


def summarize_obstacle_results(results: list[OptimizerResult], config: ObstacleConfig = ObstacleConfig()) -> dict[str, object]:
    costs = np.asarray([r.final_cost for r in results])
    margins = np.asarray([trajectory_collision_margin(r.trajectory, config) for r in results])
    successes = margins >= -1e-6
    return {
        "costs": costs,
        "margins": margins,
        "successes": successes,
        "success_rate": float(np.mean(successes)),
        "wall_times": np.asarray([r.wall_time for r in results]),
        "evaluations": np.asarray([r.evaluations for r in results]),
    }


def run_problem2_2(seeds=range(20)) -> dict[str, object]:
    config = ObstacleConfig()
    cem_runs = []
    grad_runs = []
    hybrid_runs = []
    for seed in seeds:
        cem_result, _ = run_obstacle_cem(seed=seed, config=config)
        grad_result = run_obstacle_gradient(seed=seed, config=config)
        hybrid_result, _ = run_obstacle_hybrid(seed=seed, config=config)
        cem_runs.append(cem_result)
        grad_runs.append(grad_result)
        hybrid_runs.append(hybrid_result)

    return {
        "config": config,
        "cem": cem_runs,
        "gradient": grad_runs,
        "hybrid": hybrid_runs,
        "summaries": {
            "CEM": summarize_obstacle_results(cem_runs, config),
            "Adam": summarize_obstacle_results(grad_runs, config),
            "CEM10+Adam": summarize_obstacle_results(hybrid_runs, config),
        },
    }


def cartpole_accelerations(state: jnp.ndarray, control: jnp.ndarray, config: CartPoleConfig = CartPoleConfig()) -> tuple[jnp.ndarray, jnp.ndarray]:
    x, x_dot, theta, theta_dot = state
    del x, x_dot
    mc = config.cart_mass
    mp = config.pole_mass
    length = config.pole_length
    gravity = config.gravity
    sin_theta = jnp.sin(theta)
    cos_theta = jnp.cos(theta)
    denominator = mc + mp - mp * cos_theta * cos_theta
    x_ddot = (control + mp * length * theta_dot * theta_dot * sin_theta - mp * gravity * sin_theta * cos_theta) / denominator
    theta_ddot = (gravity * sin_theta - x_ddot * cos_theta) / length
    return x_ddot, theta_ddot


def cartpole_step(
    state: jnp.ndarray,
    control: jnp.ndarray,
    config: CartPoleConfig = CartPoleConfig(),
    hard_stop: bool = False,
) -> jnp.ndarray:
    x, x_dot, theta, theta_dot = state
    x_ddot, theta_ddot = cartpole_accelerations(state, control, config)
    next_state = jnp.array(
        [
            x + config.dt * x_dot,
            x_dot + config.dt * x_ddot,
            theta + config.dt * theta_dot,
            theta_dot + config.dt * theta_ddot,
        ]
    )
    if hard_stop:
        hit_stop = (next_state[2] > jnp.pi) & (next_state[3] > 0.0)
        next_state = jnp.where(
            hit_stop,
            jnp.array([next_state[0], next_state[1], jnp.pi, 0.0], dtype=next_state.dtype),
            next_state,
        )
    return next_state


def cartpole_rollout(
    controls: jnp.ndarray,
    config: CartPoleConfig = CartPoleConfig(),
    hard_stop: bool = False,
) -> jnp.ndarray:
    controls = jnp.asarray(controls)
    start = jnp.asarray(config.start, dtype=controls.dtype)

    def scan_step(state, control):
        next_state = cartpole_step(state, control, config, hard_stop)
        return next_state, next_state

    _, states = jax.lax.scan(scan_step, start, controls)
    return jnp.concatenate([start[None, :], states], axis=0)


def cartpole_cost(
    controls: jnp.ndarray,
    config: CartPoleConfig = CartPoleConfig(),
    hard_stop: bool = False,
) -> jnp.ndarray:
    controls = jnp.asarray(controls)
    trajectory = jax.vmap(lambda u: cartpole_rollout(u, config, hard_stop))(controls) if controls.ndim == 2 and controls.shape[0] != config.horizon else None
    if trajectory is None:
        trajectory = cartpole_rollout(controls, config, hard_stop)
        final = trajectory[-1]
        return (
            100.0 * (1.0 - jnp.cos(final[2]))
            + final[0] * final[0]
            + final[1] * final[1]
            + final[3] * final[3]
            + config.control_weight * jnp.sum(controls * controls)
        )
    final = trajectory[:, -1, :]
    return (
        100.0 * (1.0 - jnp.cos(final[:, 2]))
        + final[:, 0] * final[:, 0]
        + final[:, 1] * final[:, 1]
        + final[:, 3] * final[:, 3]
        + config.control_weight * jnp.sum(controls * controls, axis=-1)
    )


def run_cartpole_cem(
    seed: int = 0,
    config: CartPoleConfig = CartPoleConfig(),
    hard_stop: bool = False,
    iterations: int = 50,
    population_size: int = 1000,
    initial_sigma: float = 12.0,
) -> tuple[OptimizerResult, CEMHistory]:
    cem_config = CEMConfig(
        population_size=population_size,
        elite_size=max(1, population_size // 10),
        iterations=iterations,
        dim=config.horizon,
        initial_mean=0.0,
        initial_sigma=initial_sigma,
        covariance="diagonal",
        min_variance=1e-5,
    )
    cost_fn = lambda u: cartpole_cost(u, config, hard_stop)
    start_time = time.perf_counter()
    history = run_cem(cost_fn, seed=seed, config=cem_config)
    wall_time = time.perf_counter() - start_time
    controls = np.asarray(history.best_samples[-1])
    trajectory = np.asarray(cartpole_rollout(jnp.asarray(controls), config, hard_stop))
    result = OptimizerResult(
        name="CEM" if not hard_stop else "CEM hard-stop",
        controls=controls,
        trajectory=trajectory,
        costs=np.asarray(history.best_costs),
        final_cost=float(history.best_costs[-1]),
        iterations=iterations,
        converged_iteration=None,
        wall_time=wall_time,
        evaluations=population_size * iterations,
    )
    return result, history


def run_cartpole_gradient(
    config: CartPoleConfig = CartPoleConfig(),
    hard_stop: bool = False,
    learning_rate: float = 0.5,
    max_iterations: int = 200,
    initial_controls: np.ndarray | None = None,
    name: str | None = None,
) -> OptimizerResult:
    cost_and_grad = jax.jit(jax.value_and_grad(lambda u: cartpole_cost(u, config, hard_stop)))
    controls = jnp.zeros((config.horizon,), dtype=jnp.float32) if initial_controls is None else jnp.asarray(initial_controls, dtype=jnp.float32)
    m = jnp.zeros_like(controls)
    v = jnp.zeros_like(controls)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    cost_and_grad(controls)

    costs = []
    start_time = time.perf_counter()
    for iteration in range(1, max_iterations + 1):
        cost, grad = cost_and_grad(controls)
        costs.append(float(cost))
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad * grad)
        m_hat = m / (1.0 - beta1**iteration)
        v_hat = v / (1.0 - beta2**iteration)
        controls = controls - learning_rate * m_hat / (jnp.sqrt(v_hat) + eps)
    wall_time = time.perf_counter() - start_time
    final_cost = float(cartpole_cost(controls, config, hard_stop))
    costs.append(final_cost)
    np_controls = np.asarray(controls)
    trajectory = np.asarray(cartpole_rollout(controls, config, hard_stop))
    return OptimizerResult(
        name=name or ("Adam" if not hard_stop else "Adam hard-stop"),
        controls=np_controls,
        trajectory=trajectory,
        costs=np.asarray(costs),
        final_cost=final_cost,
        iterations=max_iterations,
        converged_iteration=None,
        wall_time=wall_time,
        evaluations=2 * max_iterations,
    )


def cartpole_control_gradient(
    controls: np.ndarray,
    config: CartPoleConfig = CartPoleConfig(),
    hard_stop: bool = False,
) -> np.ndarray:
    grad_fn = jax.jit(jax.grad(lambda u: cartpole_cost(u, config, hard_stop)))
    return np.asarray(grad_fn(jnp.asarray(controls, dtype=jnp.float32)))


def run_cartpole_learning_rate_sweep(
    learning_rates: tuple[float, ...] = (0.15, 0.1, 0.05, 0.03),
    config: CartPoleConfig = CartPoleConfig(),
) -> dict[str, object]:
    no_contact = {
        lr: run_cartpole_gradient(
            config=config,
            hard_stop=False,
            learning_rate=lr,
            max_iterations=500,
            name=rf"Adam $\alpha={lr:g}$",
        )
        for lr in learning_rates
    }
    hard_stop = {
        lr: run_cartpole_gradient(
            config=config,
            hard_stop=True,
            learning_rate=lr,
            max_iterations=500,
            name=rf"Adam hard-stop $\alpha={lr:g}$",
        )
        for lr in learning_rates
    }
    return {
        "config": config,
        "learning_rates": learning_rates,
        "no_contact": no_contact,
        "hard_stop": hard_stop,
    }


def run_problem2_3(seed: int = 0) -> dict[str, object]:
    config = CartPoleConfig()
    cem, _ = run_cartpole_cem(seed=seed, config=config, hard_stop=False)
    gradient = run_cartpole_gradient(config=config, hard_stop=False, learning_rate=0.5)
    grad_along_cem = cartpole_control_gradient(cem.controls, config=config, hard_stop=False)

    cem_contact, _ = run_cartpole_cem(seed=seed, config=config, hard_stop=True)
    gradient_contact = run_cartpole_gradient(config=config, hard_stop=True, learning_rate=0.4)
    grad_along_cem_contact = cartpole_control_gradient(cem_contact.controls, config=config, hard_stop=True)
    same_controls_contact_grad = cartpole_control_gradient(cem.controls, config=config, hard_stop=True)

    return {
        "config": config,
        "cem": cem,
        "gradient": gradient,
        "grad_along_cem": grad_along_cem,
        "cem_contact": cem_contact,
        "gradient_contact": gradient_contact,
        "grad_along_cem_contact": grad_along_cem_contact,
        "same_controls_contact_grad": same_controls_contact_grad,
    }
