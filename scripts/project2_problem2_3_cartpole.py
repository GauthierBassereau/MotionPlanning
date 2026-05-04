"""Solve Project 2.3: cart-pole swing-up."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plotting import (
    plot_cartpole_convergence,
    plot_cartpole_frame_sequence,
    plot_cartpole_gradient_magnitude,
    plot_cartpole_learning_rate_sweep,
)
from project2 import run_cartpole_learning_rate_sweep, run_problem2_3


def print_result(name: str, result) -> None:
    terminal = result.trajectory[-1]
    print(
        f"  {name}: final cost={result.final_cost:.3f}, "
        f"wall time={result.wall_time:.3f}s, evaluations={result.evaluations}, "
        f"terminal=[x={terminal[0]:.3f}, xdot={terminal[1]:.3f}, theta={terminal[2]:.3f}, thetadot={terminal[3]:.3f}]"
    )


def main() -> None:
    results = run_problem2_3(seed=0)
    learning_rate_results = run_cartpole_learning_rate_sweep(learning_rates=(0.5, 0.4, 0.3, 0.2, 0.1))
    out_dir = ROOT / "figures" / "project2"
    convergence_output = out_dir / "problem2_3_convergence.png"
    learning_rate_output = out_dir / "problem2_3_adam_learning_rate_sweep.png"
    contact_gradient_output = out_dir / "problem2_3_contact_gradient_magnitude.png"
    frames_output = out_dir / "problem2_3_frame_sequence.png"

    plot_cartpole_convergence(results, convergence_output)
    plot_cartpole_learning_rate_sweep(learning_rate_results, learning_rate_output)
    plot_cartpole_gradient_magnitude(
        {
            "No-contact grad on CEM controls": results["grad_along_cem"],
            "Hard-stop grad on same controls": results["same_controls_contact_grad"],
            "Hard-stop grad on hard-stop CEM controls": results["grad_along_cem_contact"],
        },
        contact_gradient_output,
    )
    plot_cartpole_frame_sequence(results, frames_output)

    print(f"Wrote {convergence_output}")
    print(f"Wrote {learning_rate_output}")
    print(f"Wrote {contact_gradient_output}")
    print(f"Wrote {frames_output}")
    print("Problem 2.3 optimizer comparison:")
    print_result("CEM", results["cem"])
    print_result("Adam", results["gradient"])
    print_result("CEM hard-stop", results["cem_contact"])
    print_result("Adam hard-stop", results["gradient_contact"])
    print("Adam learning-rate sweep:")
    for case_name, case_results in (
        ("Adam", learning_rate_results["no_contact"]),
        ("Adam hard-stop", learning_rate_results["hard_stop"]),
    ):
        for learning_rate, result in case_results.items():
            print(f"  {case_name}, lr={learning_rate:g}: final cost={result.final_cost:.3f}")
    print("Gradient diagnostics:")
    for name, grad in (
        ("CEM trajectory", results["grad_along_cem"]),
        ("Hard-stop same controls", results["same_controls_contact_grad"]),
        ("Hard-stop CEM trajectory", results["grad_along_cem_contact"]),
    ):
        print(
            f"  {name}: min |grad|={np.min(np.abs(grad)):.3e}, "
            f"median |grad|={np.median(np.abs(grad)):.3e}, max |grad|={np.max(np.abs(grad)):.3e}"
        )
    print("Interpretation notes:")
    print("  The control dimension is 100, so CEM spends 50,000 cost evaluations per full run.")
    print("  Adam uses far fewer evaluation-equivalents but is sensitive to learning rate and to nonsmooth contact.")
    print("  The hard stop introduces a clamp in the dynamics; gradients can become uninformative near saturated contact states.")


if __name__ == "__main__":
    main()
