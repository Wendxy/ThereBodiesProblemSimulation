import argparse
import csv
import os
import subprocess
from typing import Tuple

import numpy as np


def sample_initial_conditions(
    rng: np.random.Generator,
    mass_scale: float,
    position_scale: float,
    velocity_scale: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    masses = mass_scale * (0.5 + rng.random(3))
    positions = rng.normal(0.0, position_scale, size=(3, 3))
    velocities = rng.normal(0.0, velocity_scale, size=(3, 3))

    # Zero center-of-mass position and velocity
    com_pos = np.sum(positions * masses[:, None], axis=0) / np.sum(masses)
    com_vel = np.sum(velocities * masses[:, None], axis=0) / np.sum(masses)
    positions -= com_pos
    velocities -= com_vel

    return masses, positions, velocities


def fmt_vec(v: np.ndarray) -> str:
    return f"{v[0]:.6e},{v[1]:.6e},{v[2]:.6e}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate three-body trajectories via C++ simulator.")
    parser.add_argument("--sim-bin", default="./threebody_opengl", help="Path to simulator binary.")
    parser.add_argument("--out-dir", default="ml/data", help="Output directory for CSV files.")
    parser.add_argument("--num-trajectories", type=int, default=50)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--dt", type=float, default=200.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--mass-scale", type=float, default=1.0e30)
    parser.add_argument("--position-scale", type=float, default=1.0e11)
    parser.add_argument("--velocity-scale", type=float, default=3.0e4)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    manifest_path = os.path.join(args.out_dir, "manifest.csv")

    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "m1", "m2", "m3", "dt", "steps"])

    for i in range(args.num_trajectories):
        masses, positions, velocities = sample_initial_conditions(
            rng, args.mass_scale, args.position_scale, args.velocity_scale
        )
        out_path = os.path.join(args.out_dir, f"traj_{i:04d}.csv")
        cmd = [
            args.sim_bin,
            "--headless",
            "--dt",
            str(args.dt),
            "--steps",
            str(args.steps),
            "--out",
            out_path,
            "--m1",
            f"{masses[0]:.6e}",
            "--m2",
            f"{masses[1]:.6e}",
            "--m3",
            f"{masses[2]:.6e}",
            "--p1",
            fmt_vec(positions[0]),
            "--p2",
            fmt_vec(positions[1]),
            "--p3",
            fmt_vec(positions[2]),
            "--v1",
            fmt_vec(velocities[0]),
            "--v2",
            fmt_vec(velocities[1]),
            "--v3",
            fmt_vec(velocities[2]),
        ]
        subprocess.run(cmd, check=True)
        with open(manifest_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    os.path.basename(out_path),
                    f"{masses[0]:.6e}",
                    f"{masses[1]:.6e}",
                    f"{masses[2]:.6e}",
                    f"{args.dt}",
                    f"{args.steps}",
                ]
            )
        print(f"[{i+1}/{args.num_trajectories}] {out_path}")


if __name__ == "__main__":
    main()
