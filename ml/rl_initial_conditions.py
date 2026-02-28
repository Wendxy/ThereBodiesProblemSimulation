import argparse
import os
import subprocess
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from utils import read_simulation_csv, stability_metrics


class Policy(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mean = nn.Linear(hidden, 18)
        self.log_std = nn.Parameter(torch.zeros(18))

    def forward(self, batch_size: int):
        x = torch.ones((batch_size, 1))
        h = self.net(x)
        mean = self.mean(h)
        std = torch.exp(self.log_std)[None, :].expand_as(mean)
        return Normal(mean, std)


def fmt_vec(v: np.ndarray) -> str:
    return f"{v[0]:.6e},{v[1]:.6e},{v[2]:.6e}"


def run_episode(
    sim_bin: str,
    out_path: str,
    dt: float,
    steps: int,
    masses: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    headless: bool,
    scale: float = 1e-8,
    record: bool = False,
) -> None:
    cmd = [
        sim_bin,
        "--headless",
        "--dt",
        str(dt),
        "--steps",
        str(steps),
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
    if not headless:
        cmd = [sim_bin] + cmd[2:]  # drop --headless
        if record:
            cmd += ["--record", out_path]
        cmd += ["--scale", str(scale)]
    subprocess.run(cmd, check=True)


def score_episode(
    csv_path: str,
    masses: np.ndarray,
    max_radius: float,
    min_separation: float,
) -> Tuple[float, dict]:
    data = read_simulation_csv(csv_path)
    positions = np.stack([data["p1"], data["p2"], data["p3"]], axis=1)
    velocities = np.stack([data["v1"], data["v2"], data["v3"]], axis=1)
    metrics = stability_metrics(positions, velocities, masses, max_radius, min_separation)

    penalty = 0.0
    penalty += max(0.0, metrics["max_radius"] - max_radius) / max_radius
    penalty += max(0.0, min_separation - metrics["min_separation"]) / min_separation
    penalty += metrics["energy_drift"]

    reward = 1.0 - penalty
    if metrics["stable"] < 0.5:
        reward -= 1.0
    return float(reward), metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="REINFORCE search for stable initial conditions.")
    parser.add_argument("--sim-bin", default="./threebody_opengl")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--dt", type=float, default=200.0)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--mass-scale", type=float, default=1.0e30)
    parser.add_argument("--max-radius", type=float, default=2.0e11)
    parser.add_argument("--min-separation", type=float, default=1.0e9)
    parser.add_argument("--out-dir", default="ml/rl_runs")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--visualize-best", action="store_true")
    parser.add_argument("--visual-scale", type=float, default=1e-8)
    parser.add_argument("--visual-out", default="ml/rl_runs/best_visual.csv")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(123)

    masses = args.mass_scale * np.array([1.0, 0.6, 0.3], dtype=np.float64)

    device = torch.device("cpu")
    policy = Policy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    baseline = 0.0

    best_reward = -1e9
    best_positions = None
    best_velocities = None

    for ep in range(args.episodes):
        dist = policy(args.batch)
        actions = dist.sample()
        log_probs = dist.log_prob(actions).sum(dim=1)

        rewards = []
        metrics_list = []
        for i in range(args.batch):
            action = actions[i].detach().cpu().numpy()
            positions = action[:9].reshape(3, 3) * 1.0e11
            velocities = action[9:].reshape(3, 3) * 3.0e4

            # Zero COM
            com_pos = np.sum(positions * masses[:, None], axis=0) / np.sum(masses)
            com_vel = np.sum(velocities * masses[:, None], axis=0) / np.sum(masses)
            positions -= com_pos
            velocities -= com_vel

            out_path = os.path.join(args.out_dir, f"ep{ep:04d}_b{i}.csv")
            run_episode(
                args.sim_bin,
                out_path,
                args.dt,
                args.steps,
                masses,
                positions,
                velocities,
                headless=True,
            )
            reward, metrics = score_episode(
                out_path, masses, args.max_radius, args.min_separation
            )
            rewards.append(reward)
            metrics_list.append(metrics)
            if reward > best_reward:
                best_reward = reward
                best_positions = positions.copy()
                best_velocities = velocities.copy()

        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        baseline = 0.9 * baseline + 0.1 * float(rewards_t.mean())
        advantages = rewards_t - baseline
        loss = -(log_probs * advantages).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        best_idx = int(np.argmax(rewards))
        best = metrics_list[best_idx]
        print(
            f"Episode {ep+1:03d} | reward={np.mean(rewards):.4f} | "
            f"best_stable={best['stable']:.0f} max_r={best['max_radius']:.3e} "
            f"min_sep={best['min_separation']:.3e} drift={best['energy_drift']:.3e}"
        )

    if args.visualize_best and best_positions is not None:
        print("Launching visual run for best candidate...")
        run_episode(
            args.sim_bin,
            args.visual_out,
            args.dt,
            args.steps,
            masses,
            best_positions,
            best_velocities,
            headless=False,
            scale=args.visual_scale,
            record=True,
        )


if __name__ == "__main__":
    main()
