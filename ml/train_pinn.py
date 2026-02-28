import argparse
import csv
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from utils import G, gravity_accelerations, pack_state, unpack_state, read_simulation_csv


class ThreeBodyDataset(Dataset):
    def __init__(self, samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray]], stats: Dict[str, np.ndarray]):
        self.samples = samples
        self.stats = stats

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, y, masses = self.samples[idx]
        x = (x - self.stats["x_mean"]) / self.stats["x_scale"]
        y = (y - self.stats["y_mean"]) / self.stats["y_scale"]
        masses = (masses - self.stats["m_mean"]) / self.stats["m_scale"]
        return torch.tensor(np.concatenate([x, masses], axis=0), dtype=torch.float32), torch.tensor(
            y, dtype=torch.float32
        )


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int, depth: int):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_manifest(manifest_path: str) -> List[Dict[str, str]]:
    with open(manifest_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def build_samples(data_dir: str, manifest: List[Dict[str, str]]) -> Tuple[List, float]:
    samples = []
    dts = set()
    for row in manifest:
        path = os.path.join(data_dir, row["path"])
        dts.add(float(row["dt"]))
        masses = np.array([float(row["m1"]), float(row["m2"]), float(row["m3"])], dtype=np.float64)
        data = read_simulation_csv(path)
        p1, v1, p2, v2, p3, v3 = data["p1"], data["v1"], data["p2"], data["v2"], data["p3"], data["v3"]
        states = pack_state(p1, v1, p2, v2, p3, v3)
        for t in range(states.shape[0] - 1):
            samples.append((states[t], states[t + 1], masses))
    if len(dts) != 1:
        raise ValueError(f"Expected one dt, found: {sorted(dts)}")
    return samples, float(list(dts)[0])


def compute_stats(samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> Dict[str, np.ndarray]:
    xs = np.stack([s[0] for s in samples], axis=0)
    ys = np.stack([s[1] for s in samples], axis=0)
    ms = np.stack([s[2] for s in samples], axis=0)
    stats = {
        "x_mean": xs.mean(axis=0),
        "y_mean": ys.mean(axis=0),
        "m_mean": ms.mean(axis=0),
        "x_scale": xs.std(axis=0) + 1e-9,
        "y_scale": ys.std(axis=0) + 1e-9,
        "m_scale": ms.std(axis=0) + 1e-9,
    }
    return stats


def physics_residual(
    pred_next: torch.Tensor,
    current: torch.Tensor,
    masses: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    # current/pred_next: (B, 18) in physical units
    p1, v1, p2, v2, p3, v3 = unpack_state(current)
    p1n, v1n, p2n, v2n, p3n, v3n = unpack_state(pred_next)

    pos = torch.stack([p1, p2, p3], dim=1)  # (B, 3, 3)
    vel = torch.stack([v1, v2, v3], dim=1)
    masses = masses.unsqueeze(-1)  # (B, 3, 1)

    acc = torch.zeros_like(pos)
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            r = pos[:, j, :] - pos[:, i, :]
            dist = torch.norm(r, dim=-1, keepdim=True) + 1e-12
            acc[:, i, :] += G * masses[:, j, 0:1] * r / (dist ** 3)

    vel_expected = vel + acc * dt
    pos_expected = pos + vel_expected * dt

    vel_pred = torch.stack([v1n, v2n, v3n], dim=1)
    pos_pred = torch.stack([p1n, p2n, p3n], dim=1)

    return torch.mean((vel_pred - vel_expected) ** 2) + torch.mean((pos_pred - pos_expected) ** 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a physics-informed next-step predictor.")
    parser.add_argument("--data-dir", default="ml/data")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-physics", type=float, default=1.0)
    parser.add_argument("--out", default="ml/model.pt")
    args = parser.parse_args()

    manifest_path = os.path.join(args.data_dir, "manifest.csv")
    manifest = load_manifest(manifest_path)
    samples, dt = build_samples(args.data_dir, manifest)
    stats = compute_stats(samples)

    dataset = ThreeBodyDataset(samples, stats)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(in_dim=21, out_dim=18, hidden=args.hidden, depth=args.depth).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            pred = model(batch_x)
            data_loss = mse(pred, batch_y)

            # Denormalize for physics residual
            x = batch_x[:, :18] * torch.tensor(stats["x_scale"], device=device) + torch.tensor(
                stats["x_mean"], device=device
            )
            masses = batch_x[:, 18:] * torch.tensor(stats["m_scale"], device=device) + torch.tensor(
                stats["m_mean"], device=device
            )
            y_pred = pred * torch.tensor(stats["y_scale"], device=device) + torch.tensor(
                stats["y_mean"], device=device
            )

            phys_loss = physics_residual(y_pred, x, masses, dt)
            loss = data_loss + args.lambda_physics * phys_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        avg = total_loss / max(1, len(loader))
        print(f"Epoch {epoch+1:03d} | loss={avg:.6e}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"model": model.state_dict(), "stats": stats, "dt": dt}, args.out)
    print(f"Saved model to {args.out}")


if __name__ == "__main__":
    main()
