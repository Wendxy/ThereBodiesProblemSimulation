import csv
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

G = 6.67430e-11


@dataclass
class SimulationConfig:
    masses: np.ndarray  # shape (3,)
    dt: float


def read_simulation_csv(path: str) -> Dict[str, np.ndarray]:
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    def col(name: str) -> np.ndarray:
        return np.array([float(r[name]) for r in rows], dtype=np.float64)

    data = {
        "time": col("time"),
        "p1": np.stack([col("body1_x"), col("body1_y"), col("body1_z")], axis=1),
        "v1": np.stack([col("body1_vx"), col("body1_vy"), col("body1_vz")], axis=1),
        "p2": np.stack([col("body2_x"), col("body2_y"), col("body2_z")], axis=1),
        "v2": np.stack([col("body2_vx"), col("body2_vy"), col("body2_vz")], axis=1),
        "p3": np.stack([col("body3_x"), col("body3_y"), col("body3_z")], axis=1),
        "v3": np.stack([col("body3_vx"), col("body3_vy"), col("body3_vz")], axis=1),
    }
    return data


def pack_state(p1, v1, p2, v2, p3, v3) -> np.ndarray:
    return np.concatenate([p1, v1, p2, v2, p3, v3], axis=-1)


def unpack_state(state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p1 = state[..., 0:3]
    v1 = state[..., 3:6]
    p2 = state[..., 6:9]
    v2 = state[..., 9:12]
    p3 = state[..., 12:15]
    v3 = state[..., 15:18]
    return p1, v1, p2, v2, p3, v3


def gravity_accelerations(positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
    # positions: (3, 3)
    acc = np.zeros_like(positions)
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            r = positions[j] - positions[i]
            dist = np.linalg.norm(r)
            if dist == 0:
                continue
            acc[i] += G * masses[j] * r / (dist ** 3)
    return acc


def compute_energy(positions: np.ndarray, velocities: np.ndarray, masses: np.ndarray) -> float:
    kinetic = 0.0
    for i in range(3):
        kinetic += 0.5 * masses[i] * float(np.dot(velocities[i], velocities[i]))

    potential = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            r = np.linalg.norm(positions[j] - positions[i])
            if r == 0:
                continue
            potential -= G * masses[i] * masses[j] / r
    return kinetic + potential


def stability_metrics(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    max_radius: float,
    min_separation: float,
) -> Dict[str, float]:
    # positions: (T, 3, 3), velocities: (T, 3, 3)
    com = np.sum(positions * masses[None, :, None], axis=1) / np.sum(masses)
    rel_pos = positions - com[:, None, :]
    max_dist = np.max(np.linalg.norm(rel_pos, axis=-1))

    min_sep = math.inf
    for t in range(positions.shape[0]):
        for i in range(3):
            for j in range(i + 1, 3):
                d = np.linalg.norm(positions[t, j] - positions[t, i])
                min_sep = min(min_sep, d)

    energies = np.array(
        [compute_energy(positions[t], velocities[t], masses) for t in range(positions.shape[0])]
    )
    energy_drift = float(np.max(np.abs(energies - energies[0])) / (abs(energies[0]) + 1e-12))

    return {
        "max_radius": float(max_dist),
        "min_separation": float(min_sep),
        "energy_drift": energy_drift,
        "stable": float(max_dist <= max_radius and min_sep >= min_separation),
    }
