"""
Synthetic 2D Dynamical World for Event-Driven Predictive Modeling.
Provides ground-truth kinematics (position, velocity, acceleration) without temporal leakage.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class DynamicalState:
    """Ground-truth kinematic state at a single timestep."""
    positions: np.ndarray  # [N_objects, 2] in [-1.0, 1.0]
    velocities: np.ndarray  # [N_objects, 2]
    accelerations: np.ndarray  # [N_objects, 2]
    radii: np.ndarray  # [N_objects]

    def to_flat_vector(self) -> np.ndarray:
        """Returns flattened [positions, velocities] vector for probing/evaluation."""
        return np.concatenate([self.positions.flatten(), self.velocities.flatten()], axis=0)


@dataclass
class Trajectory:
    """Full kinematic trajectory across time."""
    trajectory_id: int
    states: List[DynamicalState]
    times: np.ndarray  # [T]
    num_objects: int
    box_size: float = 1.0

    @property
    def positions(self) -> np.ndarray:
        """Positions tensor [T, N_objects, 2]."""
        return np.stack([s.positions for s in self.states], axis=0)

    @property
    def velocities(self) -> np.ndarray:
        """Velocities tensor [T, N_objects, 2]."""
        return np.stack([s.velocities for s in self.states], axis=0)

    @property
    def flat_kinematics(self) -> np.ndarray:
        """Flattened kinematics [T, 4 * N_objects] (x, y, vx, vy)."""
        return np.stack([s.to_flat_vector() for s in self.states], axis=0)


class MovingObjectsWorld:
    """
    2D continuous dynamical environment simulating moving circular objects.
    Supports boundary bouncing, damping, acceleration variations, and controlled noise.
    """

    def __init__(
        self,
        box_size: float = 1.0,
        default_radius: float = 0.12,
        dt: float = 0.05,
        noise_std: float = 0.01,
        damping: float = 0.0,
    ) -> None:
        self.box_size = box_size
        self.default_radius = default_radius
        self.dt = dt
        self.noise_std = noise_std
        self.damping = damping

    def generate_trajectory(
        self,
        trajectory_id: int,
        length: int = 50,
        num_objects: int = 1,
        velocity_range: Tuple[float, float] = (-1.0, 1.0),
        acceleration_std: float = 0.1,
        seed: Optional[int] = None,
    ) -> Trajectory:
        """
        Generates a continuous trajectory deterministically from a trajectory_id/seed.
        Ensures exact reproducibility across runs.
        """
        rng = np.random.default_rng(seed if seed is not None else trajectory_id)

        # Initialize positions within safe interior bounds
        margin = self.default_radius + 0.05
        positions = rng.uniform(
            -self.box_size + margin,
            self.box_size - margin,
            size=(num_objects, 2),
        )

        # Initialize velocities within specified range
        low_v, high_v = velocity_range
        velocities = rng.uniform(low_v, high_v, size=(num_objects, 2))
        # Ensure minimum velocity to avoid static objects
        min_speed = 0.2 * (abs(high_v) if abs(high_v) > 0.1 else 0.5)
        for i in range(num_objects):
            speed = np.linalg.norm(velocities[i])
            if speed < min_speed:
                direction = rng.normal(size=2)
                direction /= (np.linalg.norm(direction) + 1e-6)
                velocities[i] = direction * min_speed

        accelerations = rng.normal(0, acceleration_std, size=(num_objects, 2))
        radii = np.full(num_objects, self.default_radius)

        states: List[DynamicalState] = []
        times = np.arange(length) * self.dt

        curr_pos = positions.copy()
        curr_vel = velocities.copy()
        curr_acc = accelerations.copy()

        for t in range(length):
            states.append(
                DynamicalState(
                    positions=curr_pos.copy(),
                    velocities=curr_vel.copy(),
                    accelerations=curr_acc.copy(),
                    radii=radii.copy(),
                )
            )

            # Update kinematics
            # Slow acceleration variation (Ornstein-Uhlenbeck style perturbation)
            acc_perturbation = rng.normal(0, acceleration_std * 0.1, size=(num_objects, 2))
            curr_acc = 0.95 * curr_acc + acc_perturbation

            # Euler-Maruyama integration
            noise = rng.normal(0, self.noise_std, size=(num_objects, 2))
            curr_pos = curr_pos + curr_vel * self.dt + 0.5 * curr_acc * (self.dt ** 2)
            curr_vel = (1.0 - self.damping * self.dt) * curr_vel + curr_acc * self.dt + noise

            # Boundary reflection (elastic bounce)
            for i in range(num_objects):
                r = radii[i]
                for dim in range(2):
                    if curr_pos[i, dim] + r > self.box_size:
                        curr_pos[i, dim] = self.box_size - r
                        curr_vel[i, dim] = -abs(curr_vel[i, dim])
                    elif curr_pos[i, dim] - r < -self.box_size:
                        curr_pos[i, dim] = -self.box_size + r
                        curr_vel[i, dim] = abs(curr_vel[i, dim])

        return Trajectory(
            trajectory_id=trajectory_id,
            states=states,
            times=times,
            num_objects=num_objects,
            box_size=self.box_size,
        )


def partition_trajectories(
    total_trajectories: int = 1000,
    train_split: float = 0.8,
    val_split: float = 0.1,
) -> Dict[str, Tuple[int, int]]:
    """
    Partitions trajectory IDs strictly by trajectory index to eliminate temporal leakage.
    Example: 1000 trajectories -> train [0, 800), val [800, 900), test [900, 1000).
    """
    n_train = int(total_trajectories * train_split)
    n_val = int(total_trajectories * val_split)
    return {
        "train": (0, n_train),
        "val": (n_train, n_train + n_val),
        "test": (n_train + n_val, total_trajectories),
    }
