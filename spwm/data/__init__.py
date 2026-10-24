"""
Data generation, 3D Unity articulated robotics trajectory parsing, and neuromorphic event camera simulation.
"""

from spwm.data.event_camera import EventBatch, EventCameraSimulator, RoboticTrajectory, Trajectory
from spwm.data.datasets import (
    EventWorldDataset,
    collate_event_batches,
    create_dataloaders,
    export_unity_trajectory_file,
    parse_unity_trajectory_file,
    partition_trajectories,
    simulate_articulated_trajectory,
)

__all__ = [
    "RoboticTrajectory",
    "Trajectory",
    "EventCameraSimulator",
    "EventBatch",
    "EventWorldDataset",
    "collate_event_batches",
    "create_dataloaders",
    "partition_trajectories",
    "simulate_articulated_trajectory",
    "export_unity_trajectory_file",
    "parse_unity_trajectory_file",
]
