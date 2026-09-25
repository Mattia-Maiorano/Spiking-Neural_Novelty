"""
Generates clean, runnable research notebooks for SPWM.
Using the public package APIs as specified in Requirement 42.
"""

from pathlib import Path
import json


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }


def make_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().split("\n")]
    }


def make_md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")]
    }


def generate_all_notebooks(target_dir="notebooks"):
    out_dir = Path(target_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 01_dataset_visualization.ipynb
    nb1 = make_notebook([
        make_md_cell("# 01: Dataset Visualization and Neuromorphic Event Camera\nExploration of synthetic 2D kinematics and differential event generation."),
        make_code_cell("""
import matplotlib.pyplot as plt
from spwm.data.synthetic_world import MovingObjectsWorld
from spwm.data.event_camera import EventCameraSimulator

# 1. Instantiate environment and simulator
world = MovingObjectsWorld(box_size=1.0, default_radius=0.12)
sim = EventCameraSimulator(height=32, width=32, event_threshold=0.08)

# 2. Generate a trajectory
traj = world.generate_trajectory(trajectory_id=42, length=30, num_objects=1)
events = sim.trajectory_to_events(traj)

print(f"Trajectory states: {len(traj.states)}")
print(f"Dense events tensor shape: {events.dense_events.shape}")
"""),
        make_code_cell("""
# 3. Visualize kinematics and event channels
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].plot(traj.positions[:, 0, 0], traj.positions[:, 0, 1], 'o-', label="Continuous Trajectory")
axes[0].set_xlim(-1, 1); axes[0].set_ylim(-1, 1); axes[0].set_title("2D Ground-Truth Kinematics")

# Positive events channel at t=10
axes[1].imshow(events.dense_events[10, 0].numpy(), cmap='Blues')
axes[1].set_title("Positive Polarity Events (+1)")

# Negative events channel at t=10
axes[2].imshow(events.dense_events[10, 1].numpy(), cmap='Reds')
axes[2].set_title("Negative Polarity Events (-1)")
plt.tight_layout()
plt.show()
""")
    ])
    with open(out_dir / "01_dataset_visualization.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb1, f, indent=1)

    # 02_spiking_dynamics.ipynb
    nb2 = make_notebook([
        make_md_cell("# 02: Spiking Dynamics & Multi-Timescale Memory\nAnalysis of membrane voltage dynamics, surrogate gradient backpropagation, and multi-timescale integration."),
        make_code_cell("""
import torch
import matplotlib.pyplot as plt
from spwm.models.memory import MultiTimescaleMemory
from spwm.models.surrogate import get_surrogate

# Check surrogate gradients
x = torch.linspace(-2.0, 2.0, 200, requires_grad=True)
atan_surr = get_surrogate("atan", alpha=2.0)
s = atan_surr(x)
loss = s.sum()
loss.backward()

plt.figure(figsize=(6, 3))
plt.plot(x.detach().numpy(), x.grad.numpy(), label="Atan Surrogate Derivative dS/dx")
plt.title("Surrogate Gradient Profile")
plt.xlabel("V - V_th"); plt.legend(); plt.show()
"""),
        make_code_cell("""
# Multi-timescale decay comparison
mem = MultiTimescaleMemory(timescale_dims=(1, 1), betas=(0.7, 0.98))
state = mem.init_state(1)
_, state = mem(torch.tensor([[5.0, 5.0]]), state)

fast_trace, slow_trace = [state.v_mems[0].item()], [state.v_mems[1].item()]
for _ in range(25):
    _, state = mem(torch.zeros(1, 2), state)
    fast_trace.append(state.v_mems[0].item())
    slow_trace.append(state.v_mems[1].item())

plt.figure(figsize=(8, 4))
plt.plot(fast_trace, label="Fast Memory (β=0.70)")
plt.plot(slow_trace, label="Slow Memory (β=0.98)")
plt.title("Decay Dynamics: Fast vs Slow Spiking Populations")
plt.xlabel("Timesteps"); plt.ylabel("Membrane Potential"); plt.legend(); plt.show()
""")
    ])
    with open(out_dir / "02_spiking_dynamics.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb2, f, indent=1)

    # 03_training_analysis.ipynb
    nb3 = make_notebook([
        make_md_cell("# 03: Training Analysis & Loss Decomposition\nInspects convergence dynamics, loss terms, and gradient diagnostics."),
        make_code_cell("""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load training logs
log_path = Path("../results/spwm_v1/training_log.csv")
if not log_path.exists():
    log_path = Path("results/spwm_v1/training_log.csv")

if log_path.exists():
    df = pd.read_csv(log_path)
    print(df.head())
    
    plt.figure(figsize=(10, 4))
    plt.plot(df['epoch'], df['total_loss'], label='Total Loss')
    plt.plot(df['epoch'], df['l_pred'], label='Prediction Loss')
    plt.plot(df['epoch'], df['val_total_loss'], label='Val Loss', linestyle='--')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('SPWM-v1 Loss Trajectory'); plt.legend()
    plt.show()
else:
    print(f"Log not found at {log_path}, run training first.")
""")
    ])
    with open(out_dir / "03_training_analysis.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb3, f, indent=1)

    # 04_rollout_analysis.ipynb
    nb4 = make_notebook([
        make_md_cell("# 04: Autonomous Rollout & Physical Decoding\nEvaluates closed-loop autonomous predictions over long horizons H."),
        make_code_cell("""
import json
import matplotlib.pyplot as plt
from pathlib import Path

metrics_path = Path("../results/spwm_v1/metrics.json")
if not metrics_path.exists():
    metrics_path = Path("results/spwm_v1/metrics.json")

if metrics_path.exists():
    with open(metrics_path) as f:
        data = json.load(f)
    test_rollout = data['test']['latent_mse_per_horizon']
    extrap_rollout = data['extrapolation']['latent_mse_per_horizon']
    
    horizons = sorted([int(k) for k in test_rollout.keys() if int(k) <= 10])
    plt.figure(figsize=(7, 4))
    plt.plot(horizons, [test_rollout[str(h)] for h in horizons], 'o-', label="In-Distribution Rollout")
    plt.plot(horizons, [extrap_rollout[str(h)] for h in horizons], 's--', label="Velocity Extrapolation Rollout")
    plt.xlabel("Horizon H"); plt.ylabel("Latent MSE"); plt.title("Autonomous Long-Horizon Rollout Error")
    plt.legend(); plt.show()
else:
    print("Metrics file not found, run evaluate.py first.")
""")
    ])
    with open(out_dir / "04_rollout_analysis.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb4, f, indent=1)

    print("All 4 research notebooks generated successfully.")


if __name__ == "__main__":
    generate_all_notebooks()
