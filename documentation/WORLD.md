# SPWM Simulation & World Architecture

This document details the simulation environment, robotic rig specifications, PhysX dynamics, procedural object generation, synthetic event sensor bridges, and telemetry export pipelines implemented across Unity (`sim/Assets/Scripts/`) and Python (`spwm/data/`).

---

## 1. Simulation Engine & Physics Setup

The simulation environment is built in **Unity 6 / PhysX 4** utilizing **Featherstone `ArticulationBody`** multi-body kinematic chains.

### Physics Timestep Configuration
- Environment Macro Timestep: $\Delta t = 0.05\,\text{s}$ ($20\,\text{Hz}$).
- PhysX Physics Sub-Steps: `physicsSubSteps = 5`.
- Simulation Fixed Delta Time:
  $$\Delta t_{\text{fixed}} = \frac{\Delta t}{\text{physicsSubSteps}} = \frac{0.05}{5} = 0.01\,\text{s} \quad (100\,\text{Hz})$$
- Solver Iterations:
  - Solver Iterations: `12`
  - Solver Velocity Iterations: `4`
- Gravity Vector: $\vec{g} = [0, -9.81, 0]^T\,\text{m/s}^2$.

---

## 2. Robotic Manipulator Rig (`ArticulatedRobotRig.cs`)

The rig models a 6-DOF industrial anthropomorphic robotic manipulator arm permanently mounted to an immovable high-mass root base ($20.0\,\text{kg}$) bolted to the floor.

### Kinematic Segment Table & ArticulationBody Parameters

| Index | Segment Name | Parent | Local Anchor $(x, y, z)$ [m] | Dimensions $(w, h, d)$ [m] | Mass [kg] | Joint Friction | Angular Damping | Motion Limits [deg] | Drive Stiffness $K_p$ | Drive Damping $K_d$ | Max Force [N] | Joint Axis Euler $(x, y, z)$ | Rotation Axis Vector |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| — | **Base_Root** | -1 (Ground) | $(0.0, 1.25, 0.0)$ | $(0.088, 0.05, 0.088)$ | 20.0 | 0.05 | 0.25 | Fixed | — | — | — | — | Fixed Base |
| 0 | **Shoulder_Pan** | 0 | $(0.0, 0.05, 0.0)$ | $(0.082, 0.05, 0.082)$ | 2.5 | 0.06 | 0.25 | $[-120^\circ, +120^\circ]$ | 300 | 75 | 600 | $(0, 0, 90)$ | $(0, 1, 0)$ |
| 1 | **Shoulder_Lift** | 1 | $(0.0, 0.05, 0.0)$ | $(0.072, 0.28, 0.072)$ | 2.2 | 0.06 | 0.25 | $[-90^\circ, +185^\circ]$ | 320 | 80 | 600 | $(0, 0, 0)$ | $(1, 0, 0)$ |
| 2 | **Elbow_Flex** | 2 | $(0.0, 0.28, 0.0)$ | $(0.060, 0.25, 0.060)$ | 1.6 | 0.06 | 0.25 | $[-155^\circ, +15^\circ]$ | 260 | 65 | 500 | $(0, 56, -16)$ | $(0.5375, -0.2756, -0.7969)$ |
| 3 | **Wrist_Pitch** | 3 | $(0.0, 0.25, 0.0)$ | $(0.050, 0.04, 0.050)$ | 0.6 | 0.05 | 0.25 | $[-95^\circ, +95^\circ]$ | 200 | 50 | 350 | $(0, 56, -16)$ | $(0.5375, -0.2756, -0.7969)$ |
| 4 | **Wrist_Yaw** | 4 | $(0.0, 0.04, 0.0)$ | $(0.046, 0.04, 0.046)$ | 0.5 | 0.05 | 0.25 | $[-75^\circ, +75^\circ]$ | 180 | 45 | 300 | $(0, 90, 0)$ | $(0, 0, -1)$ |
| 5 | **Wrist_Roll** | 5 | $(0.0, 0.04, 0.0)$ | $(0.044, 0.06, 0.044)$ | 0.5 | 0.04 | 0.25 | $[-180^\circ, +180^\circ]$ | 150 | 40 | 250 | $(0, 0, 90)$ | $(0, 1, 0)$ |

### Geometry & Anchors
- **Shoulder Elevation**: $1.35\,\text{m}$ (Shoulder Mount $1.25\,\text{m}$ + Base $0.05\,\text{m}$ + Turret $0.05\,\text{m}$).
- **Upper Arm Length**: $0.28\,\text{m}$.
- **Forearm Length**: $0.25\,\text{m}$.
- **Wrist Total Length**: $0.14\,\text{m}$ (Pitch $0.04\,\text{m}$ + Yaw $0.04\,\text{m}$ + Roll $0.06\,\text{m}$).
- **End-Effector Tip**: Positioned at the extreme distal boundary of Wrist Roll, extending along local $+Y$ by $0.06\,\text{m}$.
- Maximum Joint Angle Normalization: $\theta_{\text{max}} = \pi \approx 3.14159\,\text{rad}$.

---

## 3. Four-Stage Action-Conditioned Controller (`ActionConditionedController.cs`)

Outputs 6-dimensional continuous motor actions $u_t \in [-1, 1]^6$, mapped to drive targets via target angle scaling: $\theta_{\text{target}} = u_t \cdot \theta_{\text{max}}$.

### Controller PD Mechanics
$$\tau_{\text{motor}} = K_p (\theta_{\text{target}} - \theta_{\text{current}}) + K_d (\dot{\theta}_{\text{target}} - \dot{\theta}_{\text{current}})$$
- Default $K_p = 0.35$, $K_d = 0.08$.
- Maximum motor torque: $240\,\text{N}\cdot\text{m}$.

### Phasic Workflow Stages
Every trajectory consists of 105 discrete time steps ($T = 105$, total duration $5.25\,\text{s}$):

1. **Phase 0: Neutral Resting Start ($t \in [0, 6]$)**
   - Flags: `phase_flag = 0` (`BlindApproach`).
   - The arm is positioned in a neutral resting posture with closed-loop hold steps.
2. **Phase 1: Deliberate Approach ($t \in [7, t_{\text{contact}}]$)**
   - Flags: `phase_flag = 0` (`BlindApproach`).
   - Smooth quintic polynomial trajectory interpolation sweeping the end-effector tip toward the grasp contact point on the top surface of the procedurally spawned object.
   - Anti-memorization randomization: approach duration varies between 14 and 58 steps ($t_{\text{contact}} \sim \text{step } 21\text{--}65$), with random stiffness multipliers $K_p \in [0.65, 1.45]$ and damping multipliers $K_d \in [0.65, 1.45]$.
3. **Phase 1.5: Sensory Heft & Upward Lift ($t \in [t_{\text{contact}}, t_{\text{contact}} + 12]$)**
   - Flags: `phase_flag = 1` (`SensoryHeft`).
   - Meccanismo di transizione e aggancio (`FixedJoint`):
     - **Determinazione temporale del passaggio**: La transizione della logica di controllo da Phase 1 a Phase 1.5 avviene su base temporale deterministica a un timestep pianificato della traiettoria ($t \ge \text{approachEnd} = \text{setupHoldSteps} + \text{curApproachSteps}$, preceduto da una finestra di iniezione sensoriale pre-contatto).
     - **Attivazione del `FixedJoint`**: Non dipende da una soglia geometrica hardcoded (come $<3\text{ cm}$) nel controller. All'indice di step $t = \text{approachEnd}$, il controller invoca deterministicamente `_spawner.EnsureRigidGrasp(_rig)`: se non è ancora avvenuto contatto, forza istantaneamente il vincolo `FixedJoint` tra l'oggetto e l'end-effector (eseguendo inoltre uno snap del payload nel palmo/dita). In parallelo, a livello fisico in Unity, `PayloadGraspConstraint.OnCollisionEnter` funge da trigger di collisione anticipata se l'end-effector tocca fisicamente il collider dell'oggetto prima di tale step.
   - Grounding boundary: Pre-contact sensory grounding window (7 steps) e post-contact grounding window (3 steps).
   - Execution: Executes an immediate, sharp upward lift of $\Delta y = +0.10\,\text{m}$ ($+10\,\text{cm}$) combined with an angular wrist twist $\Delta \theta = [0.08, -0.05, -0.20]\,\text{rad}$.
   - Physical effect: The object's mass and inertia exert immediate gravitational torque on the arm's joints, producing high-frequency event bursts.
4. **Phase 2: Spatial Transport & Reorientation ($t \in [t_{\text{contact}} + 13, 105]$)**
   - Flags: `phase_flag = 2` (`BlindTransport`).
   - Carries the payload along a curved 3D trajectory to a new target elevation coordinate while twisting wrist joints to reorient the object from flat to upright.

---

## 4. Procedural Object Generation (`ProceduralObjectSpawner.cs`, `unity_episode_meta.py`)

Objects are generated procedurally with deterministic pseudo-random seeds utilizing Knuth's subtractive generator (`DotNetRandom`).

### Payload Primitive Shapes
1. **Sphere** (`Shape = 0`): Radius $r \in [0.04, 0.065]\,\text{m}$.
2. **Elongated Cylinder** (`Shape = 1`): Radius $r \in [0.035, 0.05]\,\text{m}$, Length $L \in [0.12, 0.22]\,\text{m}$.
3. **Box** (`Shape = 2`): Extents $(w, h, d) \in [0.06, 0.12]\,\text{m}$.

### Payload Mass Distribution
- Overall Range: $m \in [1.20, 4.50]\,\text{kg}$.
- Mass Tiers:
  - **Light**: $[1.20, 2.30]\,\text{kg}$
  - **Normal**: $(2.30, 3.40]\,\text{kg}$
  - **Heavy**: $(3.40, 4.50]\,\text{kg}$

### Target Elevation Workspace Planes
Target workspace planes are situated in front of the robot's base:
- **Low Plane**: $y \approx 1.04\,\text{m}$ (tabletop level, $-0.31\,\text{m}$ below shoulder).
- **Mid Plane**: $y \approx 1.14\,\text{m}$ (mid manipulation workspace, $-0.21\,\text{m}$ below shoulder).
- **High Plane**: $y \approx 1.26\,\text{m}$ (upper chest reach plane, $-0.09\,\text{m}$ below shoulder).

---

## 5. Dual-Camera Neuromorphic Sensor Pipeline

The vision system utilizes two synchronized cameras feeding log-intensity temporal difference event computation.

### Camera Placements & Framing
1. **Global Static Camera (`globalCamera`)**:
   - Anthropomorphic eye-level vantage point ($y = 1.60\,\text{m}$, $25\,\text{cm}$ above shoulder mount, $22\,\text{cm}$ medial offset $-X$, $28\,\text{cm}$ posterior $-Z$).
   - Points forward-down toward workspace center: look target $(-0.12, 1.06, 0.44)\,\text{m}$.
2. **Egocentric Wrist Camera (`wristCamera`)**:
   - Rigidly mounted to the Wrist Roll link adjacent to the magnetic gripper tip.
   - Points along the end-effector grasping axis, capturing high-resolution micro-slip and local object displacement events.

### Temporal Log-Differencing Compute Shader (`EventCameraLogDiff.compute`)
Both cameras render to $32 \times 32$ RenderTextures (`RenderTextureFormat.ARGB32`).
At each timestep $\Delta t = 0.05\,\text{s}$:
1. ITU-R BT.709 Luminance Conversion:
   $$I(u, v; t) = 0.2126 \cdot R + 0.7152 \cdot G + 0.0722 \cdot B$$
2. Logarithmic Transformation:
   $$L(u, v; t) = \ln(\max(I(u, v; t), 0) + 10^{-3})$$
3. Temporal Log Difference:
   $$\Delta L(u, v; t) = L(u, v; t) - L(u, v; t - \Delta t)$$
4. Polarity Spike Emission with Threshold $\theta = 0.08$:
   $$E_{\text{ON}}(u, v; t) = \mathbb{I}(\Delta L > \theta)$$
   $$E_{\text{OFF}}(u, v; t) = \mathbb{I}(\Delta L < -\theta)$$
   If a spike occurs, reference log-intensity is updated: $L_{\text{ref}} \leftarrow L(u, v; t)$.
5. Output Concatenation: Dense 4-channel tensor $[4, 32, 32]$:
   - Channels 0–1: Global ON / OFF
   - Channels 2–3: Wrist ON / OFF

---

## 6. Telemetry Logging Contract (`RoboticTelemetryLogger.cs`)

Trajectories are exported into compressed JSON archives (`trajectory_XXXXX.json.gz`) containing:

### Top-Level Metadata
- `trajectory_id`: Integer sequence identifier.
- `archetype`: `"ManipulatorArm"`
- `payload_shape`: `"Sphere"` | `"ElongatedCylinder"` | `"Box"`
- `payload_mass`: Float in $[1.20, 4.50]\,\text{kg}$.
- `sequence_length`: $105$
- `joint_count`: $6$
- `action_dim`: $6$
- `sensor_channels`: $4$
- `sensor_height`: $32$
- `sensor_width`: $32$
- `dt`: $0.05$
- `event_threshold`: $0.08$

### Timestep Telemetry Record Schema (`steps[t]`)
- `step`: Timestep index $t \in [0, 104]$.
- `timestamp`: Float $t \cdot \Delta t$ seconds.
- `phase_flag`: Integer in $\{0, 1, 2\}$ ($0 = \text{BlindApproach}$, $1 = \text{SensoryHeft}$, $2 = \text{BlindTransport}$).
- `joint_positions`: Float array $[q_0^*, \dots, q_5^*]$ in radians.
- `joint_velocities`: Float array $[p_0^*, \dots, p_5^*]$ in rad/s.
- `actions`: Float array $[u_0, \dots, u_5]$ in $[-1, 1]^6$.
- `grayscale_frame`: Flattened $32 \times 32 = 1024$ float values (Global camera).
- `wrist_grayscale_frame`: Flattened $32 \times 32 = 1024$ float values (Wrist camera).
- `dense_events`: Flattened $4 \times 32 \times 32 = 4096$ binary float spikes (optional/regenerable via `EventCameraSimulator`).

