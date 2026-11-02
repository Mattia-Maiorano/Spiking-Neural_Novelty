# Repository Rules & Guidelines for Agents

## Mandatory Documentation Update (`./docs`) at Each Run Ending

Every agent operating in this repository MUST adhere to this rule:

### 1. Trigger
At the conclusion of **every completed run or milestone** (including training phases, pipeline executions, evaluations, simulation exports, or any completed experimental/architectural modification session):

### 2. Mandatory Update to `docs/History.md`
The agent MUST update (or append to) `docs/History.md` with the entry for the version and iteration just tested or modified.

#### Definition of Versions vs. Iterations
- **Iterations increment within the same version**: The "iteration" can and should increase also upon the same version (e.g., multiple successive iterations under `SPWM-v0.6.4` progressing through `Iteration 5`, `Iteration 6`, `Iteration 7`).
- **What constitutes a new iteration**: A new iteration is defined as the **full implementation of a feature, refactor, or fix** (or a completed training run, evaluation benchmark, or formal post-mortem).
- **Agent execution scope**: **Not each agent execution is a new iteration.** Tuning an implementation across prompts in the same chat session does **not** always account for a new iteration. Incremental prompt turns, minor parameter tweaks, and active debugging within the same chat should be consolidated into the single iteration representing that completed feature, refactor, or fix.

#### Entry Structure
Every entry in `docs/History.md` MUST strictly adhere to this 5-item structure:
- **version afflicted**: The version afflicted (e.g., `SPWM-v0.6.4`).
- **iteration on version**: The iteration or phase identifier (e.g., `Iteration 5 (Simulation Heft Smoothing & Smooth Stasis Gate)`).
- **changes made**: Concise summary of what code, architecture, parameters, or simulation scripts were modified.
- **new results (if any)**: Majorly numerical metrics (losses, evaluation errors, MAE, MSE, spike rates, horizon breakdowns). **MUST explicitly tell if no runs were made** (e.g., "No runs made - code refactoring / configuration only").
- **verdict on change made**: Explicit statement of whether the change has been **preserved**, **reverted**, or **discarded**.

> **Constraint on history format**: Keep `History.md` lean, structured, and strictly numerical. Do NOT introduce bloated narrative prose, discursive commentary, or redundant derivations.

### 3. Synchronization of Topic Documentation (`./docs/`)
If the run or change introduced modifications to contracts, models, training logic, or simulation environments, the corresponding documentation in `./docs/` MUST be kept synchronized:
- **`docs/ARCHITECTURE.md`**: For changes to neural networks, models, encoders, dynamics, decoders, or visual pipelines.
- **`docs/WORKSPACE_AND_CONTRACTS.md`**: For changes to config schemas, tensor contracts, data structures, or layer interfaces.
- **`docs/TRAINING_AND_LOSSES.md`**: For changes to loss formulations, optimization curricula, or training strategies.
- **`docs/WORLD.md`**: For changes to Unity physics, C# simulation scripts, action spaces, or robotic rigs.

