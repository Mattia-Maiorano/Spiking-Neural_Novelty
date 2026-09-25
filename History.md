# Project Evolution History: SPWM (Spiking Predictive World Model)

## Baseline State: SPWM-v1 (Surrogate BPTT Architecture)

### 1. Architecture Overview
- **Neuron Model**: Single-compartment Leaky Integrate-and-Fire (`LIFCell` in `spwm/models/neurons.py`) with membrane potential $V_t = \beta V_{t-1} + I_t$ and surrogate Heaviside thresholding (Atan, FastSigmoid, Sigmoid).
- **Timescale Organization**: Dual-timescale spiking memory (`MultiTimescaleMemory` in `spwm/models/memory.py`) splitting latent populations into 2 pools (fast $\beta \approx 0.8$, slow $\beta \approx 0.98$).
- **Latent Dynamics & World Model**:
  - `SpikingLatentDynamics` (`spwm/models/latent_dynamics.py`) receiving sensory input from `EventEncoder` and recurrent feedback $z_{t-1}$.
  - Fuses spike outputs and membrane potentials into latent state $z_t$.
  - `LatentPredictor` predicting future latent states $\hat{z}_{t+1}$.
  - `PhysicalDecoder` probing kinematic quantities (positions/velocities).
- **Learning & Optimization**:
  - Global Backpropagation Through Time (BPTT) through unrolled time sequences $T$.
  - Requires computational graph retention scaling with sequence length $\mathcal{O}(T)$ GPU/RAM memory.
  - Loss function (`SPWMLoss` in `spwm/learning/losses.py`) combining prediction error, multi-step rollout error, variance regularization, and spike sparsity penalty.

### 2. Known Limitations of SPWM-v1
1. **Memory Complexity**: $\mathcal{O}(T)$ memory footprint during training makes long or infinite streaming sequences impossible.
2. **Biological Implausibility**: Uses global backpropagation through time and non-local gradient transport.
3. **Synaptic Dynamics**: Single compartment neurons lack top-down modulatory dendritic compartments for local e-prop eligibility assignment.
4. **Weight Constraints**: Weights violate Dale's Law (arbitrary mixed excitatory/inhibitory signs).
5. **No Sleep Consolidation**: Continual online updates suffer from high-frequency noise accumulation without low-rank invariant consolidation.

---

## Upgrade State: SPWM-v2 (Non-BPTT Continuous Predictive World Model)

### 1. Architectural Blueprint & Components
- **Two-Compartment Dendritic Neurons**: Decoupled soma and modulatory dendritic potential.
- **Three-Tier Timescale Hierarchy**: 25% Fast, 50% Mid, 25% Persistent.
- **Predictive Coding Core**: Error routing $\epsilon_t = x_t - W_{pred} z_{t-1}$ with noisy mirror alignment $B = W^T + \xi$.
- **Dale's Principle & Lateral Inhibition**: Excitatory/inhibitory partitioning and Soft-WTA suppression.
- **Offline SVD Sleep Consolidation**: Periodic low-rank pruning on plasticity accumulation buffer.

---

## Release SPWM-v3: Radical Pruning & ALIF Forward-Only Architecture

### Motivazioni del Cambio di Paradigma (Post-Mortem v2)
- Rilevata divergenza caotica della loss dovuta al disallineamento geometrico causato dal consolidamento SVD offline ("fase sonno").
- Eliminato il fallimento dell'equilibrio E/I causato dalla coesistenza rigida di matrici di Dale, Soft-WTA e feedback rumoroso ($B = W^T + \xi$).
- Superato lo stallo delle metriche cinematiche e risolto il bug di out-of-bounds per orizzonti lunghi.

### Modifiche Architetturali Chiave
- **Pruning Totale:** Rimossi compartimenti dendritici, SVD offline, vincoli di Dale e rumore sinaptico.
- **ALIF Core:** Introdotto neurone Leaky Integrate-and-Fire con soglia dinamica auto-regolante ($A_t$) ed eterogeneità controllata su $\beta_{\text{adapt}}$ (0.90 / 0.985).
- **Stabilizzazione Spike Rate:** L'omeostasi intrinseca della soglia ALIF sostituisce i vincoli competitivi manuali.
- **e-prop Deterministico:** Tracce di eligibilità duali ($V$ ed $A$) accoppiate a feedback di errore privo di rumore, preservando l'ingombro di memoria $\mathcal{O}(1)$.
- **Estensione Orizzonte:** Dataset e dinamica portati a $T=150$ con validazione di rollout fino a $H=50$.
