# SPWM Training Architecture & Loss Formulations

This document provides a comprehensive specification of the training system, optimization curricula, loss functions, noise injection mechanics, and evaluation loops implemented in `spwm/training/` and `spwm/evaluation/`.

---

## 1. Two-Phase Optimization Architecture

Training is organized into two distinct, isolated phases:
1. **Phase 0: Grounded Representation Learning (`train_representation`)**
   - Trains the visual perception and state estimation pathway: `EventEncoder`, `SpikingLatentDynamics`, `sensory_predictor`, and `PhysicalDecoder`.
   - Freezes `LatentPredictor (M1)` and `LatentCritic`.
2. **Phase 1: Action-Conditioned Attractor Dynamics (`train_m1`)**
   - Freezes the perception pipeline (`EventEncoder`, `sensory_predictor`, `SpikingLatentDynamics`, and `PhysicalDecoder`).
   - Trains `LatentPredictor (M1)` and `LatentCritic` using autonomous autoregressive rollouts and closed-loop DAgger replay buffers.

---

## 2. Loss Formulations & Mathematical Objectives

### 2.1 Representation Phase Objectives (`spwm/training/phases/representation.py`)

Given an event stream $E_{1:T}$, ground-truth articulated kinematics $k^*_{1:T} = [q^*_{1:T}, p^*_{1:T}]$, and motor actions $u_{1:T}$:
$$\mathcal{L}_{\text{representation}} = \lambda_{\text{kinematic}} \mathcal{L}_{\text{kinematic}} + \lambda_{\text{derivative}} \mathcal{L}_{\text{derivative}} + \lambda_{\text{smoothness}} \mathcal{L}_{\text{smoothness}} + \lambda_{\text{sensory}} \mathcal{L}_{\text{sensory}}$$

> [!NOTE] Dynamics Action Coupling in Phase 0 & Gradient Flow Verification
> In Phase 0, `PhysicalDecoder` chiede a $q_t$ di decodificare gli angoli dei giunti. Poiché $q_t$ riceve a ogni singolo step la risposta pronta dalla telecamera tramite $k_t^{\text{sensory}}$, e poiché `action_proj` ($W_u$) in `SpikingLatentDynamics` è inizializzato a zero (`nn.init.zeros_`), all'inizio del training il gradiente di decodifica cinematica fluisce prevalentemente dal ramo visivo ($\epsilon_t$ e $k_t^{\text{sensory}}$).
> 
> **Impatto teorico in closed-loop:** Se $W_u$ non evolvesse o rimanesse trascurabile durante la fase di rappresentazione, durante il controllo closed-loop (con azioni $u_t$ pianificate da MPPI) lo stato latente $z_t$ si aggiornerebbe quasi esclusivamente assimilando gli eventi visivi generati (soggetti al ritardo intrinseco della dinamica PD dei motori) senza informazione immediata sulla coppia motoria appena comandata.
> 
> **Verifica empirica delle norme sui pesi salvati:**
> - Inizializzazione (`spwm/models/dynamics.py`):
>   $$\|W_u\|_2 = 0.0, \quad \|W_{\text{in}}\|_2 \approx 1.34 \quad (\text{Kaiming uniform})$$
> - Modelli addestrati (`results/spwm_v0_10_4_train_representation/model.pt`, `spwm_v0_10_4_train_m1/model.pt`, `seed_42`):
>   $$\|W_u\|_2 = 1.3653, \quad \|W_{\text{in}}\|_2 = 1.3388, \quad \frac{\|W_u\|_2}{\|W_{\text{in}}\|_2} \approx 1.020$$
> - Variazioni sui multiseed (`seed_101`: ratio $\approx 0.521$; `seed_2024`: ratio $\approx 0.434$).
> 
> I gradienti retropropagati attraverso la dinamica ricorrente durante Phase 0 riescono quindi ad attivare e far convergere $W_u$ a un ordine di grandezza del tutto comparabile a $W_{\text{in}}$ nei run completi, consentendo la corretta iniezione dell'azione motoria $u_t$ nella corrente ALIF $I_t$.

#### 1. Kinematic Decoding Loss ($\mathcal{L}_{\text{kinematic}}$)
Decodes continuous joint positions $\hat{q}^*$ and joint velocities $\hat{p}^*$ from latent partitions $q_t \in \mathbb{R}^{32}$ and $p_t \in \mathbb{R}^{96}$:
$$\hat{q}^*_t = \text{pos\_head}(q_t), \quad \hat{p}^*_t = \text{vel\_head}(p_t)$$
$$\mathcal{L}_{\text{pos}} = \frac{1}{B \cdot T} \sum_{b=1}^B \sum_{t=1}^T \sum_{j=1}^J (\hat{q}^*_{b,t,j} - q^*_{b,t,j})^2$$
$$\mathcal{L}_{\text{vel}} = \frac{1}{\sigma_v^2 \cdot B \cdot T} \sum_{b=1}^B \sum_{t=1}^T \sum_{j=1}^J (\hat{p}^*_{b,t,j} - p^*_{b,t,j})^2$$
$$\mathcal{L}_{\text{kinematic}} = \mathcal{L}_{\text{pos}} + \mathcal{L}_{\text{vel}}$$
where $\sigma_v = 1.0$ (configurable via `loss.velocity_loss_scale`) and $J = 6$.

#### 2. Derivative Consistency Loss ($\mathcal{L}_{\text{derivative}}$)
Enforces Hamiltonian/symplectic phase-space factorization where coordinate displacement matches integrated momentum:
$$\Delta q_t = q_t - q_{t-1}$$
$$\mathcal{L}_{\text{derivative}} = \frac{1}{B \cdot (T-1)} \sum_{b=1}^B \sum_{t=2}^T \sum_{d=1}^{q\_dim} \left(\Delta q_{b,t,d} - \Delta t \cdot p_{b,t-1,d}\right)^2$$

#### 3. Membrane Smoothness Regularization ($\mathcal{L}_{\text{smoothness}}$)
Suppresses high-frequency chatter in ALIF momentum states using smooth Huber loss ($\beta = 0.1$):
$$\mathcal{L}_{\text{smoothness}} = \frac{1}{B \cdot (T-1)} \sum_{b=1}^B \sum_{t=2}^T \sum_{d=1}^{p\_dim} \text{Huber}_{\beta=0.1}\left(p_{b,t,d} - p_{b,t-1,d}\right)$$

#### 4. Predictive Coding Sensory Reconstruction Loss ($\mathcal{L}_{\text{sensory}}$)
Minimizes prediction error neurons $\epsilon_t = s_t - \hat{s}_t$:
$$\mathcal{L}_{\text{sensory}} = \frac{1}{B \cdot T} \sum_{b=1}^B \sum_{t=1}^T \sum_{d=1}^{\text{encoder\_dim}} (\epsilon_{b,t,d} - 0)^2$$

---

### 2.2 M1 Predictor & Latent Rollout Objectives (`spwm/training/phases/m1_nominal.py`)

#### Autoregressive Target Construction
The ground-truth trajectory is generated by running sensory state assimilation over the full sequence:
$$z^*_{1:T}, \_ = \text{forward\_representation}(E_{1:T}, u_{1:T})$$
A burn-in prefix $t \in [0, t_{\text{start}}]$ grounds the initial state $z_{\text{start}} = z^*_{t_{\text{start}}}$. The model then unrolls autoregressively over horizon $H$:
$$\hat{z}_{t+1} = M_1(\hat{z}_t, u_t)$$

#### Position and Velocity Latent Losses ($\mathcal{L}_{\text{pos}}, \mathcal{L}_{\text{vel}}$)
Losses on coordinate and momentum partitions are weighted temporally to prioritize the critical Sensory Heft contact/lift phase (phase flag 1):
$$w_t^{\text{pos}} = \begin{cases} 2.0 & \text{if } t \in [t_{\text{heft\_start}} - 5, t_{\text{heft\_end}} + 10] \\ 1.0 & \text{otherwise} \end{cases}$$
$$w_t^{\text{vel}} = \begin{cases} 3.0 & \text{if } t \in [t_{\text{heft\_start}}, t_{\text{heft\_end}} + 12] \\ 1.0 & \text{otherwise} \end{cases}$$
$$\mathcal{L}_{\text{pos}} = \frac{1}{B \cdot H \cdot q\_dim} \sum_{b=1}^B \sum_{t=1}^H w_{b,t}^{\text{pos}} \|\hat{q}_{b,t} - q^*_{b,t}\|_2^2$$
$$\mathcal{L}_{\text{vel}} = \frac{1}{B \cdot H \cdot p\_dim} \sum_{b=1}^B \sum_{t=1}^H w_{b,t}^{\text{vel}} \|\hat{p}_{b,t} - p^*_{b,t}\|_2^2$$
$$\mathcal{L}_{M1} = \lambda_{\text{position}} \mathcal{L}_{\text{pos}} + \lambda_{\text{velocity}} \mathcal{L}_{\text{vel}}$$

#### Monte Carlo Critic Value Loss ($\mathcal{L}_{\text{critic}}$) (`compute_critic_mc_loss`)
LatentCritic learns the expected discounted Cartesian and geodesic rotational tracking cost:
1. Decodes predicted and target joint angles $\hat{q}_t, q^*_t$.
2. Solves forward kinematics: $T_t^{\text{pred}} = \text{FK}(\hat{q}_t)$, $T_t^* = \text{FK}(q^*_t)$.
3. Computes per-step Cartesian error $e_t^{\text{pos}} = \|p_t^{\text{pred}} - p_t^*\|_2$.
4. Computes geodesic SO(3) rotation error:
   $$\text{trace} = \text{tr}\left((R_t^{\text{pred}})^T R_t^*\right), \quad \cos \theta_t = \text{clip}\left(\frac{\text{trace} - 1}{2}, -1, 1\right)$$
   $$e_t^{\text{rot}} = 1 - \cos \theta_t$$
5. Per-step cost: $c_t = e_t^{\text{pos}} + w_{\text{rot\_critic}} \cdot e_t^{\text{rot}}$ ($w_{\text{rot\_critic}} = 1.0$).
6. Backward Bellman recursion:
   $$V_H^* = c_H, \quad V_t^* = c_t + \gamma V_{t+1}^* \quad (\gamma = 0.95)$$
7. Critic MSE Loss & Gradient Isolation:
   To ensure strict gradient isolation, predicted latent states are detached before passing to the critic network ($\hat{z}_{b,t}.\text{detach}()$), preventing critic value gradients from corrupting the M1 attractor backbone:
   $$\mathcal{L}_{\text{critic}} = \frac{1}{B \cdot H} \sum_{b=1}^B \sum_{t=1}^H \left(\text{Critic}(\hat{z}_{b,t}.\text{detach}()) - V_{b,t}^*\right)^2$$

#### Kinematic Supervision for M1 ($\mathcal{L}_{\text{kin\_M1}}$)
During the M1 training phase, predicted latents $(\hat{q}_t, \hat{p}_t)$ are passed through the frozen physical decoder heads ($\text{pos\_head}$ and $\text{vel\_head}$) to enforce joint-space kinematic consistency:
$$\hat{q}^*_t = \text{pos\_head}(\hat{q}_t), \quad \hat{p}^*_t = \text{vel\_head}(\hat{p}_t)$$
$$\mathcal{L}_{\text{kin\_M1}} = \frac{1}{B \cdot H \cdot J} \sum_{b,t,j} (\hat{q}^*_{b,t,j} - q^*_{b,t,j})^2 + \frac{0.5}{B \cdot H \cdot J} \sum_{b,t,j} (\hat{p}^*_{b,t,j} - p^*_{b,t,j})^2$$

#### Total M1 Loss
The latent attractor loss $\mathcal{L}_{M1,\text{latent}} = \lambda_{\text{position}} \mathcal{L}_{\text{pos}} + \lambda_{\text{velocity}} \mathcal{L}_{\text{vel}}$ is augmented with kinematic supervision, and critic loss is kept decoupled:
$$\mathcal{L}_{M1,\text{total}} = \mathcal{L}_{M1,\text{latent}} + 1.0 \cdot \mathcal{L}_{\text{kin\_M1}}$$
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{M1,\text{total}} + w_{\text{critic}} \mathcal{L}_{\text{critic}}$$
where $w_{\text{critic}} = \lambda_{\text{critic}} \times (5.0 \text{ if curriculum\_phase} \ge 3 \text{ else } 1.0)$.

---

## 3. Four-Phase Latch Curriculum (`spwm/training/noise.py`)

The M1 training phase employs an irreversible threshold latch state machine to bridge open-loop trajectory imitation and closed-loop control. The latch transition thresholds are evaluated **strictly against $\mathcal{L}_{M1,\text{latent}}$** (coordinate + momentum MSE), completely excluding $\mathcal{L}_{\text{critic}}$ and $\mathcal{L}_{\text{kin\_M1}}$:

1. **Curriculum Phase 1: Warmup Nominal**
   - Condition: 100% open-loop offline demonstrations.
   - Action noise level: $\alpha = 0.0$.
   - Active perception injection weight: $w_t = 0.60$.
   - Latch Transition: Triggers to Phase 2 when $\mathcal{L}_{M1,\text{latent}} < 0.01$.
2. **Curriculum Phase 2: Sensory Fade-Out**
   - Decays sensory grounding weight linearly from $0.60 \to 0.0$ over 7 epochs:
     $$w_t = 0.60 \times \left(1 - \frac{\text{epoch}}{\text{fade\_epochs}}\right)$$
   - Action noise: $\alpha = 0.0$.
   - Latch Transition: Triggers to Phase 3 when $w_t = 0.0$ and $\mathcal{L}_{M1,\text{latent}} < 0.008$.
3. **Curriculum Phase 3: Noise Ramp-Up**
   - Sensory weight: $w_t = 0.0$ (fully autonomous mental simulation).
   - Linearly ramps AR(1) Gaussian noise amplitude $\alpha$ from $0.0 \to 0.40$ over 10 epochs.
   - Action noise model:
     $$\eta_t \sim \mathcal{N}(0, \Sigma), \quad \epsilon_t = \beta \epsilon_{t-1} + \sqrt{1 - \beta^2} \eta_t \quad (\beta = 0.65)$$
     $$u_t^{\text{noisy}} = \text{clip}(u_t + \alpha \epsilon_t, -1, 1)$$
   - Latch Transition: Triggers to Phase 4 when $\alpha = 0.40$.
4. **Curriculum Phase 4: Fast DAgger Closed-Loop Integration**
   - Active parameters: $\alpha = 0.40$, $w_t = 0.0$.
   - Mixes offline training batches with online rollouts collected by Tier 2 MPPI.
   - DAgger sample ratio scales dynamically from $10\% \to 50\%$ over $2 \times \text{tier2\_frequency}$ epochs.
   - Runs until stopped by Tier 2 Early Stopping.

---

## 4. Two-Tier Validation & Early Stopping (`spwm/training/engine.py`)

Validation operates on a two-tier hierarchy:

### Tier 1 Validation (Every Epoch)
- Computes standard offline validation loss across held-out trajectories.
- Evaluates kinematic MAE:
  $$\text{MAE}_{\text{pos}} = \frac{1}{B \cdot T \cdot J} \sum |\hat{q}^* - q^*|, \quad \text{MAE}_{\text{vel}} = \frac{1}{B \cdot T \cdot J} \sum |\hat{p}^* - p^*|$$

### Tier 2 Validation: Closed-Loop MPPI Evaluation (Every `tier2_frequency` Epochs)
- Evaluates real closed-loop task execution using `LatentMPPIController` (`spwm/evaluation/mppi.py`).
- Standard receding horizon control is enabled by setting `chunk_size = 1` (re-planning at every control step).
- Candidate selection evaluated on Exponential Moving Average (EMA) weights ($\lambda_{\text{ema}} = 0.995$) or latest model weights.
- Executes online planning across procedural evaluation trajectories across Low, Mid, and High elevation target planes.

#### Definizione del Target di Traiettoria ($\mathbf{p}_{\text{target}}$) in MPPI
Il riferimento di posa $\mathbf{p}_{\text{target}}$ e orientamento fornito a `LatentMPPIController` è una **sequenza temporale sull'orizzonte di predizione** $\mathbf{p}_{\text{target},\tau} \in \mathbb{R}^{H \times 4 \times 4}$ per ciascuno step $\tau \in \{1, \dots, H\}$:
- **Approach Window ($t + \tau \le t_c$)**: Il target per lo step $\tau$ è la posa di presa/oggetto $\mathbf{p}_{\text{grasp}}$.
- **Transport Window ($t + \tau > t_c$)**: Il target per lo step $\tau$ è la posa finale di trasporto $\mathbf{p}_{\text{final}}$.
L'errore cartesiano e rotazionale di ciascuno step dell'orizzonte viene valutato direttamente contro `target_pose_seq[tau]`.

#### Meccanica e Funzione di Costo di `LatentMPPIController`
Durante il planning online in closed-loop, MPPI ottimizza le traiettorie su base geometrica e cinematica nello spazio cartesiano:
1. **Mental Simulation in Latent Space**: Esegue il rollout autoregressivo di $K$ traiettorie candidate per $H$ passi temporali usando esclusivamente il predittore latente $\mathcal{M}_1$:
   $$\mathbf{z}_{t+\tau} = \mathcal{M}_1(\mathbf{z}_{t+\tau-1}, \mathbf{u}_{t+\tau-1})$$
2. **Kinematic Decoding & FK**: A ogni step dell'orizzonte decodifica le configurazioni articolari $\hat{q} \in \mathbb{R}^6$ tramite il `PhysicalDecoder` ($\hat{k} = [\hat{q}, \hat{p}] = \mathcal{D}_{\text{phys}}(\mathbf{z})$) e calcola la posa cartesiana dell'end-effector (posizione 3D $\mathbf{p}_{\text{ee}} \in \mathbb{R}^3$ e matrice di rotazione $\mathbf{R}_{\text{ee}} \in \mathrm{SO}(3)$) tramite `DifferentiableForwardKinematics`.
3. **Composizione Esatta della Funzione di Costo**: La funzione di costo di ciascuna traiettoria candidata $k \in \{1, \dots, K\}$ è composta da:
   - **Positional Cost**: $w_{\text{pos}} \sum_{\tau=1}^H \|\mathbf{p}_{\text{ee},\tau} - \mathbf{p}_{\text{target},\tau}\|_2$ (distanza euclidea dal target sequenziale a step $\tau$).
   - **Rotational Cost**: $w_{\text{rot}} \sum_{\tau=1}^H (1 - \cos \theta_{\text{err},\tau})$ (errore geodetico $\mathrm{SO}(3)$ rispetto all'orientamento desiderato a step $\tau$).
   - **Action Smoothness / Jerk Penalty**: $w_{\text{jerk}} \sum_{\tau=1}^H \|\mathbf{u}_\tau - \mathbf{u}_{\tau-1}\|_2^2$ (penalizzazione sulla variazione brusca dei comandi di controllo).
   - **Directional Progress Reward**: $- w_{\text{progress}} \sum_{\tau=1}^H (\Delta \mathbf{p}_\tau \cdot \hat{\mathbf{d}}_{\text{target},\tau}) \cdot \text{dist\_factor}$ (premio direzionale verso il target a step $\tau$ modulato da deadband).
   - **Cartesian Velocity Damping**: $w_{\text{damping}} \sum_{\tau=1}^H \|\Delta \mathbf{p}_\tau\| \cdot (1 - \text{dist\_factor})$ (frenatura e smorzamento cinetico quando l'end-effector entra nella deadband vicina al target, sotto i $13\text{ cm}$).
   - **Terminal Value Bootstrapping (`LatentCritic`)**: Forzato a $w_{\text{critic}} = 0.0$ per tutto il controllo online fino a quando il Critic non sarà condizionato al goal.

- **Structural MPPI Score**:
  Il punteggio $S_{\text{MPPI}}$ per Early Stopping e selezione del miglior checkpoint incorpora direttamente l'errore di grasping cartesiano a $t = t_c$:
  $$S_{\text{MPPI}} = \text{grasp\_error\_cm} + \text{settling\_pos\_error\_cm} + 0.1 \cdot \text{final\_velocity} + 0.2 \cdot \text{geodesic\_rot\_error}$$

#### Modalità Autonoma vs Teacher-Forcing (`is_expert`) e Isolamento delle Metriche
Durante la valutazione di Tier 2, ciascuna traiettoria registrata nei record di prova (`FINAL_TRIALS`) opera secondo uno dei due regimi determinati dal flag `is_expert`:
- **`is_expert = False` (Natural MPPI)**: Il ciclo closed-loop interpola le azioni pianificate dal controller MPPI nello spazio latente $\mathcal{M}_1$. Tutte le metriche ufficiali di benchmark (`mppi_score`, `mppi_success_rate`, `mppi_mean_error_cm`, ecc.) e i criteri di early stopping/checkpointing sono calcolati **esclusivamente** sul sottoinsieme dei trial naturali (*Strict Metric Isolation*).
- **`is_expert = True` (Teacher-Forced / Expert Override)**: Il controller MPPI viene bypassato e ad ogni passo viene iniettata l'azione registrata dell'esperto $u_t^*$. Tali traiettorie servono a verificare la risposta cinematica oracolo e a popolare il `ClosedLoopBuffer` con rollout latenti privi di deviazioni catastrofiche nelle prime fasi di curriculum, senza alterare i punteggi statistici della policy autonoma.

- Early stopping is governed strictly by non-improving consecutive Tier 2 checks:
  $$\text{tier2\_checks\_without\_improvement} \ge \text{tier2\_patience}$$
  Minimum improvement delta required: $\Delta S < 0.001$ (`tier2_min_delta`).

---

## 5. Optimization & Hyperparameter Contracts

| Parameter | Configuration Key | Default Value |
|---|---|---|
| Optimizer | — | Adam (`torch.optim.Adam`) |
| Base Learning Rate | `training.learning_rate` | $1 \times 10^{-3}$ |
| Physical Decoder Probe LR | `training.probe_lr` | $5 \times 10^{-4}$ |
| Weight Decay | `training.weight_decay` | $1 \times 10^{-5}$ |
| Probe Weight Decay | `training.probe_weight_decay` | $1 \times 10^{-4}$ |
| Gradient Clip Norm | `training.grad_clip_norm` | $1.0$ |
| Batch Size | `training.batch_size` | $32$ |
| Sequence Length | `data.sequence_length` | $105$ |
| Phase Horizon | `training.phase_horizon` | $50$ |
| Kinematic Loss Weight | `loss.lambda_kinematic` | $1.0$ |
| Derivative Loss Weight | `loss.lambda_derivative` | $0.1$ |
| Smoothness Loss Weight | `loss.lambda_smoothness` | $0.05$ |
| Sensory Loss Weight | `loss.lambda_sensory` | $1.0$ |
| Latent Loss Weight | `loss.lambda_latent` | $1.0$ |
| Position Loss Weight | `loss.lambda_position` | $1.0$ |
| Velocity Loss Weight | `loss.lambda_velocity` | $0.5$ |
| Critic Loss Weight | `loss.lambda_critic` | $0.2$ |
| Tier 2 Check Frequency | `training.tier2_frequency` | $5$ epochs |
| Tier 2 Patience | `training.tier2_patience` | $4$ checks ($20$ epochs) |
| Model Weight EMA Decay | `training.ema_decay` | $0.995$ |

