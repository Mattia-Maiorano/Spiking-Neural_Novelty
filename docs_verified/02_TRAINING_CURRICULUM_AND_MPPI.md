# 02 — Training Curriculum, Loss Formulations & MPPI Control

---

## 1. Phase 0: Grounded State Estimator (`train_representation`)

Implemented in [RepresentationPhaseStrategy](file:///Users/Mattia/Desktop/Studies/Temp/spwm/training/phases/representation.py#L15-L123) (`123` lines) using config [results/spwm_v0_12_4_train_representation/config.yaml](file:///Users/Mattia/Desktop/Studies/Temp/results/spwm_v0_12_4_train_representation/config.yaml):

* **Gradient Isolation (`representation.py:21-31,47-51`):** `predictor` and `critic` are frozen (`requires_grad=False`); only `encoder`, `dynamics`, `sensory_predictor`, and `physical_decoder` train. `dynamics.action_proj` is zero-initialized ([dynamics.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/dynamics.py#L84)).
* **Optimizer & Cosine Schedule (`representation.py:34-45,100-103`):** Adam across 5 parameter groups:
  * `encoder`, `dynamics` (non-`beta_`), `sensory_predictor`: `lr = 1e-4` (`TrainingConfig.learning_rate`), `weight_decay = 5e-4`.
  * `dynamics` (`beta_*` leak parameters): `lr = 0.5 * 1e-4 = 5e-5`, `weight_decay = 0.0`.
  * `physical_decoder`: `probe_lr = 2e-4`, `probe_weight_decay = 1e-4`.
  * Cosine annealing scales all parameter group LRs from $1.0\times \to 0.20\times$ across `epochs = 160` (`grad_clip_norm = 0.8`).
* **Phase 0 Total Loss ([losses.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/training/losses.py#L124-L200)):**
  $$\mathcal{L}_{\text{Phase 0}} = \lambda_{\text{kin}}\mathcal{L}_{\text{kinematic}} + \lambda_{\text{deriv}}\mathcal{L}_{\text{deriv}} + \lambda_{\text{smooth}}\mathcal{L}_{\text{smooth}} + \lambda_{\text{kin\_sm}}\mathcal{L}_{\text{kin\_smooth}} + \lambda_{\text{spk}}\mathcal{L}_{\text{spike\_bal}} + \lambda_{\text{sens}}\mathcal{L}_{\text{sensory}}$$
  with defaults (`LossConfig`, `config.py:303-315`): $\lambda_{\text{kin}}=5.0$, $\lambda_{\text{deriv}}=1.0$, $\lambda_{\text{smooth}}=0.1$, $\lambda_{\text{kin\_sm}}=0.5$, $\lambda_{\text{spk}}=0.25$, $\lambda_{\text{sens}}=1.0$:
  1. `l_kinematic`: $\operatorname{MSE}_{\text{sum}}(\hat{\mathbf{q}}^*, \mathbf{q}^*) + \operatorname{MSE}_{\text{sum}}(\hat{\mathbf{p}}^*, \mathbf{p}^*)/\sigma_v^2 + 40.0\big(\operatorname{MSE}_{\text{sum}}(\mathbf{p}_{\text{ee}}, \mathbf{p}_{\text{ee}}^*) + 0.5\,\operatorname{MSE}_F(\mathbf{R}_{\text{ee}}, \mathbf{R}_{\text{ee}}^*)\big)$ via [DifferentiableForwardKinematics](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/kinematics.py#L52).
  2. `l_derivative`: $\operatorname{MSE}_{\text{sum}}(\mathbf{q}_{t+1}-\mathbf{q}_t, \Delta t\,\mathbf{p}_t[:, :q_{\text{dim}}])$.
  3. `l_smoothness`: $\operatorname{SmoothL1}_{\beta=0.1,\text{sum}}(\mathbf{p}_{t+1}, \mathbf{p}_t)$.
  4. `l_kinematic_smooth` (`losses.py:95-121,166-168`): Calls `_decoded_kinematic_temporal_loss(..., reduce_joint_sum=True)` summing across joints: finite-difference velocity loss + `0.25 * vel_mid` consistency + `0.5 * ddq_loss`, where `ddq_pred = (pos_pred[:, 2:] - 2*pos_pred[:, 1:-1] + pos_pred[:, :-2]) / dt` (dividing by `dt`, not `dt^2`).
  5. `l_spike_balance` (`losses.py:171-180`): $\sum_i (\bar{s}_i - 0.12)^2 + 4.0\sum_i \operatorname{ReLU}(0.02 - \bar{s}_i)^2 + 2.0\sum_i \operatorname{ReLU}(\bar{s}_i - 0.45)^2$ (`target_spike_rate = 0.12`).
  6. `l_sensory`: $\operatorname{MSE}_{\text{sum}}(\boldsymbol{\varepsilon}_t, \mathbf{0})$.
* **Scoped Checkpoint Handoff ([checkpoints.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/training/checkpoints.py#L35-L58)):** `load_representation_checkpoint` loads `encoder`, `sensory_predictor`, `dynamics`, and `physical_decoder` each with `strict=True`, leaving `predictor` and `critic` untouched.

---

## 2. Phase 1: Action-Conditioned Attractor $M_1$ & 4-Phase Curriculum (`train_m1`)

Implemented in [M1PhaseStrategy](file:///Users/Mattia/Desktop/Studies/Temp/spwm/training/phases/m1_nominal.py#L17-L130) (`130` lines) and orchestrated by [BaseTrainer](file:///Users/Mattia/Desktop/Studies/Temp/spwm/training/engine.py#L190-L316) using [results/spwm_v0_12_4_train_m1/config.yaml](file:///Users/Mattia/Desktop/Studies/Temp/results/spwm_v0_12_4_train_m1/config.yaml):

* **Frozen Representation & Optimizer (`m1_nominal.py:23-29`, `engine.py:236-239`):** Representation stays in `eval()` with `requires_grad=False`. Adam updates **both** `model.predictor` and `model.critic` at `lr = 1e-4`, `weight_decay = 5e-4`, dropping dynamically to `lr = 5e-5` once Latch Phase `4` engages (`engine.py:236`).
* **Phase 1 Loss ([losses.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/training/losses.py#L203-L298)):**
  $$\mathcal{L}_{\text{Phase 1}} = \mathcal{L}_{M_1,\text{latent}} + 1.0\,\mathcal{L}_{\text{kin\_M1}} + w_{\text{critic}}\,\mathcal{L}_{\text{critic}}$$
  * **Latent Rollout ($\mathcal{L}_{M_1,\text{latent}}$, `losses.py:228-241,265-267`):** Evaluates `.mean()` across `(B, H, D)` (dividing by `B * H * 32` for `q` and `B * H * 96` for `p`), with half-open contact-heft weighting where `h0, h_end` are the first/last `phase_flags == 1` indices:
    `step_weights[b, max(0, h0 - 5) : min(H, h_end + 10)] = 2.0` ($\mathbf{q}$) and `vel_step_weights[b, max(0, h0) : min(H, h_end + 12)] = 3.0` ($\mathbf{p}$), weighted by $\lambda_{\text{position}}=1.0, \lambda_{\text{velocity}}=1.5, \lambda_{\text{latent}}=1.0$.
  * **DAgger Mixed BPTT (`losses.py:246-263`, [replay_buffer.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/training/replay_buffer.py#L85-L109)):** Concatenates open-loop transitions with closed-loop transitions sampled from `ClosedLoopBuffer` (FIFO rolling window `max_eval_generations = 3`).
  * **Decoded Kinematic Loss ($\mathcal{L}_{\text{kin\_M1}}$, `losses.py:269-286`):** $\operatorname{MSE}(\hat{\mathbf{q}}^*, \mathbf{q}^*) + 0.5\,\operatorname{MSE}(\hat{\mathbf{p}}^*, \mathbf{p}^*) + 40.0(\mathcal{L}_{\text{ee\_pos}} + 0.5\,\mathcal{L}_{\text{ee\_rot}}) + 0.5\,\mathcal{L}_{\text{kin\_smooth}}$ (`reduce_joint_sum=False`).
  * **Monte Carlo Value Critic ($\mathcal{L}_{\text{critic}}$, `losses.py:54-92,264`):** Discounted return recursion $V^*_t = \|\mathbf{p}_{\text{ee}}-\mathbf{p}^*_{\text{ee}}\|_2 + (1 - \cos\theta_{\text{err}}) + 0.95\,V^*_{t+1}$, weighted by $w_{\text{critic}} = 0.2$ in Phases 1–2 and $1.0$ (`5.0 * 0.2`) in Phases 3–4.
* **4-Phase Irreversible Latch Curriculum ([noise.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/training/noise.py#L46-L154), [engine.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/training/engine.py#L194-L200,L234-L265)):**
  * `BaseTrainer.__init__` overrides `LatchCurriculumState`'s standalone class defaults (`0.01/0.008/0.4`) with `TrainingConfig` values (`phase1_loss_threshold=0.10`, `phase2_loss_threshold=0.095`, `target_alpha=0.25`, `noise_ramp_epochs=20`, `tier2_frequency=5`) and always injects `batch["curriculum_alpha"]` (bypassing `compute_mppi_noise_alpha` and `mppi_noise_warmup_epochs=10` in `M1PhaseStrategy:58`). Latch transitions depend strictly on `loss_m1_latent`:
    1. **Phase 1 (Warmup Nominal):** `w_t = 0.60`, `alpha = 0.0`, `dagger_ratio = 0.0`. Latches to Phase 2 when `loss_m1_latent < 0.10`.
    2. **Phase 2 (Sensory Fade-Out):** Linearly decays `w_t` from `0.60 -> 0.0` over `7` epochs (`alpha = 0.0`). Latches to Phase 3 when `w_t == 0.0` and `loss_m1_latent < 0.095`.
    3. **Phase 3 (AR(1) Noise Ramp):** `w_t = 0.0`, linearly ramps `alpha` from `0.0 -> 0.25` over `20` epochs via `sample_ar1_noise` ($\beta = 0.90, \sigma = 0.08$, shoulder `1.6x`, elbow `1.4x`). Latches to Phase 4 when `alpha >= 0.25`.
    4. **Phase 4 (DAgger + Fine-Tuning):** `lr = 5e-5`, `w_t = 0.0`, `alpha = 0.25`, ramps `dagger_ratio` `0.10 -> 0.50` over `2 * tier2_frequency = 10` epochs.

---

## 3. Open-Loop Soft-Fade vs. Closed-Loop Residual Handover

1. **Open-Loop Rollout Soft-Fade ([rollout.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/rollout.py#L208-L277)):** During `phase_flags == 1`, `rollout_latent` blends $\mathbf{z}_{t+1} = (1 - w_t)\mathbf{z}_{\text{pred}} + w_t \mathbf{z}_{\text{grounded}}$ where $w_t = \operatorname{clamp}(g_{\text{steps}} / 4, 0, 1) \cdot w_{\max}$ ($w_{\max} = 0.60$, reaching `0.15 -> 0.30 -> 0.45 -> 0.60` only while `act_val > activity_threshold` in `dynamics.py:192-195`). Upon entering `phase_flags == 2`, `fade_counter == 1` applies $0.50\,w_{\max} = 0.30$, then drops to `0.00`.
2. **Closed-Loop PIL Residual-Gated Handover ([benchmark_mppi_control.py](file:///Users/Mattia/Desktop/Studies/Temp/experiments/benchmark_mppi_control.py#L234-L256,L423-L445), [mppi_scorer.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/evaluation/mppi_scorer.py#L381-L394)):** During $\phi_t = 1$, causal `OnlineEventCameraStream` produces $\mathbf{z}_{\text{obs}}$ via `step_grounded`. Decoded SE(3) errors `dist_pred`, `dist_obs`, `rot_obs` against `target_final_mat` govern $w_t$:
   * If `dist_pred > tau` (`0.026 m` default; passed via `--handover-dist-threshold`, while `mppi_scorer.py:388` hardcodes `0.026`) **and** `dist_obs < 0.025 m` **and** `rot_obs < 0.049 rad`: $w_t = 0.85$.
   * Else if `dist_obs < dist_pred` and `t <= t_c + 2` (where $t_c = \texttt{p1\_indices[0]} = t_{\text{contact}} - 7$ is the $\phi_t=1$ sensory-window onset and $t_{\text{grasp}} = t_c + 7 = t_{\text{contact}}$): $w_t = 0.40 \cdot \operatorname{clip}((\text{dist\_pred} - 0.032)/0.04,\, 0,\, 1)$.
   * Otherwise: $w_t = 0.00$ (rejecting corrupted post-$t_c$ (post-attachment) event bursts where `dist_obs` is `0.16–0.22 m` while `M1` remains within `1.39–1.74 cm` median).

---

## 4. MPPI Control & Two-Tiered Validation

* **Latent MPPI Cost ([mppi_controller.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/evaluation/mppi_controller.py#L406-L551), [MPPIConfig](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/config.py#L341-L370)):** Evaluates `K=256`, `H=10`, `N_iter=3`, `temperature=0.20`, `nominal_update_gamma=0.60`, `max_action_delta=0.10`, `noise_sigma=0.08` (`1.6x` shoulder, `1.4x` elbow, `noise_beta=0.90`):
  $$\mathcal{C}_k = \sum_{h=1}^{H}\Big[w_{\text{pos}}\|\mathbf{p}_{\text{ee},h}-\mathbf{p}^*_h\|_2 + w_{\text{rot}}\big((1-\cos\theta_h) + 0.5\,\theta_h\big) + w_{\text{jerk}}\|\Delta\mathbf{u}_h\|_2^2 - w_{\text{prog}}\,\alpha_h\,(\Delta\mathbf{p}_h\cdot \hat{\mathbf{d}}_h) + w_{\text{damp}}(1-\alpha_h)\|\Delta\mathbf{p}_h\|_2\Big]$$
  $$+\; w_{\text{j\_acc}}\sum \|\Delta^2\hat{\mathbf{q}}_h\|_2^2 + w_{\text{j\_lim}}\sum \big(\operatorname{ReLU}(\hat{\mathbf{q}}_h - \mathbf{q}_{\max}) + \operatorname{ReLU}(\mathbf{q}_{\min} - \hat{\mathbf{q}}_h)\big)^2 + \gamma^H (w_{\text{critic}}\,w_{\text{pos}}\,V(\mathbf{z}_H))$$
  with `w_pos=80.0`, `w_rot=100.0`, `w_jerk=4.0`, `w_progress=50.0`, `w_damping=12.0`, `w_joint_accel=15.0`, `w_joint_limit=50.0`, `w_critic=0.0`, `gamma_discount=0.95`, and deadband factor $\alpha_h = \operatorname{clamp}((e_{\text{pos},h} - 0.03)/0.10, 0, 1)$.
  * **2-Stage SE(3) Lookahead (`benchmark_mppi_control.py:366-370`):** Horizon steps `t + tau <= t_c` target `target_grasp_mat`, while `t + tau > t_c` target `target_final_mat`.
  * **Grasp Success Criteria (`benchmark_mppi_control.py:497-513`, `mppi_scorer.py:21-61`):** An episode succeeds iff (i) end-effector position error stays `< 3.0 cm` across the final 3-step settling window, (ii) terminal $\text{SO}(3)$ orientation error `< 0.05 rad`, and (iii) terminal joint velocity $\|\dot{\mathbf{q}}(t_{\text{grasp}})\|_2 < \tau_{\text{vel}}$, where $\tau_{\text{vel}} = \max(1.5, 1.1\|\dot{\mathbf{q}}^*_{t_{\text{grasp}}}\|_2)\,\text{rad/s}$.
* **Two-Tiered Validation & EMA ([mppi_scorer.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/evaluation/mppi_scorer.py#L541-L699), [engine.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/training/engine.py#L444-L466,L649-L651)):**
  * Tier 1 open-loop MSE (`metrics.py:26-51`) is logged every epoch but decoupled from model selection.
  * Tier 2 `MPPIStructuralScorer` runs every `tier2_frequency = 5` epochs on `N=6` validation trajectories (`[3200..3205]`) with deterministic RNG seed `42000 + traj_id`, evaluates EMA weights (`ema_decay = 0.990`), and decays expert trajectories in `ClosedLoopBuffer` as `4 -> 3 -> 2 -> 1 -> 0` across Phase 4 evaluations (`mppi_scorer.py:546-549`). Emits `mppi_score`, `mppi_grasp_rot_err_rad`, and `mppi_final_rot`.
  * When `is_best=True`, `save_checkpoint` writes **both** `model.pt` and `best_model.pt` (`engine.py:649-651`). With `classic_patience: false`, training saves metrics/artifacts when `tier2_patience=5` triggers and continues to `epochs`.

---

## 5. Training Replication Commands

```bash
# 1. Phase 0: Representation Estimator (160 epochs)
.venv/bin/python experiments/train.py \
  --config results/spwm_v0_12_4_train_representation/config.yaml \
  --mode train_representation --epochs 160 --output-dir results/spwm_v0_12_4_train_representation

# 2. Phase 1: Action-Conditioned Attractor M1 (240 epochs, canonical Seed 42)
.venv/bin/python experiments/train.py \
  --config results/spwm_v0_12_4_train_m1/config.yaml \
  --mode train_m1 --epochs 240 --weighted-sampling \
  --representation-checkpoint results/spwm_v0_12_4_train_representation/model.pt \
  --output-dir results/spwm_v0_12_4_train_m1

# 3. Multi-Seed SPWM (Seeds 42, 101, 2024; Seed 2024 resumed to 260 epochs in results/multiseed/seed_2024_m1)
.venv/bin/python experiments/train_multiseed.py \
  --seeds 101 2024 --rep-epochs 160 --m1-epochs 240 --output-dir results/multiseed

# 4. Parameter-Matched ANN-Rate Ablation (per seed, 160 rep + 240 M1 epochs)
.venv/bin/python experiments/train_ann_ablation.py --seed 42 --output-dir results/ann_ablation

# 5. Continuous Recurrent Baselines (60 epochs, Seeds 42, 101, 2024)
.venv/bin/python experiments/train_baselines.py \
  --epochs 60 --total-trajectories 4000 --lr 3e-4 --seed 42 --output-dir results/baselines
```

---

## 6. Baselines & Online Observation Update Specifications

1. **Reactive Behavior Cloning (`ReactiveBCPolicy`, `322,210` params; [benchmark_classical_and_bc.py](file:///Users/Mattia/Desktop/Studies/Temp/experiments/benchmark_classical_and_bc.py#L44-L105,L295-L373)):**
   * **Architecture:** Dual-stream `EventEncoder` (`15,904` params, `4x32x32 -> 128`) + `GRUCell(134, 242)` (`274,428` params on `[enc_out, prev_action]`) + `MLP(242 -> 128 -> SiLU -> 6 -> Tanh)` (`31,878` params).
   * **Training:** `50` epochs on `[0, 3200)` / val `[3200, 3600)`, `batch_size=32`, `AdamW(lr=1e-3, weight_decay=1e-4)`, `CosineAnnealingLR(eta_min=1e-5)`, `clip_grad_norm_(1.0)`, scheduled sampling `ss_prob = min(0.5, 0.5 * ep / 50)`, and loss $\mathcal{L} = \operatorname{MSE}(\hat{u}, u) + 0.1\,\frac{1}{B(T-1)A}\sum_{t=1}^{T-1}\|\hat{u}_t - \hat{u}_{t-1}\|_2^2$.
2. **Privileged State-Oracle Jacobian-IK + PD (`mode == "fixed_pd"`; [benchmark_classical_and_bc.py](file:///Users/Mattia/Desktop/Studies/Temp/experiments/benchmark_classical_and_bc.py#L108-L138,L231-L242)):**
   * Reads `q_curr = stepper.q` directly without vision (`OnlineEventCameraStream` is never called) and uses oracle `t_c` for `payload_attached = (t >= t_c)`.
   * Computes 6D SE(3) error `[p_err, 0.35 * rot_err]` via `fk.forward_homogeneous(q_curr)`, autograd Jacobian $J \in \mathbb{R}^{6 \times 6}$, and Levenberg–Marquardt update $dq = J^\top(JJ^\top + \lambda^2 I_6)^{-1}\text{err}_{6\text{d}}$ ($\lambda = 0.05$, $\lambda^2 = 0.0025$), setting $q_{\text{des}} = \operatorname{clamp}(q_{\text{curr}} + 0.65\,dq, -3.14, 3.14)$ and $\dot{q}_{\text{des}} = \mathbf{0}$ with fixed gains `kp = 18.0`, `kd = 0.80` calibrated to `mean_calib_mass = 2.85 kg`.
3. **Online Observation Updates During $\phi_t = 1$ (`ContinuousRSSM` & `DeepGRUPredictor`; [benchmark_mppi_control.py](file:///Users/Mattia/Desktop/Studies/Temp/experiments/benchmark_mppi_control.py#L160-L193,L415-L422)):**
   * Unlike `SPWM` and `ANN-Rate` (which use `_compute_residual_handover_weight` to gate `(1 - w_t)*z_pred + w_t*z_obs`), `ContinuousRSSM` and `DeepGRUPredictor` perform an unconditional hard replacement **`curr_z = z_obs`** ($w_t = 1.0$) during `phase_flags[t] == 1` using `ev_online` and `u_pd_t`:
     * **`ContinuousRSSM`:** `h_next = gru_cell([s, u_pd_t], h)`, `e_step, _ = encoder(ev_online)`, `post_mu, _ = _get_stats(posterior_net([h_next, e_step]))`, `z_obs = [h_next, post_mu]`.
     * **`DeepGRUPredictor`:** `e_step, _ = encoder(ev_online)`, `h1_next = gru1([e_step, u_pd_t], h1)`, `h2_next = gru2(h1_next, h2)`, `z_obs = [h1_next, h2_next]`.
