# Architettura SPWM-v0.1.0 (Spiking Predictive World Model)

L'architettura **SPWM-v0.1.0** (Spiking Predictive World Model) si configura come un sistema neuromorfo ibrido a due scale temporali, progettato per fondere l'alta acuità cinematica locale con una stabilità asintotica su lunghi orizzonti attraverso un sistema di neuromodulazione del guadagno (*Gain Field System*).

Tutti i componenti del sistema sono rigorosamente tipizzati, governati da `ExperimentConfig` come *Single Source of Truth* (SSOT) e privi di fallbacks o conversioni implicite di iperparametri.

---

## 1. Rappresentazione Sensoriale e Spazio delle Fasi

Il modello elabora flussi differenziali di eventi visivi comprimendoli in uno spazio latente strutturato geometricamente, garantendo una complessità di memoria $\mathcal{O}(1)$ nel tempo.

* **Event Encoder & Bottleneck Spaziale:** I frame di eventi passano attraverso una rete convoluzionale che culmina in uno strato `SpatialSoftmax`. Questo modulo estrae analiticamente $K$ keypoint bidimensionali (`num_keypoints`) vincolati nell'arena normalizzata $[-1, 1]^2$, garantendo per costruzione l'invarianza topologica e impedendo alla rete di distorcere la geometria dell'input visivo in configurazioni non-euclidee.
* **Spazio Latente Duale ($Z \in \mathbb{R}^{\text{latent\_dim}}$):** Lo stato interno è partizionato in due sottospazi rigorosamente vincolati ($q_{\text{dim}} + p_{\text{dim}} = \text{latent\_dim}$, con $p_{\text{dim}} \pmod{q_{\text{dim}}} = 0$):
  * **Coordinate Geometriche ($q \in [-1, 1]^{q_{\text{dim}}}$):** Direttamente ancorate ai keypoint estratti dalla via dorsale sensoriale ($q_{\text{dim}} = 2 \cdot \text{num\_keypoints}$), rappresentano la configurazione spaziale istantanea.
  * **Momento Canonico / Contesto ($p \in \mathbb{R}^{p_{\text{dim}}}$):** Codifica l'inerzia, la velocità latente e la memoria a lungo termine della dinamica.
* **Decodificatore Cinematico Fisico (`PhysicalDecoder`):** Probe lineare diagnostico strutturato per disaccoppiare coordinate e velocità:
  * $\hat{pos}_t = W_{\text{pos}} q_t$
  * $\hat{vel}_t = W_{\text{vel}} r_t$, dove $r_t$ è lo stato filtrato tramite EMA sul momento canonico ($r_t = \alpha_{\text{ema}} r_{t-1} + (1-\alpha_{\text{ema}}) p_t$, governato dal parametro di smoothing $\alpha_{\text{ema}}$). Include un meccanismo di `warm_up_filter` per inizializzare coerentemente il filtro di velocità prima di rollout autonomi.

---

## 2. Fast Predictor (Acuità Cinematica e Dinamica Locale)

Il modulo veloce è responsabile dell'integrazione sensomotoria e della transizione dinamica nominale.

* **Memoria ALIF Multi-Scala:** Al cuore della dinamica latente risiede una popolazione neuromorfa divisa in pool gerarchici configurati via `MemoryConfig`:
  * Porzione reattiva: governata da decay di membrana $\beta_{\text{mem}}$ e costante rapida di adattamento per i transienti istantanei;
  * Porzione di contesto profondo: governata da costanti temporali lente per l'integrazione duratura.
* **Predictive Coding:** Il modello genera un'aspettativa sensoriale top-down $\hat{x}_t = W_{\text{pred}} z_{t-1}$. L'errore di predizione sensoriale $\epsilon_t = x_t - \hat{x}_t$ guida l'integrazione ricorrente nelle correnti somatiche.
* **Transizione Residuale Latente (`LatentPredictor`):** Un MLP a strati residui con attivazioni SiLU apprende la derivata locale della traiettoria $\Delta z_{\text{fast}} = f_\theta(z_t)$, proponendo il prossimo stato latente $z_{\text{fast\_next}} = z_t + \Delta z_{\text{fast}}$.

---

## 3. Slow Neuromodulatory Corrector (Gain Field System)

Il modulo lento sostituisce le euristiche di controllo rigido con un paradigma biologicamente plausibile di neuromodulazione di volume (Gain Mask), volto a sopprimere la deriva su traiettorie *Out-of-Distribution* (OOD).

* **Popolazione ALIF Lenta Integrativa (`SlowCorrector`):** Costituita da una popolazione di neuroni ALIF di dimensione $D_{\text{corr}}$ (`corrector.dim`) con costanti fisiologiche definite in `CorrectorConfig`:
  * $\beta_{\text{mem}}$ per l'integrazione rapida delle perturbazioni istantanee;
  * $\beta_{\text{adapt}}$ per l'accumulo persistente di evidenza di scostamento sui rollout;
  * Soglia di base $v_{\text{th0}}$ e accrescimento dinamico $\gamma$.
* **Maschera di Neuromodulazione Continua:** La popolazione emette una maschera differenziabile $\text{gain\_mask} \in [-1, 1]^{\text{latent\_dim}}$ tramite proiezione non-lineare $\tanh$.
* **Modulazione Moltiplicativa della Dinamica:** L'intervento correttivo agisce come modulatore di campo sulla transizione proposta dal predittore veloce:
  $$\Delta z_{\text{mod}} = \Delta z_{\text{fast}} \cdot (1 + \alpha_{\text{gain}} \cdot \text{gain\_mask})$$
  $$z_{t+1} = z_t + \Delta z_{\text{mod}}$$
  dove gli *hot spot* ($>0$) amplificano le dinamiche locali e i *cold spot* ($<0$) dissipano la deriva accumulata. L'autorità massima è scalata dal coefficiente $\alpha_{\text{gain}}$.
* **Vincolo Isometrico Fisico:** Le coordinate spaziali modulate $q_{\text{corr}}$ vengono vincolate nei limiti fisici euclidei dell'arena $[-1, 1]$, prevenendo deformazioni topologiche.

---

## 4. Motore Unificato di Rollout (`rollout_latent`)

Tutti i percorsi di inferenza, valutazione benchmark a multi-orizzonte e addestramento BPTT condividono la stessa funzione deterministica `rollout_latent`:
* Supporta rollout con o senza gradiente (`with_grad`).
* Gestisce in modo trasparente l'attivazione del correttore e il periodo di `burn_in_steps`.
* Integra il warm-up del filtro del probe cinematico (`initial_decoder_state`).
* Elimina ogni duplicazione logica tra test-set evaluation (`RolloutEvaluator`) e addestramento delle fasi.

---

## 5. Pipeline di Addestramento Modulare Disaccoppiata

Per garantire stabilità e impedire interferenze distruttive tra gradienti di scale temporali diverse, l'architettura adotta un curriculum a fasi disaccoppiate e una modalità congiunta per scenari unificati:

1. **`train_representation` (Rappresentazione e Probe Fisico):**
   * Predittore ricorrente ALIF congelato.
   * Addestra `SpatialSoftmax` encoder e `PhysicalDecoder` minimizzando $L_{\text{total}} = \lambda_{\text{encoder}} L_{\text{encoder}} + \lambda_{\text{decoder}} L_{\text{decoder}}$.
2. **`train_predictor` (Dinamica Nominale Multi-Step):**
   * Encoder e decoder congelati (caricati da checkpoint di rappresentazione validato).
   * Addestra `LatentPredictor` e dinamica ALIF su rollout autonomi a orizzonte $H$ (`multi_step_horizon`):
     $L_{\text{total}} = \lambda_{\text{pos}} L_{\text{pos}} + \lambda_{\text{vel}} L_{\text{vel}}$.
3. **`train_joint` (Addestramento Congiunto End-to-End):**
   * Ottimizzazione simultanea della rappresentazione sensoriale e della dinamica del predittore nominale su rollout multi-step, senza correttore attivo:
     $L_{\text{total}} = \lambda_{\text{encoder}} L_{\text{encoder}} + \lambda_{\text{decoder}} L_{\text{decoder}} + \lambda_{\text{pos}} L_{\text{pos}} + \lambda_{\text{vel}} L_{\text{vel}}$.
4. **`train_corrector` (Stabilizzazione Asintotica su Deriva OOD):**
   * Predittore, dinamica ALIF ed encoder rigorosamente congelati.
   * Addestra esclusivamente la popolazione `SlowCorrector` su rollout a orizzonte $H$ (`corrector_horizon`) soggetti a iniezione stocastica di perturbazioni cinematiche.
   * **Burn-in Iniziale:** Durante i primi passi ($t < \text{corrector\_burn\_in\_steps}$), il correttore assorbe il contesto sensoriale ma la modulazione del guadagno è in bypass.
   * **Funzione di Costo a Pettine (`CorrectorRolloutLoss`):** Ponderazione temporale ciclica gaussiana $w_t \propto \exp\left(-\frac{1}{2}\left(\frac{\min(t \pmod{P}, P - (t \pmod{P}))}{\sigma_{\text{comb}}}\right)^2\right)$ con picchi periodici centrati sul periodo $P$ (es. orizzonte di riferimento del rollout) e ampiezza $\sigma_{\text{comb}}$ (`comb_sigma`) per conferire elevata rilevanza alle finestre di transizione sopprimendo dolcemente l'intervallo mediano:
     $$L_{\text{total}} = L_{\text{pos}} + \beta_v L_{\text{vel}}$$

---

## 6. Filosofia Architetturale: Zero-Fallback e Validazione Rigorosa

L'intera codebase aderisce rigorosamente ai principi di *Zero Silent Fallbacks*:
* Nessun valore di default implicito ricavato tramite `getattr` o `.get()` permissivo nelle pipeline critiche del modello e delle loss.
* Validazione immediata dei file di configurazione (`_strict_dict_to_dataclass`): qualsiasi chiave non dichiarata o incongruente solleva immediatamente `ValueError`.
* Asserzioni topologiche rigide su dimensioni dei tensori ($q_{\text{dim}} + p_{\text{dim}} == \text{latent\_dim}$, $p_{\text{dim}} \pmod{q_{\text{dim}}} == 0$, $2 \cdot \text{num\_keypoints} == q_{\text{dim}}$).