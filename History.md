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

---

## Release SPWM-v3.1: Numerical Stabilization & Trace Normalization

### Diagnosi Empirica delle Anomalie v3 (Post-Mortem)
- **Oscillazioni e Divergenza della Val Loss:** L'estensione della sequenza a $T=150$ senza normalizzazione dell'accumulo locale ha incrementato la magnitudo del tensore $\Delta W$ di un fattore $5\times$. Ciò ha provocato continui scavalcamenti dei minimi locali durante l'ottimizzazione forward-only.
- **Collasso Predittivo (Rollout Piatto a ~0.65):** In risposta agli scossoni caotici dei pesi, il predittore latente ha collassato sulla predizione statica media dell'arena, rendendo l'errore di posizione insensibile all'orizzonte $H$ ($H=1 \approx H=50 \approx 0.65$).
- **Mantenimento Stabilità E/I:** Confermato il corretto funzionamento dell'adattamento della soglia (ALIF), con SpikeRate stabilizzato monotonicamente attorno a $0.30$.

### Interventi e Fix Implementati
- **Normalizzazione $1/T$ su e-prop:** Introdotto il fattore di scala inverso rispetto alla lunghezza temporale della sequenza ($1/T$) nell'accumulo delle tracce di eligibilità duali, rendendo l'ampiezza degli aggiornamenti invariante rispetto alla durata della traiettoria.
- **Ricalibrazione Tassi di Apprendimento:** Ridotti `learning_rate` e `local_lr` da $10^{-3}$ a $2 \cdot 10^{-4}$ per garantire un regime di variazione dei pesi compatibile con la costante biofisica di decadimento lento della soglia ALIF ($\beta_{\text{adapt}} = 0.985$).
- **Sblocco del Rollout Cinematico:** Ripristinata la dinamica predittiva attiva nello spazio latente, eliminando l'attrattore statico e consentendo la corretta propagazione temporale degli stati.

---

## Release SPWM-v3.6: Sblocco Feedback Cinematico via Feedback Alignment (Risoluzione Deadlock $W_{\text{rls}} \approx 0$)

### Diagnosi del Deadlock $W_{\text{rls}} \approx 0$
- L'analisi del training gate v3.5 ha evidenziato che la Probe Loss rimaneva inchiodata alla varianza dei dati (Val Total $\approx 0.300$, Probe Loss $\approx 0.298$, Pos/Vel Error fermi su $\approx 0.69 / 0.73$).
- Causa identificata: nel calcolo $L_{\text{kin}} = \epsilon_{\text{kin}} W_{\text{rls}}$, poiché all'inizializzazione $W_{\text{rls}} \approx 0$ e $z$ ha $R^2 \approx 0$, il feedback cinematico retroproiettato risultava $L_{\text{kin}} \approx 0$.
- Questo generava un deadlock circolare: i neuroni ALIF non ricevevano alcun segnale top-down, $W_{\text{rls}}$ non apprendeva e la dinamica latente rimaneva congelata rispetto alla cinematica.

### Interventi Implementati
- **Feedback Alignment Fisso ($B_{\text{kin\_feedback}}$):** Sostituita la retroproiezione dinamica dipendente da $W_{\text{rls}}$ con una matrice di feedback fissa, non addestrabile e registrata come buffer:
  $$B_{\text{kin\_feedback}} \sim \mathcal{N}\left(0, \frac{1}{\sqrt{4 \cdot \text{num\_objects}}}\right) \in \mathbb{R}^{(4 \cdot \text{num\_objects}) \times \text{latent\_dim}}$$
- **Sblocco del Segnale Top-Down:** Il segnale di feedback retroproiettato $L_{\text{kin}} = \epsilon_{\text{kin}} B_{\text{kin\_feedback}}$ inietta costantemente l'errore cinematico nelle tracce e-prop indipendentemente dallo stato dei pesi del decoder, abilitando l'allineamento progressivo della popolazione ALIF alle coordinate fisiche.
- **Mantenimento Online RLS:** Il modulo RLS continua ad aggiornarsi ad ogni timestep per tracciare e valutare le coordinate fisiche senza interferire negativamente con il loop di feedback delle tracce sinaptiche.

---

## Release SPWM-v3.7: Spatial Encoder Resurrection & Kinematic Unfreezing

### Diagnosi del Bottleneck Spaziale
- Il profiling layer-by-layer ha isolato la causa primaria del collasso di $R^2$ da 0.9741 (frame grezzi) a 0.0000:
  1. I layer LIF dell'`EventEncoder` erano in quiescenza totale (lif1: 0.43%, lif2: 0.0%, lif_out: 0.0%), inviando un vettore di soli zeri costanti (`enc_out = 0`) al core ALIF.
  2. I parametri dell'`EventEncoder` erano esclusi da tutti gli ottimizzatori e non ricevevano aggiornamenti (pesi congelati all'init casuale).
  3. Il core ALIF è rimasto isolato dall'input sensoriale durante l'intero addestramento.

### Modifiche Architetturali & Ottimizzazioni
- **Front-End Convoluzionale Continuo (`EventEncoder`):** Sostituita la catena di LIFCell con un encoder convoluzionale compatto continuo dotato di BatchNorm2d, LayerNorm e attivazioni SiLU, preservando l'integrità del flusso informativo spaziale ($R^2 > 0.85$).
- **Registrazione e Ottimizzazione dei Pesi (`encoder_optimizer`):**
  - I parametri dell'encoder sono registrati in un ottimizzatore AdamW dedicato (`lr=1e-3`).
  - Durante ogni batch di `train_epoch`, l'errore di predizione sensoriale top-down ($\epsilon_{\text{sensory}} = x_t - \hat{x}_t$) e l'allineamento cinematico supervisionato aggiornano i parametri dell'encoder, preservando la complessità temporale $\mathcal{O}(1)$.
- **PhysicalDecoder con Filtro Post-Sinaptico & Feedback Alignment:**
  - Mantenuto il decodificatore continuo con filtro passa-basso esponenziale e retroproiezione casuale fissa FA ($B_{\text{kin\_feedback}}$).

---

## Release SPWM-v3.7.1: Stabilization & Smooth Latent Regularization

### Diagnosi Empirica & Obiettivi
- Consolidamento dei risultati per ridurre il divario di generalizzazione tra train e validation e spingere le loss sotto 0.30.
- Risoluzione del drift delle statistiche running in `BatchNorm2d` su frame differenziali sparsi.

### Interventi Implementati
1. **Sostituzione BatchNorm -> GroupNorm/LayerNorm nell'Encoder:**
   - Sostituito `BatchNorm2d` con `GroupNorm(num_groups=min(4, c), num_channels=c)` nei blocchi convoluzionali 2D dell'`EventEncoder`.
   - Normalizzazione per singolo campione per eliminare il divario tra train e val causato dalle fluttuazioni delle statistiche di batch.
2. **Riduzione Learning Rate di Encoder e Probe:**
   - Encoder: ridotto a `2e-4`.
   - Probe: ridotto da `2e-3` a `5e-4`.
   - Garantisce un'evoluzione liscia e priva di strattoni per la rappresentazione latente.
3. **Weight Decay & Regolarizzazione:**
   - Applicato `weight_decay = 1e-4` sia sull'encoder che sul probe in AdamW per limitare la crescita smodata dei pesi convoluzionali e lineari.

---

## Release SPWM-v4.0: Topological Spatial-Softmax Frontend & Representation Stabilization

### Post-Mortem Critico di SPWM-v3.x
- **v3.0–v3.6 (Quiescent Extinction):** Identificato il fallimento rappresentazionale sistematico ($R^2 = 0.0000$ al Livello 1). L'encoder visivo spiking con soglia rigida $V_{\text{th}}=1.0$ e pesi casuali congelati estingueva l'attività (`lif2` e `lif_out` a 0.0% di firing), iniettando costantemente un tensore nullo nel core ALIF e bloccando il probe sul baricentro geometrico ($0.692$).
- **v3.7–v3.7.1 (Representation Drift):** L'introduzione di un encoder convoluzionale continuo ha ripristinato il flusso di informazione, abbattendo la Train Probe Loss a $0.14$. Tuttavia, l'uso di uno strato denso non vincolato `Linear(2048, 128)` ha causato un grave disallineamento geometrico: l'ottimizzazione latente ha deformato lo spazio in configurazioni non-euclidee, provocando overfitting del probe sul train set e divergenza in validazione (`Val Pos Err` risalito a $>0.76$).

### Innovazioni Architetturali SPWM-v4.0
- **Spatial Softmax Bottleneck (Analogia con la Via Dorsale):** Sostituito il layer fully-connected dell'encoder con una Spatial Softmax differenziabile (`SpatialSoftmax` in [encoder.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/encoder.py)). I canali convoluzionali vengono ridotti analiticamente a $K=16$ coordinate di baricentro visivo $(u_k, v_k) \in [-1, 1]^2$.
- **Invarianza Topologica e Parametri Minimi:** La geometria bidimensionale viene preservata per costruzione, eliminando centinaia di migliaia di parametri non vincolati e impedendo all'encoder di distorcere la topologia 2D.
- **Rigore O(1) e Forward-Only:** La trasformazione avviene frame-by-frame senza alcuna dipendenza temporale, preservando la complessità di memoria $\mathcal{O}(1)$ e l'assenza totale di BPTT nel modello predittivo ricorrente.

---

## Release SPWM-v4.2: Geometric Latent Space Rectification — Strada B

### Post-Mortem Critico di SPWM-v4.0 / v4.1

- **Plateau $\text{Val Pos Err} \approx 0.577$:** Nonostante la Spatial Softmax preservi la topologia 2D per costruzione, i $K=16$ keypoints $(u_k, v_k)$ vengono attratti da feature ad alto contrasto (bordi arena, artefatti di polarity) piuttosto che dall'oggetto in moto. Il frontend è corretto in struttura ma non in orientamento: la geometria euclidea esiste nel collo di bottiglia ma è disallineata rispetto alle coordinate fisiche reali.
- **Diagnosi (Strada B):** Lo spazio latente ALIF opera su un sistema di coordinate non conforme alla fisica euclidea. Prima di introdurre qualsiasi struttura simplettica o di ordine superiore, è necessario raddrizzare geometricamente lo spazio latente iniettando un gradiente direzionale che spinga le coordinate Spatial Softmax verso la posizione reale dell'oggetto.

### Innovazioni Architetturali SPWM-v4.2

#### Loss Ausiliaria di Coordinata (Strada B)

$$\mathcal{L}_{\text{total}}(t) = \mathcal{L}_{\text{pred}}(t) + \lambda_{\text{coord}} \cdot \mathcal{L}_{\text{coord}}(t) + \lambda_{\text{sparse}} \cdot \mathcal{L}_{\text{reg}}(t)$$

- **$\mathcal{L}_{\text{coord}}$:** MSE tra i keypoints raw della Spatial Softmax $(u_k, v_k) \in [-1,1]^2$ e la posizione GT dell'oggetto proiettata nel medesimo spazio immagine normalizzato. Ogni keypoint viene attraccato morbidamente al GT (soft coupling, senza assegnazione esplicita), coerente con un codice di popolazione distribuito della via dorsale biologica.
- **Coefficiente di aggancio $\lambda_{\text{coord}} = 0.15$:** Sufficiente a fornire il gradiente direzionale alle coordinate senza soffocare la dinamica predittiva libera (range operativo consigliato: $0.10 - 0.20$).

#### Flusso del Gradiente (Strada B)

Il gradiente di $\mathcal{L}_{\text{coord}}$ fluisce **direttamente** attraverso:

$$\text{GT}_{xy} \xrightarrow{\text{MSE}} \text{kp\_seq (SpatialSoftmax)} \xrightarrow{\nabla} \text{Conv2D} \xrightarrow{\nabla} \text{input\_proj}_{\text{ALIF}}$$

Nessun `detach` viene inserito tra la Spatial Softmax e il frontend Conv2D. Il percorso è ortogonale all'e-prop (non tocca i buffer $\Delta W$ né la dynamica ricorrente).

### Modifiche al Codice

- **[`encoder.py`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/encoder.py):** `forward()` espone i keypoints raw `[B, T, K*2]` tramite il flag `return_keypoints=True`; `step()` archivia `self.last_keypoints` per accesso sincrono durante il loop online.
- **[`losses.py`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/learning/losses.py):** `SPWMLoss` aggiunge `lambda_coord` (default `0.0` per retrocompatibilità) e il metodo `coordinate_loss()` che implementa il soft-coupling broadcast tra i $K$ keypoints e il GT.
- **[`trainer.py`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/learning/trainer.py):** Il blocco `encoder_optimizer` chiama `encoder(events, return_keypoints=True)` e aggiunge $\lambda_{\text{coord}} \cdot \mathcal{L}_{\text{coord}}$ a `loss_enc` con `torch.enable_grad()`. Nuovi scalari loggati per epoch: `l_coord` (train) e `val_l_coord` (val).
- **[`spwm_v4_2.yaml`](file:///Users/Mattia/Desktop/Studies/Temp/configs/experiments/spwm_v4_2.yaml):** Nuovo config con `lambda_coord: 0.15`.

### Criteri di Successo per v4.2

| Metrica | Baseline (v4.1) | Target v4.2 |
|---|---|---|
| Val Linear Pos Err | $0.577$ (plateau) | $< 0.35$ |
| Spike Rate | $\approx 0.12$ | $\in [0.11, 0.12]$ (finestra biologica, non schiacciata dalla forzatura) |
| Test TF MSE ($\mathcal{L}_{\text{pred}}$) | $0.120$ | $\leq 0.120$ (la pred loss non deve regredire) |


## Release SPWM-v4.3: Simplectic Phase Space (q, p) & EMA Buffer
- Split latent state into $(q, p)$ with $q\in\mathbb{R}^{32}$ and $p\in\mathbb{R}^{96}$.
- Added EMA buffer for spikes (`ema_spikes`) with decay $0.9$.
- Velocity update for $q$ via linear map `W_vel`.
- Exposed EMA spikes in `SPWMStepOutput`.
- Kept public API unchanged.

## Release SPWM-v4.3.1: Dual-Mode Phase Coordinates & Separated Probe Architecture
- **Dual-Mode Coordinates $q$:** In observation mode (with sensory frames), $q_t$ is clamped/assigned directly from the 16 SpatialSoftmax keypoints $(u_k, v_k) \in [-1, 1]^2$, completely eliminating open-loop accumulation drift. In autonomous prediction mode, $q_{t+1} = \text{clamp}(q_t + \tanh(W_{\text{vel}}(\text{ema\_spikes})), -1, 1)$.
- **Re-enabled $\mathcal{L}_{\text{coord}}$:** Re-activated auxiliary coordinate loss with $\lambda_{\text{coord}} = 0.15$ in `spwm_v4_3.yaml`.
- **Separated Physical Probe Architecture:** `PhysicalDecoder` now projects position directly from generalized coordinates $q_t$ (`pos_head(q_t)`) and velocity from the smoothed momentum state $p_t$ (`vel_head(smooth_p)`), maintaining input/output consistency in $[-1, 1]$.
- **Clean e-prop Dimensional Retroprojections:** Explicitly decomposed latent error feedback into $\mathcal{L}_q \in \mathbb{R}^{32}$ and $\mathcal{L}_p \in \mathbb{R}^{96}$, correctly back-projecting $\mathcal{L}_p$ through $W_{\text{fuse\_spk}}$ ($96 \to 128$) to somatic currents and updating $W_{\text{in}}$, $W_{\text{rec}}$, $W_q$, and $W_{\text{vel}}$.

## Release SPWM-v4.4 (Ablation: Scheduled Sampling & Exposure Bias)

### Obiettivi & Ipotesi Sperimentale
- Studio di ablazione volto a verificare se l'introduzione del **Curriculum Scheduled Sampling** sulla coordinata di fase $q$ ($p_{\text{auto}} \to 0.50$) potesse mitigare l'Exposure Bias e abbattere l'errore sui rollout autonomi lunghi ($H=50$).

### Risultati & Trade-off Emerso
1. **Flattening del Drift Relativo:** L'addestramento con campionamento misto ha effettivamente appiattito la pendenza del degrado temporale relativo (+16.6% di drift a $H=25$ rispetto a +58.5% della baseline sensoriale pura).
2. **Perdita di Isometria Cartesiana nello Spazio $q$:** L'integrazione di predizioni deviate durante il training ha contaminato la purezza topologica dello spazio $q$, distruggendo l'allineamento isometrico 1:1 con le coordinate reali per il probe lineare:
   - `Best Val Pos Err (H=1)`: peggiorato da **0.1475** (v4.3) a **0.4271** (v4.4).
   - `Rollout H=50 Pos Err`: peggiorato da **0.4825** (v4.3) a **0.7410** (v4.4).

### Decisione Strategica (Revert Architetturale a SPWM-v4.3)
- L'esperimento v4.4 viene archiviato e congelato come **ablazione negativa**.
- **SPWM-v4.3** viene confermato ed eletto a **riferimento fondativo (baseline ufficiale)** del progetto:
  - In fase di training e tracking sensoriale, $q_t$ rimane vincolato al 100% ai keypoint visivi di `SpatialSoftmax` ($q_{\text{sensor}}$), preservando la fedeltà e l'isometria cartesiana.
  - L'integrazione cinematica autonoma ($q_{t+1} = \text{clamp}(q_t + \tanh(W_{\text{vel}}\bar{s}_p), -1, 1)$) opera esclusivamente in modalità a sensori spenti dentro `predict_future`.
- **Direzione per SPWM-v5:** La stabilità e il controllo del drift sui rollout lunghi non verranno perseguiti per via euristica (come lo scheduled sampling), bensì attraverso principi puramente geometrici e biofisici:
  1. Vincoli di struttura simplettica conservativa (Hamiltoniani / Simplettici discreti).
  2. Garanzia di stabilità di Lyapunov e contrazione asintotica della matrice di ricorrenza $W_{\text{rec}}$ nel comparto di momento $p$.
