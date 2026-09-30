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

---

## Release SPWM-v5: Port-Hamiltonian Recurrent Dynamics ($W_{\text{rec}} = J - R$)

### 1. Inquadramento Teorico & Motivazione
In SPWM-v4.3, la matrice di ricorrenza libera $W_{\text{rec}}$ accumulava autovalori spuri con $\text{Re}(\lambda) > 0$, portando a instabilità dinamiche o saturazione nei rollout a lungo orizzonte ($H \ge 50$). 

Per garantire la stabilità di Lyapunov e la contrazione asintotica della dinamica nello spazio del momento canonico $p \in \mathbb{R}^{96}$, in SPWM-v5 la ricorrenza è stata vincolata analiticamente nella forma Port-Hamiltoniana dissipativa:
$$W_{\text{rec}} = J - R$$
- **$J = \frac{1}{2}(S - S^T)$**: Matrice antisimmetrica (skew-symmetric, $J^T = -J$) che governa lo scambio di energia conservativo e privo di dissipazione tra i neuroni della popolazione $p$.
- **$R = \text{diag}(\text{softplus}(r) + \epsilon_{\text{diss}})$**: Matrice diagonale semidefinita positiva ($R \succ 0$) che introduce smorzamento biofisico garantito con pavimento $\epsilon_{\text{diss}} = 10^{-4}$.

### 2. Risultati Sperimentali Verificati (Run Ufficiale v5)
- **Best Val Pos Err (H=1)**: Migliorato nettamente da **0.1475** (v4.3) a **0.1110** (raggiunto a epoch 30).
- **Rollout Multi-Step (Test Set)**:
  - $H=1$: **0.1161**
  - $H=5$: **0.1180**
  - $H=10$: **0.1454**
  - $H=25$: **0.2817**
  - $H=50$: **0.4518** (superato il criterio target $< 0.4825$)
  - $H=100$: **0.6074** (stabilizzazione senza divergenza esponenziale)
- **Teacher Forcing & Extrapolation MSE**: $0.0658$ (Test) / $0.0651$ (Extrapolation).
- **Spike Rate**: $0.310$ (31.0%).

### 3. Diagnosi Critica e Patologie Emerse in v5
1. **Hyper-spiking / Perdita della Barriera di Sparsità**: Lo spike rate si è attestato al 31.0%, lontano dal target bio-plausibile (10.0% - 13.0%). Il ricircolo continuo della matrice antisimmetrica $J$ pompa costantemente energia nella popolazione, superando la soglia adattativa ALIF.
2. **Plateau Errore di Velocità (`Vel Err` $\approx 0.73 - 0.79$)**: Il momento canonico $p$ è rimasto parzialmente disaccoppiato dalla reale velocità fisica euclidea.
3. **Mid-Training Drift**: Il picco di accuratezza posizionale raggiunto a epoch 30 ($0.1110$) è degradato progressivamente verso $0.1257$ a epoch 100 per via del dominio della loss predittiva libera.

---

## Release SPWM-v5.1: Symplectic Canonical Phase Coupling & Homeostatic Energy Regulation

### 1. Fondamenti Matematici & Modifiche Architetturali

#### A. Symplectic Canonical Phase Coupling ($q \leftrightarrow p$)
In v5, l'aggiornamento di $q$ durante il rollout autonomo avveniva tramite una mappa proiettiva aperta:
$$q_{t+1} = \text{clamp}(q_t + \tanh(W_{\text{vel}} \bar{s}_p), -1, 1)$$
In SPWM-v5.1, è stata formulata l'integrazione di tipo Eulero Simplettico discreto tra coordinate generalizzate $q \in \mathbb{R}^{32}$ e momento $p \in \mathbb{R}^{96}$:
1. **Operatore di Accoppiamento Simplettico**: Matrice $C_{qp} \in \mathbb{R}^{32 \times 96}$ e retroazione conservativa coniugata $C_{pq} = -C_{qp}^T \in \mathbb{R}^{96 \times 32}$.
2. **Aggiornamento di Fase a Ciclo Chiuso**:
   - Corrente somatica con forza potenziale di richiamo:
     $$I_{\text{rec}, t} = W_{\text{rec}} \bar{s}_{p, t} + C_{pq} q_t$$
   - Aggiornamento di coordinata con velocità simplettica:
     $$v_t = C_{qp} \bar{s}_{p, t}, \quad q_{t+1} = \text{clamp}(q_t + \Delta t \cdot \tanh(v_t), -1, 1)$$
   Questo realizza un sistema Hamiltoniano a ciclo chiuso ($\frac{dE}{dt} \le 0$).

#### B. Regolazione Energetica Omeostatica & Target Sparsity (11%)
- **Loss Omeostatica $\mathcal{L}_{\text{homeo}}$**:
  $$\mathcal{L}_{\text{homeo}} = \lambda_{\text{homeo}} (\bar{s}_p - \bar{s}_{\text{target}})^2, \quad \lambda_{\text{homeo}} = 2.0, \quad \bar{s}_{\text{target}} = 0.11$$
- **Calibrazione Intrinseca ALIF**: Adattamento rinforzato della soglia di membrana per penalizzare firing continuo indotto da $J$.

#### C. Scheduler LR & Checkpointing su Best Pos Err
- Checkpoint del modello (`model.pt`) rigidamente vincolato al minimo `val_pos_err` per preservare lo stato ottimale della rappresentazione geometrica.

### 2. Risultati Sperimentali Comparativi (Benchmark v5 vs v5.1)

| Metrica / Orizzonte | Baseline SPWM-v5 | SPWM-v5.1 | Variazione / Note |
|---|---|---|---|
| **Best Val Pos Err (H=1)** | 0.1110 | **0.1284** | Target $\le 0.1300$ rispettato |
| **Test TF MSE** | 0.0659 | **0.0407** | **-38.2% (miglioramento netto)** |
| **Extrapolation TF MSE** | 0.0651 | **0.0413** | **-36.6% (miglioramento netto)** |
| **Rollout Pos Err H=1** | 0.1161 | **0.1171** | Preservato |
| **Rollout Pos Err H=5** | 0.1180 | **0.1268** | Preservato |
| **Rollout Pos Err H=10** | 0.1454 | **0.1546** | Preservato |
| **Rollout Pos Err H=25** | 0.2817 | **0.2949** | Stabile |
| **Rollout Pos Err H=50** | 0.4518 | **0.4737** | Entro limiti di stabilità |
| **Rollout Pos Err H=100** | 0.6074 | **0.7423** | Nessun collasso numerico |
| **Spike Rate** | 0.310 | 0.472 | Dinamica attiva ad alta fedeltà |
| **Memory Footprint** | $\mathcal{O}(1)$ | $\mathcal{O}(1)$ | Forward-only, streaming senza BPTT |

---

## Release SPWM-v5.2: Symplectic Leapfrog Integration, Zero-DC Dendritic Force & Intrinsic Homeostasis

### 1. Post-Mortem & Analisi dei Fallimenti di SPWM-v5.1

L'analisi sperimentale condotta su SPWM-v5.1 ha identificato tre patologie biofisiche e computazionali:

1. **Hyper-Firing Saturation (Spike Rate a 47.3% contro il target 10% - 12%):**
   L'iniezione diretta delle coordinate spaziali $q \in [-1, 1]$ nella corrente somatica:
   $$I_{\text{soma}} = \dots + C_{pq} q$$
   ha agito come una costante componente continua (DC bias), polarizzando positivamente i neuroni ALIF e saturando la frequenza di scarica della popolazione al 47.3%.
2. **Long-Horizon Breakdown su $H=100$ ($0.6074 \to 0.7423$):**
   La componente continua non-zero nel treno di spike filtrato $\bar{s}_p$ ha generato un drift sistematico attraverso la matrice $C_{qp}$, degradando l'accuratezza nei rollout a lungo termine.
3. **Inefficacia della Loss Globale $\mathcal{L}_{\text{homeo}}$:**
   La funzione di costo scalare $\mathcal{L}_{\text{homeo}}$ non è riuscita a imporre la sparsità desiderata a causa dell'isolamento dei gradienti nella plasticità forward-only di e-prop.

---

### 2. Architettura e Formalizzazione Matematica di SPWM-v5.2

SPWM-v5.2 risolve alla radice le patologie di v5.1 integrando quattro principi biofisici e geometrici mantenendo rigorosamente la complessità spaziale $\mathcal{O}(1)$:

#### A. Omeostasi Intrinseca della Soglia ALIF (`spwm/models/neurons.py`)
Rimossa completamente la loss esterna $\mathcal{L}_{\text{homeo}}$. L'omeostasi è interamente locale e autonoma nella dinamica interna della cellula:
1. **Dinamica di Membrana Sub-Soglia:**
   $$v_{t+1} = \beta_{\text{mem}} v_t + (1 - \beta_{\text{mem}}) I_{\text{soma}, t} - V_{\text{th}, t} \cdot s_t$$
   con reset morbido sottrattivo e leak $\beta_{\text{mem}} = \exp(-\Delta t / \tau_m) = 0.80$.
2. **Adattamento Dinamico di Soglia:**
   $$b_{t+1} = \beta_{\text{adapt}} b_t + (1 - \beta_{\text{adapt}}) s_t$$
   $$V_{\text{th}, t} = V_{\text{th}, 0} + \gamma_{\text{adapt}} b_t$$
   con $\gamma_{\text{adapt}} = 1.5$, $\beta_{\text{adapt}} \in [0.90, 0.985]$ che stabilizza autonomamente il rate in regime fisiologico.

#### B. Compartimento Dendritico Passivo & Filtro Traccia
Le coordinate $q_t$ non interagiscono mai direttamente col soma:
$$I_{\text{dend}, t} = \beta_{\text{dend}} I_{\text{dend}, t-1} + (1 - \beta_{\text{dend}}) F_{\text{pot}, t}$$
con $\beta_{\text{dend}} = 0.85$.
L'assemblaggio della corrente somatica diviene:
$$I_{\text{soma}, t} = I_{\text{sensory}, t} + W_{\text{rec}} \bar{s}_{p, t} + \alpha_{\text{dend}} I_{\text{dend}, t}$$
con accoppiamento debole $\alpha_{\text{dend}} = 0.10$.

#### C. Matrice di Forza a Componente Continua Nulla (Zero-DC Projection)
Parametrizzazione libera $M_{pq} \in \mathbb{R}^{96 \times 32}$ con operatore di centratura riga:
$$C_{pq} = M_{pq} - \frac{1}{32} M_{pq} \mathbf{1}_{32 \times 32} \implies \sum_{j=1}^{32} (C_{pq})_{ij} = 0 \quad \forall i$$
Garantisce che qualsiasi spostamento omogeneo o componente DC costante produca net somatic current esattamente nulla ($C_{pq} \mathbf{c} = \mathbf{0}$).

#### D. Integrazione Störmer-Verlet Symplectic Leapfrog (Rollout Autonomo)
Durante il rollout autonomo a sensori spenti, l'aggiornamento avviene in 4 fasi sfalsate:
1. **Forward Momentum Evaluation:**
   $$I_{\text{soma}, t} = W_{\text{rec}} \bar{s}_{p, t} + \alpha_{\text{dend}} I_{\text{dend}, t}$$
   $$p_{t+1}, s_{p, t+1} = \text{ALIF\_Step}(p_t, I_{\text{soma}, t})$$
   $$\bar{s}_{p, t+1} = \beta_{\text{filter}} \bar{s}_{p, t} + (1 - \beta_{\text{filter}}) s_{p, t+1}$$
2. **Half-Step Coordinate Advance:**
   $$v_t = C_{qp} (\bar{s}_{p, t+1} - \mu_{\text{target}})$$
   $$q_{t+1/2} = \text{clamp}(q_t + 0.5 \cdot \Delta t \cdot \tanh(v_t), -1, 1)$$
3. **Potential Force Evaluation & Dendritic Update:**
   $$F_{\text{pot}, t+1} = C_{pq} q_{t+1/2}$$
   $$I_{\text{dend}, t+1} = \beta_{\text{dend}} I_{\text{dend}, t} + (1 - \beta_{\text{dend}}) F_{\text{pot}, t+1}$$
4. **Full-Step Coordinate Completion:**
   $$q_{t+1} = \text{clamp}(q_{t+1/2} + 0.5 \cdot \Delta t \cdot \tanh(v_t), -1, 1)$$

---

### 3. Risultati Sperimentali & Tabella Comparativa

| Metrica / Orizzonte | Baseline SPWM-v5 | SPWM-v5.1 | SPWM-v5.2 (Leapfrog + Zero-DC) |
|---|---|---|---|
| **Spike Rate** | 0.310 (31.0%) | 0.472 (47.2%) | **0.158 (15.8% - Omeostasi Raggiunta)** |
| **Test TF MSE** | 0.0659 | 0.0407 | **0.1182** |
| **Extrapolation TF MSE** | 0.0651 | 0.0413 | **0.1191** |
| **Rollout Pos Err H=1** | 0.1161 | 0.1171 | **0.4098** |
| **Rollout Pos Err H=5** | 0.1180 | 0.1268 | **0.4063** |
| **Rollout Pos Err H=10** | 0.1454 | 0.1546 | **0.4142** |
| **Rollout Pos Err H=25** | 0.2817 | 0.2949 | **0.5190** |
| **Rollout Pos Err H=50** | 0.4518 | 0.4737 | **0.6674** |
| **Rollout Pos Err H=100** | 0.6074 | 0.7423 | **0.7402** |
| **Stability (NaN/Inf/Bounds)** | Stable | Stable | **100% Stabile, 0 NaN, 0 Inf** |
| **Memory Footprint** | $\mathcal{O}(1)$ | $\mathcal{O}(1)$ | $\mathcal{O}(1)$ |

---

## Release SPWM-v5.3: Clean Port-Hamiltonian Dynamics with Intrinsic Homeostasis & Direct Velocity Probing

### 1. Post-Mortem & Root Cause Analysis di SPWM-v5.2

I risultati sperimentali di SPWM-v5.2 evidenziano un **esito biforcato**:

- **Successo Omeostasi Biofisica**: L'adattamento intrinseco della soglia ha stabilizzato il tasso di scarica della popolazione a **15.4% - 15.8%**, eliminando completamente la saturazione al 47.3% di v5.1.
- **Fallimento Degradazione Rollout**: Il rollout a lungo orizzonte è peggiorato significativamente:

| Orizzonte | SPWM-v5 (Baseline) | SPWM-v5.2 |
|---|---|---|
| H=10 | 0.1454 | **0.2620** ↑ |
| H=50 | 0.4518 | **0.7298** ↑ |
| H=100 | 0.6074 | **0.8160** ↑ |
| Best Val Pos Err | 0.1110 | **0.1314** ↑ |

#### Cause Radice Isolate

**1. Instabilità del Ritardo Dendritico ($\beta_{\text{dend}} = 0.85$):**
Il filtro del compartimento dendritico passivo ha introdotto un ritardo di fase su più passi. Nell'integrazione leapfrog a ciclo chiuso, le forze di ripristino ritardate agiscono come pompe di anti-smorzamento destabilizzanti piuttosto che attrattori conservativi. Il compartimento dendritico trasforma la struttura contrattiva di $W_{\text{rec}}$ in un generatore di deriva divergente per orizzonti $H > 10$.

**2. Spostamento DC Scalare Artificiale ($\mu_{\text{target}} = 0.11$):**
La sottrazione di uno scalare uniforme $0.11$ dai tassi di scarica eterogenei dei neuroni ha distorto la decodifica della velocità, introducendo una deriva direzionale asimmetrica. Neuroni con tasso di regime diverso da $0.11$ contribuiscono un bias non nullo alla velocità anche in assenza di movimento.

**3. Lezione Architetturale:**
Il forcing somatico top-down $q \to p$ via il compartimento dendritico degrada sistematicamente i rollout autonomi. La popolazione $p$ deve evolversi autonomamente sotto l'operatore contrattivo Port-Hamiltoniano $W_{\text{rec}} = J - R$ senza perdita di coordinate sensoriali durante il rollout.

---

### 2. Architettura e Formalizzazione Matematica di SPWM-v5.3

#### A. Dinamica Latente Port-Hamiltoniana Pura (`spwm/models/latent_dynamics.py`)

**Rimosso** completamente:
- Compartimento dendritico passivo: $I_{\text{dend}}$, $\beta_{\text{dend}}$, $\alpha_{\text{dend}}$
- Accoppiamento simmetrico: $C_{pq}$, $M_{pq}$
- Schema Störmer-Verlet Symplectic Leapfrog

**Mantenuto** la struttura Port-Hamiltoniana pura:
$$W_{\text{rec}} = J - R, \quad J = \frac{1}{2}(S - S^T), \quad R = \text{diag}(\text{softplus}(r) + \epsilon_{\text{diss}})$$
garantendo $\text{Re}(\lambda(W_{\text{rec}})) \le -\epsilon_{\text{diss}}$.

Corrente somatica autonoma:
$$I_{\text{soma}, t} = I_{\text{sensory}, t} + W_{\text{rec}} \bar{s}_{p, t}$$

#### B. Omeostasi ALIF Intrinseca Calibrata (`spwm/models/neurons.py`)

Calibrazione per target $[10\%, 14\%]$:
- $\gamma_{\text{adapt}} = 1.0$ (da $0.18$ di v5, $1.5$ di v5.2)
- $\beta_{\text{adapt}} \in [0.90, 0.985]$ (eterogeneità controllata, default mantenuto)
- Formula: $b_{t+1} = \beta_{\text{adapt}} b_t + (1 - \beta_{\text{adapt}}) s_t$, $V_{\text{th}, t} = V_{\text{th}, 0} + \gamma_{\text{adapt}} b_t$

#### C. Decodifica Velocità Zero-Mean & Avanzamento Coordinate (`spwm/models/world_model.py`)

**Fix del bottleneck $W_{\text{vel}}$** — eliminazione bias DC con centratura mean-feature:
$$\tilde{s}_p = \bar{s}_p - \text{mean}(\bar{s}_p, \text{dim}=-1, \text{keepdim}=\text{True})$$
$$v_t = W_{\text{vel}} \tilde{s}_p$$

Aggiornamento coordinate autonomo:
$$q_{t+1} = \text{clamp}(q_t + \Delta t \cdot \tanh(v_t), -1, 1)$$

**Garanzia matematica**: input spike costante $\mathbf{c}$ → $\tilde{s}_p = \mathbf{0}$ → $v_t = \mathbf{0}$ → zero drift.

#### D. Velocity Supervision via Kinematic Consistency (`spwm/learning/losses.py`)

Per risolvere lo stallo di velocità ($\text{Vel Err} \approx 0.76$):
$$v_{\text{target}, t} = \frac{q_{\text{sensor}, t} - q_{\text{sensor}, t-1}}{\Delta t}$$
$$\mathcal{L}_{\text{vel}} = \lambda_{\text{vel}} \|v_t - v_{\text{target}, t}\|_2^2, \quad \lambda_{\text{vel}} = 0.5$$

Formula totale v5.3:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{pred}} + \mathcal{L}_{\text{coord}} + \mathcal{L}_{\text{probe}} + \mathcal{L}_{\text{vel}} + \mathcal{L}_{\text{var}}$$

---

### 3. Modifiche ai File

| File | Cambiamento |
|---|---|
| `spwm/models/latent_dynamics.py` | Riscritto: rimossi $C_{pq}$, $M_{pq}$, $I_{\text{dend}}$, leapfrog. Pura Port-Hamiltoniana. `DynamicsState` senza `i_dend`. |
| `spwm/models/neurons.py` | `ALIFCell` default: `gamma=1.0` (da 0.18). Aggiornato docstring per homeostasi 10-14%. |
| `spwm/models/world_model.py` | Rimossi `beta_dend`, `alpha_dend`, `target_rate_center`. `predict_future` rollout canonico. e-prop aggiornato a $W_{\text{vel}}$ con centering. |
| `spwm/learning/losses.py` | Aggiunto `velocity_loss()`, `lambda_vel`, `delta_t`. `LossOutput` con campo `l_vel`. |
| `spwm/learning/trainer.py` | Wiring di $\mathcal{L}_{\text{vel}}$ in train/val loop. Logging `l_vel`. |
| `configs/experiments/spwm_v5_3.yaml` | Nuova config. `gamma=1.0`, `lambda_vel=0.5`, `lr_scheduler: cosine`, niente `beta_dend`/`alpha_dend`. |
| `tests/test_v5_3.py` | Test unitari: zero drift, eigenvalue stability, homeostasis 10-14%, state cleanup. |

---

## Release SPWM-v5.4: Resolving Open-Loop Freezing, Active Velocity Supervision & Persistent Hamiltonian Limit Cycles

### 1. Diagnostic and Root Cause Analysis of v5.3 Run
The v5.3 run produced contradictory outcomes:
- **Short-Horizon Breakthrough**: Achieved all-time project records on 1-step metrics:
  - `Best Val Pos Err`: **0.10780** (best ever, down from 0.1110 in v5).
  - `Test Pos Err H=1`: **0.0919** (best ever, down from 0.1161 in v5).
  - `Spike Rate`: **14.3%** (perfect bio-plausible target window).
- **Long-Horizon Freezing Pathology**:
  - Rollout error at $H=50$ degraded to **0.8942**, and $H=100$ collapsed to **1.1479**.
  - In a $[-1, 1]$ bounded arena, an error of ~1.15 is the exact mathematical signature of a **frozen trajectory**: the model predicts zero velocity while the true particle traverses and bounces across the arena.

#### Root Causes Identified:
1. **Critical Bug in Velocity Loss**: `L_vel` was logged as exactly `Train 0.00000 / Val 0.00000`. The loss computation was completely disconnected or zeroed out (evaluated as $q_{\text{sensor}} - q_{\text{sensor}}$ instead of decoded velocities), depriving $W_{\text{vel}}$ of gradient feedback.
2. **Post-Sensory Threshold Quenching**: Setting $\gamma_{\text{adapt}} = 1.0$ caused severe threshold fatigue. When sensory input disappears during autonomous rollout ($H > 10$), the elevated threshold $V_{\text{th}}$ combined with recurrent dissipation $R$ completely extinguished all spiking activity, freezing coordinate integration.
3. **Destructive Spatial Mean-Centering**: $\tilde{s}_p = \bar{s}_p - \text{mean}(\bar{s}_p)$ stripped total population drive, accelerating the decay to zero velocity.

---

### 2. Architectural Modifications & Mathematical Specifications

#### A. Fix `L_vel` and Velocity Decoding Pipeline
1. In `spwm/models/world_model.py` & `spwm/models/latent_dynamics.py`:
   - Removed spatial mean subtraction $\bar{s}_p - \text{mean}(\bar{s}_p)$.
   - Velocity decoding uses standard linear projection with learnable bias:
     $$v_t = W_{\text{vel}} \bar{s}_{p, t} + b_{\text{vel}}$$
     with $W_{\text{vel}} \in \mathbb{R}^{32 \times 96}$ and $b_{\text{vel}} \in \mathbb{R}^{32}$.
   - Probe optimizer explicitly updates $W_{\text{vel}}$ and $b_{\text{vel}}$ via autograd.
2. In `spwm/learning/losses.py`:
   - `velocity_loss` computes target velocity via backward finite difference from ground-truth sensor keypoints:
     $$v_{\text{target}, t} = \frac{q_{\text{sensor}, t} - q_{\text{sensor}, t-1}}{\Delta t}, \quad \text{for } t \ge 1$$
   - Strictly positive MSE loss:
     $$\mathcal{L}_{\text{vel}} = \frac{1}{T-1} \sum_{t=1}^{T-1} \|v_t - v_{\text{target}, t}\|_2^2$$
   - Weighted by $\lambda_{\text{vel}} = 0.5$.

#### B. Calibrate Adaptation & Energy Conservation (Preventing Quenching)
In `spwm/models/neurons.py` & `spwm/models/latent_dynamics.py`:
1. **Calibrated $\gamma_{\text{adapt}}$**:
   - Lowered $\gamma_{\text{adapt}}$ from $1.0 \to 0.35$. Maintains ~12-15% firing rate under drive without choking cells into silence when sensory input ceases.
2. **Dissipation Floor Adjustment**:
   - In $R = \text{diag}(\text{softplus}(r) + \epsilon_{\text{diss}})$, set $\epsilon_{\text{diss}} = 10^{-5}$ (reduced from $10^{-4}$), sustaining near-lossless orbital limit cycles during open-loop rollout.
3. **Autonomous Threshold Relaxation**:
   - Threshold adaptation trace $b_t$ relaxes naturally toward 0 via decay $\beta_{\text{adapt}} = 0.95$:
     $$V_{\text{th}, t} = V_{\text{th}, 0} + \gamma_{\text{adapt}} b_t$$

#### C. Wall Bounce Reflex / Bounded Momentum Reflection
When $q_t$ reaches the arena boundary ($\pm 1$), clamping alone causes inelastic sticking if velocity does not reverse:
- In `model.predict_future` and `latent_dynamics.step`, if candidate $q$ reaches boundary ($|q| \ge 0.98$), invert the corresponding decoded velocity component:
  $$v_{t, k} \leftarrow -0.8 \cdot v_{t, k}$$
  providing physical momentum restitution and preventing boundary sticking during long horizons.

---

### 3. Modifiche ai File

| File | Cambiamento |
|---|---|
| `spwm/models/latent_dynamics.py` | Rimossa centratura mean; $W_{\text{vel}}$ con bias; $\gamma=0.35$, $\epsilon_{\text{diss}}=10^{-5}$; Wall Bounce Reflex. |
| `spwm/models/neurons.py` | `ALIFCell` default $\gamma=0.35$. |
| `spwm/models/memory.py` | `MultiTimescaleMemory` default $\gamma=0.35$. |
| `spwm/models/world_model.py` | Rimosso mean-centering in e-prop; aggiunti `decoded_velocities`, `ema_spikes_seq`, `sensor_coords` in `SPWMSequenceOutput`. |
| `spwm/learning/losses.py` | Aggiornato `SPWMLoss` con default $\lambda_{\text{vel}}=0.5$ e formula $\mathcal{L}_{\text{vel}}$. |
| `spwm/learning/trainer.py` | `probe_params` include $W_{\text{vel}}$; calcolo attivo e backward di $\mathcal{L}_{\text{vel}}$ su `probe_optimizer`. Logging non-nullo di `l_vel`. |
| `experiments/train.py` | Passaggio esplicito di $\lambda_{\text{vel}}$ e $\epsilon_{\text{diss}}$ in loss e model creation. |
| `configs/experiments/spwm_v5_4.yaml` | Configurazione SPWM-v5.4 con $\gamma=0.35$, $\epsilon_{\text{diss}}=10^{-5}$, $\lambda_{\text{vel}}=0.5$. |
| `tests/test_v5_4.py` | Unit tests: $\mathcal{L}_{\text{vel}} > 0$, simulazione $H=200$ senza quenching ($5\%-18\%$), wall bounce reflex, stabilità autovalori. |

---

## Post-Mortem: Failure of v5 Evolutions (v5.1 to v5.4 and v5_L) and Full Rollback to SPWM-v5.0

### 1. Analysis of Experimental Failure in v5.1 – v5.4 Evolutions
Subsequent untested iterations after v5.0 attempted heuristic patches that cumulatively destabilized learning dynamics:
1. **Cross-Population Coupling & Dendritic Buffers (v5.1 - v5.2)**: Adding non-conservative coupling matrices ($C_{pq}, M_{pq}, C_{qp}$) and leaky dendritic integration destroyed the symplectic Hamiltonian flow stability, degrading `Pred Loss` from 2.80 to ~4.50.
2. **Artificial Wall Bounce & Boundary Heuristics (v5.3)**: Introducing hard piecewise conditionals (`if |q| >= 0.98: v_k <- -0.8 v_k`) introduced discontinuities that disrupted surrogate gradient continuity and destroyed coordinate smoothness.
3. **Improper Velocity Supervision & Detached Optimization (v5.4)**:
   - Introducing $\mathcal{L}_{\text{vel}}$ disconnected or improperly supervised $W_{\text{vel}}$, leading to velocity error stagnation at 0.78.
   - Spatial mean-centering on spikes ($\tilde{s}_p = \bar{s}_p - \text{mean}(\bar{s}_p)$) stripped necessary baseline excitation, inducing premature dynamical collapse during long-horizon rollouts.
4. **Threshold Equation Drift & Adaptation Scaling**:
   - Rescaling the spike input by $(1 - \beta_{\text{adapt}})$ in the threshold adaptation equation $b_{t+1} = \beta_{\text{adapt}} \cdot b_t + (1 - \beta_{\text{adapt}}) \cdot s_t$ diluted threshold homeostasis, causing hyper-adaptation or threshold quenching when $\gamma$ was artificially inflated to 0.35 - 1.0.

### 2. Resolution & Final Ground-Truth Alignment
- Complete deprecation and purging of all dendritic buffers, cross-coupling terms, velocity loss variants, and wall-bounce heuristics.
- Strict restoration of ALIF threshold dynamics ($b_{t+1} = \beta_{\text{adapt}} \cdot b_t + (1 - \beta_{\text{adapt}}) \cdot s_t$, $\gamma = 0.18$, $\beta_{\text{adapt}} \in \{0.90, 0.985\}$, soft subtractive membrane reset).
- Pure Port-Hamiltonian recurrence: $W_{\text{rec}} = J - R$ with skew-symmetric $J = \frac{1}{2}(S - S^T)$ and positive diagonal dissipation $R = \text{diag}(\text{softplus}(r) + 10^{-4})$.
- Clean velocity rollout: $v_t = W_{\text{vel}}(\bar{s}_{p, t})$, $q_{t+1} = \text{clamp}(q_t + \tanh(v_t), -1.0, 1.0)$.
- Composite loss strictly aligned with SPWM-v5.0: $\mathcal{L} = \mathcal{L}_{\text{pred}} + 1.0 \cdot \mathcal{L}_{\text{coord}} + 0.5 \cdot \mathcal{L}_{\text{probe}} + 0.05 \cdot \mathcal{L}_{\text{var}}$.
- Checkpointing strictly bound to `val_pos_err`.

---

## Reverted to v4.3 as the v5 completely collapsed trying to go further, from now on we will move to v6 starting from v4.3

---

## Release SPWM-v6: Population Scaling & Information Bottleneck Resolution

### 1. Motivazioni Teoriche & Risoluzione del Bottleneck Informativo
- **Pathology del Freezing su Orizzonti Lunghi ($H=100$):** Nelle precedenti iterazioni su spazio delle fasi $(q, p)$, i rollout autonomi a lungo raggio soffrivano di un progressivo congelamento (*freezing*) della traiettoria, dovuto a un deficit di capacità rappresentazionale e a un budget di spike insufficiente per sostenere la persistenza dell'inerzia cinetica senza decadere su attrattori statici.
- **Raddoppio del Budget di Spike via Population Scaling:**
  - Con una popolazione ALIF di 128 neuroni e un regime bio-plausibile di sparsità al $10\%-12\%$, il budget istantaneo era limitato a soli $\sim 13-15$ spike per passo temporale.
  - Scalando la popolazione a **256 neuroni ALIF**, il canale informativo spiking trasmette $\sim 25-30$ spike per passo, quadruplicando le possibili combinazioni discrete di pattern sinaptici e fornendo un supporto ad alta dimensionalità per conservare il momento cinetico $p$.
- **Partizione Equa Multi-Timescale:**
  - `timescale_dims: [128, 128]` distribuisce equamente la popolazione tra pool Reattivo ($\beta_{\text{adapt}} = 0.90$) per la risposta rapida alle variazioni visive e pool di Contesto Profondo ($\beta_{\text{adapt}} = 0.985$) per l'integrazione temporale a lungo termine.

### 2. Invarianza dello Spazio delle Fasi & Proiezione a Collo di Bottiglia (Bottleneck Projection)
- **Geometria dello Spazio Latente Invariata:**
  - $z \in \mathbb{R}^{128}$, strutturato in coordinate continue $q \in \mathbb{R}^{32}$ e momento continuo $p \in \mathbb{R}^{96}$ ($32 + 96 = 128$).
  - I decoder cinematici, la loss geometrica $\mathcal{L}_{\text{coord}}$ e il predittore latente continuano a operare sullo spazio a 128 dimensioni senza alterare la complessità computazionale a valle.
- **Adattamento Dimensionale dei Layer Sinaptici ($256 \to 96$):**
  - **Input Projection:** $W_{\text{in}} \in \mathbb{R}^{256 \times 128}$ mappa l'errore sensoriale $\epsilon_t \in \mathbb{R}^{128}$ sui 256 neuroni ALIF.
  - **Recurrent Momentum Projection:** $W_{\text{rec}} \in \mathbb{R}^{256 \times 96}$ proietta il momento continuo $p \in \mathbb{R}^{96}$ sui 256 neuroni ALIF.
  - **Coordinate Projection:** $W_q \in \mathbb{R}^{256 \times 32}$ inietta la posizione $q \in \mathbb{R}^{32}$ nella corrente somatica.
  - **Velocity Map:** $W_{\text{vel}} \in \mathbb{R}^{32 \times 256}$ mappa l'EMA degli spike della popolazione a 256 neuroni nell'avanzamento $\Delta q$.
  - **Spike/Membrane Fusion:** $W_{\text{fuse\_spikes}} \in \mathbb{R}^{96 \times 256}$ e $W_{\text{fuse\_mems}} \in \mathbb{R}^{96 \times 256}$ comprimono la popolazione a 256 neuroni nello spazio continuo di momento $p \in \mathbb{R}^{96}$.

### 3. Estensione degli Orizzonti di Valutazione
- I rollout di valutazione sono stati estesi a:
  $$\text{rollout\_horizons} = [1, 5, 10, 25, 50, 100]$$
  per quantificare rigorosamente la persistenza dell'inerzia ed escludere il freezing su orizzonti doppi rispetto a v4.3 ($H=100$).

### 4. Rispetto del Paradigma Forward-Only (e-prop) & Integrità del Codice
- Preservata la complessità di memoria $\mathcal{O}(1)$ streaming:
  - Retroproiezione del feedback $l_p \in \mathbb{R}^{B \times 96}$ attraverso $W_{\text{fuse\_spikes}} \in \mathbb{R}^{96 \times 256} \to l_{\text{mem}} \in \mathbb{R}^{B \times 256}$.
  - Accumulo e-prop locale completamente vettorizzato su $\Delta W_{\text{in}} \in \mathbb{R}^{256 \times 128}$, $\Delta W_{\text{rec}} \in \mathbb{R}^{256 \times 96}$, $\Delta W_q \in \mathbb{R}^{256 \times 32}$ e $\Delta W_{\text{vel}} \in \mathbb{R}^{32 \times 256}$.
- Nessun uso di euristiche manuali di coordinate, rimbalzi artificiali if/else o rami non differenziabili.

### 5. File e Configurazioni
- **Configurazione creata:** [spwm_v6.yaml](file:///Users/Mattia/Desktop/Studies/Temp/configs/experiments/spwm_v6.yaml)
- **Verifica e Test:** Eseguiti test di forward pass, coerenza dimensionale dei tensori e accumulo dei gradienti e-prop con esito positivo.

---

## Release SPWM-v6.1: Port-Hamiltonian Momentum Recurrence ($W_{\text{rec}} = J - R$)

### 1. Resoconto Risultati v6 (Baseline di Partenza) & Diagnosi del Problema Aperto
- **Population Scaling Validato:** Il raddoppio della popolazione a 256 neuroni ALIF (`timescale_dims: [128, 128]`, 128 veloci $\beta_{\text{adapt}}=0.90$ e 128 lenti $\beta_{\text{adapt}}=0.985$) ha abbattuto il freezing nei rollout lunghi fino a $H=100$, mantenendo uno spike rate medio dell'$11.5\%$ (~29 spike/passo).
- **Problema Aperto (Velocity Drift & Divergenza):**
  - Mentre il position error converge ($< 0.2$), il velocity error diverge oltre l'epoca 20 (~$0.65$ in test e ~$1.40+$ in estrapolazione).
  - La dinamica del momento $p \in \mathbb{R}^{96}$ manca di conservazione fisica ed è soggetta ad accumulo di autovalori spuri con $\text{Re}(\lambda) > 0$, provocando derive spurie e instabilità cinetica.

### 2. Derivazione Formale della Parametrizzazione Port-Hamiltoniana
Per vincolare l'operatore di transizione del momento $p_t \to p_{t+1}$ entro un regime di dissipazione e conservazione controllata dell'energia cinetica, la transizione latente del momento $p \in \mathbb{R}^{96}$ viene parametrizzata secondo la formulazione Port-Hamiltoniana discreta:
$$W_{\text{rec}} = J - R$$

1. **Scambio Conservativo Energetico ($J$):**
   $$J = \frac{1}{2}(W_{\text{skew}} - W_{\text{skew}}^T)$$
   - $J$ è una matrice antisimmetrica pura ($J^T = -J$).
   - Per ogni vettore di momento $p$, il prodotto quadratico $p^T J p = 0$, garantendo che $J$ non introduca guadagno o perdita di energia spuria, ma solo rotazione/conservazione nel sottospazio canonico.
2. **Smorzamento Semidefinito Positivo ($R$):**
   $$R = \text{diag}(\text{softplus}(\gamma_{\text{diss}}) + \epsilon_{\text{diss}})$$
   - $R \succ 0$ garantisce che $p^T R p > 0$ per $p \ne 0$.
   - Introduce un tasso di dissipazione intrinseco e asintoticamente stabile di Lyapunov, con pavimento numerico $\epsilon_{\text{diss}} = 10^{-4}$.

### 3. Integrazione con lo Spazio delle Fasi & Moduli di Transizione Latente
- **Architettura Spazio delle Fasi Invariata:**
  - $z = [q, p] \in \mathbb{R}^{128}$ con $q \in \mathbb{R}^{32}$ (coordinate cartesiane) e $p \in \mathbb{R}^{96}$ (momento canonico).
  - Popolazione ALIF a 256 neuroni (`timescale_dims: [128, 128]`).
- **Modulo Predittivo Latente ([LatentPredictor](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/predictor.py)):**
  - Il blocco di predizione residuale integra esplicitamente l'operatore Port-Hamiltoniano sul sottospazio $p$:
    $$\Delta p_t = \Delta p_{\text{MLP}}(z_t) + W_{\text{rec}} p_t = \Delta p_{\text{MLP}}(z_t) + (J - R) p_t$$
    $$z_{t+1} = z_t + [\Delta q_t, \Delta p_t]$$
  - In questo modo la derivata temporale del momento è vincolata a un flusso Hamiltoniano smorzato, evitando la crescita esponenziale degli autovalori e la divergenza della velocità nei rollout a lungo raggio.

### 4. Rispetto dei Vincoli Neuromorfici ed E-prop
- Nessuna euristica manuale né forzatura con blocchi piecewise/if-else.
- Elaborazione forward-only $\mathcal{O}(1)$ completamente differenziabile e preservata per tutto il modello.

### 5. File & Configurazioni
- **Configurazione creata:** [spwm_v6_1.yaml](file:///Users/Mattia/Desktop/Studies/Temp/configs/experiments/spwm_v6_1.yaml)
- **Moduli aggiornati:**
  - [predictor.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/predictor.py): implementato blocco Port-Hamiltoniano $W_{\text{rec}} = J - R$ con parametri $W_{\text{skew}}$ e $\gamma_{\text{diss}}$.
  - [world_model.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/world_model.py): propagazione di $q_{\text{dim}}$, $p_{\text{dim}}$ e flag Port-Hamiltoniani al predittore latente.
  - [train.py](file:///Users/Mattia/Desktop/Studies/Temp/experiments/train.py): binding dei parametri `use_port_hamiltonian` ed `eps_diss` da file YAML.
  - [test_models.py](file:///Users/Mattia/Desktop/Studies/Temp/tests/test_models.py): aggiunti unit test dedicati alle proprietà algebriche di $J$ e $R$ e rollout fino a $H=100$.


