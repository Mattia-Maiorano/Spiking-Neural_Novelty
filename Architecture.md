L'architettura **SPWM-v0.1.0** (Spiking Predictive World Model) si configura come un sistema neuromorfo ibrido a due scale temporali, progettato per fondere l'alta acuità cinematica locale con una stabilità asintotica su lunghi orizzonti attraverso un innovativo sistema di neuromodulazione del guadagno.

## 1. Rappresentazione Sensoriale e Spazio delle Fasi

Il modello elabora flussi di eventi visivi comprimendoli in uno spazio latente strutturato geometricamente, garantendo una complessità di memoria $\mathcal{O}(1)$ nel tempo.

* **Event Encoder & Bottleneck Spaziale:** I frame di eventi passano attraverso una rete convoluzionale che culmina in uno strato `SpatialSoftmax`. Questo modulo estrae analiticamente 16 keypoint bidimensionali, garantendo per costruzione l'invarianza topologica e impedendo alla rete di distorcere la geometria dell'input visivo in configurazioni non-euclidee.


* **Spazio Latente Duale ($Z \in \mathbb{R}^{128}$):** Lo stato interno è esplicitamente partizionato in due sottospazi:


* **Coordinate Geometriche ($q \in [-1, 1]^{32}$):** Direttamente ancorate ai keypoint estratti dalla via dorsale sensoriale, rappresentano la posizione spaziale istantanea.


* **Momento Canonico / Contesto ($p \in \mathbb{R}^{96}$):** Codifica l'inerzia, la velocità latente e la memoria a lungo termine della dinamica.





## 2. Fast Predictor (Acuità Cinematica e Dinamica Locale)

Il modulo veloce è responsabile dell'integrazione sensomotoria e della transizione dinamica nominale.

* **Memoria ALIF Multi-Scala:** Al cuore del predittore risiede una popolazione neuromorfa divisa in due pool di costanti di tempo: una porzione reattiva ($\beta_{\text{adapt}} = 0.90$) per i transienti rapidi e una porzione di contesto profondo ($\beta_{\text{adapt}} = 0.985$) per l'integrazione temporale.


* **Predictive Coding:** Il modello genera un'aspettativa sensoriale top-down $\hat{x}_t$ basata sullo stato precedente. L'errore di predizione $\epsilon_t = x_t - \hat{x}_t$ viene propagato nella rete per aggiornare le correnti somatiche.


* **Transizione Residuale Latente:** Un MLP compatto apprende la derivata locale della traiettoria $\Delta z_{fast} = f_\theta(z_t)$, proponendo il prossimo stato latente $z_{fast\_next} = z_t + \Delta z_{fast}$.


* **Plasticità e-prop:** L'apprendimento avviene online tramite *backpropagation* surrogata forward-only, fondendo tracce di eligibilità locali con il feedback di errore globale decodificato dai probe fisici.



## 3. Slow Neuromodulatory Corrector (Gain Field System)

Il modulo lento sostituisce le euristiche di controllo rigido con un paradigma biologicamente plausibile di modulazione di volume (Gain Mask), volto a sopprimere la deriva su traiettorie *Out-of-Distribution* (OOD).

* **Popolazione ALIF Lenta Integrativa:** Costituita da 128 neuroni ALIF con costanti biologiche marcatamente più lente ($\beta_{mem} = 0.95$, $\beta_{adapt} = 0.94$), la rete integra in modo continuo lo stato proposto dal predittore veloce e il contesto storico rallentato.


* **Maschera di Neuromodulazione Continua:** La popolazione non genera offset spaziali diretti, ma emette una maschera di attivazione limitata $\text{gain\_mask} \in [-1, 1]^{128}$.


* **Modulazione della Derivata Istantanea:** L'intervento correttivo agisce come un moltiplicatore di campo sulla transizione del predittore veloce:

$$\Delta z_{mod} = \Delta z_{fast} \cdot (1 + \alpha_{gain} \cdot \text{gain\_mask})$$



Dove gli *hot spot* ($>0$) amplificano le dinamiche locali e i *cold spot* ($<0$) fungono da dissipatori di deriva. L'autorità massima è limitata da $\alpha_{gain}.


* **Vincolo Ismetrico Fisico:** Le coordinate spaziali modulate $q_{corr}$ vengono forzatamente ri-ancorate ai confini euclidei $[-1, 1]$, prevenendo alterazioni topologiche.

## 4. Pipeline di Addestramento Disaccoppiata

Per evitare interferenze distruttive tra i gradienti di precisione locale e quelli di stabilizzazione a lungo raggio, l'architettura sfrutta due ottimizzazioni separate e seriali:

* **Fase 1: Train Predictor (Nominal Physics):** Il correttore è disattivato. Il predittore viene addestrato su dati puliti per minimizzare l'errore di ricostruzione a 1 step, regolarizzare la varianza spaziale e allineare la decodifica fisica (probe di posizione e velocità congelato per prevenire il collasso dei pesi del decoder).


* **Fase 2: Train Corrector (Drift Suppression & Burn-in):** Il predittore viene completamente congelato. Il correttore viene esposto a traiettorie con iniezione stocastica di drift impulsivo. L'ottimizzazione avviene su rollout lunghi ($H=50$) introducendo un periodo di **Burn-in** (es. 15 step iniziali). Durante il Burn-in, la rete ALIF lenta assorbe il contesto sensoriale ma la sua maschera di modulazione viene ignorata. La funzione di costo viene retro-propagata solo sugli step successivi, costringendo l'ottimizzatore a concentrarsi unicamente sulle manovre di salvataggio attivo della traiettoria.