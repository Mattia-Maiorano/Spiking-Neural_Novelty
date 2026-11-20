| Model Architecture | Memory Paradigm | Params | Spike Rate | SOPs / step | FLOPs / step | E_neuro (nJ/step) | E_GPU (nJ/step) | Energy Efficiency vs. GPU | Power @ 20Hz (mW) | E_cumul (H=100, μJ) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SPWM** | ALIF Spiking (2-tier) | **392.1k** | 11.9% | 8,001 | 134,144 | **12.00** | 60.36 | **5.0x** | **0.240** | **1.20** |
| **ANN-Rate** | Continuous Leaky Rate | **392.1k** | — | — | 134,144 | — | 60.36 | 1.0x (Ref) | 1.207 | 6.04 |
| **Continuous RSSM** | GRU + Gaussian Latent | **380.5k** | — | — | 730,112 | — | 328.55 | 1.0x (Ref) | 6.571 | 32.86 |
| **Deep GRU** | 2-Layer Deep GRU | **338.9k** | — | — | 599,040 | — | 269.57 | 1.0x (Ref) | 5.391 | 26.96 |
| **Continuous ConvLSTM** | Spatial ConvLSTM (2D) | **296.9k** | — | — | 4,718,592 | — | 2123.37 | 1.0x (Ref) | 42.467 | 212.34 |
