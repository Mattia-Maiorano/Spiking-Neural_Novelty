| Model Architecture | Memory Paradigm | Params | Spike Rate | SOPs / step | FLOPs / step | E_neuro (nJ/step) | E_GPU (nJ/step) | Energy Efficiency vs. GPU | Power @ 20Hz (mW) | E_cumul (H=100, μJ) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SPWM-v0.7.3 (Prod)** | ALIF Spiking (2-tier) | **328.7k** | 14.5% | 14,208 | 258,796 | **12.81** | 1035.18 | **80.8x** | **0.256** | **1.28** |
| **SNN-ALIF (Matched)** | ALIF Spiking (2-tier) | **328.7k** | 24.8% | 24,192 | 258,796 | **21.80** | 1035.18 | **47.5x** | **0.436** | **2.18** |
| **ANN-Rate (Ablation)** | Continuous Leaky Rate | **328.7k** | — | — | 258,796 | — | 1035.18 | 1.0x (Ref) | 20.704 | 103.52 |
| **Continuous RSSM** | GRU + Gaussian Latent | **380.5k** | — | — | 615,244 | — | 2460.98 | 1.0x (Ref) | 49.220 | 246.10 |
| **Deep GRU** | 2-Layer Deep GRU | **338.9k** | — | — | 506,772 | — | 2027.09 | 1.0x (Ref) | 40.542 | 202.71 |
| **Continuous ConvLSTM** | Spatial ConvLSTM (2D) | **296.9k** | — | — | 29,001,740 | — | 116006.96 | 1.0x (Ref) | 2320.139 | 11600.70 |
