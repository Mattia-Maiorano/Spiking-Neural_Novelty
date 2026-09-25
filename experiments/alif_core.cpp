// alif_core.cpp
#include <torch/extension.h>
#include <vector>

// Forward pass fuso per un neurone ALIF con calcolo delle tracce e-prop
std::vector<torch::Tensor>
alif_eprop_forward(torch::Tensor x_seq, // Input correnti [batch, time, neurons]
                   float beta_mem,      // Decadimento potenziale veloce
                   float beta_adapt,    // Decadimento soglia lento
                   float gamma,         // Incremento soglia dopo spike
                   float v_th_rest      // Soglia base a riposo
) {
  auto batch_size = x_seq.size(0);
  auto seq_len = x_seq.size(1);
  auto num_neurons = x_seq.size(2);
  auto options = x_seq.options();

  // Inizializzazione degli stati (allocati direttamente in memoria contigua
  // C++)
  auto v = torch::zeros({batch_size, num_neurons}, options);
  auto a = torch::zeros({batch_size, num_neurons}, options);
  auto e_v =
      torch::zeros({batch_size, num_neurons}, options); // Traccia potenziale
  auto e_a = torch::zeros({batch_size, num_neurons}, options); // Traccia soglia

  // Tensori per salvare l'output della sequenza
  auto spikes_seq = torch::zeros_like(x_seq);
  auto traces_seq = torch::zeros_like(x_seq);

  // Loop temporale fuso in C++ (Zero overhead Python)
  for (int t = 0; t < seq_len; ++t) {
    auto x_t = x_seq.select(1, t);

    // 1. Aggiornamento Dinamica ALIF
    auto v_th_t = v_th_rest + gamma * a;
    v = beta_mem * v + x_t -
        v_th_rest * spikes_seq.select(1, std::max(0, t - 1));
    a = beta_adapt * a + spikes_seq.select(1, std::max(0, t - 1));

    // 2. Emissione Spike (Heaviside standard, surrogato omesso per semplicità)
    auto spikes_t = (v > v_th_t).to(v.dtype());
    spikes_seq.select(1, t).copy_(spikes_t);

    // 3. Calcolo Tracce e-prop (Forward-Only)
    // La traccia accumula l'input in ingresso decadendo con le costanti fisiche
    e_v = beta_mem * e_v + x_t;
    e_a = beta_adapt * e_a + spikes_t;

    // Salviamo una traccia combinata (pseudocodice dipendente dalla regola
    // specifica)
    traces_seq.select(1, t).copy_(e_v + e_a);
  }

  return {spikes_seq, traces_seq};
}

// Binding (Esporta la funzione C++ in Python)
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &alif_eprop_forward, "ALIF e-prop fused forward step (C++)");
}
