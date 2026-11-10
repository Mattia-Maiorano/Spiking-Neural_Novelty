import sys
import json

def rf(v, d=4):
    """Arrotonda float rimuovendo zeri inutili."""
    if isinstance(v, float):
        return f"{v:.{d}g}"
    return str(v)

def format_table(rows, headers, delim="|"):
    """Genera una stringa tabellare senza overhead sintattico."""
    out = [delim.join(headers)]
    for r in rows:
        out.append(delim.join(rf(r.get(h, "")) for h in headers))
    return "\n".join(out)

def main():
    out_file = sys.argv[1] if len(sys.argv) > 1 else "output_compatto.txt"

    if sys.stdin.isatty():
        sys.stderr.write("Incolla il JSON, premi INVIO e poi Ctrl+D (Ctrl+Z su Win):\n\n")

    raw = sys.stdin.read().strip()
    if not raw:
        sys.stderr.write("Nessun dato in input.\n")
        return

    data = json.loads(raw)
    lines = []

    # 1. Metadati Globali
    lines.append("=== META ===")
    lines.append(f"ver:{data.get('version')} | model:{data.get('model_name')} | ep:{data.get('total_epochs')} | params:{data.get('parameters')}")
    
    # 2. Evaluation / Horizons Degradation
    ev = data.get("evaluation", {})
    if ev:
        lines.append("\n=== HORIZONS (H|latent_mse|pos_err|vel_err|q_mse|p_mse) ===")
        horizons = ev.get("horizons", [])
        lat = ev.get("latent_mse_per_horizon", {})
        pos = ev.get("position_error_per_horizon", {})
        vel = ev.get("velocity_error_per_horizon", {})
        q_m = ev.get("q_mse_per_horizon", {})
        p_m = ev.get("p_mse_per_horizon", {})
        
        lines.append("h|lat_mse|pos_err|vel_err|q_mse|p_mse")
        for h in horizons:
            sh = str(h)
            lines.append(f"{h}|{rf(lat.get(sh))}|{rf(pos.get(sh))}|{rf(vel.get(sh))}|{rf(q_m.get(sh))}|{rf(p_m.get(sh))}")

    # 3. Training Segments
    tp = data.get("training_progression", {})
    segs = tp.get("sampled_segments", [])
    if segs:
        lines.append("\n=== SEGMENTS ===")
        cols = ["epoch", "train_loss", "val_loss", "train_pos_err", "val_pos_err", "score", "train_mppi_noise_alpha"]
        lines.append(format_table(segs, cols))

    # 4. Evaluations History (MPPI)
    mp = data.get("mppi_progression", {})
    evals = mp.get("evaluations_history", [])
    if evals:
        lines.append("\n=== EVAL_HISTORY ===")
        cols = ["epoch", "mppi_score", "mean_cartesian_error_cm", "grasp_error_cm", "terminal_error_cm", "natural_blocks"]
        # Convertiamo eventuali liste interne in formato compatto (es. Low,Low)
        clean_evals = []
        for e in evals:
            row = dict(e)
            if isinstance(row.get("natural_blocks"), list):
                row["natural_blocks"] = ",".join(row["natural_blocks"])
            clean_evals.append(row)
        lines.append(format_table(clean_evals, cols))

    # 5. Final Trials (il blocco più pesante)
    trials = mp.get("expert_execution", {}).get("final_trials", [])
    if trials:
        lines.append("\n=== FINAL_TRIALS ===")
        cols = ["trajectory_id", "elevation_plane", "is_expert", "score", "mean_cartesian_err_cm", "terminal_error_cm", "grasp_error_cm"]
        lines.append(format_table(trials, cols))

    result = "\n".join(lines)

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(result)

    sys.stderr.write(f"\nFatto! Salvato come testo tabellare ultra-compresso in: {out_file}\n")

if __name__ == "__main__":
    main()
