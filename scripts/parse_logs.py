import sys
import re

def parse_spwm_log(text: str) -> str:
    lines = text.splitlines()
    header_info = []
    rows = []
    
    # Static header regex patterns
    re_device = re.compile(r"Device\s*:\s*(.+)$")
    re_epochs = re.compile(r"Epochs\s*:\s*(.+)$")
    re_save = re.compile(r"Save Directory\s*:\s*(\S+)")
    
    # Epoch block patterns
    re_ep_header = re.compile(r"Epoch \[(\d+)/\d+\](?:.*?)Time:\s*([\d\.]+)s")
    
    # Training losses (with optional Critic-MSE inline)
    re_train = re.compile(
        r"Training Losses\s*:\s*Total:\s*([\d\.]+)\s*"
        r"\[q-MSE:\s*([\d\.]+)\s*\|\s*p-MSE:\s*([\d\.]+)(?:\s*\|\s*Critic-MSE:\s*([\d\.]+))?\]\s*\|"
        r"\s*Curriculum Phase\s*(\d)/\d\s*\(w_t:\s*([\d\.]+),\s*Noise α:\s*([\d\.]+),\s*DAgger:\s*(\d+)%\)\s*\|"
        r"\s*Eval:\s*Joint Angle L2:\s*([\d\.]+)\s*\|\s*Joint Vel L2:\s*([\d\.]+)"
    )
    
    # Separate Training Critic line
    re_train_critic = re.compile(r"(?:⚡\s*)?Critic Loss\s*:\s*Critic-MSE:\s*([\d\.]+)")
    
    # Closed-Loop MPPI BPTT / Replay buffer (handles both unified and separate line formats)
    re_bptt_unified = re.compile(
        r"Trained in Epoch:\s*(\d+)\s*trajectory rollouts\s*\((\d+)\s*transitions\)"
    )
    re_bptt_legacy = re.compile(
        r"(?:Trained in Epoch:\s*(\d+)\s*trajectory rollouts|Replay Buffer:\s*(\d+)\s*trajs?)\s*\((\d+)\s*transitions"
    )
    
    # Validation losses (with optional Critic-MSE inline)
    re_val = re.compile(
        r"Validation Losses.*?Total:\s*([\d\.]+)\s*"
        r"\[q-MSE:\s*([\d\.]+)\s*\|\s*p-MSE:\s*([\d\.]+)(?:\s*\|\s*Critic-MSE:\s*([\d\.]+))?\]\s*\|"
        r"\s*Eval:\s*Joint Angle L2:\s*([\d\.]+)\s*\|\s*Joint Vel L2:\s*([\d\.]+)"
    )
    
    # Separate Validation Critic line
    re_val_critic = re.compile(r"Critic Validation.*?Critic-MSE:\s*([\d\.]+)")
    
    # Tier 2 MPPI evaluation metrics
    re_tier2 = re.compile(
        r"Tier 2 \(Closed-Loop MPPI\).*?MPPI Cost:\s*(-?[\d\.]+)\s*\|\s*Mean Cartesian Err \(cm\):\s*([\d\.]+)cm\s*\|"
        r"\s*Rotational Err \(rad\):\s*([\d\.]+)\s*rad\s*\|\s*Success Rate:\s*([\d\.]*%)"
    )
    
    # Best Checkpoint (supports values or 'inf' and epoch number or 'None')
    re_best = re.compile(r"Best Checkpoint:\s*(-?[\d\.]+|inf)\s*@\s*Ep\s*(\d+|None)")

    curr_row = {}
    
    for line in lines:
        line_clean = line.strip()
        
        if m := re_device.search(line_clean):
            header_info.append(f"device={m.group(1).strip()}")
        elif m := re_epochs.search(line_clean):
            header_info.append(f"epochs={m.group(1).strip()}")
        elif m := re_save.search(line_clean):
            header_info.append(f"save_dir={m.group(1).strip()}")
            
        elif m := re_ep_header.search(line):
            if curr_row:
                rows.append(curr_row)
            curr_row = {
                "ep": int(m.group(1)),
                "time_s": m.group(2),
                "phase": "", "w_t": "", "noise": "", "dagger": "",
                "tr_tot": "", "tr_q": "", "tr_p": "", "tr_critic": "", 
                "tr_ang_l2": "", "tr_vel_l2": "",
                "bptt_ro": "0", "bptt_tr": "0",
                "val_tot": "", "val_q": "", "val_p": "", "val_critic": "", 
                "val_ang_l2": "", "val_vel_l2": "",
                "t2_cost": "", "cart_cm": "", "rot_rad": "", "succ": "",
                "best_cost": "", "best_ep": ""
            }
        elif curr_row:
            if m := re_train.search(line):
                (curr_row["tr_tot"], curr_row["tr_q"], curr_row["tr_p"],
                 critic_mse, curr_row["phase"], curr_row["w_t"], curr_row["noise"], curr_row["dagger"],
                 curr_row["tr_ang_l2"], curr_row["tr_vel_l2"]) = m.groups()
                if critic_mse:
                    curr_row["tr_critic"] = critic_mse
            elif m := re_train_critic.search(line):
                curr_row["tr_critic"] = m.group(1)
            elif m := re_bptt_unified.search(line):
                curr_row["bptt_ro"], curr_row["bptt_tr"] = m.groups()
            elif m := re_bptt_legacy.search(line):
                ro1, ro2, transitions = m.groups()
                curr_row["bptt_ro"] = ro1 if ro1 is not None else ro2
                curr_row["bptt_tr"] = transitions
            elif m := re_val.search(line):
                (curr_row["val_tot"], curr_row["val_q"], curr_row["val_p"],
                 critic_mse, curr_row["val_ang_l2"], curr_row["val_vel_l2"]) = m.groups()
                if critic_mse:
                    curr_row["val_critic"] = critic_mse
            elif m := re_val_critic.search(line):
                curr_row["val_critic"] = m.group(1)
            elif m := re_tier2.search(line):
                (curr_row["t2_cost"], curr_row["cart_cm"],
                 curr_row["rot_rad"], curr_row["succ"]) = m.groups()
            elif m := re_best.search(line):
                curr_row["best_cost"], curr_row["best_ep"] = m.groups()

    if curr_row:
        rows.append(curr_row)

    out = [f"# CONFIG: {', '.join(header_info)}"]
    out.append(
        "# COLS: ep,time_s,phase,w_t,noise,dagger_pct,tr_tot,tr_q_mse,tr_p_mse,tr_critic_mse,"
        "tr_ang_l2,tr_vel_l2,bptt_rollouts,bptt_trans,val_tot,val_q_mse,val_p_mse,val_critic_mse,"
        "val_ang_l2,val_vel_l2,t2_cost,cart_err_cm,rot_err_rad,succ_rate,best_t2_cost,best_ep"
    )
    
    for r in rows:
        out.append(
            f"{r['ep']},{r['time_s']},{r['phase']},{r['w_t']},{r['noise']},{r['dagger']},"
            f"{r['tr_tot']},{r['tr_q']},{r['tr_p']},{r['tr_critic']},"
            f"{r['tr_ang_l2']},{r['tr_vel_l2']},"
            f"{r['bptt_ro']},{r['bptt_tr']},"
            f"{r['val_tot']},{r['val_q']},{r['val_p']},{r['val_critic']},"
            f"{r['val_ang_l2']},{r['val_vel_l2']},"
            f"{r['t2_cost']},{r['cart_cm']},{r['rot_rad']},{r['succ']},"
            f"{r['best_cost']},{r['best_ep']}"
        )
    return "\n".join(out)

if __name__ == "__main__":
    raw_log = sys.stdin.read()
    print(parse_spwm_log(raw_log))
