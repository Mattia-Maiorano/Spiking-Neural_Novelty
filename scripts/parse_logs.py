import sys
import re

def parse_spwm_log(text: str) -> str:
    lines = text.splitlines()
    header_info = []
    rows = []
    
    # Static header regex
    re_device = re.compile(r"Device\s*:\s*(\w+)")
    re_epochs = re.compile(r"Epochs\s*:\s*(\S+)")
    re_save = re.compile(r"Save Directory\s*:\s*(\S+)")
    
    # Epoch block patterns
    re_ep_header = re.compile(r"Epoch \[(\d+)/\d+\](?:.*?)Time:\s*([\d\.]+)s")
    re_train = re.compile(
        r"Training Losses\s*:\s*Total:\s*([\d\.]+)\s*\[q-MSE:\s*([\d\.]+)\s*\|\s*p-MSE:\s*([\d\.]+)\]\s*\|"
        r"\s*Curriculum Phase\s*(\d)/\d\s*\(w_t:\s*([\d\.]+),\s*Noise α:\s*([\d\.]+),\s*DAgger:\s*(\d+)%\)\s*\|"
        r"\s*Eval:\s*Joint Angle L2:\s*([\d\.]+)\s*\|\s*Joint Vel L2:\s*([\d\.]+)"
    )
    re_bptt = re.compile(r"Trained in Epoch:\s*(\d+)\s*trajectory rollouts\s*\((\d+)\s*transitions\)")
    re_val = re.compile(
        r"Validation Losses.*?Total:\s*([\d\.]+)\s*\[q-MSE:\s*([\d\.]+)\s*\|\s*p-MSE:\s*([\d\.]+)\]\s*\|"
        r"\s*Eval:\s*Joint Angle L2:\s*([\d\.]+)\s*\|\s*Joint Vel L2:\s*([\d\.]+)"
    )
    re_tier2 = re.compile(
        r"Tier 2 \(Closed-Loop MPPI\).*?MPPI Cost:\s*(-?[\d\.]+)\s*\|\s*Mean Cartesian Err \(cm\):\s*([\d\.]+)cm\s*\|"
        r"\s*Rotational Err \(rad\):\s*([\d\.]+)\s*rad\s*\|\s*Success Rate:\s*([\d\.]*%)"
    )
    re_best = re.compile(r"Best Checkpoint:\s*(-?[\d\.]+)\s*@\s*Ep\s*(\d+)")

    curr_row = {}
    
    for line in lines:
        if m := re_device.search(line):
            header_info.append(f"device={m.group(1)}")
        elif m := re_save.search(line):
            header_info.append(f"save_dir={m.group(1)}")
            
        elif m := re_ep_header.search(line):
            if curr_row:
                rows.append(curr_row)
            curr_row = {
                "ep": int(m.group(1)),
                "time_s": m.group(2),
                "phase": "", "w_t": "", "noise": "", "dagger": "",
                "tr_tot": "", "tr_q": "", "tr_p": "", "tr_ang_l2": "", "tr_vel_l2": "",
                "bptt_ro": "0", "bptt_tr": "0",
                "val_tot": "", "val_q": "", "val_p": "", "val_ang_l2": "", "val_vel_l2": "",
                "t2_cost": "", "cart_cm": "", "rot_rad": "", "succ": "",
                "best_cost": "", "best_ep": ""
            }
        elif curr_row:
            if m := re_train.search(line):
                (curr_row["tr_tot"], curr_row["tr_q"], curr_row["tr_p"],
                 curr_row["phase"], curr_row["w_t"], curr_row["noise"], curr_row["dagger"],
                 curr_row["tr_ang_l2"], curr_row["tr_vel_l2"]) = m.groups()
            elif m := re_bptt.search(line):
                curr_row["bptt_ro"], curr_row["bptt_tr"] = m.groups()
            elif m := re_val.search(line):
                (curr_row["val_tot"], curr_row["val_q"], curr_row["val_p"],
                 curr_row["val_ang_l2"], curr_row["val_vel_l2"]) = m.groups()
            elif m := re_tier2.search(line):
                (curr_row["t2_cost"], curr_row["cart_cm"],
                 curr_row["rot_rad"], curr_row["succ"]) = m.groups()
            elif m := re_best.search(line):
                curr_row["best_cost"], curr_row["best_ep"] = m.groups()

    if curr_row:
        rows.append(curr_row)

    out = [f"# CONFIG: {', '.join(header_info)}"]
    out.append("# COLS: ep,time_s,phase,w_t,noise,dagger_pct,tr_tot,tr_q_mse,tr_p_mse,tr_ang_l2,tr_vel_l2,bptt_rollouts,bptt_trans,val_tot,val_q_mse,val_p_mse,val_ang_l2,val_vel_l2,t2_cost,cart_err_cm,rot_err_rad,succ_rate,best_t2_cost,best_ep")
    
    for r in rows:
        out.append(
            f"{r['ep']},{r['time_s']},{r['phase']},{r['w_t']},{r['noise']},{r['dagger']},"
            f"{r['tr_tot']},{r['tr_q']},{r['tr_p']},{r['tr_ang_l2']},{r['tr_vel_l2']},"
            f"{r['bptt_ro']},{r['bptt_tr']},{r['val_tot']},{r['val_q']},{r['val_p']},"
            f"{r['val_ang_l2']},{r['val_vel_l2']},{r['t2_cost']},{r['cart_cm']},{r['rot_rad']},"
            f"{r['succ']},{r['best_cost']},{r['best_ep']}"
        )
    return "\n".join(out)

if __name__ == "__main__":
    raw_log = sys.stdin.read()
    print(parse_spwm_log(raw_log))
