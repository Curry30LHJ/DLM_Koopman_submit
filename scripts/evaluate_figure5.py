"""Frozen Figure 5 A/B interventions on the existing fresh simulated holdout."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from koopman_delay.evaluation import common, reproduce, figure5_protocol as protocol


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def array_sha(a):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def run(protocol_path, device, output_dir=None):
    frozen = json.loads(protocol_path.read_text())
    out = Path(output_dir).resolve() if output_dir else ROOT / frozen["result_root"]
    if (output_dir and out.exists()) or (out / "status.json").exists():
        raise FileExistsError("Choose a new output directory; frozen results cannot be overwritten")
    for path, sha in frozen["input_sha256"].items():
        if common.sha(ROOT / path) != sha:
            raise RuntimeError(f"Frozen input changed: {path}")
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    status = dict(state="running", pid=os.getpid(), started_at=now(), conditions=0, model_conditions=0, protocol_sha256=common.sha(protocol_path))
    write_json(out / "status.json", status)
    try:
        mean, std, am, ast = reproduce.scaler(ROOT, "three_tank")
        branches = []
        for row in frozen["histories"]:
            with np.load(ROOT / row["file"], allow_pickle=False) as z:
                states, actions = z["states"], z["actions"]
            env, noise, past_error = protocol.reconstruct(row["seed"], states, actions)
            branches.append((row, states, actions, env, noise, past_error))
        first = branches[0]
        fixed_req, fixed_app, fixed_clip = protocol.profiles(first[2][459:559], ast, first[3].action_low, first[3].action_high)
        shared_noise = first[4][2295:2795].copy()
        shs, ahs, future, truths, rows = [], [], [], [], []
        panel_truths = {}
        for panel in ("A", "B"):
            for hi, (row, states, actions, env, noise, past_error) in enumerate(branches):
                sh, ah = states[440:460], actions[440:459]
                if panel == "A":
                    requested, applied, clipping = protocol.profiles(actions[459:559], ast, env.action_low, env.action_high)
                    fn = noise[2295:2795].copy()
                else:
                    requested, applied, clipping = fixed_req, fixed_app, fixed_clip
                    fn = shared_noise
                condition_truth = np.asarray([protocol.branch_truth(env, a, fn) for a in applied])
                base_delta = float(np.max(np.abs(condition_truth[0]-states[460:560]))) if panel == "A" or hi == 0 else None
                if base_delta is not None and base_delta > protocol.BASE_TOLERANCE:
                    raise RuntimeError(f"Base continuation mismatch: {panel}/{hi}: {base_delta}")
                panel_truths[panel,hi] = condition_truth
                folder = out / "inputs" / panel
                folder.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(folder / f"history_{hi:03d}.npz", states_history=sh, actions_history=ah, past_actions_full=actions[:459], branch_buffer=np.asarray(env.recycle_buffer), original_process_noise=noise, future_noise=fn, requested_actions=requested, applied_actions=applied, truth_physical=condition_truth)
                for pi, name in enumerate(protocol.PROFILE_NAMES):
                    rows.append(dict(panel=panel, condition_id=f"{panel}_h{hi:02d}_p{pi:02d}", history=hi, trajectory=row["trajectory"], seed=row["seed"], window_index=440, origin=459, profile=pi, profile_name=name, clip_q1=int(clipping[pi,0]), clip_q2=int(clipping[pi,1]), clip_q3=int(clipping[pi,2]), history_sha256=array_sha(sh), past_actions_sha256=array_sha(ah), branch_buffer_sha256=array_sha(np.asarray(env.recycle_buffer)), requested_actions_sha256=array_sha(requested[pi]), applied_actions_sha256=array_sha(applied[pi]), future_noise_sha256=array_sha(fn), truth_sha256=array_sha(condition_truth[pi]), past_replay_max_abs_delta=past_error, base_future_max_abs_delta=base_delta if pi==0 else None))
                    shs.append(sh);ahs.append(ah);future.append(applied[pi]);truths.append(condition_truth[pi])
                status.update(conditions=len(rows), updated_at=now())
                write_json(out / "status.json", status)
                print(json.dumps(dict(event="conditions", panel=panel, history=hi, total=len(rows), base_delta=base_delta)), flush=True)
        assert np.array_equal(panel_truths["A",0], panel_truths["B",0])
        # Fixed variables must be identical within every panel grouping.
        for hi in range(12):
            group=[r for r in rows if r["panel"]=="A" and r["history"]==hi]
            for key in ("history_sha256", "past_actions_sha256", "branch_buffer_sha256", "future_noise_sha256"):
                assert len({r[key] for r in group})==1
        for pi in range(7):
            group=[r for r in rows if r["panel"]=="B" and r["profile"]==pi]
            for key in ("applied_actions_sha256", "future_noise_sha256"):
                assert len({r[key] for r in group})==1
        common.write_table(out / "CONDITIONS.csv", rows)
        truth = np.asarray(truths)
        if not np.isfinite(truth).all():
            raise RuntimeError("Nonfinite truth retained; no condition replacement")
        inputs = dict(states_history=(np.asarray(shs)-mean)/std, actions_history=(np.asarray(ahs)-am)/ast, future_actions=(np.asarray(future)-am)/ast)
        inputs["current_action"] = inputs["future_actions"][:,0].copy()
        assert np.array_equal(inputs["current_action"], inputs["future_actions"][:,0])
        for key, value in inputs.items():
            assert np.array_equal(value[:7], value[84:91]), key
        metrics, state_metrics, envelopes, group_envelopes, audits = [], [], [], [], []
        for modelrow in frozen["models"]:
            model = reproduce.load(ROOT, modelrow, device)
            prediction = common.inverse_scale(reproduce.predict(model, inputs, device), mean, std)
            assert np.array_equal(prediction[:7], prediction[84:91], equal_nan=True)
            np.savez_compressed(out / (modelrow["model"]+"_PREDICTIONS.npz"), prediction_physical=prediction, truth_physical=truth, condition_ids=np.asarray([r["condition_id"] for r in rows]), horizons=np.arange(1,101))
            with np.errstate(over="ignore", invalid="ignore"):
                squared=((prediction-truth)/std)**2
                ep=np.sqrt(squared.mean(axis=2))
                cu=np.sqrt(np.cumsum(squared,axis=1).mean(axis=2)/np.arange(1,101))
                se=np.sqrt(squared)
                sc=np.sqrt(np.cumsum(squared,axis=1)/np.arange(1,101)[None,:,None])
            for ci, row in enumerate(rows):
                for h in range(100):
                    metrics.append(dict(panel=row["panel"], model=modelrow["model"], condition_id=row["condition_id"], horizon=h+1, endpoint=ep[ci,h], cumulative=cu[ci,h]))
                    for c in range(9):
                        state_metrics.append(dict(panel=row["panel"], model=modelrow["model"], condition_id=row["condition_id"], component=c, horizon=h+1, endpoint=se[ci,h,c], cumulative=sc[ci,h,c]))
            for panel, offset in [("A",0),("B",84)]:
                groups=[("all_conditions",list(range(offset,offset+84)))]
                groups += [(f"history_{hi:02d}",list(range(hi*7,hi*7+7))) for hi in range(12)] if panel=="A" else [(f"profile_{pi}",list(range(84+pi,168,7))) for pi in range(7)]
                for group, ids in groups:
                    for h in range(100):
                        record=dict(panel=panel, model=modelrow["model"], group=group, conditions=len(ids), horizon=h+1, endpoint_mean=float(np.mean(ep[ids,h])), endpoint_min=float(np.min(ep[ids,h])), endpoint_max=float(np.max(ep[ids,h])), cumulative_mean=float(np.mean(cu[ids,h])), cumulative_min=float(np.min(cu[ids,h])), cumulative_max=float(np.max(cu[ids,h])))
                        (envelopes if group=="all_conditions" else group_envelopes).append(record)
            audits.append(dict(model=modelrow["model"], conditions=168, nonfinite_prediction_values=int((~np.isfinite(prediction)).sum()), nonfinite_conditions=int((~np.isfinite(prediction).all(axis=(1,2))).sum()), fresh000_A_B_prediction_exact=True))
            status.update(model_conditions=len(audits)*168, elapsed_seconds=time.monotonic()-started, updated_at=now())
            write_json(out / "status.json", status)
            print(json.dumps(dict(event="model", **audits[-1])), flush=True)
            del model
        common.write_table(out / "PER_CONDITION_H1_H100.csv", metrics)
        common.write_table(out / "PER_STATE_H1_H100.csv", state_metrics)
        common.write_table(out / "ENVELOPE_H1_H100.csv", envelopes)
        common.write_table(out / "ENVELOPE_H1_H60.csv", [r for r in envelopes if r["horizon"]<=60])
        common.write_table(out / "SUMMARY_H20_H30_H50_H60_H100.csv", [r for r in envelopes if r["horizon"] in (20,30,50,60,100)])
        common.write_table(out / "GROUP_ENVELOPE_H1_H100.csv", group_envelopes)
        common.write_table(out / "PREDICTION_AUDIT.csv", audits)
        status.update(state="completed", finished_at=now(), elapsed_seconds=time.monotonic()-started, nonfinite_conditions=sum(r["nonfinite_conditions"] for r in audits))
        write_json(out / "status.json", status)
        write_json(out / "COMPLETE.json", status)
    except Exception as exc:
        status.update(state="failed", error=repr(exc), updated_at=now())
        write_json(out / "status.json", status)
        raise


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--device",default="cpu")
    args=parser.parse_args()
    run(args.protocol,args.device,args.output_dir)
