"""Read-only Figure 5 archive and independent metric/condition verification."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"evaluation/figure5"


def sha(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def verify():
    frozen=json.loads((OUT/"PROTOCOL.json").read_text())
    complete=json.loads((OUT/"COMPLETE.json").read_text())
    assert hashlib.sha256((OUT/"PROTOCOL.json").read_bytes()).hexdigest()==complete["protocol_sha256"]
    for path,digest in frozen["input_sha256"].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    rows=list(csv.DictReader((OUT/"CONDITIONS.csv").open()))
    assert len(rows)==168
    for panel in ("A","B"):
        for hi in range(12):
            with np.load(OUT/"inputs"/panel/f"history_{hi:03d}.npz",allow_pickle=False) as z:
                assert z["branch_buffer"].shape==(26,9)
                for pi in range(7):
                    row=rows[(0 if panel=="A" else 84)+hi*7+pi]
                    for key,value in [("history_sha256",z["states_history"]),("past_actions_sha256",z["actions_history"]),("branch_buffer_sha256",z["branch_buffer"]),("future_noise_sha256",z["future_noise"]),("applied_actions_sha256",z["applied_actions"][pi]),("requested_actions_sha256",z["requested_actions"][pi]),("truth_sha256",z["truth_physical"][pi])]:
                        assert sha(value)==row[key]
                if panel=="A":
                    with np.load(ROOT/frozen["histories"][hi]["file"],allow_pickle=False) as source:
                        assert np.max(np.abs(z["truth_physical"][0]-source["states"][460:560]))<=1e-10
                if hi==0:
                    with np.load(OUT/"inputs"/"A"/"history_000.npz",allow_pickle=False) as a:
                        for key in z.files:
                            assert np.array_equal(z[key],a[key]),key
    for pi in range(7):
        b=[r for r in rows if r["panel"]=="B" and int(r["profile"])==pi]
        assert len({r["applied_actions_sha256"] for r in b})==1
        assert len({r["future_noise_sha256"] for r in b})==1
    std=np.array(json.loads((ROOT/"datasets/three_tank/manifest.json").read_text())["state_std"])
    summaries=list(csv.DictReader((OUT/"SUMMARY_H20_H30_H50_H60_H100.csv").open()))
    maximum=0.0
    for model in frozen["models"]:
        name=model["model"]
        with np.load(OUT/(name+"_PREDICTIONS.npz"),allow_pickle=False) as z:
            prediction,truth=z["prediction_physical"],z["truth_physical"]
            assert prediction.shape==truth.shape==(168,100,9)
            assert np.array_equal(prediction[:7],prediction[84:91],equal_nan=True)
            assert list(z["condition_ids"])==[r["condition_id"] for r in rows]
        error=(prediction-truth)/std
        assert np.isfinite(error).all()
        for row in [r for r in summaries if r["model"]==name]:
            h=int(row["horizon"]);part=error[:84] if row["panel"]=="A" else error[84:]
            endpoint=np.linalg.norm(part[:,h-1,:],axis=1)/3
            cumulative=np.sqrt(np.einsum("ijk,ijk->i",part[:,:h],part[:,:h])/(h*9))
            for prefix,values in [("endpoint",endpoint),("cumulative",cumulative)]:
                for suffix,fn in [("mean",np.mean),("min",np.min),("max",np.max)]:
                    expected=float(fn(values));actual=float(row[prefix+"_"+suffix]);assert np.isclose(actual,expected,rtol=1e-12,atol=1e-12);maximum=max(maximum,abs(actual-expected))
    result=dict(status="PASS",conditions=168,model_conditions=1008,full_branch_buffers=24,summary_rows=len(summaries),independent_metric_max_abs_delta=maximum,fresh000_A_B_exact=True,nonfinite_conditions=0)
    print(json.dumps(result,indent=2))
    return result


if __name__=="__main__":
    verify()
