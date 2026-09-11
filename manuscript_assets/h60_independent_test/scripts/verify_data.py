"""Read-only asset, metric and median-window checks; optional frozen raw regression."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

A=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--source-root',type=Path)
args=parser.parse_args()
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf-8'))
manifest=read(A/'ARCHIVE_MANIFEST.json')
for row in manifest['files']:
    assert sha(A/row['path'])==row['sha256'],row['path']
mapping=dict(zip(['mlp_koopman','lstm_koopman','hakan_koopman','rbf_markov','rbf_physical_delay','dlm_koopman'],['mlp','lstm','hakan','m6','m8','m5']))
curve=pd.read_csv(A/'data/independent_test_H1_H60.csv')
traj=pd.read_csv(A/'data/independent_test_per_trajectory_H1_H100.csv')
assert len(curve)==720 and len(traj)==14400
assert not curve.duplicated(['system','model','horizon']).any()
assert not traj.duplicated(['system','model','trajectory','horizon']).any()
assert np.isfinite(curve[['endpoint','cumulative']]).all().all()
assert np.isfinite(traj[['endpoint','cumulative']]).all().all()
assert (traj.groupby(['system','model','horizon']).trajectory.nunique()==12).all()
means=traj.assign(model=traj.model.map(mapping)).groupby(['system','model','horizon']).endpoint.mean()
actual=curve.set_index(['system','model','horizon']).endpoint
assert np.allclose(actual,means.loc[actual.index],rtol=0,atol=1e-12)
sel=read(A/'selection/SELECTION_RESULT.json')
tc=pd.read_csv(A/'selection/trajectory_candidates.csv')
wc=pd.read_csv(A/'selection/window_candidates.csv')
def nearest(values,keys):
    distance=np.abs(values-np.quantile(values,.5,method='linear'))
    choices=np.flatnonzero(np.isclose(distance,distance.min(),rtol=1e-12,atol=1e-15))
    return min((keys[i] for i in choices))
assert len(tc)==12 and len(wc)==882
assert nearest(tc.endpoint.to_numpy(),tc.trajectory.tolist())==sel['trajectory']
assert nearest(wc.error_step60.to_numpy(),wc.origin.tolist())==sel['origin_zero_based']
raw_max=0.
source_map={r['path']:r['sha256'] for r in read(A/'SOURCE_SNAPSHOT_MAP.json')['files']}
def source(relative):
    path=args.source_root/relative
    assert sha(path)==source_map[relative],relative
    return path
for system,name,nx,trajectory,origin in [
    ('lorenz','lorenz_fixed_origin_H60.csv',3,'fresh_000',20),
    ('three_tank','figure4_median_window_H60.csv',9,sel['trajectory'],sel['origin_zero_based'])]:
    frame=pd.read_csv(A/'data'/name,float_precision='round_trip')
    assert len(frame)==6*60*nx and not frame.duplicated(['model','horizon','component']).any()
    assert set(frame.model)==set(mapping.values()) and set(frame.horizon)==set(range(1,61))
    assert set(frame.trajectory)=={trajectory} and set(frame.origin)=={origin}
    assert np.isfinite(frame[['truth','prediction']]).all().all()
    assert (frame.groupby(['horizon','component']).truth.nunique()==1).all()
    if args.source_root:
        for model,short in mapping.items():
            with np.load(source(f'evaluation/fresh_holdout/raw/{system}/{model}/{trajectory}.npz'),allow_pickle=False) as z:
                idx=np.flatnonzero(z['origins']==origin).item()
                rows=frame[frame.model==short].sort_values(['horizon','component'])
                for csv_col,raw_key in [('truth','truth_physical'),('prediction','prediction_physical')]:
                    expected=z[raw_key][idx,:60].reshape(-1)
                    delta=float(np.max(np.abs(rows[csv_col].to_numpy()-expected)))
                    raw_max=max(raw_max,delta)
                    assert delta<1e-12
if args.source_root:
    src=pd.read_csv(source('evaluation/fresh_holdout/tables/PER_TRAJECTORY_H1_H100.csv'))
    pd.testing.assert_frame_equal(traj,src)
    std=np.array(read(source('datasets/three_tank/manifest.json'))['state_std'])
    for row in tc.itertuples():
        with np.load(source(f'evaluation/fresh_holdout/raw/three_tank/dlm_koopman/{row.trajectory}.npz'),allow_pickle=False) as z:
            errors=np.linalg.norm((z['prediction_physical'][:,59].astype(np.float64)-z['truth_physical'][:,59])/std,axis=1)/3
            assert abs(np.linalg.norm(errors)/np.sqrt(882)-row.endpoint)<1e-12
            if row.trajectory==sel['trajectory']:
                assert np.allclose(errors,wc.error_step60,rtol=0,atol=1e-12)
                assert np.array_equal(z['origins'],wc.origin)
    with np.load(source(f'datasets/fresh_holdout/three_tank/{sel["trajectory"]}.npz'),allow_pickle=False) as z:
        inputs=pd.read_csv(A/'data/selected_prescribed_inputs.csv',float_precision='round_trip')
        assert np.array_equal(inputs.to_numpy(),z['actions'][sel['origin_zero_based']:sel['origin_zero_based']+60])
print(json.dumps(dict(status='PASS',asset_files=len(manifest['files']),summary_rows=len(curve),trajectory_rows=len(traj),figure2_rows=1080,figure4_rows=3240,median_trajectory=sel['trajectory'],median_origin=sel['origin_zero_based'],source_raw_checked=bool(args.source_root),raw_prediction_max_abs_delta=raw_max if args.source_root else None,training=False,inference=False,simulation=False),indent=2))
