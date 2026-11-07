#!/usr/bin/env python3
"""Fair retrospective PGCB v1/v2 comparison on identical forecast origin windows.

No retraining or modification of inputs. The two native-unit error series are
computed from separately saved predictions with each run's train-only scaler.
Use only after *both* runs have completed successfully.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

MODELS = ('Linear', 'NLinear', 'DLinear', 'SeasonalNaive24', 'SeasonalNaive168')
PRED_LEN = 96


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def load_run(project, tag):
    root = project / 'experiments/new_data' / ('pgcb_valid_windows_' + tag)
    cfg_path = root/'protocol.json'
    cfg = json.loads(cfg_path.read_text())
    if cfg['features'] != ['demand_mw','generation_mw'] or cfg['seq_len'] != 336 or cfg['pred_len'] != PRED_LEN:
        raise RuntimeError(tag + ': incompatible protocol')
    scale = np.asarray(cfg['train_only_scaler_scale'], dtype=float)
    mu = np.asarray(cfg['train_only_scaler_mean'], dtype=float)
    with np.load(root/'test_reference.npz') as z:
        starts = z['starts'].astype(np.int64).copy()
        reference = z['truth_scaled'].astype(np.float64).copy()
        seasonal24 = z['seasonal24_scaled'].astype(np.float64).copy()
        seasonal168 = z['seasonal168_scaled'].astype(np.float64).copy()
    if len(starts)!=len(set(starts)) or len(starts)==0 or not np.all(np.diff(starts)>0):
        raise RuntimeError(tag + ': duplicate, unsorted or missing test starts')
    if reference.shape != (len(starts), PRED_LEN, 2):
        raise RuntimeError(tag + ': wrong truth shape')
    preds = {'SeasonalNaive24':seasonal24, 'SeasonalNaive168':seasonal168}
    for name in MODELS[:3]:
        with np.load(root/(name+'_test_predictions.npz')) as z:
            p=z['prediction_scaled'].astype(np.float64).copy()
            truth=z['truth_scaled'].astype(np.float64)
        if p.shape != reference.shape or not np.allclose(truth, reference, atol=1e-6,rtol=0):
            raise RuntimeError(tag + ': inconsistent '+name+' test prediction arrays')
        preds[name]=p
    return dict(tag=tag,root=root,config=cfg,config_path=cfg_path,
                starts=starts,reference=reference,preds=preds,mu=mu,scale=scale)


def score(pred, truth, scale):
    # Inverting either StandardScaler cancels the mean for prediction errors.
    delta = (pred[:,:,0]-truth[:,:,0])*scale[0]
    if not np.isfinite(delta).all():
        raise RuntimeError('Non-finite demand prediction errors')
    return {'demand_mae_mw':float(np.mean(np.abs(delta))),
            'demand_mse_mw2':float(np.mean(delta**2)),
            'n_target_values':int(delta.size)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,default=Path.home()/'Documents/COMP8240-LTSF-Linear-Reproduction')
    p.add_argument('--out-prefix',type=Path,default=Path('results/new_data/pgcb_v1_v2_common_window_v1'))
    a=p.parse_args()
    project=a.project.resolve()
    runs={x:load_run(project,x) for x in ('v1','v2')}
    one,two=runs['v1'],runs['v2']
    common,i1,i2=np.intersect1d(one['starts'],two['starts'],return_indices=True)
    if len(common)==0:
        raise RuntimeError('No matching v1/v2 forecast origins')
    if len(common)!=len(two['starts']):
        raise RuntimeError('Screened v2 windows are not a subset of v1 windows')
    original_truth=one['reference'][i1,:,:]*one['scale'][None,None,:]+one['mu'][None,None,:]
    screened_truth=two['reference'][i2,:,:]*two['scale'][None,None,:]+two['mu'][None,None,:]
    truth_max_difference=float(np.max(np.abs(original_truth-screened_truth)))
    if truth_max_difference > .10:
        raise RuntimeError('Physical target series differs on matched windows: max '+str(truth_max_difference))
    rows=[]
    for name in MODELS:
        for tag,ids in (('v1',i1),('v2',i2)):
            r=runs[tag]
            d=score(r['preds'][name][ids],r['reference'][ids],r['scale'])
            rows.append(dict(model=name,run=tag,population='same_v1_v2_test_origins',
                             n_windows=len(common),**d))
    for name in MODELS:
        v1=next(x for x in rows if x['model']==name and x['run']=='v1')
        v2=next(x for x in rows if x['model']==name and x['run']=='v2')
        print('{:<18} matched windows={} v1 MAE={:.2f} v2 MAE={:.2f} v1 MSE={:.1f} v2 MSE={:.1f}'.format(
            name,len(common),v1['demand_mae_mw'],v2['demand_mae_mw'],v1['demand_mse_mw2'],v2['demand_mse_mw2']))
    prefix=project/a.out_prefix
    prefix.parent.mkdir(parents=True,exist_ok=True)
    csvpath=prefix.with_suffix('.csv'); jsonpath=prefix.with_suffix('.json')
    if csvpath.exists() or jsonpath.exists():
        raise RuntimeError('Existing matched-window audit detected; preserve it: '+str(prefix))
    with csvpath.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    manifest=dict(v1_total_windows=len(one['starts']), v2_total_windows=len(two['starts']),
                  common_windows=len(common),common_target_points=len(common)*PRED_LEN,
                  max_native_truth_difference_mw=truth_max_difference,
                  matched_window_first_index=int(common[0]),matched_window_last_index=int(common[-1]),
                  v1_protocol_sha256=sha(one['config_path']),v2_protocol_sha256=sha(two['config_path']),
                  methods=MODELS,warning='Post-hoc screening and selected temporal windows; not independent trials or causal evidence.',
                  comparison_csv=str(csvpath.relative_to(project)))
    jsonpath.write_text(json.dumps(manifest,indent=2)+'\n')
    print('PGCB COMMON-WINDOW COMPARISON: PASS',len(common),'windows; max physical truth delta MW',truth_max_difference)
    print('OUTPUTS',csvpath,jsonpath)


if __name__=='__main__': main()
