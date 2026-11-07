#!/usr/bin/env python3
"""V2: blinded synthetic + clean-window PGCB judge cases, no v1 overwrite.

Reuses the already-installed v1 synthetic visualization code. On PGCB, use
both Linear and DLinear against the stronger 24h seasonal baseline, with
identical, evenly spaced clean full windows. Export a public, label-free
case manifest for a Vast.ai GPU; keep gold metrics locally in cases.jsonl.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

L, H = 336, 96


def load_v1(source_path):
    if not source_path.is_file():
        raise RuntimeError('Original seven-file judge kit missing: ' + str(source_path))
    spec = importlib.util.spec_from_file_location('forecast_judge_prepare_v1', str(source_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def add_pgcb_clean_cases(project, out, rows, mk):
    pgcb = project/'experiments/new_data/pgcb_valid_windows_v1'
    cfg = json.loads((pgcb/'protocol.json').read_text())
    if cfg['features'] != ['demand_mw','generation_mw']:
        raise RuntimeError('Unexpected PGCB channels')
    sigma = np.asarray(cfg['train_only_scaler_scale'], dtype=np.float64)
    mean = np.asarray(cfg['train_only_scaler_mean'], dtype=np.float64)
    with np.load(pgcb/'test_reference.npz') as ref:
        truth_scaled = ref['truth_scaled'].astype(np.float64)
        seasonal = ref['seasonal24_scaled'].astype(np.float64)
        starts = ref['starts'].astype(np.int64)
    if truth_scaled.shape != (8801,H,2) or seasonal.shape != truth_scaled.shape:
        raise RuntimeError('PGCB reference shape unexpected')
    learned = {}
    for name in ('Linear','DLinear'):
        with np.load(pgcb/(name+'_test_predictions.npz')) as f:
            learned[name] = f['prediction_scaled'].astype(np.float64)
            other_truth = f['truth_scaled']
        if learned[name].shape != truth_scaled.shape or not np.allclose(other_truth,truth_scaled,rtol=0,atol=1e-6):
            raise RuntimeError('PGCB predictions or ground truth inconsistent: '+name)
    df = pd.read_csv(project/'data/pgcb/processed/pgcb_hourly_core_audited.csv',
                     usecols=['datetime','was_observed_hour', 'demand_mw',
                              'demand_mw_scale_anomaly','generation_mw_scale_anomaly'],
                     parse_dates=['datetime'])
    seen = df['was_observed_hour'].astype(str).str.lower().isin(('true','1')).to_numpy()
    flags = (df['demand_mw_scale_anomaly'].astype(str).str.lower().isin(('true','1')).to_numpy() |
             df['generation_mw_scale_anomaly'].astype(str).str.lower().isin(('true','1')).to_numpy())
    invalid = ~seen | flags | ~np.isfinite(df['demand_mw'].to_numpy(float))
    cumulative = np.r_[0,np.cumsum(invalid,dtype=np.int64)]
    eligible = (cumulative[starts+L+H]-cumulative[starts])==0
    indices = np.flatnonzero(eligible)
    if len(indices) < 12:
        raise RuntimeError('Fewer than 12 eligible uncontaminated PGCB test windows')
    chosen = indices[np.linspace(0,len(indices)-1,12,dtype=np.int64)]
    for name in ('Linear','DLinear'):
        for j, idx in enumerate(chosen):
            start = int(starts[idx]); origin = start+L
            history = df['demand_mw'].iloc[origin-48:origin].to_numpy(float)
            future = truth_scaled[idx,:,0]*sigma[0]+mean[0]
            model = learned[name][idx,:,0]*sigma[0]+mean[0]
            naive = seasonal[idx,:,0]*sigma[0]+mean[0]
            raw_future = df['demand_mw'].iloc[origin:origin+H].to_numpy(float)
            if not np.allclose(future,raw_future,atol=.02,rtol=1e-5):
                raise RuntimeError('PGCB inverse-scale truth mismatch: '+str(idx))
            if not all(np.isfinite(x).all() for x in (history,future,model,naive)):
                raise RuntimeError('Non-finite PGCB chart values')
            mk.add_pair(rows,out,'pgcb_group{}_{}'.format(1 if name == 'Linear' else 2, j),
                        'pgcb','real_electricity_demand','demand_mw',
                        history, future,model,naive,(name,'SeasonalNaive24'),
                        timestamp=str(df['datetime'].iloc[origin]),window_index=int(idx))
    print('PGCB clean test windows available:', len(indices), flush=True)
    print('PGCB comparisons created: 24 underlying (12 Linear + 12 DLinear)', flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--project',type=Path,
                   default=Path.home()/'Documents/COMP8240-LTSF-Linear-Reproduction')
    p.add_argument('--out',type=Path,default=None)
    a=p.parse_args()
    project = a.project.resolve()
    out = (a.out or project/'experiments/new_data/vlm_judge_v3').resolve()
    # Preserve any existing judge study; in particular do not modify v1.
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('Judge v2 output already exists; preserving it: '+str(out))
    out.mkdir(parents=True,exist_ok=True)
    mk = load_v1(project/'tools/vlm_judge/prepare_cases.py')
    records=[]
    mk.synthetic_cases(project,out,records)
    add_pgcb_clean_cases(project,out,records,mk)
    if len(records)!=144:
        raise RuntimeError('Expected exactly 144 A/B panels, found '+str(len(records)))
    manifest = out/'cases.jsonl'
    public = out/'public_cases.jsonl'
    with manifest.open('w',encoding='utf-8') as f, public.open('w',encoding='utf-8') as g:
        for rec in records:
            f.write(json.dumps(rec,sort_keys=True)+'\n')
            # NEVER export the gold judgments, model names or error metrics to GPU.
            blinded={k:rec[k] for k in ('id','base_id','variant','image','image_sha256')}
            g.write(json.dumps(blinded,sort_keys=True)+'\n')
    print('\n=== VLM JUDGE V3 CASES READY ===')
    print('Synthetic underlying comparisons: 48 (8 conditions x 6)')
    print('PGCB underlying comparisons: 24 (12 per trained model)')
    print('A/B + B/A panels: 144')
    print('PRIVATE labeled manifest (KEEP ON MAC):',manifest)
    print('PUBLIC blind manifest (GPU transfer):',public)
    print('VLM CASE PREPARATION: PASS')

if __name__=='__main__':
    main()
