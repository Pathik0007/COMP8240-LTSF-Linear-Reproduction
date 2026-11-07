#!/usr/bin/env python3
"""Post-run audit for COMP8240 PGCB experiment; no retraining or changes to metrics.
Prints split-level extreme observations and independently verifies saved test scores.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path.home() / 'Documents/COMP8240-LTSF-Linear-Reproduction'
OUT = ROOT / 'experiments/new_data/pgcb_valid_windows_v1'
RAW = ROOT / 'data/pgcb/processed/pgcb_hourly_core_audited.csv'

cfg = json.loads((OUT / 'protocol.json').read_text())
mu = np.asarray(cfg['train_only_scaler_mean'], dtype=np.float64)
sigma = np.asarray(cfg['train_only_scaler_scale'], dtype=np.float64)
cols = ['demand_mw', 'generation_mw']
assert cfg['features'] == cols

frame = pd.read_csv(RAW, usecols=['datetime', 'was_observed_hour',
    'demand_mw', 'generation_mw', 'demand_mw_scale_anomaly',
    'generation_mw_scale_anomaly'], parse_dates=['datetime'])
values = frame[cols].to_numpy(dtype=np.float64)
observed = frame['was_observed_hour'].astype(str).str.lower().isin(['true', '1']).to_numpy()
valid = observed & np.isfinite(values).all(axis=1)
flags = {col: frame[col + '_scale_anomaly'].astype(str).str.lower().isin(['true', '1']).to_numpy() for col in cols}
train_end = int(cfg['train_end_index'])
val_end = int(cfg['validation_end_index'])
assert len(frame) == cfg['n_grid_rows']
assert np.isfinite(sigma).all() and np.all(sigma > 0)

print('=== TRAIN-ONLY SCALER ===')
for k, name in enumerate(cols):
    print('{}: mean {:.3f}, std {:.3f}'.format(name, mu[k], sigma[k]))

print('\n=== SPLIT EXTREMES AND SCALE FLAGS ===')
for label, (start, stop) in {
    'train':(0, train_end), 'validation':(train_end, val_end),
    'test':(val_end, len(frame))}.items():
    print('\n' + label.upper() + ':', 'rows', stop-start, 'valid', int(valid[start:stop].sum()))
    for k, name in enumerate(cols):
        flagged = np.flatnonzero(flags[name][start:stop]) + start
        z = np.abs((values[start:stop, k] - mu[k]) / sigma[k])
        z[~np.isfinite(z)] = -1
        extreme_local = np.argsort(z)[-3:][::-1]
        print(' ', name, 'flagged rows', len(flagged), 'maximum |train-z| {:.2f}'.format(z.max()))
        for j in extreme_local:
            index = start + int(j)
            print('    ', str(frame['datetime'].iloc[index]), 'raw', values[index,k], '|z|', round(z[j], 2), 'flagged', bool(flags[name][index]))

print('\n=== INDEPENDENT SAVED TEST METRIC AUDIT ===')
reference = np.load(OUT / 'test_reference.npz')
truth = reference['truth_scaled'].astype(np.float64)
assert truth.shape == (8801,96,2), truth.shape
for name in ['Linear','NLinear','DLinear']:
    arr = np.load(OUT / (name + '_test_predictions.npz'))
    pred = arr['prediction_scaled'].astype(np.float64)
    saved_truth = arr['truth_scaled'].astype(np.float64)
    if pred.shape != truth.shape or not np.allclose(saved_truth,truth,rtol=0,atol=1e-6):
        raise RuntimeError(name + ': inconsistent truth or shape')
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise RuntimeError(name + ': non-finite predictions/targets')
    delta = (pred-truth) * sigma.reshape(1,1,2)
    mae = float(np.mean(np.abs(delta[:,:,0])))
    mse = float(np.mean(delta[:,:,0]**2))
    recorded = json.loads((OUT/(name+'_metrics.json')).read_text())
    if not np.isclose(mae, recorded['demand_mae_mw'],atol=0.05,rtol=0):
        raise RuntimeError(name+': MAE mismatch')
    if not np.isclose(mse, recorded['demand_mse_mw2'],atol=1.0,rtol=0):
        raise RuntimeError(name+': MSE mismatch')
    print(name, 'Demand MAE', round(mae,3), 'MSE', round(mse,3), 'PASS')

print('\n=== TARGET-WINDOW ANOMALY EXPOSURE ===')
starts = reference['starts'].astype(np.int64)
assert len(starts) == len(truth)
target_idx = starts[:,None] + 336 + np.arange(96)[None,:]
for name in cols:
    affected = np.any(flags[name][target_idx],axis=1)
    print('Test forecast windows whose targets contain flagged '+name+':',int(affected.sum()),'of',len(starts))

print('\nPOSTRUN NUMERICAL AUDIT: PASS')
print('Note: PASS means saved test metrics agree; it does not resolve validation outliers.')
