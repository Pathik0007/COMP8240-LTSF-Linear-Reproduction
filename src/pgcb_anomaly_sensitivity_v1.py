#!/usr/bin/env python3
"""Post-hoc PGCB anomaly sensitivity analysis, without retraining or modifying v1.

Recomputes all 6 forecasting methods from saved test outputs under three
explicitly defined evaluation populations; measures anomalous validation
observation exposure using a 24-h seasonal-naive diagnostic only.
Python >=3.9; numpy, pandas. Does not edit original training outputs.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

L, H = 336, 96
METHODS = ('Linear', 'NLinear', 'DLinear', 'Repeat', 'SeasonalNaive24', 'SeasonalNaive168')
FEATURES = ('demand_mw', 'generation_mw')


def sha256(p):
    d = hashlib.sha256()
    with p.open('rb') as fh:
        for piece in iter(lambda: fh.read(1024 * 1024), b''):
            d.update(piece)
    return d.hexdigest()


def valid_starts(valid, a, b):
    candidates = np.arange(a, b-L-H+1, dtype=np.int64)
    cumulative = np.r_[0, np.cumsum(valid, dtype=np.int64)]
    count = cumulative[candidates+L+H] - cumulative[candidates]
    return candidates[count == L+H]


def metrics(pred_native, truth_native, target_mask):
    err = (pred_native-truth_native).astype(np.float64)
    arr = err[target_mask]
    if not arr.size:
        raise RuntimeError('Empty evaluation mask')
    return float(np.mean(np.abs(arr))), float(np.mean(np.square(arr))), int(arr.size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path,
                        default=Path.home()/'Documents/COMP8240-LTSF-Linear-Reproduction')
    args = parser.parse_args()
    root = args.project.resolve()
    out = root/'experiments/new_data/pgcb_valid_windows_v1'
    csv_data = root/'data/pgcb/processed/pgcb_hourly_core_audited.csv'
    results = root/'results/new_data/pgcb_anomaly_sensitivity_v1.csv'
    details = root/'experiments/new_data/pgcb_anomaly_sensitivity_v1/summary.json'
    if results.exists() or details.exists():
        raise RuntimeError('Sensitivity results already exist; preserving them (do not overwrite).')

    cfg = json.loads((out/'protocol.json').read_text())
    if sha256(csv_data) != cfg['pgcb_sha256']:
        raise RuntimeError('PGCB dataset does not match source used in v1 benchmark')
    if list(cfg['features']) != list(FEATURES):
        raise RuntimeError('Unexpected model feature list')
    frame = pd.read_csv(csv_data, usecols=['datetime','was_observed_hour',
                       'demand_mw','generation_mw',
                       'demand_mw_scale_anomaly', 'generation_mw_scale_anomaly'],
                       parse_dates=['datetime'])
    if len(frame) != cfg['n_grid_rows']:
        raise RuntimeError('Hourly grid length changed')
    raw = frame[list(FEATURES)].to_numpy(dtype=np.float64)
    observed = frame['was_observed_hour'].astype(str).str.lower().isin(('true', '1')).to_numpy()
    valid = observed & np.isfinite(raw).all(axis=1)
    flags = np.stack([
        frame[c+'_scale_anomaly'].astype(str).str.lower().isin(('true', '1')).to_numpy()
        for c in FEATURES
    ], axis=1)
    anyflag = flags.any(axis=1)
    mu = np.asarray(cfg['train_only_scaler_mean'], dtype=np.float64)
    sigma = np.asarray(cfg['train_only_scaler_scale'], dtype=np.float64)
    if not np.isfinite(sigma).all() or not (sigma > 0).all():
        raise RuntimeError('Invalid scaler')
    if np.any(flags[~observed]):
        print('NOTE: Some anomaly flags also appear on unobserved hours')

    reference = np.load(out/'test_reference.npz')
    starts = reference['starts'].astype(np.int64)
    truth_scaled = reference['truth_scaled'].astype(np.float64)
    if truth_scaled.shape != (8801, H, 2) or starts.shape != (8801,):
        raise RuntimeError('Unexpected test shapes')
    valend = int(cfg['validation_end_index'])
    if not np.all(starts + L >= valend):
        raise RuntimeError('Test target boundary leakage')
    if not np.array_equal(valid_starts(valid, valend-L, len(frame)), starts):
        raise RuntimeError('Saved test window locations disagree with independent reconstruction')
    target_ix = starts[:,None]+L+np.arange(H, dtype=np.int64)[None,:]
    truth_native = truth_scaled*sigma + mu
    if not np.allclose(truth_native, raw[target_ix], atol=.02, rtol=1e-5):
        raise RuntimeError('Inverse-scaled targets disagree with raw PGCB observed data')
    # Pre-declared sensitivity subsets: (i) all observed points,
    # (ii) exclude *flagged demand target points only*,
    # (iii) exclude entire windows if either series is flagged in input or target.
    mask_all = np.ones((len(starts), H), dtype=bool)
    mask_clean_targets = ~flags[target_ix, 0]
    cumulative = np.r_[0, np.cumsum(anyflag.astype(np.int64))]
    clean_window = (cumulative[starts+L+H]-cumulative[starts] == 0)
    mask_clean_windows = np.broadcast_to(clean_window[:,None], mask_all.shape)
    if not clean_window.any():
        raise RuntimeError('No clean complete test windows')
    print('=== PGCB SENSITIVITY POPULATIONS ===')
    print('Test windows:', len(starts), 'hours/target:', H)
    print('Test target points:', int(mask_all.sum()))
    print('Flagged demand target points:', int((~mask_clean_targets).sum()))
    print('Test windows with flagged demand targets:', int((~mask_clean_targets.all(axis=1)).sum()))
    print('Clean input+target windows, BOTH channels:', int(clean_window.sum()))

    preds = {}
    for name in ('Linear', 'NLinear', 'DLinear'):
        with np.load(out/(name+'_test_predictions.npz')) as data:
            p = data['prediction_scaled'].astype(np.float64)
            t = data['truth_scaled'].astype(np.float64)
        if p.shape != truth_scaled.shape or not np.allclose(t, truth_scaled, rtol=0, atol=1e-6):
            raise RuntimeError('Truth mismatch: '+name)
        preds[name] = p*sigma+mu
    preds['SeasonalNaive24'] = reference['seasonal24_scaled'].astype(np.float64)*sigma+mu
    preds['SeasonalNaive168'] = reference['seasonal168_scaled'].astype(np.float64)*sigma+mu
    # Repeat baseline: two channels, last observed input value held for all 96 h.
    repeat = raw[starts+L-1,:][:,None,:]
    preds['Repeat'] = np.broadcast_to(repeat, truth_native.shape)
    assert set(preds) == set(METHODS)
    baseline_csv = root/'results/new_data/pgcb_valid_windows_v1_metrics.csv'
    saved = pd.read_csv(baseline_csv).set_index('model')
    output_rows = []
    print('\n=== INDEPENDENT SIX-METHOD SENSITIVITY AUDIT ===')
    for name in METHODS:
        pred = preds[name]
        if not np.isfinite(pred).all():
            raise RuntimeError('Non-finite predicted values: '+name)
        full_mae, full_mse, _ = metrics(pred[:,:,0], truth_native[:,:,0], mask_all)
        if not np.isclose(full_mae, float(saved.loc[name,'demand_mae_mw']), atol=.05, rtol=0):
            raise RuntimeError('Original MAE mismatch: '+name)
        if not np.isclose(full_mse, float(saved.loc[name,'demand_mse_mw2']), atol=1., rtol=0):
            raise RuntimeError('Original MSE mismatch: '+name)
        for label, mask in (('all_observed_targets', mask_all),
                            ('exclude_flagged_demand_target_points', mask_clean_targets),
                            ('exclude_any_flagged_input_or_target_window', mask_clean_windows)):
            mae, mse, n = metrics(pred[:,:,0], truth_native[:,:,0], mask)
            output_rows.append(dict(model=name, population=label,
                                    n_target_points=n,
                                    n_windows=(len(starts) if label != 'exclude_any_flagged_input_or_target_window'
                                               else int(clean_window.sum())),
                                    demand_mae_mw=mae, demand_mse_mw2=mse))
        clean = next(r for r in output_rows if r['model']==name and
                     r['population']=='exclude_flagged_demand_target_points')
        print('{:<18} all MAE {:>9.2f} MSE {:>13.1f} | unflagged-target MAE {:>9.2f} MSE {:>13.1f} PASS'.format(
            name, full_mae, full_mse, clean['demand_mae_mw'], clean['demand_mse_mw2']))

    train_end = int(cfg['train_end_index'])
    val_starts = valid_starts(valid, train_end-L, valend)
    if len(val_starts) != int(cfg['valid_windows']['validation']):
        raise RuntimeError('Validation window count disagrees with original run')
    val_future_ix = val_starts[:,None]+L+np.arange(H, dtype=np.int64)[None,:]
    val_flag_gen = flags[val_future_ix,1]
    val_scaled_truth_gen = (raw[val_future_ix,1]-mu[1])/sigma[1]
    val_pred24_gen = (raw[val_starts[:,None]+L-24+(np.arange(H, dtype=np.int64)%24)[None,:],1]-mu[1])/sigma[1]
    err2_gen = (val_pred24_gen-val_scaled_truth_gen)**2
    total_gen = float(err2_gen.sum())
    flagged_gen = float(err2_gen[val_flag_gen].sum())
    if not (total_gen >= flagged_gen >= 0):
        raise RuntimeError('Validation diagnostic calculation invalid')
    # A validation window is sensitivity-clean only if input and future contain
    # no flagged observation in either channel.
    valclean = (cumulative[val_starts+L+H]-cumulative[val_starts] == 0)
    val_info = {
        'validation_windows':len(val_starts),
        'validation_windows_without_any_input_or_target_flags':int(valclean.sum()),
        'seasonal24_generation_validation_mse_scaled_clean_windows':
            float(err2_gen[valclean].mean()) if valclean.any() else None,
        'flagged_generation_target_occurrences':int(val_flag_gen.sum()),
        'flagged_generation_unique_hours':int(np.unique(val_future_ix[val_flag_gen]).size),
        'seasonal24_generation_validation_mse_scaled':float(err2_gen.mean()),
        'seasonal24_generation_target_flag_share_of_squared_error':
            float(flagged_gen/total_gen) if total_gen else None,
        'seasonal24_generation_validation_mse_scaled_nonflagged_target_points':
            float(err2_gen[~val_flag_gen].mean()),
        'NOTE':'This is the seasonal-naive 24h diagnostic, not any trained-model validation loss.',
    }
    print('\n=== VALIDATION OUTLIER EXPOSURE (24H BASELINE DIAGNOSTIC) ===')
    for key in ('validation_windows','validation_windows_without_any_input_or_target_flags',
                'seasonal24_generation_validation_mse_scaled_clean_windows',
                'flagged_generation_target_occurrences',
                'flagged_generation_unique_hours',
                'seasonal24_generation_validation_mse_scaled',
                'seasonal24_generation_target_flag_share_of_squared_error',
                'seasonal24_generation_validation_mse_scaled_nonflagged_target_points'):
        print(key+':', val_info[key])

    results.parent.mkdir(parents=True, exist_ok=True)
    details.parent.mkdir(parents=True, exist_ok=True)
    with results.open('w',newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(output_rows[0]))
        writer.writeheader();writer.writerows(output_rows)
    summary = {'raw_source_sha256':cfg['pgcb_sha256'], 'n_test_windows':len(starts),
               'n_test_target_points':int(mask_all.sum()),
               'n_flagged_test_demand_target_points':int((~mask_clean_targets).sum()),
               'n_clean_test_input_target_windows':int(clean_window.sum()),
               'validation_diagnostic':val_info,
               'note':'Post-hoc filtering of SAVED predictions only; training/scaler and checkpoint selections unchanged. This is a sensitivity analysis, not a retrained cleaned experiment.',
               'metrics_csv':str(results)}
    details.write_text(json.dumps(summary, indent=2))
    print('\nPGCB ANOMALY SENSITIVITY: PASS')
    print('Saved:', results)
    print('Saved:', details)

if __name__ == '__main__':
    main()
