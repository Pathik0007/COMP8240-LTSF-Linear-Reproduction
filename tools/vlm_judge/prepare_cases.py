#!/usr/bin/env python3
"""Create reproducible, blinded visual forecasting evaluation panels.

Synthetic: DLinear vs 24-hour seasonal naive across 8 conditions.
PGCB (optional): DLinear vs 168-hour seasonal naive on valid test windows.
Numerical ground truth is saved ONLY in manifest, never inserted into judge prompts.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

CONDITIONS = ['stable', 'level_shift', 'regime_change', 'noise_medium',
              'noise_high', 'outliers_low', 'outliers_high', 'nonlinear']
L, H = 336, 96
SEED = 8240
PAIRS_PER_CONDITION = 6
EPS = 1e-12


def hash_file(path):
    dig = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            dig.update(block)
    return dig.hexdigest()


def locate_pred(directory):
    paths = list(directory.glob('results/*/pred.npy'))
    if len(paths) != 1:
        raise RuntimeError('{}: expected exactly one pred.npy, got {}'.format(directory, len(paths)))
    return np.load(paths[0])


def mae_mse(y, p):
    err = np.asarray(y, float) - np.asarray(p, float)
    return float(np.mean(np.abs(err))), float(np.mean(err ** 2))


def winner(a, b, threshold=.05):
    # A relative advantage of <5% is treated as a practical tie.
    if abs(a-b) <= threshold * max(a, b, EPS):
        return 'tie'
    return 'A' if a < b else 'B'


def make_plot(history, truth, a, b, path, units):
    if len(truth) != H or len(a) != H or len(b) != H:
        raise RuntimeError('Expected 96 hourly target and candidate values')
    fig, ax = plt.subplots(figsize=(10.6, 4.1), dpi=130)
    ax.plot(np.arange(-len(history), 0), history, label='Observed history',
            color='#555555', linewidth=1.6)
    ax.plot(np.arange(H), truth, label='Observed future', color='#111111',
            linewidth=2.4, zorder=5)
    ax.plot(np.arange(H), a, label='Forecast A', color='#1565c0', linewidth=1.7, alpha=.9)
    ax.plot(np.arange(H), b, label='Forecast B', color='#d97706', linewidth=1.7, alpha=.9)
    ax.axvline(-.5, color='#999999', linestyle='--', linewidth=1)
    ax.set_title('Blinded 96-hour forecast comparison')
    ax.set_xlabel('Hours relative to forecast origin')
    ax.set_ylabel(units)
    ax.grid(True, alpha=.18)
    ax.legend(loc='best', ncol=2, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def add_pair(records, out, base_id, family, condition, channel, history,
             truth, model_pred, baseline_pred, candidate_names,
             timestamp=None, window_index=None):
    for variant in ('ab', 'ba'):
        # Fixed complementary ordering; each pair's true and model IDs are hidden from VLM.
        a, b = ((model_pred, baseline_pred) if variant == 'ab'
                else (baseline_pred, model_pred))
        a_model, b_model = (candidate_names if variant == 'ab'
                            else tuple(reversed(candidate_names)))
        a_mae, a_mse = mae_mse(truth, a)
        b_mae, b_mse = mae_mse(truth, b)
        img_path = out / 'images' / '{}_{}.png'.format(base_id, variant)
        make_plot(history, truth, a, b, img_path,
                  'MW' if family == 'pgcb' else 'Synthetic native units')
        records.append({
            'id': '{}_{}'.format(base_id, variant),
            'base_id': base_id,
            'variant': variant,
            'family': family,
            'condition': condition,
            'channel': channel,
            'window_index': window_index,
            'origin_timestamp': timestamp,
            'image': str(img_path.relative_to(out)),
            'image_sha256': hash_file(img_path),
            'model_A': a_model,
            'model_B': b_model,
            'A_mae': a_mae, 'B_mae': b_mae,
            'A_mse': a_mse, 'B_mse': b_mse,
            'gold_mae': winner(a_mae, b_mae),
            'gold_mse': winner(a_mse, b_mse),
            'gold_tie_threshold': .05,
        })


def synthetic_cases(project, out, records):
    base = project / 'experiments/new_data'
    for condition in CONDITIONS:
        csv_path = project / 'data/synthetic' / (condition + '.csv')
        df = pd.read_csv(csv_path)
        values = df.iloc[:, 1:].to_numpy(dtype=np.float64)
        if values.shape != (10000, 4):
            raise RuntimeError('Unexpected synthetic shape: {} {}'.format(condition, values.shape))
        scaler = StandardScaler().fit(values[:7000])
        if condition == 'stable':
            pred_dir = base / 'pilot_stable_dlinear_h96'
        else:
            pred_dir = base / 'synthetic_family_v1' / condition / 'DLinear'
        preds = locate_pred(pred_dir)
        if preds.shape != (1905, H, 4):
            raise RuntimeError('Unexpected prediction shape: {} {}'.format(condition, preds.shape))
        if condition in ('level_shift', 'regime_change'):
            indices = [0, 1, 12, 48, 256, 1600]
        else:
            indices = [0, 64, 256, 512, 1024, 1800]
        for j, idx in enumerate(indices):
            ch = j % 4
            origin = 8000 + idx
            history = values[origin-48:origin, ch]
            truth = values[origin:origin+H, ch]
            learned = preds[idx, :, ch].astype(float) * scaler.scale_[ch] + scaler.mean_[ch]
            baseline = np.tile(values[origin-24:origin, ch], H//24)
            add_pair(records, out, 'syn_{}_{}'.format(condition, j), 'synthetic',
                     condition, 'channel_{}'.format(ch + 1), history, truth,
                     learned, baseline, ('DLinear', 'SeasonalNaive24'),
                     timestamp=str(df.iloc[origin, 0]), window_index=int(idx))
        print('Synthetic cases created:', condition, 'base comparisons', len(indices), flush=True)


def pgcb_cases(project, out, records):
    directory = project / 'experiments/new_data/pgcb_valid_windows_v1'
    ref_path = directory / 'test_reference.npz'
    pred_path = directory / 'DLinear_test_predictions.npz'
    cfg_path = directory / 'protocol.json'
    if not all(p.is_file() for p in (ref_path, pred_path, cfg_path)):
        print('PGCB results unavailable: skip PGCB judge cases (not a failure)', flush=True)
        return 0
    with cfg_path.open() as f:
        cfg = json.load(f)
    scales = np.asarray(cfg['train_only_scaler_scale'], float)
    means = np.asarray(cfg['train_only_scaler_mean'], float)
    reference = np.load(ref_path)
    predictions = np.load(pred_path)
    truth = reference['truth_scaled']
    preds = predictions['prediction_scaled']
    baseline = reference['seasonal168_scaled']
    starts = reference['starts']
    if not (truth.shape == preds.shape == baseline.shape) or truth.shape[-1] != 2:
        raise RuntimeError('PGCB prediction/reference shape mismatch')
    df = pd.read_csv(project / 'data/pgcb/processed/pgcb_hourly_core_audited.csv',
                     usecols=['datetime', 'demand_mw'], parse_dates=['datetime'])
    cases = np.linspace(0, len(truth)-1, 12, dtype=int)
    for j, idx in enumerate(cases):
        start = int(starts[idx]); origin = start + L
        x = df['demand_mw'].iloc[origin-48:origin].to_numpy(float)
        y = truth[idx, :, 0] * scales[0] + means[0]
        p = preds[idx, :, 0] * scales[0] + means[0]
        b = baseline[idx, :, 0] * scales[0] + means[0]
        if not all(np.isfinite(v).all() for v in (x, y, p, b)):
            raise RuntimeError('PGCB non-finite case at {}'.format(idx))
        add_pair(records, out, 'pgcb_demand_{}'.format(j), 'pgcb',
                 'real_electricity_demand', 'demand_mw', x, y, p, b,
                 ('DLinear', 'SeasonalNaive168'),
                 timestamp=str(df['datetime'].iloc[origin]), window_index=int(idx))
    print('PGCB base comparisons:', len(cases), flush=True)
    return len(cases)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path.home() / 'Documents/COMP8240-LTSF-Linear-Reproduction')
    parser.add_argument('--out', type=Path, default=None)
    parser.add_argument('--synthetic-only', action='store_true')
    args = parser.parse_args()
    project = args.project.resolve()
    out = (args.out or project / 'experiments/new_data/vlm_judge_v1').resolve()
    manifest = out / 'cases.jsonl'
    if manifest.exists():
        raise RuntimeError('Judge manifest already exists. Preserve it: {}'.format(manifest))
    out.mkdir(parents=True, exist_ok=True)
    records = []
    synthetic_cases(project, out, records)
    if not args.synthetic_only:
        pgcb_cases(project, out, records)
    with manifest.open('w') as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + '\n')
    print('JUDGE CASES PREPARED', 'original cases', len(records)//2,
          'A/B panel variants', len(records), 'manifest:', manifest, flush=True)
    print('NOTE: Ground-truth MAE/MSE exists only in private manifest; never include it in VLM prompts.')


if __name__ == '__main__':
    main()
