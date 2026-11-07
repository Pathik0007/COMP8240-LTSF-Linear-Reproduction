#!/usr/bin/env python3
"""COMP8240 PGCB complete-window benchmark (Python 3.9, PyTorch 1.9).

Uses unmodified original LTSF-Linear Linear/NLinear/DLinear model classes with
chronological complete-window filtering. No imputation, test-time fitting or
future context. This is a new-data adaptation, not an unmodified replication.
"""

import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

EXPECTED_SOURCE_COMMIT = '0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6'
FEATURES = ['demand_mw', 'generation_mw']
SEQ_LEN, PRED_LEN, LABEL_LEN = 336, 96, 48
SEED = 2021
BATCH_SIZE = 32
EPOCHS = 10
PATIENCE = 3
LR = 0.005
# Post-hoc anomaly-screen sensitivity bounds; NOT physically verified grid limits.
SCREEN_MIN_MW = 1000.0
SCREEN_MAX_MW = 30000.0
FLAG_COLS = ['demand_mw_scale_anomaly', 'generation_mw_scale_anomaly']


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def assert_source(path):
    if not (path / 'run_longExp.py').is_file():
        raise RuntimeError('Cannot find original LTSF-Linear source: ' + str(path))
    sha = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    if sha != EXPECTED_SOURCE_COMMIT:
        raise RuntimeError('Official source commit differs: ' + sha)
    for extra in ([], ['--cached']):
        cmd = ['git', '-C', str(path), 'diff'] + extra + ['--quiet']
        if subprocess.call(cmd) != 0:
            raise RuntimeError('Tracked official source has modifications; stop before running')
    return sha


def load_data(path):
    df = pd.read_csv(path, usecols=['datetime', 'was_observed_hour'] + FEATURES + FLAG_COLS,
                     parse_dates=['datetime'])
    times = df['datetime']
    if times.isna().any() or not times.is_monotonic_increasing or times.duplicated().any():
        raise RuntimeError('Timestamp order, duplicates or parsing invalid')
    if not times.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
        raise RuntimeError('Data does not represent a continuous real hourly grid')
    observed = df['was_observed_hour'].astype(str).str.lower().isin(('true', '1')).to_numpy()
    raw = df[FEATURES].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float64)
    flags = np.zeros(len(df), dtype=bool)
    for col in FLAG_COLS:
        flags |= df[col].astype(str).str.lower().isin(('true', '1')).to_numpy()
    observed_finite = observed & np.isfinite(raw).all(axis=1)
    plausible = ((raw >= SCREEN_MIN_MW) & (raw <= SCREEN_MAX_MW)).all(axis=1)
    valid = observed_finite & ~flags & plausible
    print('POST-HOC SCREEN (both channels; min/max MW):', SCREEN_MIN_MW, SCREEN_MAX_MW)
    print('Observed finite hours:', int(observed_finite.sum()),
          '| flagged among these:', int((flags & observed_finite).sum()),
          '| outside heuristic limits:', int((~plausible & observed_finite).sum()),
          '| kept:', int(valid.sum()), flush=True)
    n_train, n_val_end = int(len(df)*.7), len(df)-int(len(df)*.2)
    for split, lo, hi in [('train',0,n_train),('validation',n_train,n_val_end),('test',n_val_end,len(df))]:
        print('V2 SCREEN', split, 'rows',hi-lo,'kept',int(valid[lo:hi].sum()),
              'rejected among observed finite',int((observed_finite[lo:hi] & ~valid[lo:hi]).sum()),flush=True)
    return df, raw, valid


def find_starts(valid, n, l, h):
    ntrain = int(n * .7)
    ntest = int(n * .2)
    nvalend = n - ntest
    boundaries = {'train': (0, ntrain),
                  'validation': (ntrain - l, nvalend),
                  'test': (nvalend - l, n)}
    cumulative = np.concatenate(([0], np.cumsum(valid.astype(np.int64))))
    result = {}
    for split, (a, b) in boundaries.items():
        candidate = np.arange(a, b - l - h + 1, dtype=np.int64)
        counts = cumulative[candidate + l + h] - cumulative[candidate]
        starts = candidate[counts == l + h]
        if len(starts) == 0:
            raise RuntimeError('No complete windows for ' + split)
        target_start = starts + l
        target_end = target_start + h
        if split == 'train' and not np.all(target_end <= ntrain):
            raise RuntimeError('Training target leaks into validation')
        if split == 'validation' and not (np.all(target_start >= ntrain)
                                           and np.all(target_end <= nvalend)):
            raise RuntimeError('Validation target crosses split boundary')
        if split == 'test' and not np.all(target_start >= nvalend):
            raise RuntimeError('Test target crosses split boundary')
        result[split] = starts
    return result, ntrain, nvalend


class Windows(Dataset):
    def __init__(self, scaled, starts):
        self.data = scaled
        self.starts = starts

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, k):
        s = int(self.starts[k])
        return (torch.from_numpy(self.data[s:s + SEQ_LEN]),
                torch.from_numpy(self.data[s + SEQ_LEN:s + SEQ_LEN + PRED_LEN]))


def score(pred, truth, scales):
    if pred.shape != truth.shape or pred.ndim != 3 or pred.shape[-1] != len(FEATURES):
        raise RuntimeError('Predictions do not match expected truth shape')
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise RuntimeError('Non-finite numbers encountered during evaluation')
    delta = pred.astype(np.float64) - truth.astype(np.float64)
    native = delta * scales.reshape(1, 1, -1)
    return {
        'mse_scaled_all': float(np.mean(delta ** 2)),
        'mae_scaled_all': float(np.mean(np.abs(delta))),
        'mse_native_all_mw2': float(np.mean(native ** 2)),
        'mae_native_all_mw': float(np.mean(np.abs(native))),
        'demand_mse_mw2': float(np.mean(native[:, :, 0] ** 2)),
        'demand_mae_mw': float(np.mean(np.abs(native[:, :, 0]))),
        'generation_mse_mw2': float(np.mean(native[:, :, 1] ** 2)),
        'generation_mae_mw': float(np.mean(np.abs(native[:, :, 1]))),
    }


def baseline_predictions(scaled, starts):
    # Values at s+SEQ_LEN and later are NEVER supplied as model inputs.
    xlast = np.stack([scaled[s + SEQ_LEN - 1] for s in starts]).astype(np.float32)
    last24 = np.stack([scaled[s + SEQ_LEN - 24:s + SEQ_LEN] for s in starts]).astype(np.float32)
    week = np.stack([scaled[s + SEQ_LEN - 168:s + SEQ_LEN - 168 + PRED_LEN]
                     for s in starts]).astype(np.float32)
    return {'Repeat': np.repeat(xlast[:, None, :], PRED_LEN, axis=1),
            'SeasonalNaive24': np.tile(last24, (1, PRED_LEN // 24, 1)),
            'SeasonalNaive168': week}


def calc_val(model, loader):
    model.eval()
    sum_sq, count = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            err = model(xb) - yb
            sum_sq += float((err ** 2).sum().item())
            count += err.numel()
    return sum_sq / count


def train_and_predict(name, model_class, train_loader, val_loader, test_loader, out, scales):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    model = model_class.Model(SimpleNamespace(seq_len=SEQ_LEN, pred_len=PRED_LEN,
                                               enc_in=len(FEATURES), individual=False)).float()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best_loss, best_epoch, bad = float('inf'), None, 0
    checkpoints = out / ('checkpoint_' + name + '.pth')
    tic = time.monotonic()
    print('\nMODEL', name, 'parameters', sum(p.numel() for p in model.parameters()), flush=True)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = torch.mean((model(xb) - yb) ** 2)
            loss.backward()
            opt.step()
        v = calc_val(model, val_loader)
        print(' ', name, 'epoch', epoch, 'validation MSE', format(v, '.8f'), flush=True)
        if v < best_loss - 1e-12:
            best_loss, best_epoch, bad = v, epoch, 0
            torch.save(model.state_dict(), str(checkpoints))
        else:
            bad += 1
            if bad >= PATIENCE:
                print(' ', name, 'early stopping', flush=True)
                break
        # Mirrors the original type1 schedule: next epoch uses half the LR.
        for group in opt.param_groups:
            group['lr'] = LR * (.5 ** (epoch - 1))
    model.load_state_dict(torch.load(str(checkpoints), map_location='cpu'))
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            preds.append(model(xb).numpy())
            trues.append(yb.numpy())
    pred = np.concatenate(preds, axis=0)
    truth = np.concatenate(trues, axis=0)
    vals = score(pred, truth, scales)
    vals.update({'model': name, 'best_epoch': int(best_epoch),
                 'validation_mse_scaled': float(best_loss),
                 'training_seconds': float(time.monotonic() - tic)})
    np.savez_compressed(str(out / (name + '_test_predictions.npz')),
                        prediction_scaled=pred, truth_scaled=truth)
    (out / (name + '_metrics.json')).write_text(json.dumps(vals, indent=2))
    print('RESULT', name, 'Demand MAE MW', round(vals['demand_mae_mw'], 3),
          'Demand MSE MW2', round(vals['demand_mse_mw2'], 3), flush=True)
    return vals, truth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path,
                        default=Path.home() / 'Documents/COMP8240-LTSF-Linear-Reproduction')
    parser.add_argument('--official', type=Path,
                        default=Path.home() / 'Documents/LTSF-Linear')
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    project, official = args.project.resolve(), args.official.resolve()
    sha = assert_source(official)
    raw_path = project / 'data/pgcb/processed/pgcb_hourly_core_audited.csv'
    df, raw, valid = load_data(raw_path)
    starts, trainend, valend = find_starts(valid, len(df), SEQ_LEN, PRED_LEN)
    print('=== PGCB COMPLETE-WINDOW AUDIT ===')
    print('Source commit:', sha)
    print('Raw SHA256:', digest(raw_path))
    print('Grid rows:', len(df), 'complete observed rows:', valid.sum())
    for split in ('train', 'validation', 'test'):
        print(split, 'valid windows:', len(starts[split]))
    if args.audit_only:
        return
    # Fit exclusively to observed (finite) rows in the training period.
    scaler = StandardScaler().fit(raw[:trainend][valid[:trainend]])
    scaled = ((raw - scaler.mean_) / scaler.scale_).astype(np.float32)
    print('V2 test windows (screened):', len(starts['test']))
    out = project / 'experiments/new_data/pgcb_valid_windows_v2'
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / 'protocol.json'
    if config_path.exists():
        raise RuntimeError('Existing PGCB run detected, preserving it: ' + str(out))
    config = {'official_source_sha': sha, 'pgcb_sha256': digest(raw_path),
              'features': FEATURES, 'n_grid_rows': len(df),
              'valid_windows': {key: len(value) for key, value in starts.items()},
              'train_end_index': trainend, 'validation_end_index': valend,
              'train_only_scaler_mean': scaler.mean_.tolist(),
              'train_only_scaler_scale': scaler.scale_.tolist(),
              'seq_len': SEQ_LEN, 'label_len': LABEL_LEN, 'pred_len': PRED_LEN,
              'seed': SEED, 'batch_size': BATCH_SIZE, 'max_epochs': EPOCHS,
              'early_stopping_patience': PATIENCE, 'learning_rate': LR,
              'learning_rate_schedule': 'original type1: epoch 1,2 at initial LR, halve every later epoch',
              'feature_policy': 'POST-HOC screening: demand and generation windows exclude missing/nonfinite, supplied scale flags, and BOTH channels outside heuristic 1000-30000 MW; sensitivity analysis, not validated physical thresholds',
              'metric_policy': 'all 2 channels plus primary demand and secondary generation in MW',
              'torch': torch.__version__, 'python': sys.version.split()[0]}
    config_path.write_text(json.dumps(config, indent=2))
    torch.set_num_threads(min(4, torch.get_num_threads()))
    train_loader = DataLoader(Windows(scaled, starts['train']), batch_size=BATCH_SIZE,
                              shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(Windows(scaled, starts['validation']), batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0)
    test_loader = DataLoader(Windows(scaled, starts['test']), batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0)
    sys.path.insert(0, str(official))
    from models import Linear, NLinear, DLinear
    classes = {'Linear': Linear, 'NLinear': NLinear, 'DLinear': DLinear}
    all_scores = []
    for name in ('Linear', 'NLinear', 'DLinear'):
        vals, truth = train_and_predict(name, classes[name], train_loader, val_loader,
                                        test_loader, out, scaler.scale_)
        all_scores.append(vals)
    # Baselines on precisely the same test windows (no fitted parameters).
    baselines = baseline_predictions(scaled, starts['test'])
    for name, pred in baselines.items():
        vals = score(pred, truth, scaler.scale_)
        vals.update({'model': name, 'best_epoch': '', 'validation_mse_scaled': '',
                     'training_seconds': 0.0})
        all_scores.append(vals)
    csv_path = project / 'results/new_data/pgcb_valid_windows_v2_metrics.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_scores[0].keys()))
        writer.writeheader()
        writer.writerows(all_scores)
    np.savez_compressed(str(out / 'test_reference.npz'),
                        truth_scaled=truth,
                        starts=starts['test'],
                        timestamps=df['datetime'].iloc[starts['test'] + SEQ_LEN].astype(str).to_numpy(),
                        seasonal24_scaled=baselines['SeasonalNaive24'],
                        seasonal168_scaled=baselines['SeasonalNaive168'])
    print('\n=== PGCB SIX-METHOD BENCHMARK ===')
    for val in all_scores:
        print('{:<18} demand MAE (MW) {:>11.3f} demand MSE (MW^2) {:>16.3f}'.format(
            val['model'], val['demand_mae_mw'], val['demand_mse_mw2']))
    print('PGCB BENCHMARK: PASS')
    print('CSV:', csv_path)
    print('Evidence:', out)


if __name__ == '__main__':
    main()
