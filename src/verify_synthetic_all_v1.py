from pathlib import Path
import argparse
import csv
import json
import re
import sys

import numpy as np

parser = argparse.ArgumentParser(description="Verify all eight synthetic conditions without overwriting published results")
parser.add_argument("--official", type=Path, default=Path.home() / "Documents/LTSF-Linear")
parser.add_argument("--run-root", type=Path, help="Uniform condition/model tree from the rerun driver")
parser.add_argument("--output-dir", type=Path)
args = parser.parse_args()
PROJECT = Path(__file__).resolve().parents[1]
OFFICIAL = args.official.resolve()
RUN_ROOT = args.run_root.resolve() if args.run_root else None
OUTPUT_DIR = args.output_dir.resolve() if args.output_dir else PROJECT / "experiments/reruns/synthetic_verification"
for filename in ("synthetic_all_conditions_v1_metrics.csv", "synthetic_all_conditions_v1_summary.json"):
    if (OUTPUT_DIR / filename).exists():
        raise RuntimeError("Existing verification preserved; choose a fresh output directory")
sys.path.insert(0, str(OFFICIAL))

from data_provider.data_loader import Dataset_Custom

CONDITIONS = [
    "stable",
    "level_shift",
    "regime_change",
    "noise_medium",
    "noise_high",
    "outliers_low",
    "outliers_high",
    "nonlinear",
]
MODELS = ["Linear", "NLinear", "DLinear"]
SEQ_LEN = 336
PRED_LEN = 96

def get_run_dir(condition, model):
    if RUN_ROOT is not None:
        return RUN_ROOT / condition / model
    base = PROJECT / "experiments/new_data"
    if condition == "stable":
        if model == "DLinear":
            return base / "pilot_stable_dlinear_h96"
        return base / "stable_family_v1" / model
    return base / "synthetic_family_v1" / condition / model

def calculate_metrics(pred, truth, scale):
    if pred.shape != truth.shape:
        raise ValueError(
            f"Prediction shape {pred.shape} != truth {truth.shape}"
        )
    if not np.isfinite(pred).all():
        raise ValueError("Non-finite prediction values")
    if not np.isfinite(truth).all():
        raise ValueError("Non-finite test values")

    diff = pred.astype(np.float64) - truth.astype(np.float64)
    native_diff = diff * scale.reshape(1, 1, -1)

    return {
        "mse_scaled": float(np.mean(diff ** 2)),
        "mae_scaled": float(np.mean(np.abs(diff))),
        "mse_native": float(np.mean(native_diff ** 2)),
        "mae_native": float(np.mean(np.abs(native_diff))),
    }

def get_predictions(condition, model):
    directory = get_run_dir(condition, model)
    matches = sorted((directory / "results").glob("*/pred.npy"))
    if len(matches) != 1:
        raise RuntimeError(
            f"{condition}/{model}: expected one pred.npy, "
            f"found {len(matches)}"
        )
    return np.load(matches[0]), directory

def verify_log(directory, metrics, condition, model):
    log_path = directory / "run.log"
    content = log_path.read_text(encoding="utf-8")
    matches = re.findall(
        r"^mse:\s*([0-9.eE+-]+),\s*mae:\s*([0-9.eE+-]+)",
        content,
        flags=re.MULTILINE,
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"{condition}/{model}: missing or ambiguous final metrics"
        )

    mse_logged, mae_logged = map(float, matches[0])
    if not np.isclose(
        metrics["mse_scaled"], mse_logged, rtol=0, atol=1e-6
    ):
        raise RuntimeError(f"{condition}/{model}: MSE mismatch")
    if not np.isclose(
        metrics["mae_scaled"], mae_logged, rtol=0, atol=1e-6
    ):
        raise RuntimeError(f"{condition}/{model}: MAE mismatch")

rows = []
summaries = []

for condition in CONDITIONS:
    ds = Dataset_Custom(
        root_path=str(PROJECT / "data/synthetic"),
        flag="test",
        size=[SEQ_LEN, 48, PRED_LEN],
        features="M",
        data_path=f"{condition}.csv",
        target="channel_1",
        scale=True,
        timeenc=1,
        freq="h",
        train_only=False,
    )

    truth = np.stack([
        ds[i][1][-PRED_LEN:]
        for i in range(len(ds))
    ]).astype(np.float32)

    history = np.stack([
        ds[i][0]
        for i in range(len(ds))
    ]).astype(np.float32)

    assert truth.shape == (1905, 96, 4), truth.shape
    assert history.shape == (1905, 336, 4), history.shape

    scale = ds.scaler.scale_

    repeat = np.repeat(history[:, -1:, :], PRED_LEN, axis=1)
    seasonal = np.tile(history[:, -24:, :], (1, 4, 1))

    scored = {}

    for model in MODELS:
        pred, directory = get_predictions(condition, model)
        score = calculate_metrics(pred, truth, scale)
        verify_log(directory, score, condition, model)
        scored[model] = score

    scored["Repeat"] = calculate_metrics(repeat, truth, scale)
    scored["SeasonalNaive24"] = calculate_metrics(
        seasonal, truth, scale
    )

    seasonal_mse = scored["SeasonalNaive24"]["mse_native"]

    for model, score in scored.items():
        improvement = (
            100.0 * (seasonal_mse - score["mse_native"])
            / seasonal_mse
        )
        rows.append({
            "condition": condition,
            "model": model,
            "test_windows": len(ds),
            "seq_len": SEQ_LEN,
            "pred_len": PRED_LEN,
            **score,
            "improvement_over_seasonal_pct": improvement,
        })

    winner = min(
        MODELS,
        key=lambda m: scored[m]["mse_native"]
    )

    summaries.append({
        "condition": condition,
        "best_trained_model": winner,
        "best_trained_native_mse": scored[winner]["mse_native"],
        "seasonal_native_mse": seasonal_mse,
        "best_vs_seasonal_pct": (
            100.0 * (seasonal_mse -
                     scored[winner]["mse_native"])
            / seasonal_mse
        ),
    })

    print(f"VERIFIED: {condition} — 3 trained models + 2 baselines")

if len(rows) != 40:
    raise RuntimeError(f"Expected 40 rows, got {len(rows)}")

out_dir = OUTPUT_DIR
out_dir.mkdir(parents=True, exist_ok=True)

csv_path = out_dir / "synthetic_all_conditions_v1_metrics.csv"
json_path = out_dir / "synthetic_all_conditions_v1_summary.json"

with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

json_path.write_text(
    json.dumps({
        "conditions": summaries,
        "trained_runs_verified": 24,
        "baseline_evaluations": 16,
        "verification": "PASS",
    }, indent=2),
    encoding="utf-8",
)

print()
print("=== EIGHT-CONDITION BENCHMARK SUMMARY ===")
print(
    f"{'Condition':<17} {'Best model':<10} "
    f"{'Best native MSE':>16} {'Seasonal MSE':>14} "
    f"{'Gain %':>9}"
)
print("-" * 75)

for item in summaries:
    print(
        f"{item['condition']:<17} "
        f"{item['best_trained_model']:<10} "
        f"{item['best_trained_native_mse']:>16.6f} "
        f"{item['seasonal_native_mse']:>14.6f} "
        f"{item['best_vs_seasonal_pct']:>9.2f}"
    )

print()
print("Trained models verified:", 24)
print("Baseline evaluations:", 16)
print("Total result rows:", len(rows))
print("VERIFICATION: PASS")
print("Saved CSV:", csv_path)
print("Saved JSON:", json_path)
