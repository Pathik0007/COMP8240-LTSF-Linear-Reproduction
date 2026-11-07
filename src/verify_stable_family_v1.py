from pathlib import Path
import csv
import re
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
OFFICIAL = Path.home() / "Documents/LTSF-Linear"
sys.path.insert(0, str(OFFICIAL))

from data_provider.data_loader import Dataset_Custom

dataset = Dataset_Custom(
    root_path=str(PROJECT / "data/synthetic"),
    flag="test",
    size=[336, 48, 96],
    features="M",
    data_path="stable.csv",
    target="channel_1",
    scale=True,
    timeenc=1,
    freq="h",
    train_only=False,
)

truth = np.stack([
    dataset[i][1][-96:]
    for i in range(len(dataset))
]).astype(np.float32)

history = np.stack([
    dataset[i][0]
    for i in range(len(dataset))
]).astype(np.float32)

assert truth.shape == (1905, 96, 4)
assert history.shape == (1905, 336, 4)

scales = dataset.scaler.scale_[None, None, :]

def metrics(prediction):
    if prediction.shape != truth.shape:
        raise ValueError(
            f"Shape mismatch: {prediction.shape} != {truth.shape}"
        )

    if not np.isfinite(prediction).all():
        raise ValueError("Predictions contain non-finite values")

    difference = (
        prediction.astype(np.float64)
        - truth.astype(np.float64)
    )

    native_difference = difference * scales

    return {
        "mse_scaled": float(np.mean(difference ** 2)),
        "mae_scaled": float(np.mean(np.abs(difference))),
        "mse_native": float(np.mean(native_difference ** 2)),
        "mae_native": float(np.mean(np.abs(native_difference))),
    }

runs = {
    "DLinear": (
        PROJECT / "experiments/new_data/pilot_stable_dlinear_h96"
    ),
    "Linear": (
        PROJECT / "experiments/new_data/stable_family_v1/Linear"
    ),
    "NLinear": (
        PROJECT / "experiments/new_data/stable_family_v1/NLinear"
    ),
}

results = {}

for model, directory in runs.items():
    predictions = list(
        (directory / "results").glob("*/pred.npy")
    )

    if len(predictions) != 1:
        raise RuntimeError(
            f"{model}: expected one pred.npy, "
            f"found {len(predictions)}"
        )

    prediction = np.load(predictions[0])
    score = metrics(prediction)

    log_path = directory / "run.log"
    log_text = log_path.read_text()

    matches = re.findall(
        r"mse:([-+0-9.eE]+),\s*mae:([-+0-9.eE]+)",
        log_text
    )

    if not matches:
        raise RuntimeError(f"{model}: no final metrics in log")

    logged_mse, logged_mae = map(float, matches[-1])

    if not np.isclose(
        score["mse_scaled"], logged_mse, atol=1e-6
    ):
        raise RuntimeError(f"{model}: MSE verification FAILED")

    if not np.isclose(
        score["mae_scaled"], logged_mae, atol=1e-6
    ):
        raise RuntimeError(f"{model}: MAE verification FAILED")

    results[model] = score

repeat = np.repeat(
    history[:, -1:, :],
    96,
    axis=1
)

seasonal_naive = np.tile(
    history[:, -24:, :],
    (1, 4, 1)
)

results["Repeat"] = metrics(repeat)
results["SeasonalNaive24"] = metrics(seasonal_naive)

out_dir = PROJECT / "results/new_data"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "stable_family_v1_metrics.csv"

fields = [
    "model",
    "dataset",
    "test_windows",
    "mse_scaled",
    "mae_scaled",
    "mse_native",
    "mae_native",
]

with out_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()

    for model, score in results.items():
        writer.writerow({
            "model": model,
            "dataset": "synthetic_stable",
            "test_windows": len(dataset),
            **score
        })

print()
print("=== VERIFIED FIVE-METHOD COMPARISON ===")
print(
    f"{'Model':<18} {'Scaled MSE':>14} "
    f"{'Scaled MAE':>14} {'Native MSE':>14}"
)
print("-" * 65)

for model, score in results.items():
    print(
        f"{model:<18} "
        f"{score['mse_scaled']:>14.8f} "
        f"{score['mae_scaled']:>14.8f} "
        f"{score['mse_native']:>14.8f}"
    )

season_mse = results["SeasonalNaive24"]["mse_scaled"]
dlinear_mse = results["DLinear"]["mse_scaled"]

improvement = (
    100.0 * (season_mse - dlinear_mse) / season_mse
)

print()
print(
    "DLinear MSE improvement over seasonal naive:",
    f"{improvement:.2f}%"
)
print("All trained-model metrics matched their saved logs.")
print("VERIFICATION: PASS")
print("Saved:", out_path)
