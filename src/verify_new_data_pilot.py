from pathlib import Path
import json
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
OFFICIAL = Path.home() / "Documents" / "LTSF-Linear"

sys.path.insert(0, str(OFFICIAL))
from data_provider.data_loader import Dataset_Custom

RUN = PROJECT / "experiments/new_data/pilot_stable_dlinear_h96"

files = list((RUN / "results").glob("*/pred.npy"))
if len(files) != 1:
    raise RuntimeError(
        "Expected exactly one prediction file, found: "
        + str(len(files))
    )

pred = np.load(files[0])

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
    dataset[i][1][-96:].astype(np.float32)
    for i in range(len(dataset))
])

last_observed = np.stack([
    dataset[i][0][-1].astype(np.float32)
    for i in range(len(dataset))
])

repeat_pred = np.repeat(
    last_observed[:, None, :], 96, axis=1
)

if pred.shape != truth.shape:
    raise RuntimeError(
        "Shape mismatch: {} vs {}".format(
            pred.shape, truth.shape
        )
    )

if not np.isfinite(pred).all():
    raise RuntimeError("Non-finite model predictions")

if not np.isfinite(truth).all():
    raise RuntimeError("Non-finite ground truth")

def metrics(a, b):
    error = a.astype(np.float64) - b.astype(np.float64)
    return {
        "mse": float(np.mean(error ** 2)),
        "mae": float(np.mean(np.abs(error))),
    }

dlinear = metrics(pred, truth)
repeat = metrics(repeat_pred, truth)

reported_mse = 0.009757786989212036
reported_mae = 0.07783430814743042

if not np.isclose(dlinear["mse"], reported_mse, atol=1e-6):
    raise RuntimeError("Reported MSE not independently reproduced")

if not np.isclose(dlinear["mae"], reported_mae, atol=1e-6):
    raise RuntimeError("Reported MAE not independently reproduced")

summary = {
    "dataset": "synthetic_stable",
    "model": "DLinear",
    "seq_len": 336,
    "pred_len": 96,
    "test_windows": len(dataset),
    "predictions_shape": list(pred.shape),
    "metric_space": "training-standardised",
    "dlinear": dlinear,
    "repeat_baseline": repeat,
    "mse_improvement_over_repeat_percent":
        100 * (repeat["mse"] - dlinear["mse"]) / repeat["mse"],
    "verification": "PASS",
}

out = RUN / "independent_verification.json"
out.write_text(json.dumps(summary, indent=2))

print("=== INDEPENDENT VERIFICATION ===")
print("Prediction shape:", pred.shape)
print("DLinear MSE:", dlinear["mse"])
print("DLinear MAE:", dlinear["mae"])
print("Repeat MSE:", repeat["mse"])
print("Repeat MAE:", repeat["mae"])
print("MSE improvement over Repeat (%):",
      summary["mse_improvement_over_repeat_percent"])
print("VERIFICATION: PASS")
print("Saved:", out)
