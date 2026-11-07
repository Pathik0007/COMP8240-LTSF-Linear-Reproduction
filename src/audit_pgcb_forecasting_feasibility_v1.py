from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/pgcb/processed/pgcb_hourly_core_audited.csv"

df = pd.read_csv(
    SOURCE,
    usecols=[
        "datetime",
        "demand_mw",
        "generation_mw",
        "was_observed_hour",
        "demand_mw_scale_anomaly",
    ],
    parse_dates=["datetime"],
)

if not df["datetime"].is_monotonic_increasing:
    raise RuntimeError("Datetime values are not sorted")

if df["datetime"].duplicated().any():
    raise RuntimeError("Duplicate timestamps remain")

expected = pd.date_range(
    df["datetime"].iloc[0],
    df["datetime"].iloc[-1],
    freq="h",
)
if not np.array_equal(
    df["datetime"].to_numpy(),
    expected.to_numpy()
):
    raise RuntimeError("Hourly grid is not continuous")

n = len(df)
train_end = int(n * 0.7)
test_size = int(n * 0.2)
val_end = n - test_size

seq_len = 336
pred_len = 96

observed = (
    df["was_observed_hour"]
    .astype(str)
    .str.lower()
    .isin(["true", "1"])
    .to_numpy()
)

anomaly = (
    df["demand_mw_scale_anomaly"]
    .astype(str)
    .str.lower()
    .isin(["true", "1"])
    .to_numpy()
)

demand = pd.to_numeric(
    df["demand_mw"], errors="coerce"
).to_numpy(dtype=float)

generation = pd.to_numeric(
    df["generation_mw"], errors="coerce"
).to_numpy(dtype=float)

valid_demand = observed & np.isfinite(demand)
valid_pair = (
    valid_demand & np.isfinite(generation)
)

def counts_for_windows(mask, split_start, split_end):
    starts = np.arange(
        split_start,
        split_end - seq_len - pred_len + 1,
        dtype=np.int64,
    )

    if len(starts) == 0:
        return {
            "total": 0,
            "full_target": 0,
            "full_input_and_target": 0,
            "at_least_95pct_target": 0,
        }

    cumsum = np.concatenate((
        [0], np.cumsum(mask.astype(np.int64))
    ))

    input_valid = (
        cumsum[starts + seq_len]
        - cumsum[starts]
    )
    target_valid = (
        cumsum[starts + seq_len + pred_len]
        - cumsum[starts + seq_len]
    )

    return {
        "total": int(len(starts)),
        "full_target": int(
            np.count_nonzero(target_valid == pred_len)
        ),
        "full_input_and_target": int(
            np.count_nonzero(
                (input_valid == seq_len)
                & (target_valid == pred_len)
            )
        ),
        "at_least_95pct_target": int(
            np.count_nonzero(
                target_valid >= int(np.ceil(0.95 * pred_len))
            )
        ),
    }

splits = {
    "train": (0, train_end),
    "validation": (train_end - seq_len, val_end),
    "test": (val_end - seq_len, n),
}

report = {
    "total_grid_hours": n,
    "observed_hours": int(observed.sum()),
    "missing_hours": int((~observed).sum()),
    "demand_valid_hours": int(valid_demand.sum()),
    "demand_anomaly_flags": int(anomaly.sum()),
    "split_cutoffs": {
        "train_end": str(df["datetime"].iloc[train_end]),
        "validation_end": str(df["datetime"].iloc[val_end]),
    },
    "splits": {},
}

print("=== PGCB DATA INTEGRITY ===")
for key in [
    "total_grid_hours", "observed_hours",
    "missing_hours", "demand_valid_hours",
    "demand_anomaly_flags",
]:
    print(f"{key}: {report[key]}")

print()
print("=== FORECAST WINDOW FEASIBILITY ===")

for name, (start, end) in splits.items():
    single = counts_for_windows(
        valid_demand, start, end
    )
    pair = counts_for_windows(
        valid_pair, start, end
    )

    report["splits"][name] = {
        "demand_only": single,
        "demand_and_generation": pair,
    }

    print(f"\n{name.upper()}")
    print("Demand-only:", single)
    print("Demand+generation:", pair)

print()
print("=== LONGEST COMPLETE DEMAND SEGMENTS ===")

padded = np.concatenate(
    ([False], valid_demand, [False])
)
start_indices = np.flatnonzero(
    np.diff(padded.astype(int)) == 1
)
end_indices = np.flatnonzero(
    np.diff(padded.astype(int)) == -1
)

segments = sorted(
    zip(start_indices, end_indices),
    key=lambda item: item[1] - item[0],
    reverse=True,
)

for start, end in segments[:5]:
    print(
        f"{end-start:6d} hours: "
        f"{df['datetime'].iloc[start]} to "
        f"{df['datetime'].iloc[end-1]}"
    )

output = (
    ROOT / "experiments/new_data/"
    "pgcb_forecasting_feasibility_v1.json"
)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)

print()
print("PGCB AUDIT: PASS")
print("Saved:", output)
