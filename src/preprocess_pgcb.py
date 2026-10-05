from pathlib import Path
import json

import numpy as np
import pandas as pd


RAW_PATH = Path("data/pgcb/raw/PGCB_date_power_demand.xlsx")
PROCESSED_DIR = Path("data/pgcb/processed")
EXPERIMENT_DIR = Path("experiments/pgcb")

OUTPUT_PATH = PROCESSED_DIR / "pgcb_hourly_core_audited.csv"
SUMMARY_PATH = EXPERIMENT_DIR / "preprocessing_summary.json"

CORE_COLS = [
    "generation_mw",
    "demand_mw",
    "load_shedding",
    "gas",
    "liquid_fuel",
    "coal",
    "hydro",
    "india_bheramara_hvdc",
    "india_tripura",
]

STRUCTURALLY_PARTIAL_COLS = [
    "solar",
    "wind",
    "india_adani",
    "nepal",
]


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(RAW_PATH, engine="openpyxl")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="raise")

    raw_rows = len(df)

    # --------------------------------------------------------------
    # Cadence audit
    # --------------------------------------------------------------
    minute_counts = (
        df["datetime"]
        .dt.minute
        .value_counts()
        .sort_index()
        .to_dict()
    )

    # Benchmark is explicitly hourly.
    hourly = df.loc[df["datetime"].dt.minute.eq(0)].copy()
    half_hour_rows_removed = raw_rows - len(hourly)

    # --------------------------------------------------------------
    # Consistent decade-wide feature set
    # --------------------------------------------------------------
    hourly = hourly[["datetime"] + CORE_COLS].copy()

    for col in CORE_COLS:
        hourly[col] = pd.to_numeric(hourly[col], errors="coerce")

    # --------------------------------------------------------------
    # Duplicate timestamps
    #
    # Median aggregation is deterministic and robust to occasional
    # extreme duplicate entries. source_rows preserves provenance.
    # --------------------------------------------------------------
    source_counts = (
        hourly.groupby("datetime")
        .size()
        .rename("source_rows")
    )

    duplicate_groups = int((source_counts > 1).sum())
    rows_in_duplicate_groups = int(
        source_counts[source_counts > 1].sum()
    )

    collapsed = (
        hourly.groupby("datetime", as_index=True)[CORE_COLS]
        .median()
        .sort_index()
    )

    collapsed = collapsed.join(source_counts)

    # --------------------------------------------------------------
    # Construct complete hourly grid
    # --------------------------------------------------------------
    full_index = pd.date_range(
        start=collapsed.index.min(),
        end=collapsed.index.max(),
        freq="H",
        name="datetime",
    )

    audited = collapsed.reindex(full_index)

    audited["was_observed_hour"] = audited["source_rows"].notna()
    audited["was_duplicate_hour"] = audited["source_rows"].fillna(0).gt(1)

    audited["source_rows"] = (
        audited["source_rows"]
        .fillna(0)
        .astype(int)
    )

    missing_hour_count = int((~audited["was_observed_hour"]).sum())

    # --------------------------------------------------------------
    # Data-quality annotations
    #
    # These are FLAGS only. No values are repaired or removed here.
    # A deliberately conservative threshold of 5 x raw q99 is used
    # to identify obvious scale anomalies for later investigation.
    # --------------------------------------------------------------
    anomaly_summary = {}

    any_scale_anomaly = pd.Series(
        False,
        index=audited.index,
    )

    for col in CORE_COLS:
        observed = collapsed[col].dropna()

        q99 = float(observed.quantile(0.99))
        upper_screen = float(5.0 * q99)

        flag_col = f"{col}_scale_anomaly"

        audited[flag_col] = (
            (audited[col] < 0)
            | (audited[col] > upper_screen)
        ).fillna(False)

        count = int(audited[flag_col].sum())

        any_scale_anomaly = (
            any_scale_anomaly
            | audited[flag_col]
        )

        anomaly_summary[col] = {
            "q99": q99,
            "screening_upper_bound_5x_q99": upper_screen,
            "flagged_rows": count,
        }

    audited["any_scale_anomaly"] = any_scale_anomaly

    # --------------------------------------------------------------
    # Internal consistency annotation
    # demand approximately equals generation + load shedding.
    # Again: annotation only, not correction.
    # --------------------------------------------------------------
    audited["balance_residual_mw"] = (
        audited["demand_mw"]
        - audited["generation_mw"]
        - audited["load_shedding"]
    )

    audited["abs_balance_residual_mw"] = (
        audited["balance_residual_mw"].abs()
    )

    # --------------------------------------------------------------
    # Save reconstructable dataset
    # --------------------------------------------------------------
    audited.reset_index().to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # --------------------------------------------------------------
    # Structural availability metadata from original file
    # --------------------------------------------------------------
    structural_availability = {}

    for col in STRUCTURALLY_PARTIAL_COLS:
        nonnull = df.loc[df[col].notna(), ["datetime", col]]

        structural_availability[col] = {
            "non_missing_rows": int(df[col].notna().sum()),
            "missing_rows": int(df[col].isna().sum()),
            "missing_percent": float(df[col].isna().mean() * 100),
            "first_non_missing_datetime": (
                str(nonnull["datetime"].min())
                if len(nonnull)
                else None
            ),
            "last_non_missing_datetime": (
                str(nonnull["datetime"].max())
                if len(nonnull)
                else None
            ),
        }

    summary = {
        "source_file": str(RAW_PATH),
        "raw_rows": raw_rows,
        "raw_columns": int(len(df.columns)),
        "raw_start": str(df["datetime"].min()),
        "raw_end": str(df["datetime"].max()),
        "minute_of_hour_counts": {
            str(k): int(v)
            for k, v in minute_counts.items()
        },
        "exact_hour_rows_retained": int(len(hourly)),
        "off_hour_rows_excluded": int(half_hour_rows_removed),
        "duplicate_timestamp_groups_exact_hour": duplicate_groups,
        "rows_in_duplicate_timestamp_groups": rows_in_duplicate_groups,
        "unique_hourly_observations_after_collapse": int(len(collapsed)),
        "full_hourly_grid_rows": int(len(audited)),
        "missing_hours_after_reindex": missing_hour_count,
        "observed_hour_coverage_percent": float(
            100.0
            * audited["was_observed_hour"].mean()
        ),
        "core_features": CORE_COLS,
        "structurally_partial_features_excluded_from_core": (
            STRUCTURALLY_PARTIAL_COLS
        ),
        "duplicate_resolution": (
            "median of numeric core variables at identical timestamps"
        ),
        "missing_hour_policy": (
            "preserved as missing; no interpolation performed"
        ),
        "anomaly_policy": (
            "values preserved; conservative 5x-q99 flags added only"
        ),
        "scale_anomaly_summary": anomaly_summary,
        "total_rows_with_any_scale_anomaly": int(
            audited["any_scale_anomaly"].sum()
        ),
        "structural_availability": structural_availability,
    }

    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("PGCB conservative preprocessing complete")
    print("----------------------------------------")
    print("raw rows:", raw_rows)
    print("exact-hour rows retained:", len(hourly))
    print("off-hour rows excluded:", half_hour_rows_removed)
    print("duplicate exact-hour groups:", duplicate_groups)
    print("unique observed hours:", len(collapsed))
    print("complete hourly grid rows:", len(audited))
    print("missing hours:", missing_hour_count)
    print(
        "hourly coverage:",
        f"{100.0 * audited['was_observed_hour'].mean():.3f}%"
    )
    print(
        "rows flagged with scale anomalies:",
        int(audited["any_scale_anomaly"].sum()),
    )
    print("\nSaved:")
    print(OUTPUT_PATH)
    print(SUMMARY_PATH)


if __name__ == "__main__":
    main()
