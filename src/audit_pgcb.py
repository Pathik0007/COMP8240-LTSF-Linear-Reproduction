from pathlib import Path
import pandas as pd

RAW_PATH = Path("data/pgcb/raw/PGCB_date_power_demand.xlsx")
AUDIT_DIR = Path("experiments/pgcb")
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = AUDIT_DIR / "time_integrity_audit.txt"
MISSINGNESS_PATH = AUDIT_DIR / "missingness_summary.csv"


def main():
    df = pd.read_excel(RAW_PATH, engine="openpyxl")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    with REPORT_PATH.open("w", encoding="utf-8") as out:
        def p(*args):
            print(*args)
            print(*args, file=out)

        p("PGCB TIME-INTEGRITY AUDIT")
        p("=" * 80)

        p("\nBASIC SHAPE")
        p("rows:", len(df))
        p("columns:", len(df.columns))
        p("datetime_parse_failures:", int(df["datetime"].isna().sum()))

        valid = df[df["datetime"].notna()].copy()

        p("\nTIME RANGE")
        p("minimum_datetime:", valid["datetime"].min())
        p("maximum_datetime:", valid["datetime"].max())
        p("monotonic_increasing:", valid["datetime"].is_monotonic_increasing)
        p("monotonic_decreasing:", valid["datetime"].is_monotonic_decreasing)

        off_hour = (
            (valid["datetime"].dt.minute != 0)
            | (valid["datetime"].dt.second != 0)
            | (valid["datetime"].dt.microsecond != 0)
        )
        p("off_hour_timestamp_rows:", int(off_hour.sum()))

        counts = valid["datetime"].value_counts()
        duplicate_groups = counts[counts > 1]

        p("\nTIMESTAMP UNIQUENESS")
        p("unique_timestamps:", int(valid["datetime"].nunique()))
        p(
            "rows_in_duplicate_timestamp_groups:",
            int(valid["datetime"].duplicated(keep=False).sum()),
        )
        p("duplicate_timestamp_groups:", int(len(duplicate_groups)))
        p(
            "extra_rows_beyond_unique_timestamps:",
            int(len(valid) - valid["datetime"].nunique()),
        )
        p("exact_duplicate_full_rows:", int(valid.duplicated().sum()))

        dedup_full = valid.drop_duplicates()
        conflicting = dedup_full[
            dedup_full["datetime"].duplicated(keep=False)
        ]
        conflicting_groups = conflicting["datetime"].nunique()

        p(
            "duplicate_timestamp_groups_remaining_after_exact_row_dedup:",
            int(conflicting_groups),
        )

        p("\nMOST FREQUENT DUPLICATE TIMESTAMPS")
        if len(duplicate_groups):
            p(duplicate_groups.head(20).to_string())
        else:
            p("none")

        unique_time = pd.DatetimeIndex(
            valid["datetime"].drop_duplicates().sort_values()
        )

        expected = pd.date_range(
            start=unique_time.min(),
            end=unique_time.max(),
            freq="H",
        )

        missing_hours = expected.difference(unique_time)

        p("\nHOURLY COVERAGE")
        p("expected_hourly_timestamps_inclusive:", len(expected))
        p("observed_unique_timestamps:", len(unique_time))
        p("missing_hourly_timestamps:", len(missing_hours))
        p(
            "hourly_coverage_percent:",
            round(100.0 * len(unique_time) / len(expected), 6),
        )

        gaps = unique_time.to_series().diff().dropna()

        p("\nGAP ANALYSIS")
        p("one_hour_gaps:", int((gaps == pd.Timedelta(hours=1)).sum()))
        p("gaps_greater_than_one_hour:", int((gaps > pd.Timedelta(hours=1)).sum()))
        p("largest_gap:", gaps.max())

        p("\nGAP SIZE DISTRIBUTION — TOP 20")
        p(gaps.value_counts().head(20).to_string())

        p("\nFIRST 30 MISSING HOURS")
        if len(missing_hours):
            for ts in missing_hours[:30]:
                p(ts)
        else:
            p("none")

        p("\nROWS AND UNIQUE TIMESTAMPS BY YEAR")
        annual = (
            valid.assign(year=valid["datetime"].dt.year)
            .groupby("year")
            .agg(
                rows=("datetime", "size"),
                unique_timestamps=("datetime", "nunique"),
            )
        )
        p(annual.to_string())

        p("\nSTRUCTURAL AVAILABILITY OF PARTIALLY MISSING SERIES")
        partial_cols = [
            "solar",
            "wind",
            "india_adani",
            "nepal",
            "remarks",
        ]

        for col in partial_cols:
            nonnull = valid.loc[valid[col].notna(), ["datetime", col]]
            p(f"\n{col}:")
            p("  non_missing:", int(valid[col].notna().sum()))
            p("  missing:", int(valid[col].isna().sum()))
            p("  missing_percent:", round(valid[col].isna().mean() * 100, 3))

            if len(nonnull):
                p("  first_non_missing_datetime:", nonnull["datetime"].min())
                p("  last_non_missing_datetime:", nonnull["datetime"].max())

        p("\nDEMAND / GENERATION CONSISTENCY")
        residual = (
            valid["demand_mw"]
            - valid["generation_mw"]
            - valid["load_shedding"]
        )

        p("exact_balance_rows:", int((residual == 0).sum()))
        p(
            "exact_balance_percent:",
            round(100.0 * (residual == 0).mean(), 3),
        )
        p(
            "within_1_mw_percent:",
            round(100.0 * (residual.abs() <= 1).mean(), 3),
        )
        p("mean_absolute_balance_residual:", float(residual.abs().mean()))
        p("maximum_absolute_balance_residual:", float(residual.abs().max()))

    missingness = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": [int(df[c].isna().sum()) for c in df.columns],
            "missing_percent": [
                float(df[c].isna().mean() * 100) for c in df.columns
            ],
        }
    )

    missingness.to_csv(MISSINGNESS_PATH, index=False)

    print("\nSaved:")
    print(REPORT_PATH)
    print(MISSINGNESS_PATH)


if __name__ == "__main__":
    main()
