from pathlib import Path
import pandas as pd
import numpy as np

RAW_PATH = Path("data/pgcb/raw/PGCB_date_power_demand.xlsx")
OUT_PATH = Path("experiments/pgcb/cleaning_decision_audit.txt")

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

ALL_NUMERIC = CORE_COLS + [
    "solar",
    "wind",
    "india_adani",
    "nepal",
]


def longest_hourly_run(timestamps):
    idx = pd.DatetimeIndex(pd.Series(timestamps).dropna().drop_duplicates().sort_values())

    if len(idx) == 0:
        return None

    diffs = pd.Series(idx).diff()
    groups = diffs.ne(pd.Timedelta(hours=1)).cumsum()

    runs = (
        pd.DataFrame({"datetime": idx, "group": groups.values})
        .groupby("group")
        .agg(
            start=("datetime", "min"),
            end=("datetime", "max"),
            hours=("datetime", "size"),
        )
        .sort_values("hours", ascending=False)
    )

    return runs.iloc[0], runs.head(10)


def main():
    df = pd.read_excel(RAW_PATH, engine="openpyxl")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    with OUT_PATH.open("w", encoding="utf-8") as out:
        def p(*args):
            print(*args)
            print(*args, file=out)

        p("PGCB CLEANING-DECISION AUDIT")
        p("=" * 80)

        # ------------------------------------------------------------
        # Minute / cadence structure
        # ------------------------------------------------------------
        p("\nMINUTE-OF-HOUR DISTRIBUTION")
        p(df["datetime"].dt.minute.value_counts().sort_index().to_string())

        off_hour = df[df["datetime"].dt.minute != 0]

        p("\nOFF-HOUR TIMESTAMP RANGE")
        p("off_hour_rows:", len(off_hour))
        if len(off_hour):
            p("first_off_hour:", off_hour["datetime"].min())
            p("last_off_hour:", off_hour["datetime"].max())

        cadence_year = (
            df.assign(
                year=df["datetime"].dt.year,
                exact_hour=df["datetime"].dt.minute.eq(0),
            )
            .groupby("year")
            .agg(
                total_rows=("datetime", "size"),
                exact_hour_rows=("exact_hour", "sum"),
            )
        )
        cadence_year["off_hour_rows"] = (
            cadence_year["total_rows"] - cadence_year["exact_hour_rows"]
        )

        p("\nCADENCE BY YEAR")
        p(cadence_year.to_string())

        # ------------------------------------------------------------
        # Duplicate timestamp conflicts
        # ------------------------------------------------------------
        timestamp_counts = df["datetime"].value_counts()
        duplicated_times = timestamp_counts[timestamp_counts > 1].index

        duplicate_df = df[df["datetime"].isin(duplicated_times)]

        core_conflicting_groups = 0
        numeric_conflicting_groups = 0

        for _, group in duplicate_df.groupby("datetime"):
            core_nunique = group[CORE_COLS].nunique(dropna=False)
            all_nunique = group[ALL_NUMERIC].nunique(dropna=False)

            if (core_nunique > 1).any():
                core_conflicting_groups += 1

            if (all_nunique > 1).any():
                numeric_conflicting_groups += 1

        p("\nDUPLICATE TIMESTAMP CONFLICTS")
        p("duplicate_timestamp_groups:", len(duplicated_times))
        p(
            "groups_with_conflicting_core_numeric_values:",
            core_conflicting_groups,
        )
        p(
            "groups_with_conflicting_any_numeric_values:",
            numeric_conflicting_groups,
        )

        exact = df[df["datetime"].dt.minute.eq(0)].copy()
        exact_counts = exact["datetime"].value_counts()

        p("duplicate_groups_among_exact_hour_rows:", int((exact_counts > 1).sum()))

        # ------------------------------------------------------------
        # Hourly coverage using ONLY exact-hour timestamps
        # ------------------------------------------------------------
        exact_unique = pd.DatetimeIndex(
            exact["datetime"].drop_duplicates().sort_values()
        )

        expected_exact = pd.date_range(
            exact_unique.min(),
            exact_unique.max(),
            freq="H",
        )

        missing_exact = expected_exact.difference(exact_unique)

        p("\nEXACT-HOUR GRID COVERAGE")
        p("exact_hour_rows:", len(exact))
        p("unique_exact_hour_timestamps:", len(exact_unique))
        p("expected_hourly_timestamps:", len(expected_exact))
        p("missing_exact_hours:", len(missing_exact))
        p(
            "coverage_percent:",
            round(100 * len(exact_unique) / len(expected_exact), 6),
        )

        # ------------------------------------------------------------
        # Longest runs
        # ------------------------------------------------------------
        best_all, top_all = longest_hourly_run(exact_unique)

        p("\nLONGEST CONTIGUOUS HOURLY RUN — EXACT-HOUR TIMESTAMPS")
        p(best_all.to_string())

        p("\nTOP 10 CONTIGUOUS HOURLY RUNS — EXACT-HOUR TIMESTAMPS")
        p(top_all.to_string())

        unique_only_times = exact_counts[exact_counts == 1].index
        best_unique, top_unique = longest_hourly_run(unique_only_times)

        p("\nLONGEST CONTIGUOUS RUN — EXACT-HOUR + NON-DUPLICATED TIMESTAMPS")
        p(best_unique.to_string())

        p("\nTOP 10 RUNS — EXACT-HOUR + NON-DUPLICATED TIMESTAMPS")
        p(top_unique.to_string())

        # ------------------------------------------------------------
        # Value distributions / obvious anomalies
        # ------------------------------------------------------------
        p("\nCORE VARIABLE RANGE AUDIT")

        rows = []

        for col in CORE_COLS:
            s = pd.to_numeric(df[col], errors="coerce")

            rows.append(
                {
                    "column": col,
                    "min": s.min(),
                    "q01": s.quantile(0.01),
                    "median": s.median(),
                    "q99": s.quantile(0.99),
                    "max": s.max(),
                    "negative_rows": int((s < 0).sum()),
                    "zero_rows": int((s == 0).sum()),
                }
            )

        range_df = pd.DataFrame(rows)
        p(range_df.to_string(index=False))

        # ------------------------------------------------------------
        # Balance residual audit
        # demand ~= generation + load shedding
        # ------------------------------------------------------------
        df["balance_residual"] = (
            df["demand_mw"]
            - df["generation_mw"]
            - df["load_shedding"]
        )

        df["abs_balance_residual"] = df["balance_residual"].abs()

        p("\nBALANCE RESIDUAL QUANTILES")
        p(
            df["abs_balance_residual"]
            .quantile([0, 0.5, 0.9, 0.95, 0.99, 0.999, 1.0])
            .to_string()
        )

        p("\nTOP 20 ABSOLUTE BALANCE RESIDUAL ROWS")

        cols = [
            "datetime",
            "generation_mw",
            "demand_mw",
            "load_shedding",
            "balance_residual",
        ]

        p(
            df.nlargest(20, "abs_balance_residual")[cols]
            .to_string(index=False)
        )

        # ------------------------------------------------------------
        # Very large demand/generation observations
        # ------------------------------------------------------------
        p("\nTOP 20 GENERATION VALUES")
        p(
            df.nlargest(
                20,
                "generation_mw",
            )[["datetime", "generation_mw", "demand_mw", "load_shedding"]]
            .to_string(index=False)
        )

        p("\nTOP 20 DEMAND VALUES")
        p(
            df.nlargest(
                20,
                "demand_mw",
            )[["datetime", "generation_mw", "demand_mw", "load_shedding"]]
            .to_string(index=False)
        )

    print("\nSaved:")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
