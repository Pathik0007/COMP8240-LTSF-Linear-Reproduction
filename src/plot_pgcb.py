from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATA_PATH = Path("data/pgcb/processed/pgcb_hourly_core_audited.csv")
OUT_DIR = Path("results/figures/pgcb")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = OUT_DIR / "pgcb_demand_generation_sample.png"


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["datetime"])

    # Selected from the audited long contiguous 2025 hourly segment.
    start = pd.Timestamp("2025-05-04 00:00:00")
    end = pd.Timestamp("2025-05-10 23:00:00")

    sample = df.loc[
        (df["datetime"] >= start)
        & (df["datetime"] <= end)
        & df["was_observed_hour"]
    ].copy()

    if sample.empty:
        raise RuntimeError("Selected PGCB plotting window is empty.")

    expected = pd.date_range(start, end, freq="H")

    if len(sample) != len(expected):
        raise RuntimeError(
            f"Selected window is not fully observed: "
            f"{len(sample)} rows vs {len(expected)} expected."
        )

    if sample["datetime"].duplicated().any():
        raise RuntimeError("Duplicate timestamps found in plotting window.")

    if sample[["generation_mw", "demand_mw"]].isna().any().any():
        raise RuntimeError("Missing demand/generation values in plotting window.")

    fig, ax = plt.subplots(figsize=(11, 5.5))

    ax.plot(
        sample["datetime"],
        sample["demand_mw"],
        label="Demand",
        linewidth=1.6,
    )

    ax.plot(
        sample["datetime"],
        sample["generation_mw"],
        label="Generation",
        linewidth=1.6,
    )

    ax.set_title(
        "PGCB hourly electricity demand and generation\n"
        "Bangladesh, 4–10 May 2025"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("MW")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.autofmt_xdate()
    fig.tight_layout()

    fig.savefig(
        OUT_PATH,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Rows plotted:", len(sample))
    print("Start:", sample["datetime"].min())
    print("End:", sample["datetime"].max())
    print("Missing demand:", int(sample["demand_mw"].isna().sum()))
    print("Missing generation:", int(sample["generation_mw"].isna().sum()))
    print("Saved:", OUT_PATH)


if __name__ == "__main__":
    main()
