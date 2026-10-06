from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_PATH = Path("results/comparison_table.csv")
OUT_DIR = Path("results/figures/replication")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIGURE_PATH = OUT_DIR / "etth1_replication_mse_comparison.png"
SUMMARY_PATH = OUT_DIR / "etth1_replication_summary.csv"


def main():
    df = pd.read_csv(INPUT_PATH)

    model_order = ["Linear", "NLinear", "DLinear"]
    horizons = [96, 192, 336, 720]

    # --------------------------------------------------------------
    # Compact numerical summary for presentation / narration
    # --------------------------------------------------------------
    summary = (
        df.groupby("model", as_index=False)
        .agg(
            mean_abs_mse_pct_diff=("mse_pct_diff", lambda x: x.abs().mean()),
            max_abs_mse_pct_diff=("mse_pct_diff", lambda x: x.abs().max()),
            mean_abs_mae_pct_diff=("mae_pct_diff", lambda x: x.abs().mean()),
            max_abs_mae_pct_diff=("mae_pct_diff", lambda x: x.abs().max()),
        )
    )

    summary["model"] = pd.Categorical(
        summary["model"],
        categories=model_order,
        ordered=True,
    )

    summary = summary.sort_values("model")

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
        float_format="%.3f",
    )

    # --------------------------------------------------------------
    # Presentation figure
    # --------------------------------------------------------------
    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(13.5, 4.8),
        sharey=False,
    )

    width = 0.34
    x = np.arange(len(horizons))

    for ax, model in zip(axes, model_order):
        part = (
            df.loc[df["model"] == model]
            .set_index("horizon")
            .loc[horizons]
            .reset_index()
        )

        ax.bar(
            x - width / 2,
            part["paper_mse"],
            width,
            label="Published MSE",
        )

        ax.bar(
            x + width / 2,
            part["reproduced_mse"],
            width,
            label="Reproduced MSE",
        )

        for i, row in part.iterrows():
            pct = row["mse_pct_diff"]

            ax.text(
                x[i],
                max(row["paper_mse"], row["reproduced_mse"]) + 0.012,
                f"{pct:+.2f}%",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

        ax.set_title(model)
        ax.set_xticks(x)
        ax.set_xticklabels(horizons)
        ax.set_xlabel("Forecast horizon")
        ax.set_ylabel("MSE")
        ax.grid(axis="y", alpha=0.22)

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.suptitle(
        "ETTh1 replication: published vs reproduced MSE",
        fontsize=15,
        y=1.08,
    )

    fig.text(
        0.5,
        -0.02,
        "Labels show percentage difference from the published result.",
        ha="center",
        fontsize=10,
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_PATH,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Replication presentation evidence created")
    print("-----------------------------------------")

    print("\nSummary:")
    print(summary.to_string(index=False))

    print("\nSaved:")
    print(FIGURE_PATH)
    print(SUMMARY_PATH)


if __name__ == "__main__":
    main()
