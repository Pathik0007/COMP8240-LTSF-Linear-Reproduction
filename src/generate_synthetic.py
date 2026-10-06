from pathlib import Path
import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Reproducibility configuration
# ---------------------------------------------------------------------

MASTER_SEED = 8240

N = 10_000
N_CHANNELS = 4
START_DATE = "2020-01-01 00:00:00"
FREQ = "h"

TRAIN_END = 7_000
VAL_END = 8_000
TEST_START = VAL_END

BASE_PERIOD = 24.0
BASE_NOISE_SIGMA = 0.15

CHANNEL_NAMES = [f"channel_{i}" for i in range(1, N_CHANNELS + 1)]

AMPLITUDES = np.array([1.8, 2.2, 1.6, 2.0], dtype=float)
SLOPES = np.array([0.00045, 0.00060, 0.00035, 0.00050], dtype=float)
PHASES = np.array([0.0, 0.35, 0.70, 1.05], dtype=float)


ROOT = Path(".")
DATA_DIR = ROOT / "data" / "synthetic"
ANNOTATION_DIR = ROOT / "experiments" / "synthetic" / "annotations"
EXPERIMENT_DIR = ROOT / "experiments" / "synthetic"
METADATA_DIR = ROOT / "metadata"
FIGURE_DIR = ROOT / "results" / "figures" / "synthetic"

MANIFEST_PATH = METADATA_DIR / "synthetic_manifest.json"
SUMMARY_PATH = EXPERIMENT_DIR / "construction_summary.csv"
FIGURE_PATH = FIGURE_DIR / "synthetic_conditions_overview.png"


def ensure_directories():
    for path in [
        DATA_DIR,
        ANNOTATION_DIR,
        EXPERIMENT_DIR,
        METADATA_DIR,
        FIGURE_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def make_dates():
    return pd.date_range(
        start=START_DATE,
        periods=N,
        freq=FREQ,
        name="date",
    )


def make_split_labels():
    split = np.empty(N, dtype=object)
    split[:TRAIN_END] = "train"
    split[TRAIN_END:VAL_END] = "validation"
    split[VAL_END:] = "test"
    return split


def deterministic_signal(t):
    trend = t[:, None] * SLOPES[None, :]

    seasonal = (
        AMPLITUDES[None, :]
        * np.sin(
            2.0 * np.pi * t[:, None] / BASE_PERIOD
            + PHASES[None, :]
        )
    )

    return trend + seasonal


def generate_base_noise():
    rng = np.random.default_rng(MASTER_SEED)

    return rng.normal(
        loc=0.0,
        scale=1.0,
        size=(N, N_CHANNELS),
    )


def generate_stable(base_signal, innovations):
    values = base_signal + BASE_NOISE_SIGMA * innovations

    metadata = {
        "noise_sigma": BASE_NOISE_SIGMA,
        "description": (
            "Stable linear trend plus shared 24-hour periodic structure "
            "with low Gaussian noise."
        ),
    }

    return values, metadata, np.zeros((N, N_CHANNELS), dtype=bool)


def generate_level_shift(base_signal, innovations):
    values = base_signal + BASE_NOISE_SIGMA * innovations

    offsets = 1.5 * AMPLITUDES

    values = values.copy()
    values[TEST_START:] += offsets[None, :]

    metadata = {
        "noise_sigma": BASE_NOISE_SIGMA,
        "change_index": TEST_START,
        "change_split": "test",
        "level_shift_offsets": offsets.tolist(),
        "description": (
            "Stable trend and periodicity retained, with a fixed level "
            "offset introduced exactly at the test boundary."
        ),
    }

    return values, metadata, np.zeros((N, N_CHANNELS), dtype=bool)


def generate_regime_change(base_signal, innovations, t):
    values = base_signal + BASE_NOISE_SIGMA * innovations
    values = values.copy()

    # Preserve continuity at the change point but increase trend slope.
    slope_increment = 3.0 * SLOPES

    post_t = (
        t[TEST_START:] - t[TEST_START]
    )[:, None]

    values[TEST_START:] += post_t * slope_increment[None, :]

    metadata = {
        "noise_sigma": BASE_NOISE_SIGMA,
        "change_index": TEST_START,
        "change_split": "test",
        "post_change_slope_increment": slope_increment.tolist(),
        "description": (
            "Abrupt regime change at the test boundary: trend slope "
            "increases while level continuity and periodicity are retained."
        ),
    }

    return values, metadata, np.zeros((N, N_CHANNELS), dtype=bool)


def generate_noise_condition(base_signal, innovations, sigma):
    values = base_signal + sigma * innovations

    metadata = {
        "noise_sigma": float(sigma),
        "description": (
            f"Same underlying trend and periodicity with Gaussian "
            f"noise sigma={sigma}."
        ),
    }

    return values, metadata, np.zeros((N, N_CHANNELS), dtype=bool)


def generate_outliers(
    base_signal,
    innovations,
    *,
    seed,
    outlier_rate,
):
    values = base_signal + BASE_NOISE_SIGMA * innovations

    rng = np.random.default_rng(seed)

    mask = rng.random((N, N_CHANNELS)) < outlier_rate

    signs = rng.choice(
        np.array([-1.0, 1.0]),
        size=(N, N_CHANNELS),
    )

    spike_scale = 6.0 * AMPLITUDES[None, :]

    spikes = signs * spike_scale

    values = values.copy()
    values[mask] += spikes[mask]

    metadata = {
        "noise_sigma": BASE_NOISE_SIGMA,
        "outlier_seed": seed,
        "outlier_rate": float(outlier_rate),
        "outlier_cells": int(mask.sum()),
        "spike_scale_in_amplitudes": 6.0,
        "description": (
            "Stable baseline with sparse additive spikes. Gaussian noise "
            "level is unchanged so outlier frequency is isolated from "
            "ordinary noise severity."
        ),
    }

    return values, metadata, mask


def generate_nonlinear(base_signal, innovations):
    residual = np.zeros((N, N_CHANNELS), dtype=float)

    residual[0] = BASE_NOISE_SIGMA * innovations[0]
    residual[1] = (
        0.55 * residual[0]
        + BASE_NOISE_SIGMA * innovations[1]
    )

    for i in range(2, N):
        interaction = np.tanh(
            residual[i - 1] * residual[i - 2]
        )

        residual[i] = (
            0.55 * residual[i - 1]
            - 0.15 * residual[i - 2]
            + 0.35 * interaction
            + BASE_NOISE_SIGMA * innovations[i]
        )

    values = base_signal + residual

    metadata = {
        "noise_sigma": BASE_NOISE_SIGMA,
        "ar1": 0.55,
        "ar2": -0.15,
        "nonlinear_interaction": 0.35,
        "interaction_function": "tanh(z[t-1] * z[t-2])",
        "description": (
            "Trend and periodicity plus a stable nonlinear autoregressive "
            "residual. The bounded lag interaction cannot be represented "
            "by a purely linear lag mapping."
        ),
    }

    return values, metadata, np.zeros((N, N_CHANNELS), dtype=bool)


def save_condition(
    name,
    values,
    condition_metadata,
    outlier_mask,
    dates,
    split_labels,
):
    data_path = DATA_DIR / f"{name}.csv"
    annotation_path = ANNOTATION_DIR / f"{name}_annotations.csv"

    data = pd.DataFrame(
        values,
        columns=CHANNEL_NAMES,
    )

    data.insert(0, "date", dates)

    data.to_csv(
        data_path,
        index=False,
        float_format="%.8f",
    )

    is_change_condition = name in {"level_shift", "regime_change"}

    is_post_change = np.zeros(N, dtype=bool)

    if is_change_condition:
        change_index = int(condition_metadata["change_index"])
        is_post_change[change_index:] = True

    annotations = pd.DataFrame({
        "date": dates,
        "split": split_labels,
        "condition": name,
        "is_test_period": np.arange(N) >= TEST_START,
        "is_post_change": is_post_change,
        "is_outlier_any": outlier_mask.any(axis=1),
        "outlier_channel_count": outlier_mask.sum(axis=1),
    })

    for j, channel in enumerate(CHANNEL_NAMES):
        annotations[f"{channel}_is_outlier"] = outlier_mask[:, j]

    annotations.to_csv(annotation_path, index=False)

    record = {
        "condition": name,
        "rows": N,
        "channels": N_CHANNELS,
        "data_file": str(data_path),
        "annotation_file": str(annotation_path),
        "data_sha256": sha256_file(data_path),
        "annotation_sha256": sha256_file(annotation_path),
        **condition_metadata,
    }

    return record

def make_figure(condition_values, dates):
    names = [
        "stable",
        "level_shift",
        "regime_change",
        "noise_high",
        "outliers_high",
        "nonlinear",
    ]

    titles = {
        "stable": "Stable trend + periodicity",
        "level_shift": "Level shift at test boundary",
        "regime_change": "Trend-regime change",
        "noise_high": "High Gaussian noise",
        "outliers_high": "Sparse outlier spikes",
        "nonlinear": "Nonlinear temporal dependence",
    }

    start = TEST_START - 600
    end = TEST_START + 600

    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(13, 10),
        sharex=True,
    )

    axes = axes.ravel()

    for ax, name in zip(axes, names):
        ax.plot(
            dates[start:end],
            condition_values[name][start:end, 0],
            linewidth=1.0,
        )

        ax.axvline(
            dates[TEST_START],
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
        )

        ax.set_title(titles[name])
        ax.grid(alpha=0.2)

    fig.suptitle(
        "Controlled synthetic time-series conditions\n"
        "Channel 1; dashed line = validation/test boundary",
        fontsize=14,
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_PATH,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    ensure_directories()

    dates = make_dates()
    split_labels = make_split_labels()

    t = np.arange(N, dtype=float)

    base_signal = deterministic_signal(t)
    innovations = generate_base_noise()

    conditions = {}

    conditions["stable"] = generate_stable(
        base_signal,
        innovations,
    )

    conditions["level_shift"] = generate_level_shift(
        base_signal,
        innovations,
    )

    conditions["regime_change"] = generate_regime_change(
        base_signal,
        innovations,
        t,
    )

    conditions["noise_medium"] = generate_noise_condition(
        base_signal,
        innovations,
        sigma=0.45,
    )

    conditions["noise_high"] = generate_noise_condition(
        base_signal,
        innovations,
        sigma=0.90,
    )

    conditions["outliers_low"] = generate_outliers(
        base_signal,
        innovations,
        seed=8241,
        outlier_rate=0.0025,
    )

    conditions["outliers_high"] = generate_outliers(
        base_signal,
        innovations,
        seed=8242,
        outlier_rate=0.0100,
    )

    conditions["nonlinear"] = generate_nonlinear(
        base_signal,
        innovations,
    )

    manifest_records = []
    condition_values = {}

    for name, (
        values,
        condition_metadata,
        outlier_mask,
    ) in conditions.items():

        condition_values[name] = values

        record = save_condition(
            name=name,
            values=values,
            condition_metadata=condition_metadata,
            outlier_mask=outlier_mask,
            dates=dates,
            split_labels=split_labels,
        )

        manifest_records.append(record)

    summary = pd.DataFrame(manifest_records)

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    make_figure(
        condition_values,
        dates,
    )

    manifest = {
        "project": "COMP8240 LTSF-Linear reproduction and generalisation",
        "purpose": (
            "Controlled synthetic datasets for testing when the relative "
            "performance of linear long-horizon forecasting models changes."
        ),
        "master_seed": MASTER_SEED,
        "rows_per_condition": N,
        "channels": CHANNEL_NAMES,
        "sampling_frequency": "hourly",
        "start_date": START_DATE,
        "base_equation": (
            "y_t = a*t + A*sin(2*pi*t/P + phase) + epsilon_t"
        ),
        "base_period_hours": BASE_PERIOD,
        "base_noise_sigma": BASE_NOISE_SIGMA,
        "split": {
            "train": {
                "start_index": 0,
                "end_index_exclusive": TRAIN_END,
                "rows": TRAIN_END,
            },
            "validation": {
                "start_index": TRAIN_END,
                "end_index_exclusive": VAL_END,
                "rows": VAL_END - TRAIN_END,
            },
            "test": {
                "start_index": VAL_END,
                "end_index_exclusive": N,
                "rows": N - VAL_END,
            },
        },
        "experimental_design": {
            "chronological_split": True,
            "fixed_random_seeds": True,
            "shared_base_innovations_where_possible": True,
            "one_primary_generating_factor_varied_at_a_time": True,
            "ground_truth_annotations_saved_separately": True,
        },
        "conditions": manifest_records,
        "summary_file": str(SUMMARY_PATH),
        "figure_file": str(FIGURE_PATH),
    }

    with MANIFEST_PATH.open("w", encoding="utf-8") as handle:
        json.dump(
            manifest,
            handle,
            indent=2,
        )

    print("Synthetic dataset construction complete")
    print("--------------------------------------")
    print("master seed:", MASTER_SEED)
    print("rows per condition:", N)
    print("channels:", N_CHANNELS)
    print(
        "split rows:",
        f"train={TRAIN_END},",
        f"validation={VAL_END - TRAIN_END},",
        f"test={N - VAL_END}",
    )

    print("\nConditions:")
    for record in manifest_records:
        print(
            f"- {record['condition']}: "
            f"{record['rows']} rows"
        )

    print("\nSaved:")
    print(DATA_DIR)
    print(ANNOTATION_DIR)
    print(SUMMARY_PATH)
    print(MANIFEST_PATH)
    print(FIGURE_PATH)


if __name__ == "__main__":
    main()
