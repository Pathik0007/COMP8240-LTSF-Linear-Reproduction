from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


DATA_DIR = Path("data/synthetic")
ANNOTATION_DIR = Path("experiments/synthetic/annotations")
MANIFEST_PATH = Path("metadata/synthetic_manifest.json")
AUDIT_PATH = Path("experiments/synthetic/construction_audit.txt")

EXPECTED_CONDITIONS = [
    "stable",
    "level_shift",
    "regime_change",
    "noise_medium",
    "noise_high",
    "outliers_low",
    "outliers_high",
    "nonlinear",
]

EXPECTED_ROWS = 10_000
EXPECTED_CHANNELS = 4
EXPECTED_SPLITS = {
    "train": 7000,
    "validation": 1000,
    "test": 2000,
}


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    manifest = json.loads(MANIFEST_PATH.read_text())

    records = {
        record["condition"]: record
        for record in manifest["conditions"]
    }

    lines = []

    def log(text=""):
        print(text)
        lines.append(str(text))

    log("SYNTHETIC DATASET CONSTRUCTION AUDIT")
    log("=" * 80)

    # ------------------------------------------------------------------
    # Global manifest checks
    # ------------------------------------------------------------------
    assert_true(
        manifest["master_seed"] == 8240,
        "Unexpected master seed.",
    )

    assert_true(
        manifest["rows_per_condition"] == EXPECTED_ROWS,
        "Unexpected rows_per_condition.",
    )

    assert_true(
        manifest["channels"]
        == [f"channel_{i}" for i in range(1, 5)],
        "Unexpected channel definition.",
    )

    assert_true(
        list(records.keys()) == EXPECTED_CONDITIONS,
        "Condition list/order differs from expected design.",
    )

    log(f"master_seed: {manifest['master_seed']}")
    log(f"rows_per_condition: {manifest['rows_per_condition']}")
    log(f"number_of_conditions: {len(records)}")
    log()

    stable = None

    for condition in EXPECTED_CONDITIONS:
        record = records[condition]

        data_path = Path(record["data_file"])
        ann_path = Path(record["annotation_file"])

        assert_true(data_path.exists(), f"Missing {data_path}")
        assert_true(ann_path.exists(), f"Missing {ann_path}")

        assert_true(
            sha256_file(data_path) == record["data_sha256"],
            f"Data hash mismatch: {condition}",
        )

        assert_true(
            sha256_file(ann_path) == record["annotation_sha256"],
            f"Annotation hash mismatch: {condition}",
        )

        df = pd.read_csv(data_path, parse_dates=["date"])
        ann = pd.read_csv(ann_path, parse_dates=["date"])

        channel_cols = [
            c for c in df.columns if c.startswith("channel_")
        ]

        assert_true(
            len(df) == EXPECTED_ROWS,
            f"Wrong row count: {condition}",
        )

        assert_true(
            len(channel_cols) == EXPECTED_CHANNELS,
            f"Wrong channel count: {condition}",
        )

        assert_true(
            len(ann) == EXPECTED_ROWS,
            f"Wrong annotation count: {condition}",
        )

        assert_true(
            df["date"].is_monotonic_increasing,
            f"Timestamps not ordered: {condition}",
        )

        assert_true(
            not df["date"].duplicated().any(),
            f"Duplicate timestamps: {condition}",
        )

        expected_delta = pd.Timedelta(hours=1)

        deltas = df["date"].diff().dropna()

        assert_true(
            deltas.eq(expected_delta).all(),
            f"Non-hourly timestamp gap: {condition}",
        )

        assert_true(
            not df[channel_cols].isna().any().any(),
            f"Missing numeric values: {condition}",
        )

        assert_true(
            np.isfinite(df[channel_cols].to_numpy()).all(),
            f"Non-finite numeric values: {condition}",
        )

        assert_true(
            df["date"].equals(ann["date"]),
            f"Data/annotation timestamp mismatch: {condition}",
        )

        split_counts = ann["split"].value_counts().to_dict()

        assert_true(
            split_counts == EXPECTED_SPLITS,
            f"Wrong split counts: {condition}: {split_counts}",
        )

        if condition in {"level_shift", "regime_change"}:
            assert_true(
                ann["is_post_change"].sum() == 2000,
                f"Wrong post-change count: {condition}",
            )

            first_change = int(
                np.flatnonzero(
                    ann["is_post_change"].to_numpy()
                )[0]
            )

            assert_true(
                first_change == 8000,
                f"Wrong change point: {condition}",
            )
        else:
            assert_true(
                ann["is_post_change"].sum() == 0,
                f"Unexpected change labels: {condition}",
            )

        per_channel_flags = [
            f"channel_{i}_is_outlier"
            for i in range(1, 5)
        ]

        assert_true(
            all(c in ann.columns for c in per_channel_flags),
            f"Missing per-channel outlier labels: {condition}",
        )

        reconstructed_any = (
            ann[per_channel_flags]
            .astype(bool)
            .any(axis=1)
        )

        assert_true(
            reconstructed_any.equals(
                ann["is_outlier_any"].astype(bool)
            ),
            f"Inconsistent outlier ground truth: {condition}",
        )

        if stable is None:
            stable = df[channel_cols].copy()

        log(condition)
        log(f"  rows: {len(df)}")
        log(f"  channels: {len(channel_cols)}")
        log(f"  start: {df['date'].min()}")
        log(f"  end: {df['date'].max()}")
        log(f"  missing_values: {int(df[channel_cols].isna().sum().sum())}")
        log(
            "  outlier_cells: "
            f"{int(ann[per_channel_flags].sum().sum())}"
        )
        log()

    # ------------------------------------------------------------------
    # Controlled-design checks
    # ------------------------------------------------------------------
    stable_df = pd.read_csv(
        DATA_DIR / "stable.csv",
        parse_dates=["date"],
    )

    level_df = pd.read_csv(
        DATA_DIR / "level_shift.csv",
        parse_dates=["date"],
    )

    regime_df = pd.read_csv(
        DATA_DIR / "regime_change.csv",
        parse_dates=["date"],
    )

    medium_df = pd.read_csv(
        DATA_DIR / "noise_medium.csv",
        parse_dates=["date"],
    )

    high_df = pd.read_csv(
        DATA_DIR / "noise_high.csv",
        parse_dates=["date"],
    )

    channels = [f"channel_{i}" for i in range(1, 5)]

    # Level-shift condition must be exactly identical to stable before test.
    pre_level_diff = (
        level_df.loc[:7999, channels]
        - stable_df.loc[:7999, channels]
    ).abs().to_numpy().max()

    assert_true(
        pre_level_diff < 1e-10,
        "Level-shift condition differs before change point.",
    )

    level_delta = (
        level_df.loc[8000:, channels].to_numpy()
        - stable_df.loc[8000:, channels].to_numpy()
    )

    expected_offsets = np.array(
        records["level_shift"]["level_shift_offsets"],
        dtype=float,
    )

    observed_offsets = np.median(level_delta, axis=0)

    assert_true(
        np.allclose(
            observed_offsets,
            expected_offsets,
            atol=1e-7,
        ),
        "Level-shift offsets do not match manifest.",
    )

    # Regime-change condition must match stable up to change point.
    pre_regime_diff = (
        regime_df.loc[:8000, channels]
        - stable_df.loc[:8000, channels]
    ).abs().to_numpy().max()

    assert_true(
        pre_regime_diff < 1e-10,
        "Regime-change condition differs before change point.",
    )

    # Noise severity should rise monotonically relative to deterministic signal.
    t = np.arange(EXPECTED_ROWS, dtype=float)

    amplitudes = np.array([1.8, 2.2, 1.6, 2.0])
    slopes = np.array([0.00045, 0.00060, 0.00035, 0.00050])
    phases = np.array([0.0, 0.35, 0.70, 1.05])

    deterministic = (
        t[:, None] * slopes[None, :]
        + amplitudes[None, :]
        * np.sin(
            2.0 * np.pi * t[:, None] / 24.0
            + phases[None, :]
        )
    )

    rms = {}

    for name, frame in [
        ("stable", stable_df),
        ("noise_medium", medium_df),
        ("noise_high", high_df),
    ]:
        residual = frame[channels].to_numpy() - deterministic

        rms[name] = float(
            np.sqrt(np.mean(residual ** 2))
        )

    assert_true(
        rms["stable"] < rms["noise_medium"] < rms["noise_high"],
        "Noise severity is not monotonically increasing.",
    )

    # Outlier severity counts must increase.
    low_ann = pd.read_csv(
        ANNOTATION_DIR / "outliers_low_annotations.csv"
    )

    high_ann = pd.read_csv(
        ANNOTATION_DIR / "outliers_high_annotations.csv"
    )

    low_cells = int(low_ann["outlier_channel_count"].sum())
    high_cells = int(high_ann["outlier_channel_count"].sum())

    assert_true(
        low_cells < high_cells,
        "Outlier severity does not increase.",
    )

    nonlinear_df = pd.read_csv(
        DATA_DIR / "nonlinear.csv"
    )

    nonlinear_values = nonlinear_df[channels].to_numpy()

    assert_true(
        np.isfinite(nonlinear_values).all(),
        "Nonlinear process diverged or became non-finite.",
    )

    log("CONTROLLED DESIGN CHECKS")
    log("-" * 80)
    log(f"pre_change_level_max_abs_difference: {pre_level_diff:.12g}")
    log(
        "observed_level_shift_offsets: "
        + ", ".join(f"{x:.6f}" for x in observed_offsets)
    )
    log(f"pre_change_regime_max_abs_difference: {pre_regime_diff:.12g}")
    log(f"stable_noise_rms: {rms['stable']:.6f}")
    log(f"medium_noise_rms: {rms['noise_medium']:.6f}")
    log(f"high_noise_rms: {rms['noise_high']:.6f}")
    log(f"low_outlier_cells: {low_cells}")
    log(f"high_outlier_cells: {high_cells}")
    log(
        "nonlinear_value_range: "
        f"{nonlinear_values.min():.6f} to "
        f"{nonlinear_values.max():.6f}"
    )
    log()
    log("AUDIT RESULT: PASS")

    AUDIT_PATH.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
