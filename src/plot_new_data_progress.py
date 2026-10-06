from pathlib import Path
import json

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd


PGCB_SUMMARY_PATH = Path("experiments/pgcb/preprocessing_summary.json")
PGCB_FIG_PATH = Path("results/figures/pgcb/pgcb_demand_generation_sample.png")

SYN_MANIFEST_PATH = Path("metadata/synthetic_manifest.json")
SYN_AUDIT_PATH = Path("experiments/synthetic/construction_audit.txt")
SYN_FIG_PATH = Path("results/figures/synthetic/synthetic_conditions_overview.png")

OUT_DIR = Path("results/figures/new_data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FIG_PATH = OUT_DIR / "new_data_progress_overview.png"
OUT_TEXT_PATH = OUT_DIR / "new_data_progress_summary.txt"


def read_pgcb_summary():
    with PGCB_SUMMARY_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_synthetic_summary():
    with SYN_MANIFEST_PATH.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    audit_text = SYN_AUDIT_PATH.read_text(encoding="utf-8")

    lines = audit_text.splitlines()

    extracted = {
        "pre_change_level_max_abs_difference": None,
        "observed_level_shift_offsets": None,
        "pre_change_regime_max_abs_difference": None,
        "stable_noise_rms": None,
        "medium_noise_rms": None,
        "high_noise_rms": None,
        "low_outlier_cells": None,
        "high_outlier_cells": None,
        "nonlinear_value_range": None,
        "audit_result": None,
    }

    for line in lines:
        line = line.strip()

        if line.startswith("AUDIT RESULT:"):
            extracted["audit_result"] = line.split(":", 1)[1].strip()
            continue

        for key in list(extracted.keys()):
            prefix = key + ":"
            if line.startswith(prefix):
                extracted[key] = line[len(prefix):].strip()

    return manifest, extracted


def add_text_block(ax, title, lines):
    ax.axis("off")
    text = title + "\n\n" + "\n".join(lines)
    ax.text(
        0.0,
        1.0,
        text,
        va="top",
        ha="left",
        fontsize=10.5,
        family="monospace",
    )


def main():
    pgcb = read_pgcb_summary()
    manifest, syn_audit = read_synthetic_summary()

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1.15, 1])

    ax_pgcb_img = fig.add_subplot(gs[0, 0])
    ax_pgcb_text = fig.add_subplot(gs[0, 1])
    ax_syn_img = fig.add_subplot(gs[1, 0])
    ax_syn_text = fig.add_subplot(gs[1, 1])

    # PGCB image
    pgcb_img = mpimg.imread(PGCB_FIG_PATH)
    ax_pgcb_img.imshow(pgcb_img)
    ax_pgcb_img.axis("off")
    ax_pgcb_img.set_title("PGCB external dataset sample window", fontsize=12)

    # PGCB text
    pgcb_lines = [
        "Source: UCI / PGCB Bangladesh hourly power dataset",
        f"Raw rows: {pgcb['raw_rows']}",
        f"Exact-hour rows retained: {pgcb['exact_hour_rows_retained']}",
        f"Off-hour rows excluded: {pgcb['off_hour_rows_excluded']}",
        f"Duplicate exact-hour groups: {pgcb['duplicate_timestamp_groups_exact_hour']}",
        f"Missing hours after reindex: {pgcb['missing_hours_after_reindex']}",
        f"Observed-hour coverage: {pgcb['observed_hour_coverage_percent']:.3f}%",
        f"Rows with scale-anomaly flags: {pgcb['total_rows_with_any_scale_anomaly']}",
        "",
        "Preprocessing policy:",
        "- exact-hour benchmark only",
        "- duplicate timestamps collapsed by median",
        "- missing hours preserved",
        "- no interpolation",
        "- anomaly values retained with flags",
    ]
    add_text_block(ax_pgcb_text, "PGCB preprocessing progress", pgcb_lines)

    # Synthetic image
    syn_img = mpimg.imread(SYN_FIG_PATH)
    ax_syn_img.imshow(syn_img)
    ax_syn_img.axis("off")
    ax_syn_img.set_title("Controlled synthetic conditions overview", fontsize=12)

    # Synthetic text
    conditions = [c["condition"] for c in manifest["conditions"]]
    syn_lines = [
        f"Master seed: {manifest['master_seed']}",
        f"Rows per condition: {manifest['rows_per_condition']}",
        f"Channels: {len(manifest['channels'])}",
        "Split: train=7000, validation=1000, test=2000",
        "",
        "Conditions:",
        "- " + ", ".join(conditions[:4]),
        "- " + ", ".join(conditions[4:]),
        "",
        "Audit evidence:",
        f"- level pre-change diff: {syn_audit['pre_change_level_max_abs_difference']}",
        f"- regime pre-change diff: {syn_audit['pre_change_regime_max_abs_difference']}",
        f"- noise RMS: {syn_audit['stable_noise_rms']} -> {syn_audit['medium_noise_rms']} -> {syn_audit['high_noise_rms']}",
        f"- outlier cells: {syn_audit['low_outlier_cells']} -> {syn_audit['high_outlier_cells']}",
        f"- nonlinear range: {syn_audit['nonlinear_value_range']}",
        f"- audit result: {syn_audit['audit_result']}",
    ]
    add_text_block(ax_syn_text, "Synthetic construction progress", syn_lines)

    fig.suptitle(
        "New-data progress: external dataset preparation and controlled synthetic construction",
        fontsize=15,
        y=0.98,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.965])

    fig.savefig(
        OUT_FIG_PATH,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)

    summary_text = []
    summary_text.append("NEW DATA PROGRESS SUMMARY")
    summary_text.append("=" * 80)
    summary_text.append("")
    summary_text.append("PGCB external dataset")
    summary_text.append(f"- raw_rows: {pgcb['raw_rows']}")
    summary_text.append(f"- exact_hour_rows_retained: {pgcb['exact_hour_rows_retained']}")
    summary_text.append(f"- off_hour_rows_excluded: {pgcb['off_hour_rows_excluded']}")
    summary_text.append(f"- duplicate_timestamp_groups_exact_hour: {pgcb['duplicate_timestamp_groups_exact_hour']}")
    summary_text.append(f"- missing_hours_after_reindex: {pgcb['missing_hours_after_reindex']}")
    summary_text.append(f"- observed_hour_coverage_percent: {pgcb['observed_hour_coverage_percent']:.6f}")
    summary_text.append(f"- total_rows_with_any_scale_anomaly: {pgcb['total_rows_with_any_scale_anomaly']}")
    summary_text.append("")
    summary_text.append("Synthetic dataset")
    summary_text.append(f"- master_seed: {manifest['master_seed']}")
    summary_text.append(f"- rows_per_condition: {manifest['rows_per_condition']}")
    summary_text.append(f"- channels: {len(manifest['channels'])}")
    summary_text.append(f"- conditions: {', '.join(conditions)}")
    summary_text.append(f"- stable_noise_rms: {syn_audit['stable_noise_rms']}")
    summary_text.append(f"- medium_noise_rms: {syn_audit['medium_noise_rms']}")
    summary_text.append(f"- high_noise_rms: {syn_audit['high_noise_rms']}")
    summary_text.append(f"- low_outlier_cells: {syn_audit['low_outlier_cells']}")
    summary_text.append(f"- high_outlier_cells: {syn_audit['high_outlier_cells']}")
    summary_text.append(f"- audit_result: {syn_audit['audit_result']}")

    OUT_TEXT_PATH.write_text("\n".join(summary_text) + "\n", encoding="utf-8")

    print("New-data progress presentation evidence created")
    print("----------------------------------------------")
    print("Saved:")
    print(OUT_FIG_PATH)
    print(OUT_TEXT_PATH)


if __name__ == "__main__":
    main()
