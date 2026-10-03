from pathlib import Path
import csv

INPUT = Path("results/original_replication.csv")
OUTPUT = Path("results/comparison_table.csv")

required_columns = {
    "model",
    "dataset",
    "horizon",
    "paper_mse",
    "paper_mae",
    "reproduced_mse",
    "reproduced_mae",
}

with INPUT.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    if reader.fieldnames is None:
        raise ValueError("Input CSV has no header.")

    missing = required_columns - set(reader.fieldnames)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    rows = list(reader)

output_rows = []

for row in rows:
    paper_mse = float(row["paper_mse"])
    paper_mae = float(row["paper_mae"])
    reproduced_mse = float(row["reproduced_mse"])
    reproduced_mae = float(row["reproduced_mae"])

    mse_abs_diff = reproduced_mse - paper_mse
    mae_abs_diff = reproduced_mae - paper_mae

    mse_pct_diff = 100 * mse_abs_diff / paper_mse
    mae_pct_diff = 100 * mae_abs_diff / paper_mae

    output_rows.append(
        {
            **row,
            "mse_abs_diff": f"{mse_abs_diff:.6f}",
            "mae_abs_diff": f"{mae_abs_diff:.6f}",
            "mse_pct_diff": f"{mse_pct_diff:.2f}",
            "mae_pct_diff": f"{mae_pct_diff:.2f}",
        }
    )

fieldnames = list(output_rows[0].keys())

with OUTPUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(output_rows)

print(
    f"{'Model':<10} {'Dataset':<8} {'H':>5} "
    f"{'Paper MSE':>10} {'Run MSE':>10} {'MSE %':>8} "
    f"{'Paper MAE':>10} {'Run MAE':>10} {'MAE %':>8}"
)

print("-" * 94)

for row in output_rows:
    print(
        f"{row['model']:<10} "
        f"{row['dataset']:<8} "
        f"{int(row['horizon']):>5} "
        f"{float(row['paper_mse']):>10.3f} "
        f"{float(row['reproduced_mse']):>10.3f} "
        f"{float(row['mse_pct_diff']):>7.2f}% "
        f"{float(row['paper_mae']):>10.3f} "
        f"{float(row['reproduced_mae']):>10.3f} "
        f"{float(row['mae_pct_diff']):>7.2f}%"
    )

print()
print(f"Saved comparison table to: {OUTPUT}")
