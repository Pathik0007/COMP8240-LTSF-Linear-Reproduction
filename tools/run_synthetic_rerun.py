#!/usr/bin/env python3
"""Plan or execute all 24 synthetic runs using the pinned official runner."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHA = "0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6"
CONDITIONS = (
    "stable", "level_shift", "regime_change", "noise_medium",
    "noise_high", "outliers_low", "outliers_high", "nonlinear"
)
MODELS = ("Linear", "NLinear", "DLinear")

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--official", type=Path, default=Path.home() / "Documents/LTSF-Linear")
    p.add_argument("--output-root", type=Path, default=ROOT / "experiments/reruns/synthetic_all")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true", help="Actually train; default is a plan only")
    a = p.parse_args()

    official, output = a.official.resolve(), a.output_root.resolve()
    head = subprocess.check_output(
        ["git", "-C", str(official), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != SHA:
        raise RuntimeError("Unexpected official source revision: " + head)

    for flags in ([], ["--cached"]):
        subprocess.run(
            ["git", "-C", str(official), "diff", *flags, "--quiet"], check=True
        )

    entry = official / "run_longExp.py"
    source_hash = hashlib.sha256(entry.read_bytes()).hexdigest()
    plan = []

    for condition in CONDITIONS:
        data = ROOT / "data/synthetic" / (condition + ".csv")
        if not data.is_file():
            raise RuntimeError("Missing input: " + str(data))
        data_hash = hashlib.sha256(data.read_bytes()).hexdigest()

        for model in MODELS:
            dest = output / condition / model
            if dest.exists():
                raise RuntimeError("Existing run preserved; choose another output root: " + str(dest))
            plan.append((condition, model, data, data_hash, dest))
            print("PLAN", condition, model, dest)

    if not a.execute:
        print("SYNTHETIC PLAN: PASS (24 runs; no training or files written)")
        return

    for condition, model, data, data_hash, dest in plan:
        dest.mkdir(parents=True, exist_ok=False)
        metadata = dict(
            condition=condition, model=model, official_sha=SHA,
            runner_sha256=source_hash, data_sha256=data_hash,
            seed=2021, seq_len=336, pred_len=96, status="running"
        )
        (dest / "protocol.json").write_text(json.dumps(metadata, indent=2) + "\n")

        cmd = [
            sys.executable, "-u", str(entry), "--is_training", "1",
            "--model_id", "COMP8240_" + condition + "_336_96",
            "--model", model, "--data", "custom",
            "--root_path", str(data.parent), "--data_path", data.name,
            "--features", "M", "--target", "channel_1", "--freq", "h",
            "--seq_len", "336", "--label_len", "48", "--pred_len", "96",
            "--enc_in", "4", "--dec_in", "4", "--c_out", "4",
            "--moving_avg", "25", "--batch_size", "32",
            "--learning_rate", "0.005", "--train_epochs", "10",
            "--patience", "3", "--num_workers", "0", "--itr", "1",
            "--des", "COMP8240_A3_SYN_V1",
            "--checkpoints", str(dest / "checkpoints")
        ]
        with (dest / "run.log").open("w") as log:
            completed = subprocess.run(
                cmd, cwd=dest, stdout=log, stderr=subprocess.STDOUT
            )

        metadata.update(
            status="PASS" if completed.returncode == 0 else "FAILED",
            return_code=completed.returncode, command=cmd
        )
        (dest / "protocol.json").write_text(json.dumps(metadata, indent=2) + "\n")
        if completed.returncode:
            raise RuntimeError("Training failed; inspect " + str(dest / "run.log"))

    print("SYNTHETIC TRAINING: COMPLETE; run the numerical verifier next")

if __name__ == "__main__":
    main()
