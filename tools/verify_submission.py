#!/usr/bin/env python3
"""Verify final judge statistics without training or API access."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

r = Path(__file__).resolve().parents[1]
b = r / "experiments/new_data/vlm_judge_v3"

with tempfile.TemporaryDirectory(prefix="comp8240_verify_") as t:
    out = Path(t) / "judge_study_v3"
    cmd = [
        sys.executable, "-B",
        str(r / "tools/vlm_judge/analyze_judge_v3.py"),
        "--cases", str(b / "cases.jsonl"),
        "--judge", "claude46=" + str(b / "claude46_judgments.jsonl"),
        "--human", "rater1=" + str(b / "human_annotations_rater1.csv"),
        "--human", "rater1:batch2=" + str(b / "human_annotations_rater1_batch2.csv"),
        "--human-skip-first", "2",
        "--out-prefix", str(out),
    ]
    run = subprocess.run(cmd, cwd=r, capture_output=True, text=True)
    if run.returncode:
        raise SystemExit(run.stdout + run.stderr)

    expected = json.loads(
        (r / "results/new_data/judge_study_v3.summary.json").read_text()
    )
    actual = json.loads(out.with_suffix(".summary.json").read_text())
    expected.pop("provenance", None)
    actual.pop("provenance", None)

    if actual != expected:
        raise SystemExit("FAIL: statistics differ from the published summary.")

print("JUDGE STATISTICS: EXACT MATCH")
print("No API calls, training or changes to published results.")
