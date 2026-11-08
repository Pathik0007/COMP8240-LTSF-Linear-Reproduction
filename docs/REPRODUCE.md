# Reproduce and audit the project

## Quick offline verification

From the repository root:

    python3 tools/verify_submission.py

Expected: JUDGE STATISTICS: EXACT MATCH.
This recomputes every statistical field from saved cases, responses and both
human batches, excluding the two assisted labels. Machine-specific provenance
is excluded from equality. Temporary outputs are removed automatically.
No API calls, training, external datasets or Conda environment are needed.

## Forecasting prerequisites

Run commands below from the project root. The historical training environment
is named comp8240-ltsf. Install the workbook dependency there if needed:

    conda run -n comp8240-ltsf python -m pip install -r environment/pip-requirements.txt

For a new environment, environment/environment.yml is the portable starting
specification. See environment/environment_notes.md and preserved exports for
the distinct Mac and Kaggle runtimes. Different platforms/package builds may
produce different results.

For the historical CPU setting, disable CUDA before optional training:

    export CUDA_VISIBLE_DEVICES=""

Use a separate clean official checkout at the pinned revision:

    git clone https://github.com/cure-lab/LTSF-Linear.git "$HOME/Documents/LTSF-Linear"
    git -C "$HOME/Documents/LTSF-Linear" checkout 0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6

Reuse an existing clean checkout at that revision instead of cloning again.
Place canonical ETTh1.csv in its dataset directory.
Source: https://github.com/zhouhaoyi/ETDataset/tree/main/ETT-small.
Expected ETTh1 SHA-256:
f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066.

PGCB source: https://doi.org/10.24432/C59P6V (UCI 1175; CC BY 4.0).
Download PGCB_date_power_demand.xlsx into data/pgcb/raw/.
Expected workbook SHA-256:
cd10a95b98033594b70ff176d69e9feaa2ade5f43b8ae4e5640a0f9deabc9da1.
The workbook hash differs from the processed-input hash recorded by training.

Prepare local data in the training environment:

    conda run -n comp8240-ltsf python src/generate_synthetic.py
    conda run -n comp8240-ltsf python src/audit_synthetic.py
    conda run -n comp8240-ltsf python src/preprocess_pgcb.py

Construction scripts also write their documented audits, manifests and figures.

## Separate rerun destinations

Forecasting examples write under experiments/reruns/, ignored by Git.
Published evidence remains unchanged. Choose a fresh output directory when
repeating a completed run.

Training commands below are optional reproduction instructions; they do not
need to be executed again for submission.

## ETTh1 replication and seed sensitivity

Plan all twelve DLinear sensitivity runs without training:

    conda run -n comp8240-ltsf python tools/run_dlinear_seed_sweep_v1.py --project "$PWD" --seeds 2021,2022,2023 --horizons 96,192,336,720 --output-root "$PWD/experiments/reruns/dlinear_seed_sweep" --dry-run

For actual execution, remove --dry-run.
Seeds 2021-2023 are three total seeds; seed 2021 repeats the baseline.
Inspect each run.log and config.json.

Linear/NLinear replication, using separate working directories and logs:

    PROJECT_REPO="$PWD" LOG_DIR="$PWD/experiments/reruns/original_etth1" conda run --no-capture-output -n comp8240-ltsf bash experiments/original_etth1/run_linear_nlinear_etth1.sh

This executes eight runs: two models and four horizons.
The four DLinear seed-2021 configurations complete the baseline scope.
The official source remains unchanged.

## All eight synthetic conditions

Plan 24 runs: eight conditions including Stable, three learned models:

    python3 tools/run_synthetic_rerun.py --dry-run

Optional training:

    conda run --no-capture-output -n comp8240-ltsf python tools/run_synthetic_rerun.py --execute

The driver uses official run_longExp.py and its custom-dataset loader:
seed 2021, input 336, horizon 96, batch 32, LR 0.005, maximum ten epochs,
patience three. Each run is stored under:
experiments/reruns/synthetic_all/condition/model.

The historical seven-condition shell runner remains as provenance;
use the new driver to reproduce all eight conditions.

After training, verify predictions against logs and recompute all 40 evaluations:

    conda run -n comp8240-ltsf python src/verify_synthetic_all_v1.py --run-root "$PWD/experiments/reruns/synthetic_all" --output-dir "$PWD/experiments/reruns/synthetic_check"

Expected: VERIFICATION: PASS, 24 learned-model runs and 16 baseline evaluations.
Compare the new CSV with results/new_data/synthetic_all_conditions_v1_metrics.csv.
Scaled and native-unit MSE/MAE are separate fields.
Without --run-root, the verifier can inspect the historical split layout
when the local prediction arrays exist.

## PGCB and common-window comparison

PGCB uses adapted loading/training around unchanged official model classes.
v1 uses complete observed windows; v2 is the post-hoc screening sensitivity.

Audit eligibility without training:

    conda run -n comp8240-ltsf python src/pgcb_valid_window_benchmark_v2.py --project "$PWD" --audit-only

Optional full reproduction:

    conda run --no-capture-output -n comp8240-ltsf python src/pgcb_valid_window_benchmark_v1.py --project "$PWD" --output-dir "$PWD/experiments/reruns/pgcb_v1"
    conda run --no-capture-output -n comp8240-ltsf python src/pgcb_valid_window_benchmark_v2.py --project "$PWD" --output-dir "$PWD/experiments/reruns/pgcb_v2"
    conda run -n comp8240-ltsf python src/compare_pgcb_v1_v2.py --project "$PWD" --v1-dir "$PWD/experiments/reruns/pgcb_v1" --v2-dir "$PWD/experiments/reruns/pgcb_v2" --out-prefix experiments/reruns/pgcb_common_window

Each training directory contains protocol.json, metrics.csv and prediction arrays.
Expected test counts: v1 8,801; v2/common 8,050.
Compare new metrics against the published CSV summaries.

Repeat is reconstructed from the final value of the saved last-24-hour block.
The original five-method common-window CSV is preserved. New comparison output
also includes Repeat, whose report values were previously checked separately.

## Verification and access

Existing fixture checks, without forecasting or paid API calls:

    conda run -n comp8240-ltsf python -m unittest discover -s tests -p test_verified_fixes_v1.py -v

GitHub remains public; a marker needs no collaborator access to read evidence.
Full forecast reproduction requires external input data and compatible packages.
Published summaries alone are not sufficient to recompute prediction-level metrics.

Overlapping windows and post-hoc screening do not establish independent or
causal results. The completed Claude study requires no new calls.
Historical settings, six parsing failures and valid-response cost scope
are documented in tools/vlm_judge/README.md.
