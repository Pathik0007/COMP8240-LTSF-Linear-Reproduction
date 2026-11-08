# Reproduce and audit the COMP8240 project

## Quick offline verification

From the repository root:

    python3 tools/verify_submission.py

This uses standard-library Python to recompute every statistical field in the
final judge summary from committed cases, responses and both annotation batches.
The two assisted labels are excluded. Machine-specific provenance is excluded
from the equality check. Temporary outputs are removed automatically.

No Conda environment, external dataset or API key is needed for this check.
Forecasting stages below have separate data and training dependencies.
The portable environment YAML omits the original installation prefix and includes
the Excel reader. Historical platform exports remain original runtime evidence.

## Prerequisites

1. Obtain the original [LTSF-Linear repository](https://github.com/cure-lab/LTSF-Linear) into `~/Documents/LTSF-Linear`, checkout commit **`0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6`**, and place `ETTh1.csv` at `~/Documents/LTSF-Linear/dataset/ETTh1.csv`. The project code checks this revision for several experiments.
2. Clone this repository at `~/Documents/COMP8240-LTSF-Linear-Reproduction` on macOS, or adjust the absolute default paths/CLI flags in the documented scripts.
3. Use a Conda environment called `comp8240-ltsf` with **Python 3.9.19, PyTorch 1.9.0.post2, NumPy 1.21.6 and pandas 1.3.5**, plus scikit-learn, matplotlib and openpyxl as required. The historical environment may need compatible system packages. Verify installed versions before trying to reproduce metric values.
4. For PGCB, obtain the source workbook from its rightful publisher and place it at `data/pgcb/raw/PGCB_date_power_demand.xlsx`; original raw workbook and any redistribution rights are **not supplied** here. Run preprocessing from the project root to create `data/pgcb/processed/pgcb_hourly_core_audited.csv`.

## Reproduce or audit stages

```bash
# No GPU/API calls; code test using synthetic fixtures
conda run --no-capture-output -n comp8240-ltsf python -m unittest discover -s tests -p test_verified_fixes_v1.py -v

# Original source preconditions and controlled DLinear seed-run plan, no training
conda run --no-capture-output -n comp8240-ltsf python tools/run_dlinear_seed_sweep_v1.py --seeds 2021,2022,2023 --horizons 96,192,336,720 --dry-run

# Generate and audit eight controlled synthetic datasets
conda run --no-capture-output -n comp8240-ltsf python src/generate_synthetic.py
conda run --no-capture-output -n comp8240-ltsf python src/audit_synthetic.py

# Prepare PGCB from separately acquired source workbook
conda run --no-capture-output -n comp8240-ltsf python src/preprocess_pgcb.py

# No training: inspect screened PGCB windows
conda run --no-capture-output -n comp8240-ltsf python src/pgcb_valid_window_benchmark_v2.py --audit-only

# Offline forecast-judge analysis from saved cases and responses
conda run --no-capture-output -n comp8240-ltsf python tools/vlm_judge/analyze_judge_v3.py \
  --cases experiments/new_data/vlm_judge_v3/cases.jsonl \
  --judge claude46=experiments/new_data/vlm_judge_v3/claude46_judgments.jsonl \
  --human rater1=experiments/new_data/vlm_judge_v3/human_annotations_rater1.csv \
  --human rater1:batch2=experiments/new_data/vlm_judge_v3/human_annotations_rater1_batch2.csv \
  --human-skip-first 2 --out-prefix results/new_data/judge_study_v3
```

The complete CPU synthetic benchmark is orchestrated by `src/run_synthetic_family_v1.sh` for seven conditions; the stable condition is in earlier scripts. Run only if the complete input datasets and official source checkout exist. **Historical outputs are preserved by the scripts: a completed run will generally not be overwritten.** The DLinear seed sweep similarly refuses existing output folders. Do not rerun all jobs without a clean, separately versioned output destination.

The PGCB v1/v2 training commands are `python src/pgcb_valid_window_benchmark_v1.py` and `python src/pgcb_valid_window_benchmark_v2.py` from the project root in the pinned environment; each requires the processed workbook above and writes results plus large local prediction `.npz` files. Matched-window comparison uses `python src/compare_pgcb_v1_v2.py` after both are complete. The public small summary files are sufficient to *inspect* the reported results but are not sufficient to recompute every prediction-level PGCB metric independently without the raw workbook and locally saved large arrays.

## Privacy, integrity and access

Never commit provider API keys, `.env` files, raw provider workbooks, payment details, model checkpoints or personal temporary archives. The LLM evaluation has concluded; the gold-label `cases.jsonl` manifest was blinded during judging and is included in the curated post-judging release for full offline metric reproducibility. Never reveal it to a future rater before their annotations are collected. Versioned logs and source revisions are evidence; don't silently edit them to match headline results.

Check the public GitHub URL without signing in before final submission and include it in the final report's replication footnote. Do not promise an external user can run full data experiments without obtaining the original input datasets and compatible environment.
