# COMP8240 — Replication, Generalisation, and Reliability Audits of Linear Time-Series Forecasting

This repository contains the code and curated evidence for a COMP8240 research reproduction and extension of Zeng, Chen, Zhang, and Xu, **“Are Transformers Effective for Time Series Forecasting?”** (AAAI 2023, [paper](https://doi.org/10.1609/aaai.v37i9.26317), [official code](https://github.com/cure-lab/LTSF-Linear)). The original implementation used for the project is pinned to commit `0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6` in a **separate checkout**; it is not vendored into this repository.

## Start here

Author: Pathik Ahmed — COMP8240, Macquarie University.

- [Reproduction guide](docs/REPRODUCE.md): prerequisites, commands and limitations.
- [Experiment interpretation](docs/EXPERIMENTS.md): protocols and findings.
- [Data index](data/README.md): sources and provenance.
- [Judge protocol and API evidence](tools/vlm_judge/README.md).

Quick verification using standard-library Python, with no training or API key:

    python3 tools/verify_submission.py

Expected: JUDGE STATISTICS: EXACT MATCH.
Published results remain unchanged.

## Scope and completed status

- **Original-paper replication:** ETTh1 multivariate forecasts, Linear/NLinear/DLinear at horizons 96, 192, 336 and 720, look-back 336. These are **12 ETTh1 runs**, not replications of every source-paper dataset. See [`results/comparison_table.csv`](results/comparison_table.csv).
- **DLinear seed sensitivity:** 12 controlled ETTh1 runs, three explicitly set seeds (2021–2023) × four horizons, using a narrowly modified seed assignment in memory with original on-disk code unchanged. Seed 2023 at horizon 336 selected an early validation checkpoint with relatively poor test performance. Logs/configurations provide the details.
- **Controlled new data:** eight four-channel synthetic conditions (10,000 hourly steps each), three trained linear models per condition and two naive baselines (24 trained runs plus 16 baseline evaluations). See [`results/new_data/synthetic_all_conditions_v1_metrics.csv`](results/new_data/synthetic_all_conditions_v1_metrics.csv).
- **Real new data:** Bangladesh Power Grid Company/PGCB public-grid-source demand and generation, two-channel hourly grid, 336-hour look-back and 96-hour forecast. The original complete-window protocol evaluated 8,801 test windows with six methods; a **retrospective** screened protocol evaluated 8,050. See [`results/new_data/pgcb_valid_windows_v1_metrics.csv`](results/new_data/pgcb_valid_windows_v1_metrics.csv), [`results/new_data/pgcb_valid_windows_v2_metrics.csv`](results/new_data/pgcb_valid_windows_v2_metrics.csv), and [`results/new_data/pgcb_v1_v2_common_window_v1.csv`](results/new_data/pgcb_v1_v2_common_window_v1.csv).
- **Vision-language evaluation:** Claude Sonnet 4.6 judged **72 underlying forecast comparisons × two A/B orders = 144 valid image responses**; decisions were compared against 5%-tolerance single-channel MAE/MSE references. There are 72 student annotations, of which **two were AI-assisted and excluded** from unassisted agreement (70 remaining, one rater, partly post-hoc). See [`results/new_data/judge_study_v3.summary.json`](results/new_data/judge_study_v3.summary.json).

**Not completed:** no new Transformer training experiments are claimed here. The original paper's Transformer comparisons are described in the report as *published* baselines, not models newly executed for this extension. Optional future work includes multi-dataset, multi-seed and independent-rater studies.

## Selected findings (descriptive, not inferential)

| Audit | Finding |
|---|---|
| ETTh1 original reproduction | Linear/NLinear single-run MSEs are close to the source-paper ETTh1 values; DLinear deviations warrant seed and checkpoint investigation. |
| DLinear at horizon 96 | Three-seed MSE mean **0.376026**, versus paper **0.375**. |
| DLinear at horizon 192 | Three-seed MSE mean **0.427396**, versus paper **0.405**; a residual gap remains. |
| Synthetic level shift | Best trained MSE **0.473047**, seasonal-naive MSE **0.100460**, for the tested protocol. |
| PGCB common 8,050-window cohort | Linear v1/v2 MSE **1,472,984 / 1,497,298 MW²**. Screening substantially reduces the full-test headline MSE mostly by changing the evaluated window set. |
| Claude visual judge | MAE agreement **39/72 (54.2%)**, Cohen κ **0.271**, reversal consistency **54/72 (75%)**, position-robust MAE accuracy **36/72 (50%)**. |
| Human visual judging | Unassisted human vs MAE **28/70 (40%)**, human vs Claude **39/70 (55.7%)**. One non-expert rater; no inter-human agreement claim. |

These results are tied to the defined datasets, windows, thresholds and protocols. In particular, overlapping forecast windows must not be treated as independent statistical trials, and the best synthetic learned model was selected retrospectively for descriptive summaries.

## Repository guide

- `src/`: data creation, PGCB preprocessing, forecasting, audits and matching-window comparisons.
- `tools/run_dlinear_seed_sweep_v1.py`: controlled ETTh1 seed sensitivity.
- `tools/vlm_judge/`: blinded forecast chart construction, hosted judge execution and offline analysis.
- `tests/`: PGCB screening and cohort-matching fixture tests.
- `data/`: locally generated synthetic and processed PGCB data (raw external files are not redistributed).
- `results/`: curated numerical summaries, comparisons and figures.
- `experiments/`: protocol manifests and selected logs. Heavy checkpoints/arrays stay local.
- [`docs/REPRODUCE.md`](docs/REPRODUCE.md): data prerequisites, commands and reproducibility limits.
- [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md): methodological caveats and selected numeric findings.

## Reproducibility and source provenance

Use Python 3.9 and the project-pinned environment (Torch 1.9.0.post2, NumPy 1.21.6, pandas 1.3.5), a separate official code checkout at the exact commit above, and the original ETTh1 file in the official checkout. PGCB raw data must be obtained lawfully from its source and placed at the local path documented in `docs/REPRODUCE.md`. API keys are never included; offline judge analysis requires no API credits.

**Public code/results do not guarantee zero-effort full reproduction:** original-paper data, the raw PGCB workbook and some large model prediction artifacts are external/local prerequisites. This is stated rather than silently omitted.

## Responsible reporting

We report original unscreened results alongside post-hoc screening sensitivity, avoid picking test-optimal checkpoints, disclose human-label assistance/post-hoc timing, and do not treat a vision judge as a substitute for numerical forecasting accuracy. Code/data licenses and rights to redistribute external data should be checked at their source.

For source-paper attribution, see the [AAAI 2023 article](https://doi.org/10.1609/aaai.v37i9.26317). The COMP8240 final report contains citations and fuller research-ethics discussion.
