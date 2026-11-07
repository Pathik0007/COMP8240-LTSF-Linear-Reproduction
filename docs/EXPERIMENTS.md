# Research evidence and limitations

This file gives the concise meaning of the completed experiments. **The authoritative values are the saved CSV/JSON/logs**, not rounded text below.

## 1. Original-paper replication

AAAI 2023 Zeng et al., ETTh1, 7 variates, look-back 336; Linear, NLinear and DLinear at horizons 96/192/336/720. All 12 baseline replication MSE/MAE entries are in `results/comparison_table.csv`. Original repository commit is `0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6`.

DLinear seed sensitivity (three explicit seeds per horizon; *not* three iterations from `--itr=3`):

| Horizon | Published MSE | 2021 | 2022 | 2023 |
|---|---:|---:|---:|---:|
| 96 | .375 | .384144 | .372845 | .371089 |
| 192 | .405 | .443487 | .422079 | .416621 |
| 336 | .439 | .446925 | .439067 | .527392 |
| 720 | .472 | .504245 | .509362 | .491901 |

At horizon 336, the seed-2023 validation-selected checkpoint came from epoch 1 with validation loss ≈1.05178 and final test MSE ≈0.527392; a later epoch had smaller *logged test* loss but higher validation loss. Never substitute the test-best epoch for validation-selected results. Differences may involve seed and temporal validation/test mismatch; they do not prove a specific mechanism caused the source-paper discrepancy.

## 2. Controlled synthetic data

Eight conditions × 10,000 hourly steps × four channels, train/validation/test 70%/10%/20%; look-back 336, horizon 96. Three learned models (Linear/NLinear/DLinear) and Repeat/SeasonalNaive24 for each condition. 24 trained + 16 baseline results. Descriptive best-trained native-unit MSE vs seasonal-naive MSE: level shift `.473047` vs `.100460`, regime change `.105292` vs `.059862`, stable `.024997` vs `.044657`; full table in `results/new_data/synthetic_all_conditions_v1_summary.json`.

Best-trained is selected retrospectively using test results and must **never** be described as a deployable model selection strategy. The source generated data can be recreated with `src/generate_synthetic.py` and the manifest; results are single-seed unless stated otherwise.

## 3. PGCB electricity demand and generation

The processed grid spans 89,101 hourly timestamps, of which 88,046 were observed finite before screening. v1 uses only fully observed input+target windows: train 21,316, validation 4,228, test 8,801. Its Linear demand-only test MSE is 2,734,295 MW² and MAE 889.176 MW. No imputation is performed, and scaler fitting is on training observations only.

v2 **post-hoc** screening rejects 23 observed finite hours using both pre-existing flags and a researcher-selected 1,000–30,000 MW range on *both* channels; its valid-window counts are train 19,229, validation 3,083, test 8,050. The thresholds are heuristic, not established engineering bounds and not a deployable policy. Do not claim the retrospectively derived pre-existing flags were fit exclusively on training data.

The full-test headline MSE changes greatly because eligible test windows change. On the **same 8,050 test windows**, Linear v1 MSE ≈1,472,984 versus v2 ≈1,497,298 MW²; NLinear v1 ≈1,521,933 vs v2 ≈1,515,718; DLinear v1 ≈1,507,228 vs v2 ≈1,503,592. This is a descriptive comparison that controls the tested origin set, not a randomised causal treatment effect. `Repeat` is independently verified on the same windows: MAE 1,471.526 MW, MSE 3,576,310.848 MW².

Data licensing, rights and preprocessing must be checked before redistribution; **do not push the raw PGCB workbook** or any protected provider file without permission.

## 4. Forecast-image judging

Claude Sonnet 4.6 judged 144 A/B and B/A image panels for 72 base cases. With a 5% relative MAE tie threshold, original-order agreement is 39/72 (54.17%), κ .2707; position consistency is 54/72, and position-robust correct MAE judgments are 36/72. Human versus MAE is 28/70 after excluding the first two AI-assisted labels; one rater and later annotation after seeing aggregate Claude results. The 5%-tie threshold is an evaluation choice: see `results/new_data/judge_study_v3.summary.json` for 0–20% threshold sensitivity and uncertainty estimates.

This study assesses **reliability of visual preference against a numerical ranking rule**, not universal ability to recognise the best forecast. Chart visual similarity, axis scaling, and selected correlated windows constrain interpretation.

## 5. Research integrity

Keep the original model checkpoints and historical failed JSON logs locally; do not rewrite past outputs. Distinguish reproduced original implementation from extension code, distinguish per-window from independent trial counts, document exclusion rules and human-assistance disclosure, and cite the source paper and original dataset owners. The final ACL report should contain a separate ethics/limitations section.
