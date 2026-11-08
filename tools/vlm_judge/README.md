# Blinded hosted vision-language forecast judging — completed study

This directory contains the versioned preparation, execution and analysis code for a completed **Claude Sonnet 4.6** image-judging experiment. The early `prepare_cases.py`/`run_vlm_judge.py` files document a previous planned local-VLM workflow; **they are not the hosted provider used for the reported results**.

## Actual protocol

- Charts: 72 underlying comparisons, each shown as both `ab` and `ba` = 144 image panels; 48 synthetic comparisons (six per condition) plus 24 PGCB comparisons. Single-channel 96-hour forecast plots show observed future and anonymous A/B forecasts, with no model identifiers or numeric errors sent to the judge.
- Model/provider: Anthropic Claude `claude-sonnet-4-6`; prompt version `comp8240_forecast_visual_judge_v1_2026-11-02` in `run_hosted_api_judge_v3.py`.
- Output: 144 valid JSON responses. 142 were saved with 256 max output tokens using v2; two retries required 1024 max output tokens using v3. Earlier malformed responses remain in error logs.
- Numerical reference: relative MAE/MSE difference ≤5% is a tie; reference calculated on the plotted **single target channel**, not the aggregate training loss. Compare original order only for primary numerical agreement; reversed order is a repeated measurement for position consistency.
- Student annotations: 24 original + 48 later = 72 charts, but first two were AI-assisted and excluded. The later 48 were collected after seeing aggregate Claude results; one rater, not an expert panel.

## Published descriptive audit

- Claude–MAE: **39/72**, 54.17%, κ **0.2707**.
- Claude–MSE: **40/72**, 55.56%.
- Reversal consistency: **54/72**, 75%.
- Position-robust and MAE-correct: **36/72**, 50%.
- Human–MAE: **28/70**, 40%; human–Claude **39/70**, 55.71%.
- Analysis supports alternative numerical tie thresholds and descriptive confidence strata. Selected windows can overlap; exploratory intervals must not be interpreted as independent-population inferential guarantees.

## Re-run the offline analysis (no API key needed)

From the project root with the pinned environment:

```bash
conda run --no-capture-output -n comp8240-ltsf python tools/vlm_judge/analyze_judge_v3.py \
  --cases experiments/new_data/vlm_judge_v3/cases.jsonl \
  --judge claude46=experiments/new_data/vlm_judge_v3/claude46_judgments.jsonl \
  --human rater1=experiments/new_data/vlm_judge_v3/human_annotations_rater1.csv \
  --human rater1:batch2=experiments/new_data/vlm_judge_v3/human_annotations_rater1_batch2.csv \
  --human-skip-first 2 --out-prefix results/new_data/judge_study_v3
```

`cases.jsonl` is the **post-judging numerical gold manifest**; keep it hidden from any future blinded rater or judge until their annotations are final. `public_cases.jsonl` is the blinded input. The gold case manifest and anonymised charts can be published after the completed annotations; never expose gold labels to a judge or annotator before their decision. Confirm redistribution and privacy implications before publication.

For any new API runs: obtain the image inputs, supply your own provider key through an environment variable, use `--dry-run` and a small `--limit` pilot, and monitor provider billing. Do **not** commit keys, `.env` files or billing records. The existing results can be analysed without further API calls.


## Historical API evidence and cost scope

The completed study has 144 valid final responses in
`experiments/new_data/vlm_judge_v3/claude46_judgments.jsonl`.

Recorded output-token limits were 256 for 142 valid responses and
1,024 for two recovered responses. The existing failure log,
`experiments/new_data/vlm_judge_v3/claude46_judgments_errors.jsonl`,
records six parsing failures across four image presentations.

The sum of estimated costs recorded for the 144 valid responses is
US$0.68568. Historical failed-response token usage was not recorded,
so this is not a verified total charge for every attempted request.

These historical responses and annotations are preserved unchanged.
Offline statistical verification requires no provider key or new API calls.
