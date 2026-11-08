# Project data and evidence

| Data | Source | Documentation |
|---|---|---|
| ETTh1 | Canonical ETT release; separate official checkout | [Reproduction guide](../docs/REPRODUCE.md) |
| PGCB workbook | [UCI dataset 1175](https://doi.org/10.24432/C59P6V); CC BY 4.0 | [PGCB provenance and preprocessing](pgcb/README.md) |
| Synthetic conditions | Eight generated four-channel datasets; master seed 8240 | [Construction and annotations](synthetic/README.md) |
| Judge evidence | Completed Claude Sonnet 4.6 study | [Protocol and evidence](../tools/vlm_judge/README.md) |

ETTh1 SHA-256:
f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066.

Place the downloaded PGCB workbook at:
data/pgcb/raw/PGCB_date_power_demand.xlsx.

The workbook hash and processed-CSV hash identify different files.
Raw workbooks, generated datasets, prediction arrays and checkpoints remain local.
The repository publishes construction scripts, manifests and curated evidence.
Existing source licences continue to apply.

From the repository root, run the offline statistical check:

    python3 tools/verify_submission.py

It uses standard-library Python, removes temporary outputs automatically,
and makes no API calls or changes to published results.
