# PGCB Hourly Generation Dataset

## Source

Power Grid Company of Bangladesh (PGCB) hourly generation dataset,
distributed through the UCI Machine Learning Repository.

UCI dataset ID: 1175

Dataset file:
`PGCB_date_power_demand.xlsx`

Accessed for COMP8240 in October 2026.

Licence:
Creative Commons Attribution 4.0 International (CC BY 4.0).

## Raw-file integrity

Downloaded UCI archive SHA-256:

`056152d333735c27b88eec0192d4ce3dba80a61b6fc92ac62c00cf592609fb4b`

Extracted workbook SHA-256:

`cd10a95b98033594b70ff176d69e9feaa2ade5f43b8ae4e5640a0f9deabc9da1`

The raw files are not committed to this repository.

## Raw dataset audit

The workbook contains 92,650 rows and 15 columns spanning
19 April 2015 to 17 June 2025.

The dataset is predominantly hourly but is not a perfectly regular
hourly series. The audit identified:

- 88,469 exact-hour observations;
- 4,181 half-hour observations;
- 381 duplicated timestamps in the raw workbook;
- 373 duplicated timestamp groups among exact-hour records;
- 1,055 missing timestamps relative to the complete exact-hour grid;
- substantial structural missingness in solar, wind, India-Adani and
  Nepal import variables;
- several obvious scale anomalies in raw numeric fields.

These properties are documented rather than silently removed.

## Conservative preprocessing

`src/preprocess_pgcb.py` constructs an auditable hourly representation.

The preprocessing:

1. retains exact-hour observations for the hourly forecasting benchmark;
2. uses the consistently available core generation/demand variables;
3. collapses duplicate timestamps using the median of numeric values;
4. sorts observations chronologically;
5. reindexes onto the complete hourly grid;
6. preserves missing hours rather than automatically interpolating them;
7. attaches duplicate, missing-hour, balance-residual and scale-anomaly
   quality annotations.

No forecasting-specific imputation is performed at this stage.

This separates data construction from later model fitting so that
imputation and any additional transformations can be fitted without
test-set leakage.
