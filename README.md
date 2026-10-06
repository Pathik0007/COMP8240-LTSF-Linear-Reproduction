# COMP8240 LTSF-Linear Reproduction and Generalisation

This repository supports a reproduction and extension of:

Ailing Zeng, Muxi Chen, Lei Zhang, and Qiang Xu.  
"Are Transformers Effective for Time Series Forecasting?"  
AAAI 2023.

## Project Objective

This project investigates:

1. whether selected LTSF-Linear results can be reproduced using the authors' released code and original data;
2. whether the reported findings generalise to new public time-series datasets; and
3. under which controlled data conditions the relative performance of Linear, NLinear, DLinear, and selected Transformer baselines changes.

## Current Status

- [x] Official LTSF-Linear repository obtained and executed
- [x] Canonical ETTh1 dataset obtained
- [x] DLinear executed on ETTh1 for forecast horizons 96, 192, 336, and 720
- [x] Linear reproduced under matched ETTh1 settings
- [x] NLinear reproduced under matched ETTh1 settings
- [x] Primary external dataset prepared
- [x] Controlled synthetic dataset generated
- [ ] New-data experiments completed

## Repository Structure

- `src/` - preprocessing, generation, evaluation, and experiment scripts
- `data/` - data documentation and generated datasets
- `experiments/` - experiment-specific logs and outputs
- `results/` - numerical comparisons and figures
- `metadata/` - metadata describing constructed datasets
- `environment/` - software and reproducibility information

## Original Implementation

Official LTSF-Linear repository:  
https://github.com/cure-lab/LTSF-Linear

## Reproducibility

Experiment settings, software versions, preprocessing decisions, generated-data parameters, and random seeds will be recorded so that the reported results can be reconstructed.
