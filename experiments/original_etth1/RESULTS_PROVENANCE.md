# Results Provenance

The DLinear ETTh1 reproduction results recorded in this repository
originate from an execution of the official LTSF-Linear implementation.

Official repository:
https://github.com/cure-lab/LTSF-Linear

Repository commit:
0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6

Original dataset:
ETTh1 from the canonical ETT dataset release.

Configuration:
- model: DLinear
- features: M
- seq_len: 336
- pred_len: 96, 192, 336, 720
- enc_in: 7
- individual: False
- moving_avg: 25
- batch_size: 32
- learning_rate: 0.005
- loss: MSE
- seed: 2021
- execution device: CPU

The four-horizon official script completed successfully.

The values in `results/original_replication.csv` are the final
MSE and MAE values produced after evaluation using the selected
validation checkpoint; they are not intermediate epoch test losses.

The original execution was completed before this repository was
created. The numerical results are retained here as provenance for
the subsequent COMP8240 replication and extension work.
