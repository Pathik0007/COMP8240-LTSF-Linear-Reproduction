# macOS arm64 Validation

A fresh validation run of DLinear on ETTh1 was performed on macOS arm64
using the official LTSF-Linear source at commit:

0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6

Environment:
- macOS architecture: arm64
- Python: 3.9.19
- PyTorch: 1.9.0.post2
- NumPy: 1.21.6
- pandas: 1.3.5
- scikit-learn: 1.0.2
- matplotlib: 3.5.3
- CUDA: False

Dataset:
- ETTh1
- SHA-256:
  f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066

The original num_workers=10 configuration failed on macOS because the
released runner interacts incompatibly with multiprocessing spawn.
The author source code was not modified. The run was repeated with
num_workers=0, changing only DataLoader parallelism.

Validation configuration:
- model: DLinear
- features: M
- seq_len: 336
- label_len: 48
- pred_len: 96
- enc_in: 7
- moving_avg: 25
- batch_size: 32
- train_epochs: 10
- patience: 3
- learning_rate: 0.005
- seed: 2021
- CPU execution

Fresh macOS arm64 result:
- MSE: 0.3841443955898285
- MAE: 0.40471312403678894

These values exactly match the earlier recorded DLinear h=96 reproduction
to the precision printed by the official implementation.

The tracked files in the official LTSF-Linear repository remained
unchanged.
