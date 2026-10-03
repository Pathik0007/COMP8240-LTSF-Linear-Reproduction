# Reproduction Environment

The initial official DLinear ETTh1 reproduction was executed using:

- Python 3.9.25
- PyTorch 1.9.0+cu102
- NumPy 1.21.6
- pandas 1.3.5
- scikit-learn 1.0.2
- matplotlib 3.5.3
- CPU execution

The official LTSF-Linear README recommends Python 3.6.9.

PyTorch 1.9 matches the implementation generation used by the repository, while Python 3.9 is a documented deviation from the README recommendation.

The environment information is retained explicitly so that differences between the published and reproduced results can be investigated rather than hidden.
