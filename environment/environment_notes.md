# Recorded environments

Mac runtime, documented in runtime_versions.txt and the osx-arm64 export:
Python 3.9.19, PyTorch 1.9.0.post2, NumPy 1.21.6, pandas 1.3.5,
scikit-learn 1.0.2, matplotlib 3.5.3, openpyxl 3.1.5; CPU execution.

Earlier Kaggle CPU execution:
Python 3.9.25, PyTorch 1.9.0+cu102, NumPy 1.21.6, pandas 1.3.5,
scikit-learn 1.0.2 and matplotlib 3.5.3.
The CUDA-capable package build does not mean that experiment used a GPU.

environment.yml is the portable starting specification. Historical exports
record platform-specific installations and should remain unchanged.
Exact reproduction can depend on platform and package builds.
The official README recommends Python 3.6.9; Python 3.9 is a documented deviation.

The judge-statistics verifier uses standard-library Python independently
of the forecasting environment.
