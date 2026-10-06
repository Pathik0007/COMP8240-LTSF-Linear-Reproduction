# Controlled Synthetic Time-Series Dataset

## Purpose

This dataset family was constructed programmatically for the COMP8240
LTSF-Linear reproduction and generalisation project.

Its purpose is to test forecasting behaviour under controlled, known
data-generating conditions rather than relying only on naturally
occurring benchmark datasets.

## Base process

Each multivariate series contains four aligned hourly channels.

The stable base follows:

`y_t = a*t + A*sin(2*pi*t/P + phase) + epsilon_t`

where:

- `a` controls trend slope;
- `A` controls seasonal amplitude;
- `P = 24` hours is the shared base period;
- `epsilon_t` is Gaussian noise.

The four channels have different amplitudes, phases and trend slopes,
but share the same base period so that channel heterogeneity is not
confounded with the controlled condition being tested.

## Dataset dimensions

Each condition contains:

- 10,000 hourly timestamps;
- 4 numeric channels;
- 7,000 training rows;
- 1,000 validation rows;
- 2,000 test rows.

Splits are chronological.

## Controlled conditions

### Stable

Low-noise trend plus periodicity.

### Level shift

The test period receives a fixed channel-specific level offset while
trend and periodicity are retained.

### Regime change

The trend slope changes at the test boundary while the series remains
continuous at the transition.

### Gaussian noise

Two additional severity levels are generated while the underlying
signal and random innovation sequence remain controlled:

- medium noise: sigma = 0.45;
- high noise: sigma = 0.90.

The stable baseline uses sigma = 0.15.

### Sparse outliers

Sparse additive spikes are introduced separately from Gaussian-noise
severity:

- low frequency: 0.25% probability per channel-time cell;
- high frequency: 1.00% probability per channel-time cell.

The ordinary Gaussian noise remains fixed at sigma = 0.15.

### Nonlinear temporal dependence

A bounded nonlinear autoregressive residual is added to the deterministic
trend/periodic component.

The residual depends on its previous two values and contains the
interaction:

`tanh(z[t-1] * z[t-2])`

This creates nonlinear lag dependence while preventing unstable
divergence.

## Reproducibility

The master random seed is `8240`.

Outlier-mask seeds are recorded separately.

The generator records:

- generating parameters;
- chronological split boundaries;
- condition-specific parameters;
- SHA-256 hashes for generated data and annotations;
- ground-truth change-point and outlier annotations.

To reproduce the construction:

    python src/generate_synthetic.py
    python src/audit_synthetic.py

The complete machine-readable construction manifest is stored in:

`metadata/synthetic_manifest.json`

The construction audit is stored in:

`experiments/synthetic/construction_audit.txt`

## Ground-truth annotations

Every condition has a matching annotation CSV containing:

- chronological split;
- condition name;
- test-period indicator;
- post-change indicator;
- any-outlier indicator;
- number of outlier channels at each timestamp;
- per-channel outlier flags.

These labels are generated directly from the construction process rather
than inferred after generation.
