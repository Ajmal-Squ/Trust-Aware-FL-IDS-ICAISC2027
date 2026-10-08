# Trust-Aware Federated Intrusion Detection Against Poisoning Attacks in Edge Networks

Reproducibility repository for **ICAISC 2027 Paper 47**:

> Ajmal Khan, *Trust-Aware Federated Intrusion Detection Against Poisoning Attacks in Edge Networks*.

This repository supports the conference paper's controlled trust operating-point study. The main question is how the client trust threshold changes the trade-off between poisoning rejection and retention of benign non-IID updates.

## Measured conference protocol

- Dataset: official UNSW-NB15 training/test split
- Task: binary normal-versus-attack detection
- Clients: 20
- Non-IID partition: Dirichlet `alpha = 0.3`
- Malicious participation: 20% (4/20 clients)
- Attacks: label flip, sign flip, backdoor
- Trust thresholds: `0.35, 0.45, 0.55, 0.65, 0.75`
- Independent measured seeds: `11, 29`
- Local epochs: 1
- Model: compact convolutional-Transformer

The threshold study is a **controlled first-round diagnostic**. For each seed/attack condition, client updates are generated once and then re-evaluated at all five trust thresholds. This isolates the trust decision from changes in client partitioning or local stochastic training.

## Repository layout

```text
.
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── .gitignore
├── code/
│   ├── trust_aware_core.py
│   ├── conference_threshold_experiment.py
│   └── generate_figures.py
├── results/
│   ├── threshold_raw.csv
│   ├── threshold_summary.csv
│   └── evidence_raw.csv
└── docs/
    └── experiment_protocol.md
```

Running `code/generate_figures.py` creates the five paper figures in a local `figures/` directory.

## Dataset

The UNSW-NB15 CSV files are **not redistributed**. Obtain the official files separately and place them as:

```text
data/UNSW_NB15_training-set.csv
data/UNSW_NB15_testing-set.csv
```

Expected official split sizes are 175,341 training records and 82,332 testing records.

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

## Reproduce the conference threshold experiment

From the repository root:

```bash
python code/conference_threshold_experiment.py \
  data/UNSW_NB15_training-set.csv \
  data/UNSW_NB15_testing-set.csv \
  results
```

Generate the paper figures from the measured CSV files:

```bash
python code/generate_figures.py
```

## Trust evidence

Each client update is evaluated using four complementary terms: directional agreement with a coordinate-wise robust center, loss behavior on a trusted coordinator reference subset, client reputation, and update-norm consistency. Low-trust updates are rejected. Accepted updates are clipped around the coordinate-wise median with a three-MAD envelope and aggregated using trust- and sample-dependent weights.

## Measured outputs

The committed CSV files are the measured outputs used for the conference analysis. At `tau = 0.55`, the two-seed mean malicious-update rejection is 87.5% for label flipping and 100% for sign flipping, while benign rejection is 3.13% and 6.25%, respectively. Backdoor updates are harder to separate, with 37.5% malicious rejection at the same threshold. These values are first-round operating-point measurements, not converged multi-round IDS scores.

## Reproducibility boundary

- Preprocessing is fitted on the official training file only.
- The coordinator reference subset is drawn from the training-side validation split.
- The official test set is used only for evaluation.
- Client updates are held fixed across the threshold sweep within each seed/attack case.
- Only two independent measured seeds are used in the conference study, so the results are reported as controlled measurements rather than strong population-level statistical claims.

## License

Code is released under the MIT License. The UNSW-NB15 dataset remains subject to its original terms and is not included here.

## Citation

Please cite the conference paper after final proceedings metadata (pages/DOI) are available. `CITATION.cff` contains the current manuscript-level metadata.
