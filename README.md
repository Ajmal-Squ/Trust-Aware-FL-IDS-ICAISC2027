# Trust-Aware Federated Intrusion Detection Against Poisoning Attacks in Edge Networks

Reproducibility repository for **ICAISC 2027 Paper 47**:

> Ajmal Khan, *Trust-Aware Federated Intrusion Detection Against Poisoning Attacks in Edge Networks*.

This repository contains the code, measured result files, and figures used for the conference paper's trust operating-point study. The conference study focuses on how the client trust threshold controls the trade-off between poisoning rejection and retention of benign non-IID updates.

## Experimental scope

The reported conference experiment uses:

- UNSW-NB15 official training and testing files;
- binary normal-versus-attack classification;
- 20 federated clients;
- Dirichlet non-IID partitioning with `alpha = 0.3`;
- 20% malicious participation (4 of 20 clients);
- label-flip, sign-flip, and backdoor attacks;
- trust thresholds `0.35, 0.45, 0.55, 0.65, 0.75`;
- seeds `11, 29, 47`;
- one local epoch per client;
- a compact convolutional-Transformer classifier.

The threshold experiment is a **controlled first-round operating-point analysis**. For each seed/attack condition, local client updates are generated once and the same updates are re-evaluated across all trust thresholds. This isolates the effect of the trust decision from different local-training trajectories.

## Repository layout

```text
.
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── .gitignore
├── code/
│   ├── conference_threshold_experiment.py
│   └── metra_fl_binary_rerun.py
├── results/
│   ├── threshold_raw.csv
│   ├── threshold_summary.csv
│   └── evidence_raw.csv
├── figures/
│   ├── Fig1_threshold_rejection.png
│   ├── Fig2_threshold_screening_f1.png
│   ├── Fig3_threshold_macro_f1.png
│   ├── Fig4_backdoor_asr.png
│   └── Fig5_evidence_auroc.png
└── docs/
    └── experiment_protocol.md
```

## Dataset

The UNSW-NB15 CSV files are **not redistributed** in this repository. Obtain the official files separately and place them as:

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

## Run the conference threshold experiment

From the repository root:

```bash
python code/conference_threshold_experiment.py \
  data/UNSW_NB15_training-set.csv \
  data/UNSW_NB15_testing-set.csv \
  results
```

The script writes raw threshold results, evidence values, and threshold summaries to `results/`.

## Trust evidence

Each client update is scored using complementary evidence:

1. directional agreement with a coordinate-wise robust center;
2. loss-based behavior on a trusted coordinator reference subset;
3. client reputation history;
4. update-norm consistency.

Low-trust updates are rejected. Accepted updates are coordinate-wise clipped around the median using a three-MAD envelope and aggregated with trust- and sample-dependent weights.

## Reproduced measured outputs

The committed CSV files are the measured outputs used to prepare the conference figures. They are included to permit direct checking of the reported values without rerunning the full experiment.

At trust threshold `0.55`, the measured screening behavior across the reported seeds includes strong rejection of label-flip and sign-flip updates with substantially lower benign rejection, while backdoor updates are more difficult to separate. The evidence-level analysis also shows that no single evidence component behaves identically across all poisoning mechanisms; see `results/evidence_raw.csv` and the paper for the exact interpretation.

## Relation to the broader METRA-FL implementation

`code/metra_fl_binary_rerun.py` provides the underlying model, preprocessing, local-training, attack, metric, and aggregation utilities used by the conference experiment. The conference paper itself is intentionally narrower: it studies trust-threshold calibration rather than presenting the broader full-round evaluation matrix.

## Reproducibility notes

- The official test set is not used to fit preprocessing or train local models.
- The coordinator reference subset is drawn from the training-side validation split.
- The threshold sweep reuses the same client updates within each seed/attack case.
- The committed result CSVs should be treated as the authoritative measured outputs for the conference figures.
- Exact runtime depends on CPU/GPU, PyTorch version, and deterministic-kernel availability.

## License

Code is released under the MIT License. The UNSW-NB15 dataset remains subject to its original terms and is not included here.

## Citation

If this repository supports your work, please cite the conference paper after the final proceedings metadata (year, pages, DOI) become available. `CITATION.cff` contains the current manuscript-level citation metadata.
