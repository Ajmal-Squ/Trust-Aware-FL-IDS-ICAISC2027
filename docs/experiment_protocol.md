# Conference experiment protocol

## Objective

Characterize the trust-threshold operating point of a poisoning-resilient federated intrusion detector while holding first-round client updates fixed within each seed/attack case.

## Data and preprocessing

The experiment uses the official UNSW-NB15 training and testing CSV files. Numeric variables are median-imputed and standardized. Categorical variables are mode-imputed and one-hot encoded with unknown categories ignored. The preprocessor is fitted only on the official training file. The binary target is normal versus attack.

## Federation

- Clients: 20
- Dirichlet alpha: 0.3
- Local epochs: 1
- Batch size: 1024
- Seeds: 11, 29, 47
- Malicious fraction: 0.20

Development records are partitioned among clients using a class-wise Dirichlet split. A coordinator validation subset is retained on the training side, and up to 1,500 validation records are sampled as the trusted reference set.

## Poisoning conditions

Three attacks are evaluated: label flipping, sign flipping, and backdoor injection. Malicious client identities are drawn seed-wise and kept fixed across the threshold sweep for each attack case.

## Trust score

The implementation combines directional agreement, trusted-reference validation behavior, reputation, and norm consistency. In the first round, reputation starts from 0.5 for every client.

The implemented trust score is

`T = clip(0.30*S + 0.35*V + 0.15*R + 0.20*(1-O), 0, 1)`

where `S` is robust-center directional agreement, `V` is the validation term, `R` is reputation, and `O` is the norm-outlier score.

## Threshold sweep

The thresholds are `0.35, 0.45, 0.55, 0.65, 0.75`. For a fixed seed and attack, the local updates and their evidence values are computed once. Only the accept/reject threshold is changed. This design isolates threshold effects from stochastic retraining.

## Aggregation

Accepted updates are clipped coordinate-wise around the median using a three-MAD envelope. Aggregation weights are proportional to client sample count multiplied by trust. If all clients fall below the threshold, the maximum-trust client is retained as a fallback.

## Reported measures

The result files contain malicious-update rejection, benign-update rejection, screening precision/recall/F1, accuracy, macro-F1, balanced accuracy, MCC, FPR, and backdoor ASR where applicable. `evidence_raw.csv` records per-client trust and evidence values for evidence-level analysis.

## Interpretation boundary

This is a first-round operating-point experiment. It should not be interpreted as a replacement for a full multi-round convergence study.
