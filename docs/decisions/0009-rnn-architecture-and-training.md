# 0009: LSTM/GRU Architecture and Training Protocol

Date: 2026-10-28

## Decision

Follow outline 3.4 for both recurrent models (`src/cdss_project/rnn.py`):

- **Input:** the 6-hour window from decision `0007`. Numeric features are
  standardized, and categorical codes are one-hot encoded per hour (unseen
  codes become all zeros). `SequenceEncoder` is fit on the training stays used
  for fitting only.
- **Architecture (`SepsisRNN`):**
  - single-head self-attention over the 6 time steps, with a residual
    connection (`x + attention(x)`);
  - 3 recurrent layers (LSTM or GRU, 16 units), each followed by
    `BatchNorm1d` and dropout 0.2;
  - the last time step goes to Linear(16→8), BatchNorm, ReLU, dropout 0.2,
    then Linear(8→1), giving one logit.
  - GRU differs only in the cell.
- **Training:**
  - `BCEWithLogitsLoss` with `pos_weight = negatives / positives` (no
    resampling, so the outline's "weighted if imbalance persists" applies);
  - Adam with lr 1e-3, batch size 64 windows, at most 100 epochs, seed 42;
  - early stopping when validation loss has not improved for 10 epochs; the
    best weights are restored.
- **Validation:** the same fold as the baselines (decision `0008`), i.e. the
  training fold after the test fold. It drives early stopping and threshold
  selection (max F1, or `--target-sensitivity`). Unlike the baselines, the
  RNNs are not refit on all training stays, because early stopping needs the
  held-out fold.
- **Evaluation:** the same metrics as the baselines.
  `eda_vs_baselines.csv` holds the paired EDA (decision `0008`) against every
  `*_stay_alerts.csv` in `outputs/baselines/test_fold_<k>/`.
- Fitted weights are not saved (same policy as `0008`).

## Rationale

The architecture and training values are the outline's. The outline leaves
open the residual connection, the ReLU in the head, the one-hot encoding and
the meaning of a batch ("64 patients"). These were chosen as standard,
low-risk defaults: the residual keeps the raw inputs available if attention
is uninformative over only 6 steps, and the batch is read as 64 windows.
Sharing the validation fold with the baselines keeps thresholds and EDA
comparable.

## Consequences

- On ~1.14M windows with batch 64, one epoch is about 13k optimizer steps.
  CPU training is slow; use `--device cuda` where available.
  `--batch-size` can be raised, but that deviates from the outline.
- TimeSHAP (milestone 6) will need saved weights. Add `torch.save` of the
  `state_dict` then, loaded with `weights_only=True`, and update
  `SECURITY_MODEL.md`.
