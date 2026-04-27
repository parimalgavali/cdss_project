# CDSS Project: Early Sepsis Onset Prediction

This repository supports a clinical decision support systems project on early
sepsis onset prediction from ICU time-series data. The project compares
classical baselines with recurrent neural network models across prediction
horizons from 25 hours before the decision point down to 1 hour.

## Project Goal

The core research question is:

> How does predictive performance change across prediction horizons, and what
> is the minimum horizon where LSTM and GRU models achieve clinically meaningful
> results?

The planned evaluation combines standard metrics such as AUROC, AUPRC, F1,
sensitivity, and specificity with Early Detection Advantage (EDA), which
measures how many hours earlier sequential models detect sepsis compared with
classical baselines.

## Repository Layout

```text
.
├── AGENTS.md              # Instructions for future coding-agent sessions
├── docs/                  # Project notes and data contracts
├── notebooks/             # Exploratory analysis notebooks
├── scripts/               # Command-line utilities
├── src/cdss_project/      # Reusable project code
├── tests/                 # Automated tests
└── requirements.txt       # Python dependencies
```

## Project Memory

Future sessions should start with:

```text
docs/project_memory.md
docs/decisions/README.md
```

These files record durable assumptions and decisions so project context is not
lost when chat history is compacted or a new session begins.

## Local Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The MIMIC-derived dataset should stay local and must not be committed to Git.
Place it at:

```text
data/mimic_sepsis_onset_prediction.csv
```

## Validate the Dataset

Run the inspection script before building preprocessing or model code:

```bash
python scripts/inspect_data.py
```

The script checks the expected columns, stay/hour structure, `hour_before`
range, and reports `NaN` labels in `sepsis` explicitly.

Run the preprocessing dry run:

```bash
python scripts/preprocess_data.py --dry-run
```

Create a separate processed CSV copy without modifying the original:

```bash
python scripts/preprocess_data.py
```

The generated file is written to `data/processed/mimic_sepsis_preprocessed.csv`
and remains ignored by Git.

## Planned Milestones

1. Scaffold, data contract, and validation.
2. Preprocessing and 6-hour sliding-window construction.
3. Logistic Regression and Random Forest baselines.
4. LSTM and GRU training pipeline in PyTorch.
5. Horizon-wise metrics and Early Detection Advantage.
6. Plots, report outputs, and optional TimeSHAP analysis.

## Parallel Agent Strategy

Parallel agents should be used selectively after the data contract and scaffold
are stable. Good future splits are baselines, recurrent model modules,
evaluation/EDA, and documentation/report plots.
