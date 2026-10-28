# Threat model: parimalgavali/cdss_project

_Threat model for security scans and reviews of this repository. Built from the code at commit 991d5e6; items marked (assumption) have not yet been confirmed by the owner. Update it when the system, its data, or its deployment changes._

## System overview
Offline, reproducible research pipeline for early sepsis-onset prediction from ICU hourly time series. Python (pandas, NumPy, scikit-learn, imbalanced-learn, PyTorch planned). Researchers run CLI scripts locally against a MIMIC-derived CSV.

- No network services, no authentication, no web UI, no CI/CD workflows, no infrastructure-as-code (no infrastructure repository identified).
- Data flow: `data/mimic_sepsis_onset_prediction.csv` -> `scripts/inspect_data.py` / `scripts/preprocess_data.py` -> `src/cdss_project/*` (pandas `read_csv`) -> `data/processed/mimic_sepsis_preprocessed.csv` (`to_csv`) -> `scripts/build_windows.py` -> `data/processed/windows_w6.npz` (`np.savez_compressed`, read back with `np.load(allow_pickle=False)`).
- Planned: LR/RF baselines, PyTorch LSTM/GRU training, model artifacts in `models/`, outputs in `outputs/` and `reports/figures/`.

## Components
| Name | Covers |
|---|---|
| **cli-scripts** | `scripts/` (argparse entry points: `--csv`, `--output`, `--overwrite`, `--n-folds`, `--test-fold`, `--random-state`) |
| **data-pipeline** | `src/cdss_project/` (data_contract, audit, preprocessing, features; future windowing/model code) |
| **notebooks** | `notebooks/` (exploration/figures) |
| **tests** | `tests/`, `pytest.ini` |
| **repo-hygiene** | `.gitignore`, `requirements.txt`, `docs/`, `AGENTS.md`, committed PDFs |

## Trust boundaries and entry points
1. **Local user -> CLI scripts**: arguments are supplied by the researcher running the script on their own machine (trusted, assumption).
2. **Dataset file -> pipeline**: the CSV comes from the course/professor (PhysioNet/MIMIC-derived); parsed with pandas only (no pickle/eval).
3. **Third-party packages -> process**: `requirements.txt` is unpinned (no versions, no hashes), so every install resolves latest from PyPI.
4. **Repository -> GitHub**: anything committed becomes visible to whoever can read the repo (visibility not confirmed).
5. **Future: model checkpoints -> `torch.load`/pickle**: loading checkpoints or joblib/pickle files executes code if the file is untrusted.

## Assets
- **Clinical dataset** (MIMIC-derived, credentialed access under the PhysioNet Data Use Agreement): high sensitivity; must never be committed or shared. Currently protected by `.gitignore` (`data/`, `models/`, `outputs/`).
- **Processed data and model artifacts** derived from it: same sensitivity as the dataset (models can memorise records).
- **Unpublished reference material** (`docs/Sepsis_Paper_2025_by_Prof.pdf`, git-ignored; `Tentative_CDSS_Project_Outline.pdf` is committed).
- **Integrity of research results** (leakage-free splits, correct labels, reproducibility).
- Researcher workstation (code execution via dependencies or unsafe deserialization).

## Attacker profiles (assumption until confirmed)
- **Supply-chain attacker**: compromised/typosquatted PyPI package reached through unpinned `requirements.txt`.
- **Malicious artifact provider**: someone supplying a crafted model checkpoint/pickle or CSV (relevant once model loading exists).
- **Repository reader**: anyone who can read the GitHub repo, if clinical data or secrets were ever committed.
- Not considered: remote network attackers (no exposed services).

## Existing controls
- `.gitignore` excludes `data/`, `models/`, `outputs/`, `reports/figures/`, venvs, caches and the professor's paper.
- `AGENTS.md` / decision 0001 forbid committing clinical data or generated artifacts.
- Preprocessing never overwrites the source CSV; output refuses to overwrite without `--overwrite`.
- Only pandas CSV parsing and NumPy `.npz` arrays loaded with `allow_pickle=False`; no pickle, eval, subprocess or network calls in current code.
- Train-only fitting of imputation statistics (result integrity / leakage).

## Out of scope / accepted risk
- None stated by the owner yet.

## Open questions
- Where does the code run (laptop only, shared server/HPC, future service)?
- Which assets matter most: data confidentiality, result integrity, or both?
- Which attackers matter: supply chain only, untrusted input files, other users on a shared machine?
- Is the GitHub repository public or private?
- Anything out of scope (CLI argument misuse, notebooks, tests)?
