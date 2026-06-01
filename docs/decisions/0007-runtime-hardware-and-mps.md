# 0007: Runtime Hardware and Apple MPS

Date: 2026-06-01

## Decision

For neural model training, select the PyTorch device in this order:

1. Apple Metal/MPS, when available;
2. CUDA, when available;
3. CPU fallback.

The user's local laptop is an Apple M1 MacBook with 8GB RAM. Local runs can
handle fixed 3-hour and fixed 6-hour experiments, but full expanding-window
training may exceed local memory. Heavy experiments can be run on stronger
hardware after cloning the repository, provided the dataset is placed locally
and remains outside Git.

## Rationale

The previous neural training code only checked for CUDA before falling back to
CPU. CUDA is not available on Apple Silicon Macs, but PyTorch can use Apple's
Metal backend through MPS. MPS can speed up LSTM/GRU training on the user's
MacBook compared with CPU-only training.

The expanding-window dataset is substantially heavier because it stores up to
30 time steps per patient-hour. Even with MPS acceleration, memory pressure can
still kill full expanding-window runs on an 8GB laptop.

## Consequences

- New LSTM/GRU experiment outputs should record `device: "mps"` on the user's
  MacBook when run outside restricted sandbox contexts.
- Logistic Regression and Random Forest are not accelerated by MPS.
- Generated data, outputs, models, and local reference PDFs remain ignored by
  Git.
- If another machine is used, the friend/collaborator should clone the repo,
  create the virtual environment, install dependencies, and place the dataset
  locally under `data/`.
