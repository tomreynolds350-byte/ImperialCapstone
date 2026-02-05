# Imperial Capstone - Bayesian Optimisation (Stage 2)

## Overview
This repository tracks my weekly Bayesian-optimisation-style submissions for eight black-box functions (2D through 8D). The workflow is intentionally lightweight and explainable: I append each round's observed inputs/outputs to the existing dataset, then propose one new candidate per function using a simple surrogate-guided heuristic. The aim is steady learning and iterative improvement rather than immediate optimality.

## Data Layout
- `initial_data/function_*/initial_inputs.npy` and `initial_data/function_*/initial_outputs.npy` now include the initial samples plus Round 01 (appended).
- Backup copies of the original initial-only datasets are preserved as `initial_data/function_*/initial_inputs_round00.npy` and `initial_data/function_*/initial_outputs_round00.npy`.
- Round-level submissions live in `deliverables/submissions/`, including portal-ready strings and raw vectors.
- Reflections are in `deliverables/reflections/`.

## Current Round 2 Approach (Summary)
I moved from pure distance-based exploration (Round 1) to a true Gaussian process (GP) surrogate for Round 2:
- Fit a GP with a Matern kernel per function (scikit-learn).
- Tune kernel hyperparameters via a small randomized search (as in the shared repo examples).
- Choose the acquisition per function (EI vs UCB) based on whether the best observed output is a clear outlier (z-score >= 2.5).
- Generate candidate points from two sources:
  1. Global random samples in [0.001, 0.98].
  2. Local perturbations around the current best point.
- Score candidates using EI or UCB (selected per function), then apply a soft boundary penalty (0.05 margin) to avoid extreme 0.00/0.98 corners.

Round 2 acquisition choices:
- EI: Functions 5 and 7
- UCB: Functions 1, 2, 3, 4, 6, 8
- Select the highest-EI candidate and format it for portal submission.

This gives a principled trade-off between exploration (uncertainty) and exploitation (predicted improvement), while staying consistent with the tutor advice to remain exploration-heavy until mid-stage.

## Key Files for Round 2
- Portal strings: `deliverables/submissions/round_02_portal_strings.txt`
- Raw vectors: `deliverables/submissions/round_02_portal_strings.json`
- Round 2 inputs list: `deliverables/submissions/round_02_inputs.txt`
- Reflection (Part 1 + Part 2): `deliverables/reflections/round_02_reflection.md`

## Helpful Scripts
- `execution/summarize_initial_data.py` - summarises current data for all functions.
- `execution/plot_round_comparison.py` - compares pre/post round distributions (if run).
- `execution/propose_gp_candidates.py` - GP + EI candidate generation for the next round.

## Reference Repos (shared by Murari)
- https://github.com/jdchen5/machinelearninglabs
- https://github.com/jdchen5/machinelearninglabs/blob/main/CCompetition/old_codes/function_1.ipynb

## Portal Format Reminder
Each function input is a hyphen-separated string with six decimals per component, e.g.:
- Function 1: `x1-x2`
- Function 8: `x1-x2-x3-x4-x5-x6-x7-x8`

All values must be in `[0, 1)` and formatted to exactly six decimals.
