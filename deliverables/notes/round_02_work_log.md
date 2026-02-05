# Round 02 Work Log (ML Student Notes)

## Why this document exists
You asked for a clear explanation of what I did and why, written for an ML student. This is a short, structured record you can refer back to.

## 1) Appending Round 01 data
Goal: make all execution scripts see the full dataset (initial + Round 01) without manual concatenation each time.

What I did:
- Parsed `deliverables/submissions/round_01_inputs.txt` and `deliverables/submissions/round_01_outputs.txt`.
- Appended one new row per function into `initial_data/function_*/initial_inputs.npy` and `initial_data/function_*/initial_outputs.npy`.
- Saved backups of the original initial-only arrays as:
  - `initial_data/function_*/initial_inputs_round00.npy`
  - `initial_data/function_*/initial_outputs_round00.npy`

Why it matters:
- Scripts such as `execution/summarize_initial_data.py` now operate on the full dataset automatically.
- Backups preserve the clean baseline if we ever need to compare pre/post rounds.

## 2) Strategy update (Round 02)
Goal: keep exploration dominant, but start using a simple model to guide where to explore.

Approach used:
- Fit a Gaussian process (GP) model for each function using scikit-learn (Matern kernel).
- Tune kernel hyperparameters with a small randomized search (as shown in the shared repo notebook).
- Kernel bounds used: constant in [1e-3, 1e3], length_scale in [1e-4, 10], noise in [1e-9, 1.0].
- Choose the acquisition per function:
  - Use EI when the best observed output is a clear outlier vs the median (z-score >= 2.5).
  - Use UCB otherwise to keep exploration dominant.
- Generate candidate points from two sources:
  - Global random points in [0.001, 0.98] (to explore new areas).
  - Local perturbations around the current best point (to test exploitation).
- Score candidates using EI or UCB (selected per function), then apply a soft boundary penalty (0.05 margin) to avoid extreme 0.00/0.98 corners.

Acquisition choices for Round 2 (z-score rule):
- EI: Functions 5 and 7 (best output is a strong outlier).
- UCB: Functions 1, 2, 3, 4, 6, 8.

Rationale:
- A GP is the standard Bayesian-optimisation surrogate and provides uncertainty, which linear regression cannot.
- Randomized kernel tuning reduces arbitrary hyperparameter choices and follows the repo example.
- EI naturally balances exploration and exploitation without a hand-tuned distance penalty.
- This matches the tutor guidance: stay exploration-heavy early on but begin model-based search.

## 3) Round 02 submission artefacts
Files created:
- Portal strings: `deliverables/submissions/round_02_portal_strings.txt`
- JSON with raw vectors: `deliverables/submissions/round_02_portal_strings.json`
- Round 02 inputs list: `deliverables/submissions/round_02_inputs.txt`

These contain the exact 6-decimal, hyphen-separated strings required for manual portal entry.

## 4) Reflection write-up
I wrote a combined reflection in `deliverables/reflections/round_02_reflection.md` that includes:
- Part 1: same format as Round 01 (per-function rationale, challenges, next adjustments).
- Part 2: answers to the provided prompts about strategy shifts, exploration vs exploitation, regression assumptions, and interpretability.

## 5) Documentation update
The README was updated (`README.md.txt`) to document:
- The current data layout (including backups).
- The Round 02 strategy in plain language.
- Where the key submission files live.

## Quick takeaway (for future weeks)
- Keep appending new outputs to the dataset each round.
- Move gradually from linear heuristics to GP + acquisition once sample sizes grow.
- Use interpretability (coefficients, simple plots) as guidance, not truth.
