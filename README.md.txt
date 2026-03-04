# Imperial Capstone - Bayesian Optimisation (Stage 2)

## Overview
This repository tracks weekly submissions for eight black-box functions (2D to 8D). The workflow is append-only and reproducible:
- ingest the latest portal feedback,
- update `initial_data/function_*/initial_inputs.npy` and `initial_outputs.npy`,
- generate one new query per function,
- export portal-ready strings and reflection notes.

## Current Strategy Baseline (as of Round 4)
Default policy is now exploration-heavy for this stage:
- UCB-first acquisition with configurable `kappa`.
- Strong novelty weighting and novelty-floor filtering.
- Large global candidate pools with reduced local clustering around incumbents.
- Hybrid surrogate stack: GP + MLP + logistic/SVM boundary models.

This is implemented in:
- `execution/propose_round_04_candidates.py`

## Key Data and Deliverables
- Data store: `initial_data/function_*/initial_inputs.npy`, `initial_outputs.npy`
- Submissions: `deliverables/submissions/`
- Reflections: `deliverables/reflections/`
- Notes/work logs: `deliverables/notes/`

## Round 4 Canonical Files
- Portal strings: `deliverables/submissions/round_04_portal_strings.txt`
- Raw vectors: `deliverables/submissions/round_04_portal_strings.json`
- Input arrays: `deliverables/submissions/round_04_inputs.txt`
- Diagnostics: `deliverables/submissions/round_04_hybrid_debug.json`
- Part 2 reflection: `deliverables/reflections/round_04_reflection.md`
- Work log: `deliverables/notes/round_04_work_log.md`

## Script Usage
Generate canonical Round 4 outputs (without re-ingesting data):
- `python execution/propose_round_04_candidates.py --strategy explore --prefix round_04 --skip-ingest`

Generate an alternate-prefixed set:
- `python execution/propose_round_04_candidates.py --strategy explore --prefix round_04_explore --skip-ingest`

## Portal Format Reminder
Each function input must be a hyphen-separated string with six decimals per component and values in `[0, 1)`.
Example: `0.123456-0.654321`.
