# Round 05 Work Log (ML Student Notes)

## Why this document exists
You asked for the same style of round documentation as prior weeks, including a stats snapshot of current challenge performance and the official next query set.

## 1) Round 4 ingestion from local submission artifacts
What I used:
- `deliverables/submissions/round_04_inputs_batched.txt`
- `deliverables/submissions/round_04_outputs.txt` (latest batch = Round 4 outputs)

What was done:
- Parsed the batched files and selected the latest batch (Round 4 feedback).
- Appended one `(x, y)` row per function into:
  - `initial_data/function_*/initial_inputs.npy`
  - `initial_data/function_*/initial_outputs.npy`
- Wrote canonical Round 4 outputs to `deliverables/submissions/round_04_outputs_canonical.txt`.

Validation:
- All 8 functions were appended exactly once (no duplicates detected).
- Sample counts after append:
  - f1: 14, f2: 14, f3: 19, f4: 34, f5: 24, f6: 24, f7: 34, f8: 44.

## 2) Round 4 outcomes and challenge stats
Observed change (`round_04_output - round_03_output`):
- f1: down very slightly (near-zero magnitude)
- f2: down
- f3: up
- f4: down strongly
- f5: down
- f6: down
- f7: down
- f8: down

Objective impact (maximize each function):
- 1/8 improved vs Round 3 (f3).
- 7/8 improved vs Round 1 baseline.
- New bests by round across submitted rounds:
  - Round 1: 1 function
  - Round 2: 2 functions
  - Round 3: 4 functions
  - Round 4: 1 function

Current all-time best source (initial data + rounds):
- f1: initial set
- f2: initial set
- f3: initial set
- f4: Round 3
- f5: Round 1
- f6: Round 3
- f7: Round 3
- f8: Round 2

Interpretation:
- Round 4 was a lower-yield exploratory round by immediate score, but it expanded coverage and updated posterior uncertainty for Round 5 decisions.

## 3) Final Round 5 strategy (exploration-heavy, as requested)
Final policy used for this submission:
- UCB-only acquisition (`kappa=3.2`) to keep exploration pressure high.
- Hybrid surrogate stack retained:
  - GP regressor for uncertainty-aware scoring.
  - MLP regressor for non-linear response structure and gradient refinement.
  - Logistic regression + RBF SVM for boundary structure (`good` vs `bad`).
- Novelty floor filtering retained to prevent local re-sampling.
- Larger global candidate pools retained with reduced incumbent clustering.

Implementation used:
- `execution/propose_round_05_candidates.py`

## 4) Official Round 5 query set submitted
Canonical portal strings (`deliverables/submissions/round_05_portal_strings.txt`):
- function_1: `0.038800-0.689887`
- function_2: `0.677496-0.474733`
- function_3: `0.944654-0.423830-0.603601`
- function_4: `0.485936-0.240452-0.037372-0.040856`
- function_5: `0.592532-0.914214-0.884356-0.945220`
- function_6: `0.827078-0.045831-0.944861-0.132362-0.917946`
- function_7: `0.207515-0.096458-0.400414-0.129468-0.117823-0.914403`
- function_8: `0.825829-0.174258-0.074230-0.051112-0.152122-0.898703-0.070768-0.058783`

## 5) Artifacts created/updated
- New execution script: `execution/propose_round_05_candidates.py`
- Round 4 canonical output snapshot:
  - `deliverables/submissions/round_04_outputs_canonical.txt`
- Round 5 canonical files:
  - `deliverables/submissions/round_05_inputs.txt`
  - `deliverables/submissions/round_05_portal_strings.txt`
  - `deliverables/submissions/round_05_portal_strings.json`
  - `deliverables/submissions/round_05_hybrid_debug.json`
- Batch helper used for ingestion parity:
  - `deliverables/submissions/round_04_inputs_batched.txt`

## 6) Key evidence this set is exploration-forward
Chosen candidates satisfied novelty-floor constraints in all functions.
Examples from debug diagnostics (`chosen_candidate_min_dist` vs `novelty_floor_applied`):
- f4: 0.4880 vs 0.3001
- f6: 0.8120 vs 0.4413
- f8: 1.1001 vs 0.6277

Interpretation:
- Round 5 queries remain model-guided, but are intentionally pushed into less-sampled regions.
- This is aligned with your preference to stay exploration-heavy at this stage.
