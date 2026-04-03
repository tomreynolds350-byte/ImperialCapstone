# Round 08 Work Log (ML Student Notes)

## Why this document exists
You asked for Round 7 performance stats and the same style of round documentation as previous weeks, while moving into Round 8 with 17 data points and a more selective exploitation policy.

## 1) Round 7 ingestion from local submission artifacts
What I used:
- `deliverables/submissions/round_07_inputs_batched.txt`
- `deliverables/submissions/round_07_outputs.txt` (latest batch = Round 7 outputs)

What was done:
- Created `round_07_inputs_batched.txt` by appending `round_07_inputs.txt` to the prior batched input file.
- Parsed the latest batch (Round 7 feedback).
- Appended one `(x, y)` row per function into:
  - `initial_data/function_*/initial_inputs.npy`
  - `initial_data/function_*/initial_outputs.npy`
- Wrote canonical Round 7 outputs to `deliverables/submissions/round_07_outputs_canonical.txt`.

Validation:
- All 8 functions were appended exactly once (no duplicates detected).
- Sample counts after append:
  - f1: 17, f2: 17, f3: 22, f4: 37, f5: 27, f6: 27, f7: 37, f8: 47.

## 2) Round 7 outcomes and challenge stats
Observed change (`round_07_output - round_06_output`):
- f1: down (tiny near-zero sign flip)
- f2: down by `-0.3566`
- f3: down by `-0.0072`
- f4: up by `+0.6484`
- f5: down by `-86.9100`
- f6: up by `+0.3187`
- f7: down by `-0.2461`
- f8: up by `+0.1307`

Objective impact (maximize each function):
- 3/8 improved vs Round 6.
- 7/8 improved vs Round 1 baseline.
- Round 7 rank among submitted rounds (1=best):
  - f1: 7th, f2: 4th, f3: 2nd, f4: 1st, f5: 2nd, f6: 2nd, f7: 3rd, f8: 1st.
- Average submitted-round rank for Round 7: `2.75`.
- Round 7 currently holds the best submitted result in 2/8 functions:
  - f4 and f8.
- Round 7 set new all-time bests (initial data + rounds) in 2/8 functions:
  - f4 and f8.

Current all-time best source:
- f1: initial set
- f2: Round 6
- f3: initial set
- f4: Round 7
- f5: Round 6
- f6: Round 3
- f7: Round 6
- f8: Round 7

Interpretation:
- Round 7 showed that exploitation is still the right direction, but not as a blanket policy.
- The exploit move paid off in f4, f6, and f8, but clearly overfit weaker or less stable cases such as f2 and slightly softened already-strong incumbents such as f5 and f7.

## 3) Round 8 tuning: selective exploitation instead of global exploitation
Post-ingest `z_best` values on the 17-point state were:
- f1: `0.00`
- f2: `1.64`
- f3: `0.73`
- f4: `1.72`
- f5: `3.30`
- f6: `1.69`
- f7: `3.29`
- f8: `1.85`

This made the main tuning question much clearer:
- keep f2 below the EI threshold,
- keep f4/f6/f8 above it,
- retain strong exploitation for f5/f7,
- leave f1/f3 on UCB.

Settings tested on a temporary 17-point copy:
- `balanced_k1.45_t1.9_b030`
- `balanced_k1.35_t1.75_b030`
- `balanced_k1.25_t1.75_b028`
- `exploit_k1.25_t1.8_b030`
- `exploit_k1.15_t1.75_b030`
- second sweep around `t=1.65` to separate f2 from f4/f6/f8

Final policy chosen:
- `strategy=exploit`
- `kappa=1.25`
- `z_best_threshold=1.65`
- `boundary_margin=0.03`

Selection rationale:
- `z_best_threshold=1.65` was the key value:
  - f2 stayed on UCB (`z_best=1.64`)
  - f4, f6, and f8 stayed exploitative (`1.72`, `1.69`, `1.85`)
- This preserved the benefits of Round 7 while reducing the chance of repeating its main overfit failure.

## 4) Manual safeguard for function 5
One additional guardrail was needed for f5.

Observed issue:
- The surrogate repeatedly proposed a candidate that pulled one coordinate far away from the established near-corner optimum, despite the top two empirical outputs coming from the same tight corner basin.

Safeguard used:
- If the top two observed f5 points are both in the near-corner basin (`>= 0.97` in every coordinate) and the model candidate exits that basin on at least one coordinate, override the proposal with the midpoint of the top two observed points.

This safeguard was triggered for Round 8 and produced:
- override anchors:
  - `[0.98, 0.98, 0.98, 0.98]`
  - `[0.978540, 0.977812, 0.979000, 0.979404]`
- final override candidate:
  - `0.979270-0.978906-0.979500-0.979702`

Interpretation:
- This is a manual anti-hallucination check on the optimiser itself: trust the data when the surrogate starts contradicting a very strong repeated empirical pattern.

## 5) Final Round 8 strategy
Final policy used for this submission:
- UCB on unresolved / weaker-signal regimes:
  - f1, f2, f3
- EI on stronger local basins:
  - f4, f5, f6, f7, f8
- Manual corner-basin safeguard on f5 only
- Hybrid surrogate stack retained:
  - GP regressor for uncertainty-aware scoring
  - MLP regressor for non-linear local structure
  - logistic regression + RBF SVM for region discrimination

Implementation used:
- `execution/propose_round_08_candidates.py`

## 6) Official Round 8 query set prepared
Canonical portal strings (`deliverables/submissions/round_08_portal_strings.txt`):
- function_1: `0.806450-0.756325`
- function_2: `0.701073-0.065274`
- function_3: `0.050543-0.037158-0.093648`
- function_4: `0.377194-0.383384-0.370038-0.469160`
- function_5: `0.979270-0.978906-0.979500-0.979702`
- function_6: `0.489991-0.315136-0.565761-0.941643-0.031525`
- function_7: `0.031678-0.407209-0.398641-0.042540-0.255976-0.730075`
- function_8: `0.161552-0.031331-0.174070-0.047935-0.820760-0.579446-0.128132-0.589476`

## 7) Artifacts created/updated
- New execution wrapper:
  - `execution/propose_round_08_candidates.py`
- New ingestion helper:
  - `deliverables/submissions/round_07_inputs_batched.txt`
- Round 7 canonical output snapshot:
  - `deliverables/submissions/round_07_outputs_canonical.txt`
- Round 8 canonical files:
  - `deliverables/submissions/round_08_inputs.txt`
  - `deliverables/submissions/round_08_portal_strings.txt`
  - `deliverables/submissions/round_08_portal_strings.json`
  - `deliverables/submissions/round_08_hybrid_debug.json`

Validation:
- Existing BO unit tests passed:
  - `python -m unittest tests/test_bo_core.py`

## 8) Key evidence this set is selective-exploit rather than globally greedy
From Round 8 debug diagnostics:
- UCB retained on:
  - f1, f2, f3
- EI retained on:
  - f4, f5, f6, f7, f8
- The threshold split is evidence-based:
  - f2 `z_best=1.64` stayed below the EI threshold
  - f4 `1.72`, f6 `1.69`, f8 `1.85` stayed above it
- Local exploitation remains strong where it worked:
  - f4 min distance: `0.0451`
  - f6 min distance: `0.0437`
  - f8 min distance: `0.2994`
- f5 did not follow the unstable surrogate jump:
  - manual corner override was applied from the top two observed points.

Interpretation:
- Round 8 keeps the exploit gains from Round 7, but with a cleaner separation between weak-signal functions and proven local basins.
