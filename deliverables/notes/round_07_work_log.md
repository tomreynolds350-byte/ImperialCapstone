# Round 07 Work Log (ML Student Notes)

## Why this document exists
You asked for Round 6 performance stats and the same style of round documentation as previous weeks, while moving more deliberately from exploration into exploitation for Round 7 and grounding the reasoning in Module 18 hyperparameter tuning ideas.

## 1) Round 6 ingestion from local submission artifacts
What I used:
- `deliverables/submissions/round_06_inputs_batched.txt`
- `deliverables/submissions/round_06_outputs.txt` (latest batch = Round 6 outputs)

What was done:
- Created `round_06_inputs_batched.txt` by appending `round_06_inputs.txt` to the prior batched input file.
- Parsed the latest batch (Round 6 feedback).
- Appended one `(x, y)` row per function into:
  - `initial_data/function_*/initial_inputs.npy`
  - `initial_data/function_*/initial_outputs.npy`
- Wrote canonical Round 6 outputs to `deliverables/submissions/round_06_outputs_canonical.txt`.

Validation:
- All 8 functions were appended exactly once (no duplicates detected).
- Sample counts after append:
  - f1: 16, f2: 16, f3: 21, f4: 36, f5: 26, f6: 26, f7: 36, f8: 46.

## 2) Round 6 outcomes and challenge stats
Observed change (`round_06_output - round_05_output`):
- f1: up (tiny near-zero magnitude)
- f2: up by `+0.1639`
- f3: up by `+0.0487`
- f4: up by `+14.2646`
- f5: up by `+5031.8634`
- f6: up by `+1.7806`
- f7: up by `+0.8879`
- f8: up by `+1.3284`

Objective impact (maximize each function):
- 8/8 improved vs Round 5.
- 8/8 improved vs Round 1 baseline.
- Round 6 rank among submitted rounds (1=best):
  - f1: 3rd, f2: 1st, f3: 1st, f4: 3rd, f5: 1st, f6: 3rd, f7: 1st, f8: 1st.
- Average submitted-round rank for Round 6: `1.75` (best round so far).
- Round 6 currently holds the best submitted result in 5/8 functions:
  - f2, f3, f5, f7, f8.
- Round 6 set new all-time bests (initial data + rounds) in 4/8 functions:
  - f2, f5, f7, f8.

Current all-time best source:
- f1: initial set
- f2: Round 6
- f3: initial set
- f4: Round 3
- f5: Round 6
- f6: Round 3
- f7: Round 6
- f8: Round 6

Interpretation:
- Round 6 strongly validated the exploit shift.
- The remaining stubborn functions are f1, f3, f4, and f6, so Round 7 should be selective exploitation rather than uniform corner-chasing.

## 3) Module 18-informed hyperparameter tuning for Round 7
Hyperparameters tuned this week:
- strategy profile: moved from `balanced` to `exploit`
- UCB pressure: lowered `kappa` from `1.6` to `1.25`
- EI/UCB switch sensitivity: lowered `z_best_threshold` from `2.2` to `1.6`
- boundary behavior: tightened `boundary_margin` from `0.035` to `0.028`

Methods used:
- Manual adjustment to define a more exploit-forward policy.
- Small grid search over policy settings on a temporary copy of the 16-point dataset:
  - `balanced_k1.6_t2.2_b035`
  - `balanced_k1.35_t2.0_b030`
  - `exploit_k1.4_t2.0_b030`
  - `exploit_k1.25_t1.8_b030`
  - `exploit_k1.1_t1.6_b028`
  - `exploit_k1.25_t1.6_b028`
- The per-function search itself remains Bayesian optimization:
  - GP surrogate + EI/UCB acquisition
  - random/global candidate sampling
  - local perturbation around strong incumbents
  - hybrid reranking with MLP and boundary classifiers

Selection rationale:
- `exploit_k1.25_t1.6_b028` gave the strongest exploitation profile without collapsing everything to one extreme.
- It reduced average distance to current incumbents versus the weaker exploit settings.
- It activated EI on 6/8 functions while keeping UCB on the two weakest-signal cases.

## 4) Final Round 7 strategy
Final policy used for this submission:
- Strategy mode switched to `exploit`.
- `kappa=1.25`
- `z_best_threshold=1.6`
- `boundary_margin=0.028`
- Hybrid surrogate stack retained:
  - GP regressor for uncertainty-aware scoring
  - MLP regressor for non-linear local structure
  - logistic regression + RBF SVM for `good` vs `bad` region discrimination
- No PyTorch migration yet:
  - scikit-learn remains the production baseline for this submission round.

Implementation used:
- `execution/bo_core.py`
- `execution/propose_round_07_candidates.py`

## 5) Official Round 7 query set prepared
Canonical portal strings (`deliverables/submissions/round_07_portal_strings.txt`):
- function_1: `0.703237-0.943635`
- function_2: `0.642872-0.036917`
- function_3: `0.945262-0.910815-0.402281`
- function_4: `0.370921-0.387088-0.378369-0.425395`
- function_5: `0.978540-0.977812-0.979000-0.979404`
- function_6: `0.499860-0.302567-0.526474-0.951831-0.033665`
- function_7: `0.029747-0.306322-0.257998-0.079994-0.312747-0.755726`
- function_8: `0.100894-0.051417-0.109188-0.058883-0.948183-0.583779-0.236027-0.358484`

## 6) Artifacts created/updated
- New execution wrapper:
  - `execution/propose_round_07_candidates.py`
- BO core updated for Round 7 tuning:
  - `execution/bo_core.py`
- New ingestion helper:
  - `deliverables/submissions/round_06_inputs_batched.txt`
- Round 6 canonical output snapshot:
  - `deliverables/submissions/round_06_outputs_canonical.txt`
- Round 7 canonical files:
  - `deliverables/submissions/round_07_inputs.txt`
  - `deliverables/submissions/round_07_portal_strings.txt`
  - `deliverables/submissions/round_07_portal_strings.json`
  - `deliverables/submissions/round_07_hybrid_debug.json`

Validation:
- Existing BO unit tests passed:
  - `python -m unittest tests/test_bo_core.py`

## 7) Key evidence this set is exploitation-forward
From Round 7 debug diagnostics:
- EI is now active on 6/8 functions:
  - f2, f4, f5, f6, f7, f8.
- UCB was retained only for the weakest/least-settled regimes:
  - f1 and f3.
- Several candidates are intentionally close to current incumbents:
  - f5 min distance: `0.0029`
  - f4 min distance: `0.0442`
  - f2 min distance: `0.0900`
  - f7 min distance: `0.1198`
- High-confidence functions still carry strong classifier support:
  - f5 `p_good=0.965`
  - f7 `p_good=0.786`
  - f8 `p_good=0.977`

Interpretation:
- Round 7 is materially more exploitative than Round 6, but not fully greedy.
- The policy still preserves UCB on unresolved functions and keeps one wider jump in 8D space where uncertainty remains structurally high.
