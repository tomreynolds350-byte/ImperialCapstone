# Round 06 Work Log (ML Student Notes)

## Why this document exists
You asked for Round 5 performance stats and the same style of round documentation as previous weeks, while shifting strategy from exploration toward exploitation for Round 6.

## 1) Round 5 ingestion from local submission artifacts
What I used:
- `deliverables/submissions/round_05_inputs_batched.txt`
- `deliverables/submissions/round_05_outputs.txt` (latest batch = Round 5 outputs)

What was done:
- Parsed the batched files and selected the latest batch (Round 5 feedback).
- Appended one `(x, y)` row per function into:
  - `initial_data/function_*/initial_inputs.npy`
  - `initial_data/function_*/initial_outputs.npy`
- Wrote canonical Round 5 outputs to `deliverables/submissions/round_05_outputs_canonical.txt`.

Validation:
- All 8 functions were appended exactly once (no duplicates detected).
- Sample counts after append:
  - f1: 15, f2: 15, f3: 20, f4: 35, f5: 25, f6: 25, f7: 35, f8: 45.

## 2) Round 5 outcomes and challenge stats
Observed change (`round_05_output - round_04_output`):
- f1: up (tiny near-zero magnitude)
- f2: up
- f3: down
- f4: down
- f5: down
- f6: down
- f7: up
- f8: down

Objective impact (maximize each function):
- 3/8 improved vs Round 4 (f1, f2, f7).
- 6/8 improved vs Round 1 baseline (f1, f2, f3, f4, f7, f8).
- Round 5 rank among submitted rounds (1=best):
  - f1: 3rd, f2: 2nd, f3: 4th, f4: 4th, f5: 5th, f6: 5th, f7: 3rd, f8: 4th.
- New bests by round across submitted rounds:
  - Round 1: 1 function
  - Round 2: 2 functions
  - Round 3: 4 functions
  - Round 4: 1 function
  - Round 5: 0 functions

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
- Round 5 improved some functions but was net weaker than earlier peak rounds.
- That supports moving into a more exploitative policy for Round 6.

## 3) Final Round 6 strategy (shift from exploration to exploitation)
Final policy used for this submission:
- Strategy mode switched from exploration-heavy to `balanced`.
- UCB pressure reduced (`kappa=1.6`, down from prior exploration-heavy settings).
- EI allowed where incumbent dominance is strong (f5 and f7), with UCB elsewhere.
- Hybrid surrogate stack retained:
  - GP regressor for uncertainty-aware scoring.
  - MLP regressor for non-linear response structure and gradient refinement.
  - Logistic regression + RBF SVM for boundary structure (`good` vs `bad`).
- Novelty retained as a guardrail, but with lower weight than exploration rounds.

Implementation used:
- `execution/propose_round_06_candidates.py`

## 4) Official Round 6 query set prepared
Canonical portal strings (`deliverables/submissions/round_06_portal_strings.txt`):
- function_1: `0.944613-0.791862`
- function_2: `0.695161-0.896365`
- function_3: `0.671332-0.942951-0.050580`
- function_4: `0.422239-0.357431-0.465260-0.430476`
- function_5: `0.980000-0.980000-0.980000-0.980000`
- function_6: `0.244306-0.236834-0.907955-0.942766-0.041013`
- function_7: `0.068002-0.394924-0.281174-0.125062-0.300620-0.707610`
- function_8: `0.214123-0.179436-0.036257-0.159980-0.552780-0.739544-0.089287-0.273649`

## 5) Artifacts created/updated
- New execution script: `execution/propose_round_06_candidates.py`
- New batched helper for ingestion parity:
  - `deliverables/submissions/round_05_inputs_batched.txt`
- Round 5 canonical output snapshot:
  - `deliverables/submissions/round_05_outputs_canonical.txt`
- Round 6 canonical files:
  - `deliverables/submissions/round_06_inputs.txt`
  - `deliverables/submissions/round_06_portal_strings.txt`
  - `deliverables/submissions/round_06_portal_strings.json`
  - `deliverables/submissions/round_06_hybrid_debug.json`

## 6) Key evidence this set is exploitation-forward
From debug diagnostics:
- EI (exploitative acquisition) activated where incumbent is strongest:
  - f5: `acquisition=ei`, `z_best=3.191`
  - f7: `acquisition=ei`, `z_best=3.628`
- Novelty weights are low in strongest-incumbent functions:
  - f5 novelty weight: `0.05`
  - f7 novelty weight: `0.05`
- Several chosen points are close to known regions:
  - f2 min distance: `0.0311`
  - f5 min distance: `0.1000`
  - f7 min distance: `0.0848`

Interpretation:
- Round 6 policy now favors model confidence and local improvement where signal is strong, while still preserving limited coverage in uncertain areas.
