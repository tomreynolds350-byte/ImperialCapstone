# Round 04 Work Log (ML Student Notes)

## Why this document exists
You asked for the same style of round documentation as before, with clear methodology and outcomes for the next capstone submission.

## 1) Round 3 ingestion from portal exports
What I used:
- `c:\Users\tom_m\Downloads\inputs.txt`
- `c:\Users\tom_m\Downloads\outputs.txt`

What was done:
- Parsed the batched files and selected the latest batch (Round 3).
- Wrote canonical Round 3 outputs to `deliverables/submissions/round_03_outputs.txt`.
- Appended one `(x, y)` row per function into:
  - `initial_data/function_*/initial_inputs.npy`
  - `initial_data/function_*/initial_outputs.npy`

Validation:
- All 8 functions were appended exactly once (no duplicates detected).
- Sample counts after append:
  - f1: 13, f2: 13, f3: 18, f4: 33, f5: 23, f6: 23, f7: 33, f8: 43.

## 2) Round 3 outcome summary versus Round 2
Observed change (`round_03_output - round_02_output`):
- f1: down slightly (tiny near-zero magnitude)
- f2: up strongly
- f3: down slightly
- f4: up
- f5: down (expected after forcing exploration in Round 3)
- f6: up
- f7: up
- f8: down slightly

Impact on objective (maximize):
- 4/8 improved vs Round 2 (f2, f4, f6, f7).
- 7/8 improved vs Round 1.
- 4/8 set new global bests in Round 3 (f2, f4, f6, f7).

## 3) Final Round 4 strategy (exploration-heavy)
Final policy used for this week's submission:
- UCB-only acquisition (no EI switching), with higher kappa for exploration pressure.
- Higher novelty weight in candidate scoring.
- Novelty floor filtering: candidates must be sufficiently far from existing points.
- Larger global candidate pools and reduced concentration around incumbent-best neighborhoods.
- Boundary-aware classification retained, but exploitation weights reduced versus novelty.

Models retained in the stack:
- GP regressor (uncertainty signal).
- MLP regressor (non-linear fit; backprop-trained).
- Logistic regression + RBF SVM (`good` vs `bad`) for boundary structure.

## 4) Official Round 4 query set submitted
Canonical portal strings (`deliverables/submissions/round_04_portal_strings.txt`):
- function_1: `0.940497-0.035466`
- function_2: `0.943199-0.361781`
- function_3: `0.937642-0.945531-0.045005`
- function_4: `0.050747-0.334864-0.705454-0.507004`
- function_5: `0.942745-0.498680-0.943096-0.895238`
- function_6: `0.122455-0.861277-0.072306-0.899635-0.035296`
- function_7: `0.093978-0.807689-0.076030-0.035905-0.063327-0.905769`
- function_8: `0.068040-0.751590-0.037700-0.760163-0.113308-0.905472-0.130826-0.072029`

## 5) Artifacts created/updated
- New execution script (updated with strategy modes): `execution/propose_round_04_candidates.py`
- Round 3 canonical outputs: `deliverables/submissions/round_03_outputs.txt`
- Canonical Round 4 files (exploration-heavy):
  - `deliverables/submissions/round_04_inputs.txt`
  - `deliverables/submissions/round_04_portal_strings.txt`
  - `deliverables/submissions/round_04_portal_strings.json`
  - `deliverables/submissions/round_04_hybrid_debug.json`
- Alternate archived copy (same exploration mode, explicit suffix):
  - `deliverables/submissions/round_04_explore_*`

## 6) Key evidence that this set is more exploratory
Minimum distance from prior samples increased materially versus the earlier balanced set in most functions (largest jumps in f4, f5, f6, f7).

Interpretation:
- Queries are now probing less-visited regions while still preserving model guidance.
- This aligns with the stated Stage 2 preference: exploration-heavy before late-stage tightening.
