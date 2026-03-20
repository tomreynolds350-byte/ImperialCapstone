# Software Architecture

## 1) Design goals
- Keep optimization decisions reproducible and auditable.
- Separate data, model logic, and submission artifacts.
- Preserve flexibility for strategy updates across rounds.

## 2) High-level pipeline
1. Ingest latest portal feedback (inputs and outputs).
2. Append one new `(x, y)` pair per function into canonical data arrays.
3. Fit surrogate models per function.
4. Score candidate pools with acquisition-driven policy.
5. Export portal-ready strings and debug diagnostics.
6. Document rationale and outcomes in a work log + reflection.

## 3) Component map
- Data layer:
  - `initial_data/function_*/initial_inputs.npy`
  - `initial_data/function_*/initial_outputs.npy`
- Modeling/orchestration layer:
  - `execution/bo_core.py` (shared BO core: ingestion, GP fitting, scoring, refinement, artifact writing)
  - `execution/propose_round_06_candidates.py` (current default round workflow wrapper)
- Artifact/output layer:
  - `deliverables/submissions/round_XX_inputs.txt`
  - `deliverables/submissions/round_XX_portal_strings.txt`
  - `deliverables/submissions/round_XX_hybrid_debug.json`
- Documentation layer:
  - `deliverables/notes/round_XX_work_log.md`
  - `deliverables/reflections/*.md`

## 4) Modeling stack
- Gaussian Process Regressor:
  - Provides mean + uncertainty for acquisition scoring.
  - Uses an anisotropic Matern kernel with direct optimizer restarts and diagnostics-first fit reporting.
- MLP Regressor:
  - Captures nonlinear response patterns for shortlist reranking.
- Logistic Regression + RBF SVM:
  - Classifies high-vs-low response regions and highlights boundary uncertainty.

## 5) Strategy policy
- Current policy is controlled exploitation (after exploration-heavy rounds):
  - Balanced acquisition mode with reduced UCB pressure.
  - EI enabled where incumbent dominance is strong.
  - Novelty retained as guardrail, but with reduced weight.
  - GP acquisition selects the shortlist first; hybrid reranking and boundary override decide the final point.

## 6) Architecture trade-offs
- Benefits:
  - Better uncertainty handling than purely deterministic heuristics.
  - Better local sample efficiency in stronger regions.
  - Clear artifact trail for each round.
- Costs:
  - More moving parts than a single-model baseline.
  - Hyperparameter sensitivity in small-data/high-dimensional functions.
  - Additional compute for multi-model scoring.

## 7) Technical evidence links
- [Technical Foundations](TECHNICAL_FOUNDATIONS.md)
- NeurIPS references used in this project include HEBO (1st place), NVIDIA RAPIDS ensemble (2nd place), and JetBrains SPBOpt paper (3rd place).
