# Stage 2 High-Level Directive

Goal
- Maximize eight unknown black-box functions under strict evaluation limits.
- Propose exactly one new input per function per round and produce reflection-ready evidence.

When to Use
- At the start of each weekly round.
- When resetting strategy after new portal feedback.

Inputs
- Current datasets per function (`initial_data/function_*/initial_inputs.npy`, `initial_outputs.npy`).
- Latest portal feedback files (inputs/outputs export).
- Round number and user preference (currently: exploration-heavy).

Outputs
- Eight candidate inputs (one per function).
- Portal-ready strings (hyphen-separated, six decimals).
- Reflection-ready evidence (support vectors, gradients, model trade-offs, next-step rationale).

Process
1. Parse latest portal exports and append one `(x, y)` row per function (unless explicitly skipped).
2. Recompute per-function state: incumbent best, spread, uncertainty, and density.
3. Generate candidates with hybrid surrogates (GP + MLP + logistic/SVM boundary models).
4. Score candidates with exploration-heavy policy:
   - UCB-focused acquisition,
   - novelty weighting,
   - novelty-floor filtering,
   - boundary sanity checks.
5. Select one in-bounds candidate per function.
6. Write `round_XX_*` submission artifacts and reflection notes.

Current Strategy Directive (Active)
- Default to exploration-heavy unless the user asks otherwise.
- Prefer broader state-space coverage over short-term local gain.
- Keep exploitation constrained until later rounds.

Constraints and Scope
- One query per function per round.
- Dimensions: 2D, 2D, 3D, 4D, 4D, 5D, 6D, 8D.
- Outputs may be noisy; avoid overfitting to single-point jumps.
- Keep all portal values in `[0, 1)` with six-decimal formatting.

Edge Cases
- Missing/incomplete round files: stop and request correct files.
- Duplicate input with mismatched output: do not append; flag data integrity issue.
- Dimension mismatch: fail fast and inspect function-specific files.
- Formatting rejection by portal: revalidate token count and decimal formatting.

Change History

| Version | Date | Changes |
| --- | --- | --- |
| 1.0.0 | January 28, 2026 | Initial directive created |
| 1.0.1 | January 28, 2026 | Added change history section |
| 1.0.2 | February 2, 2026 | Clarified portal feedback loop and noise handling |
| 1.1.0 | February 18, 2026 | Set exploration-heavy policy as active default; added hybrid-surrogate and novelty-floor process |

Last Updated: February 18, 2026
Status: Active
