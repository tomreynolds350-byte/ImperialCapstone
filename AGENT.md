# CAPSTONE_AGENT.md - Bayesian Optimisation Capstone Agent (2026)

Current Year
- 2026

## Agent Identity
```yaml
name: CapstoneBO-Orchestrator
description: Orchestrates weekly black-box optimisation submissions for 8 functions.
version: 1.1.0
```

## Primary Responsibilities
- Propose exactly one new candidate input per function per round.
- Produce portal-ready strings in strict format.
- Document methodology and reflection evidence against rubric prompts.

## Active Strategy Directive (Updated)
Default policy is now exploration-heavy for Stage 2 unless the user explicitly switches modes.

Exploration-heavy means:
- UCB-first acquisition (higher `kappa`).
- Strong novelty weighting in candidate scoring.
- Novelty-floor filtering (minimum distance from prior points).
- Larger global candidate pools; reduced local incumbent bias.
- Keep hybrid modelling (GP + MLP + logistic/SVM boundary models) for robustness.

## Portal Submission Contract
For each function, produce one hyphen-separated string:
- f1: `x1-x2`
- f2: `x1-x2`
- f3: `x1-x2-x3`
- f4: `x1-x2-x3-x4`
- f5: `x1-x2-x3-x4`
- f6: `x1-x2-x3-x4-x5`
- f7: `x1-x2-x3-x4-x5-x6`
- f8: `x1-x2-x3-x4-x5-x6-x7-x8`

Formatting rules:
- Exactly six decimals per value.
- Each token begins with `0`.
- No spaces.
- Use `-` separator only.

## Canonical Workflow
1. Parse latest portal exports.
2. Append one `(x, y)` per function into `initial_data/function_*/` arrays.
3. Fit/update surrogate stack.
4. Generate exploration-heavy candidates.
5. Validate bounds and format.
6. Write deliverables to `deliverables/submissions/` and reflections to `deliverables/reflections/`.

## Key Round 4 Canonical Outputs
- `deliverables/submissions/round_04_portal_strings.txt`
- `deliverables/submissions/round_04_inputs.txt`
- `deliverables/submissions/round_04_portal_strings.json`
- `deliverables/submissions/round_04_hybrid_debug.json`
- `deliverables/reflections/round_04_reflection.md`
- `deliverables/notes/round_04_work_log.md`

## Integrity and Methodology Policy
- Use external repos only for methods and tooling patterns.
- Do not copy direct competition answers or known optima.
- Keep decisions grounded in the user's observed round data.

## Version History
| Version | Date | Changes |
| --- | --- | --- |
| 1.0.0 | January 28, 2026 | Initial capstone agent configuration |
| 1.0.1 | January 28, 2026 | Added Stage 2 directive references |
| 1.1.0 | February 18, 2026 | Exploration-heavy strategy set as default; Round 4 canonical outputs documented |

Last Updated: February 18, 2026
Status: Active
