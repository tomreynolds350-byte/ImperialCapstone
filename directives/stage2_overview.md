# Stage 2 High-Level Directive

Goal
- Maximize eight unknown black-box functions using limited evaluations.
- Propose exactly one new input per function per round and produce a short reflection.

When to Use
- At the start of Stage 2 to set expectations and constraints.
- When the user asks for the overall process, rules, or a reset of strategy.

Inputs
- Initial data: 10 (x, y) points per function.
- Round number and any new outputs from the portal.
- Downloaded portal feedback for the most recent round (often delivered via email and/or as a downloadable file; may be a NumPy binary).
- Any user constraints (time, preferred methods, reporting format).
- Portal submission format requirements, if provided.

Outputs
- Eight candidate inputs (one per function) for the next round.
- Portal-ready strings (hyphen-separated, six decimals per component).
- A brief reflection covering method, exploration vs exploitation, what was learned, and next steps.

Process
1. Download portal feedback and append the new (x, y) row to each function's dataset.
2. Review all available data per function (initial + prior rounds).
3. Identify the current best y and its x for each function.
4. Pick a strategy per function (explore, exploit, or balance) based on dimension and data density.
5. Propose one new x per function (manually adjust if needed for sanity/bounds).
6. Validate format and bounds; produce portal-ready strings.
7. Write the reflection tied to decisions and evidence.

Constraints and Scope
- One query per function per round.
- Functions are unknown and increase in dimension from 2D to 8D.
- Evaluations are limited; focus on smart, evidence-based guesses.
- Use any ML method (random, grid, Bayesian optimization, manual reasoning, surrogate models).
- Perfection is not required; the goal is a thoughtful iterative process.
- Stage 2 runs from Module 2 through Module 24; Module 25 is for final tuning and optional sharing.

Not Required
- Building a full optimizer from scratch.
- Submitting code as part of the portal.
- Finding the global maximum for every function.

Edge Cases
- Missing or incomplete round data: ask for clarification before proposing.
- Dimension mismatch: verify expected dimension per function before formatting.
- Noise is expected (outputs include random noise): avoid overreacting to a single point; note it in reflection and hedge with exploration.
- Same input may return a slightly different output: treat y as noisy and focus on trends over rounds.
- Portal format rejection: revalidate precision and separator rules.

Notes
- Reflections should document method choice, exploration vs exploitation, and what the last result taught you.
- Capstone "success" is completing the iterative submit -> feedback -> append -> resubmit loop each round; optimality is not required.
- Keep a clear, repeatable weekly workflow for consistency.

Change History

| Version | Date | Changes |
| --- | --- | --- |
| 1.0.0 | January 28, 2026 | Initial directive created |
| 1.0.1 | January 28, 2026 | Added change history section |
| 1.0.2 | February 2, 2026 | Clarified portal feedback loop, manual adjustment, and expected noise |

Last Updated: February 2, 2026
Status: Active
