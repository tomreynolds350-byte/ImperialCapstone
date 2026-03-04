# Round 7 Evolution Note
Date: 2026-03-04

- User preference: begin transitioning parts of the modeling stack toward PyTorch later in the capstone.
- Timing: do not switch now; Round 6 is still early-stage.
- Trigger: revisit after Round 6 outputs are ingested and Round 7 planning starts.
- Execution plan:
  1. Keep current scikit-learn workflow as the production baseline.
  2. Prototype PyTorch surrogate(s) in parallel for comparison.
  3. Promote PyTorch components only if they improve diagnostics and round-to-round outcome stability.
