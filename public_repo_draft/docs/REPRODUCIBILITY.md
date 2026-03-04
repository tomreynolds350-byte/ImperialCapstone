# Reproducibility Guide

## Preconditions
- Python environment with dependencies used in the execution scripts (`numpy`, `scipy`, `scikit-learn`).
- Latest portal feedback files available locally.

## Standard round workflow (current: Round 6 script)
Run from repository root:

```powershell
python execution/propose_round_06_candidates.py `
  --inputs-path deliverables/submissions/round_05_inputs_batched.txt `
  --outputs-path deliverables/submissions/round_05_outputs.txt `
  --round-index -1 `
  --strategy balanced `
  --kappa 1.6 `
  --prefix round_06
```

What this does:
1. Parses the latest batched round data.
2. Appends round feedback into `initial_data/function_*/`.
3. Trains per-function surrogate stack.
4. Generates one candidate per function.
5. Writes canonical round artifacts.

## Expected output artifacts
- `deliverables/submissions/round_06_inputs.txt`
- `deliverables/submissions/round_06_portal_strings.txt`
- `deliverables/submissions/round_06_portal_strings.json`
- `deliverables/submissions/round_06_hybrid_debug.json`
- `deliverables/submissions/round_05_outputs_canonical.txt` (latest ingested batch snapshot)

## Validation checklist
- Confirm each function has exactly one new row appended in `initial_data`.
- Confirm portal string dimensions match function dimensions.
- Confirm all values remain inside configured bounds.
- Confirm `round_06_hybrid_debug.json` includes config + ingest summary.

## Operational notes
- Keep workflow append-only; avoid manual edits to historical round outputs.
- If rerunning candidate generation without ingesting again, use `--skip-ingest`.
- Keep work-log and reflection files in sync with the generated round artifacts.
