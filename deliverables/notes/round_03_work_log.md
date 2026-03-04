# Round 03 Work Log (ML Student Notes)

## Why this document exists
You asked for the same style of post-round documentation we used before, with emphasis on how the code works and what changed after ingesting Round 2 outputs.

## 1) Round 2 ingestion and duplication handling
What happened:
- The files in `.tmp` (`round 2 inputs.txt` and `round 2 outputs.txt`) each contained **two batches**.
- Batch 1 matched prior round artifacts (`round_01_inputs.txt`, `round_01_outputs.txt`).
- Batch 2 was the actual new Round 2 feedback.

What was done:
- Wrote canonical Round 2 outputs to `deliverables/submissions/round_02_outputs.txt`.
- Normalized Round 2 inputs in `deliverables/submissions/round_02_inputs.txt`.
- Appended exactly one new `(x, y)` row per function into:
  - `initial_data/function_*/initial_inputs.npy`
  - `initial_data/function_*/initial_outputs.npy`

Validation:
- Row counts now match "initial + 2 rounds" exactly for all 8 functions.
- No exact duplicate input rows or output values were introduced.

## 2) Round 2 outcome summary versus Round 1
Observed performance changes (`round_02_output - round_01_output`):
- f1: near flat (tiny positive, still near zero)
- f2: down
- f3: up
- f4: up strongly
- f5: down (still very high overall)
- f6: up strongly
- f7: up strongly
- f8: up strongly

Capstone objective impact (maximize each black-box function):
- 6/8 Round 2 outputs improved over Round 1.
- 3/8 set new global bests (f4, f6, f8).
- This is a healthy exploration/exploitation tradeoff at this stage.

## 3) Round 3 candidate generation and Function 5 adjustment
Base generation:
- Used `execution/propose_gp_candidates.py` to produce GP-guided Round 3 candidates.
- Artifacts created in `deliverables/submissions/round_03_*`.

Your concern:
- You flagged that Function 5 proposals looked too concentrated near the same high region.

Adjustment made:
- Replaced `function_5` Round 3 candidate with a more exploratory alternative:
  - New f5 input: `0.973381-0.951652-0.977831-0.659398`
- Why this is more exploratory:
  - Minimum Euclidean distance to existing f5 inputs is `0.2661` (substantially non-local).
  - Still inside bounds `[0.001, 0.98]`.
- Updated files consistently:
  - `deliverables/submissions/round_03_inputs.txt`
  - `deliverables/submissions/round_03_portal_strings.txt`
  - `deliverables/submissions/round_03_portal_strings.json`

## 4) Plot/markdown package regenerated (same structure as before)
Generated Round 2 pre/post comparison package:
- Plots root: `deliverables/round_02/plots`
  - `pre/` = initial + Round 1
  - `post/` = initial + Round 1 + Round 2
- Summary markdown: `deliverables/round_02/plots/round_02_pre_post_summary.md`
- Summary PDF: `deliverables/round_02/plots/round_02_pre_post_summary.pdf`
- Student plot guide: `deliverables/notes/round_02_plot_guide.md`

## 5) How the code works (quick map)
- `execution/plot_round_comparison.py`
  - Loads a "pre" dataset and one round's inputs/outputs.
  - Builds `pre` and `post` arrays per function.
  - Calls plotting utilities to render all charts.
- `execution/plot_initial_data.py`
  - Contains plotting functions (histograms, dim-vs-y, PCA, correlation heatmap, scatter matrix, parallel coordinates).
- `execution/generate_plot_guide.py`
  - Reads current data and writes student-readable interpretation notes.
  - Now supports custom title and plot path prefix for per-round docs.
- `execution/build_round_01_plot_pdf.py`
  - Builds side-by-side pre/post PDF pages from selected panels.
  - Now supports custom pre/post labels.
