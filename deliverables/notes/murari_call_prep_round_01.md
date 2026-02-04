# Murari Call Script — Round 01 Review + Round 02 Strategy

## Goal (30 seconds)
I want feedback on whether my interpretation of the pre/post plots is reasonable and confirm a GP-based strategy for Round 02.

## Quick agenda (10–15 min)
- Show two pre/post plot comparisons (functions 2 and 5) to demonstrate learning.
- Summarize what seems stable vs uncertain across all 8 functions.
- Walk through my Round 02 plan: GP surrogate + acquisition to propose one new point per function.
- Ask for feedback on exploration/exploitation balance and any pitfalls.

## Plots to show (high signal)
- Function 2: `deliverables/round_01/plots/post/function_2/scatter_x1_x2.png`
- Function 5: `deliverables/round_01/plots/post/function_5/parallel_coords.png`
- If time: Function 1 or 3 pre/post scatter (`deliverables/round_01/plots/pre/...` vs `post/...`)

## Strategy for Round 02 (what I will do)
Iterative loop for each function:
1) Fit GP surrogate on all available points.
2) Use an acquisition function to pick the next input.
3) Query the black-box with that input.
4) Append new data and refit GP.
5) Repeat each round.

## Python functions I plan to use (core building blocks)
- Data handling: `numpy.load`, `numpy.vstack`, `numpy.concatenate`
- Scaling (optional): `sklearn.preprocessing.StandardScaler`
- GP model:
  - `sklearn.gaussian_process.GaussianProcessRegressor`
  - Kernels: `sklearn.gaussian_process.kernels.Matern`, `RBF`, `WhiteKernel`, `ConstantKernel`
- Acquisition:
  - Expected Improvement (EI): `scipy.stats.norm.cdf`, `scipy.stats.norm.pdf`
  - Upper Confidence Bound (UCB): simple formula using mean/std from GP
- Candidate generation:
  - Random: `numpy.random.default_rng().random((n_candidates, d))`
  - Optional low‑D grid: `numpy.linspace` + `numpy.meshgrid`

## Open questions for Murari
- Is EI or UCB better for the noisy outputs we see so far?
- Should I emphasize exploration for the higher‑D functions (6–8D), even after a strong result?
- Is it acceptable to use random candidate sets for acquisition, or should I add low‑D grid searches?
- Any advice on kernel choice or noise term size at this stage?
