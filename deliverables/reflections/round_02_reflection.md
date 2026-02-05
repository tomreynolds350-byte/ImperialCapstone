Round 02 Reflection (Stage 2 - Part 1 and Part 2)

Part 1 - Submission reflection (same format as Round 01)

This is my second submission for Stage 2 and the approach has shifted from pure distance-based exploration to a true Gaussian process (GP) surrogate with acquisition functions. I fit a GP (Matern kernel) per function using scikit-learn, tune the kernel hyperparameters via a small randomized search (following the example techniques in the shared repo), then choose between Expected Improvement (EI) and Upper Confidence Bound (UCB) based on how extreme the current best output is relative to the median (a simple z-score rule). Candidate points come from a mix of global random points and local perturbations around the incumbent. I also cap candidate inputs to the range [0.001, 0.98] and apply a soft boundary penalty so the optimizer does not collapse onto extreme 0.00/0.98 corners. The intent is to keep exploration dominant early on, but now with a probabilistic model that quantifies uncertainty rather than a purely geometric rule.

Per-function principle (what guided each query):
- Function 1 (2D): UCB for exploration; uncertainty is high and outputs are near zero, so UCB is safer than a narrow EI peak.
- Function 2 (2D, noisy): UCB to hedge against noise and preserve exploration in a small space.
- Function 3 (3D): UCB to explore; the best value is not extreme relative to the median.
- Function 4 (4D): UCB to explore; the best value is modest relative to spread and the surface is likely multi-modal.
- Function 5 (4D): EI to exploit; the current best is a strong outlier, so I prioritised improvement around the high region while still avoiding boundary collapse.
- Function 6 (5D): UCB to explore; moderate best-vs-median gap and higher dimensionality.
- Function 7 (6D): EI to exploit; the best value is a large outlier and worth probing locally.
- Function 8 (8D): UCB to explore; high dimensionality and sparse coverage.

Most challenging functions and why:
- Functions 7 and 8 remain the hardest due to dimensionality and sparse coverage; any surrogate is fragile and likely to be misled by noise.
- Function 1 is also tricky because outputs are near zero and the signal is weak; this makes directional inference highly uncertain.

How I will adjust in future rounds:
- Keep the GP + EI/UCB split, but tune the z-score threshold and UCB kappa based on how stable the next outputs are.
- Expand the candidate pool with Sobol sequences and a slightly denser local cloud around the incumbent in high-performing functions.
- For functions with large output variance (e.g., Function 5), test small local perturbations around the best point to distinguish true trend from noise.

Part 2 - Reflection on strategy (prompts)

1) Main change in strategy and what prompted it
The main change was the move from a max-distance rule (Round 1) to a true GP surrogate with EI/UCB acquisition functions (Round 2), plus kernel tuning via randomized search. The prompt for this change was twofold: the Round 1 outputs, which revealed at least one strong signal (Function 5), and the tutor advice to begin using model-based heuristics while remaining exploration-heavy in the early weeks. A GP lets me quantify uncertainty, and switching between EI and UCB lets me decide when to exploit a strong outlier versus keep exploring.

2) Exploration vs exploitation and trade-offs
I still prioritised exploration overall, but with selective exploitation for functions that showed a clearer trend (notably Functions 5 and 7). The trade-off is straightforward: early exploration helps avoid local maxima, but ignoring obvious high-performing regions is inefficient. UCB is my default for exploration; EI is used when the best observed value is a clear outlier and worth local refinement.

3) Influence from participants, discussions, or recent outputs
The tutor call directly influenced the balance: continue exploration until around weeks 5-6, but start using model-based search rather than random coverage. The large positive output in Function 5 also pushed me toward a mild exploitation step to test whether the trend is monotonic or just a lucky outlier.

4) Likely assumption violations for linear/logistic regression
For several functions the linearity and additivity assumptions are likely violated (nonlinear response surfaces, interaction terms, and possible multi-modality). Function 2 is explicitly noisy, so homoscedasticity is unlikely; residual variance is probably input-dependent. In higher-dimensional functions (7 and 8), the sample size is very small relative to the number of features, so multicollinearity and unstable coefficients are probable.

5) Regions that might be roughly linear or support a decision boundary
There are hints of roughly linear behaviour in some low-dimensional regions (e.g., Function 2 appears to increase with x1 in the current data), so a linear model can be directionally useful even if imperfect. A logistic regression classifier could be applied by thresholding outputs (e.g., "top 20%" vs "not top"), but the decision boundary would likely be noisy and curved; logistic would struggle to capture multi-modal or interaction-driven structure without engineered features.

6) Interpretability and feature effects
Yes, but cautiously. I used simple per-dimension plots and correlations (from the Round 1 diagnostics) to sanity-check which features might matter, then let the GP + EI selection drive the final choice. The interpretability was a guide, not the decision rule.
