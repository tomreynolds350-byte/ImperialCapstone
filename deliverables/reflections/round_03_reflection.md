# Round 03 Reflection (Stage 2 - Part 2)

After the third iteration, my strategy is now clearly different from Round 1.

In Round 1 I used an exploration-first heuristic (distance from prior points). In Rounds 2 and 3 I shifted to a model-guided approach: a Gaussian Process (GP) surrogate with Matern kernel, uncertainty-aware acquisition, and light hyperparameter tuning through randomised search. I still use heuristics, but they now sit on top of model predictions instead of replacing them. For example, I use a rule to switch between UCB and EI depending on how "outlier-like" the current best output is, and I keep a boundary penalty to avoid collapsing to extreme corners.

I currently balance exploration and exploitation as a controlled mix, not a 50/50 split. UCB is my default for broad coverage because uncertainty is still high in many functions (especially 6D and 8D). EI is used more selectively where I observe a strong incumbent and want local improvement. Round 3 highlighted why this balance matters: function 5 kept proposing near the same high-value region, which is efficient for short-term exploitation but risks missing other strong basins. I therefore manually increased exploration for function 5 by selecting a farther candidate (while keeping it in bounds). This was a deliberate policy choice: preserve good model signal, but avoid over-committing to one neighborhood too early.

How would SVMs change the approach? I would not use SVM regression as my primary optimiser here, but soft-margin and kernel SVM classification can still be useful in a helper role. I can binarise outputs into "high-performing" vs "not high-performing" (for example, top quartile threshold) and train a soft-margin SVM to get a robust, noise-tolerant decision boundary. A kernel SVM (RBF) would help if the response surface is nonlinear and the high-performing region is curved or disconnected. In practice, this would be a region-filtering step: identify promising zones first, then apply GP-based acquisition within those zones. The limitation is that converting to classes throws away ranking detail among high values, so I would keep SVM as a complement, not a replacement.

As data grows, several model limitations become clearer. First, local overfitting risk appears when GP length scales shrink too much: the model can become overly confident around isolated highs and keep sampling nearby. Second, in higher dimensions the sample size remains small relative to space volume, so uncertainty estimates are fragile and acquisition may chase artifacts. Third, some dimensions likely have weak influence for certain functions, but with this data volume it is hard to separate truly irrelevant features from features that are simply under-sampled. These limitations appear in diagnostics as unstable best-point neighborhoods, acquisition clustering, and weakly consistent correlation/PCA narratives across rounds.

This black-box setup is very close to real data-science work under incomplete knowledge. In real projects we rarely know the true functional form, noise process, or whether observed patterns are stable. We have to make decisions with partial evidence, manage tradeoffs (explore new hypotheses vs exploit current wins), and update strategy when diagnostics contradict assumptions. The iterative submit-feedback-adjust loop here mirrors that exact process: propose, test, learn, recalibrate. It also reinforces reproducibility habits (artifact versioning, append-only data updates, explicit assumptions) that are transferable beyond optimization tasks.

Planned next adjustments:
- Keep GP + acquisition as core, but enforce a per-function diversity constraint when repeated proposals cluster too tightly.
- Add simple rank-based diagnostics (e.g., top-k stability across candidate pools) to reduce sensitivity to single noisy highs.
- Continue tracking interpretable summaries (dim-vs-y, PCA, correlation) as directional evidence only, not proof.

Completion note for this module:
- Query set prepared for portal submission (`round_03_portal_strings.txt`).
- Reflection prepared for discussion board posting.
- Next action is peer engagement with focused comments on strategy tradeoffs and model-risk handling.
