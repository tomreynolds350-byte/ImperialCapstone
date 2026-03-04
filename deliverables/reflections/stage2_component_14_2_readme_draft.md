# Stage 2 Component 14.2 README Draft (Section A)

## Section 1: Project overview
This capstone tackles a Black-Box Optimisation (BBO) challenge: I must optimise eight unknown functions when I cannot inspect their formulas or gradients. In each round, I submit one candidate input per function and receive a scalar output from the portal. The goal is to improve decisions iteratively using limited feedback.

The overall objective is to learn a practical optimisation workflow under uncertainty: propose, observe, update, and propose again. This is directly relevant to real-world ML because many production objectives are effectively black-box (e.g., model performance after expensive training runs, simulation-based engineering metrics, or policy outcomes with delayed feedback).

For my career, this project strengthens core data-science habits: building reproducible experiments, balancing exploration vs exploitation, documenting assumptions, and communicating evolving strategy clearly to technical and non-technical audiences.

## Section 2: Inputs and outputs
Each round requires one input vector for each function:
- Function 1: 2D
- Function 2: 2D
- Function 3: 3D
- Function 4: 4D
- Function 5: 4D
- Function 6: 5D
- Function 7: 6D
- Function 8: 8D

Submission format is a hyphen-separated string with fixed precision (six decimals), for example:
- `function_3: 0.371493-0.055539-0.554166`
- `function_8: 0.122449-0.358034-0.107950-0.477987-0.874061-0.820614-0.154373-0.913123`

In my workflow I constrain candidate values to a safe interior region `[0.001, 0.98]` to avoid extreme boundary behavior. The returned output is a single scalar value per function (performance signal), and the process is noisy, so repeated nearby points can still differ in score.

## Section 3: Challenge objectives
The challenge objective is to **maximise** each unknown function using a limited query budget (one query per function per round). Constraints include:
- Unknown function structure (no formula, no gradients)
- Sparse data in early rounds, especially in higher dimensions
- Noisy outputs
- Sequential feedback loop (cannot evaluate all possibilities upfront)

Given these constraints, success is not “guaranteed global optimum.” Success is a disciplined, evidence-based iterative process that improves results over time and justifies each round’s query choices.

## Section 4: Technical approach
My approach across the first three submissions evolved in stages:

1. Round 1 (exploration-first baseline)
- Random/LHS-style candidate generation with distance-based selection.
- Principle: map unexplored regions before committing to local exploitation.

2. Round 2 (model-guided strategy)
- Gaussian Process (GP, Matern kernel) surrogate per function.
- Randomised hyperparameter search for kernel settings.
- Acquisition: switch between UCB and EI using a simple outlier rule on current best vs median.
- Added soft boundary penalty and duplicate avoidance.

3. Round 3 (diversity-aware refinement)
- Continued GP + acquisition strategy.
- Added manual diversity correction where proposals became too concentrated (notably function 5) by selecting a farther, still high-potential candidate.

Exploration vs exploitation balance:
- Early rounds: heavier exploration (uncertainty is high).
- Later rounds: selective exploitation around strong incumbents, while preserving diversity constraints to avoid local traps.

Potential role of regressions and SVMs:
- Linear/logistic regression can provide interpretable directional signals but may underfit nonlinear response surfaces.
- SVMs can be useful as a helper model by classifying “high vs low performance” regions.
- Kernel SVMs could capture nonlinear decision boundaries, but I treat them as complementary filters, not full replacements for GP-based acquisition.

What makes this approach thoughtful:
- It is incremental, auditable, and intentionally adaptable.
- Every round leaves artifacts (inputs/outputs, plots, reflections) that explain not only what I submitted, but why.

---

## Short self-reply reflection (for the discussion thread)
After drafting this README, my biggest improvement area is formalising diversity constraints earlier so exploitation does not over-concentrate in one basin. My next iteration will keep GP-based guidance but add explicit per-function spread checks before finalising candidates, especially for higher-dimensional functions where local confidence can be misleading with limited data.
