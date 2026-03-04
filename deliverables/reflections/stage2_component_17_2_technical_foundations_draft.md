# Stage 2 Component 17.2 - Technical Foundations (Draft)

## 1) Main technical justification for my current BBO approach
My current approach is a hybrid Bayesian optimization workflow designed for low-budget black-box settings: one Gaussian Process (GP) surrogate for uncertainty-aware search, one neural surrogate (MLP) for non-linear local structure, and lightweight boundary classifiers (logistic regression + RBF SVM) for region discrimination. The technical justification is that Bayesian optimization is specifically built for expensive objective evaluations and small data regimes, where uncertainty estimates are central to sample-efficient decisions. I use acquisition-driven search (UCB and EI) plus novelty constraints to control explore-exploit balance instead of relying on one heuristic.

This is consistent with established BO practice: model the response surface, quantify uncertainty, and choose points that trade off expected gain versus information gain.

## 2) Academic papers guiding my design
I used the following ideas from literature:

- Snoek, Larochelle, Adams (2012), *Practical Bayesian Optimization of Machine Learning Algorithms*: supports GP surrogate + Expected Improvement style acquisition for ML tuning.
- Shahriari et al. (2016), *Taking the Human Out of the Loop: A Review of Bayesian Optimization*: frames BO as a sequential decision process and clarifies acquisition trade-offs.
- Bergstra et al. (2011), *Algorithms for Hyper-Parameter Optimization*: motivates alternatives to pure GP and the need for robust behavior in higher-dimensional, mixed settings.
- Eriksson et al. (2019), *Scalable Global Optimization via Local Bayesian Optimization (TuRBO)*: reinforces trust-region/local BO logic when global smoothness assumptions are weak.
- Cowen-Rivers et al. (HEBO, JAIR 2022), *HEBO: Pushing the Limits of Sample-Efficient Hyperparameter Optimisation*: supports hybrid and practical BO design in mixed-variable tasks.
- Sazanovich et al. (PMLR 2021), *Solving Black-Box Optimization Challenge via Learning Search Space Partition for Local Bayesian Optimization*: shows competition-relevant value of local partitioning in low-budget rounds.

I also reviewed NeurIPS 2020 winning-solution repositories in this workspace (HEBO and NVIDIA RAPIDS 2nd-place) to validate practical patterns such as optimizer ensembling and robust acquisition behavior.

## 3) Third-party libraries/frameworks and why they were selected
Central stack:

- `numpy`: array operations and deterministic artifact handling.
- `scipy`: probabilistic functions used in acquisition computation.
- `scikit-learn`: GP, MLP, logistic regression, SVM, CV and search utilities.

Why this stack was right for my stage:

- Fast iteration for weekly rounds with limited data.
- Strong classical baselines with low implementation overhead.
- Reproducible APIs and simple integration into append-only round workflows.

Why not primarily PyTorch/TensorFlow here:

- They are excellent for custom deep models, but my immediate bottleneck is query efficiency under low sample budgets, not large-scale gradient training.
- For this capstone stage, scikit-learn gives better speed-to-insight and easier auditability.
- Planned evolution: begin evaluating PyTorch surrogate variants from Round 7 onward, once Round 6 outcomes are ingested and used as a stronger baseline checkpoint.

## 4) How I will document/present this in GitHub
I will make reasoning explicit at three levels:

1. Top-level README: concise model stack, current strategy mode, and links to technical foundations.
2. `docs/TECHNICAL_FOUNDATIONS.md`: paper-backed rationale, library trade-offs, and design decisions mapped to implementation.
3. Round artifacts + work logs: each round keeps canonical inputs/outputs/debug files and a decision log, so claims are traceable to evidence.

This structure helps peers/facilitators see both "what I ran" and "why I ran it," and helps employers evaluate methodological rigor and reproducibility.

## 5) Additional sources for ongoing refinement
Next sources I plan to use:

- More robust/high-dimensional BO methods (e.g., trust-region and decomposition variants beyond baseline TuRBO usage).
- Recent BO benchmarking frameworks for mixed-variable settings (for stronger ablation discipline).
- Ensemble and portfolio optimization research for adaptive optimizer selection across function types.
- Open-source implementations from NeurIPS BBO participants in this workspace as practical engineering references.

Planned upgrades from these sources include adaptive acquisition scheduling by function regime, clearer uncertainty calibration checks, and benchmarked policy switching criteria between exploration-heavy and exploitation-heavy phases.


