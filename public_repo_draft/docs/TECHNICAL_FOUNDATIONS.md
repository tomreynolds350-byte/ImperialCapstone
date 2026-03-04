# Technical Foundations

## Objective
This document explains why the current black-box optimization (BBO) approach is technically justified, what evidence supports it, and how decisions are documented for reproducibility.

## Core approach and justification
Current strategy uses a hybrid BO stack per function:
- Gaussian Process surrogate for predictive mean + uncertainty.
- MLP surrogate for non-linear local structure.
- Logistic regression and RBF SVM for boundary-aware region scoring.
- Acquisition-driven point selection (UCB/EI) with novelty constraints.

Why this is justified:
- BO is designed for expensive black-box objectives with low sample budgets.
- Uncertainty-aware acquisition provides principled explore-exploit control.
- Hybrid surrogates improve robustness when one model class underfits or oversmooths.

## Research basis
Key references used to shape decisions:

1. Snoek, Larochelle, Adams (2012), *Practical Bayesian Optimization of Machine Learning Algorithms*.
2. Shahriari et al. (2016), *Taking the Human Out of the Loop: A Review of Bayesian Optimization*.
3. Bergstra et al. (2011), *Algorithms for Hyper-Parameter Optimization*.
4. Eriksson et al. (2019), *Scalable Global Optimization via Local Bayesian Optimization (TuRBO)*.
5. Cowen-Rivers et al. (2022), *HEBO: Pushing the Limits of Sample-Efficient Hyperparameter Optimisation*.
6. Sazanovich et al. (2021), *Solving Black-Box Optimization Challenge via Learning Search Space Partition for Local Bayesian Optimization*.
7. Liu, Tunguz, Titericz (2020), *GPU Accelerated Exhaustive Search for Optimal Ensemble of Black-Box Optimization Algorithms*.

Competition-grounded external references reviewed in this workspace:
- HEBO repository: https://github.com/huawei-noah/HEBO
- NVIDIA RAPIDS 2nd-place repository: https://github.com/daxiongshu/rapids-ai-BBO-2nd-place-solution
- JetBrains NeurIPS competition paper (local copy): `NeurIPS-solutions/JetBrains-paper.pdf`

## Library and framework choices
Chosen stack:
- `numpy` and `scipy` for numerics and acquisition math.
- `scikit-learn` for GP, MLP, logistic regression, SVM, CV, and parameter search.

Rationale:
- Fast weekly iteration with low engineering overhead.
- Reliable classical estimators in low-data settings.
- Strong reproducibility and easier audit trails than heavier custom deep-learning loops.

Alternative frameworks considered:
- PyTorch/TensorFlow are valuable for custom deep surrogates and large training regimes, but were not primary for this capstone stage because sample efficiency and traceability were the first-order constraints.

## How this is presented in the repo
- `README.md`: concise strategy and status summary.
- `docs/ARCHITECTURE.md`: end-to-end data/model/artifact flow.
- `docs/REPRODUCIBILITY.md`: exact commands and validation checks.
- Round artifacts under `deliverables/submissions/` and per-round logs under `deliverables/notes/` provide evidence-backed traceability.

## Next evidence to incorporate
- Additional trust-region/decomposition BO studies for higher-dimensional tasks.
- Mixed-variable benchmark suites for more rigorous ablations.
- Portfolio/ensemble optimizer research for adaptive per-function policy selection.

These will be integrated as explicit ablation comparisons and documented policy-change criteria.
