# Stage 2 Component 17.2 Reflection

My current BBO strategy is technically grounded in Bayesian optimization for low-budget, expensive black-box settings. I combine a GP surrogate (uncertainty-aware global guidance), an MLP surrogate (non-linear local structure), and boundary classifiers (logistic/SVM) to improve robustness when function behavior differs across dimensions.

The strongest justification comes from established BO literature: model uncertainty explicitly, then optimize an acquisition function that balances exploration and exploitation. I operationalize this through UCB/EI switching plus novelty constraints to avoid pathological local re-sampling.

The papers that most directly influenced my design are: Snoek et al. (2012), Shahriari et al. (2016), Bergstra et al. (2011), Eriksson et al. (2019, TuRBO), Cowen-Rivers et al. (2022, HEBO), and Sazanovich et al. (2021, NeurIPS competition paper on learned partitioning for local BO). These sources collectively support my choice of uncertainty-driven search, local refinement under low budgets, and practical robustness over single-model purity.

My implementation stack is NumPy + SciPy + scikit-learn. This was the right choice for this stage because it offers strong baselines, high iteration speed, and clear reproducibility for weekly round cycles. PyTorch/TensorFlow remain useful alternatives for deeper custom surrogates, but they were not first priority under my current sample-budget and traceability constraints.

In GitHub, I present justifications at multiple levels: a concise top-level README, architecture and reproducibility docs, and round-level artifacts/work logs that tie design claims to concrete evidence. This makes reasoning visible to peers/facilitators and professionally interpretable for employers.

Next, I plan to add stronger benchmark-backed ablations using mixed-variable BO frameworks and further trust-region/decomposition methods, plus adaptive policy switching criteria learned from optimizer portfolio research.
