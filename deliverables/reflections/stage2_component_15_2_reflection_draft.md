# Stage 2 Component 15.2 Reflection Draft (Section A)

In this round, I focused on how neural-network hyperparameters change surrogate quality for the BBO task, then linked those lessons to my next capstone decisions.

## 1) Hyperparameter effects (convergence, stability, performance)
The main neural-network hyperparameters I used/observed were:
- Hidden-layer size (`hidden_layer_sizes`)
- Solver (`lbfgs` vs `adam`)
- Regularization strength (`alpha`)
- Learning rate (`learning_rate_init`, for `adam`)
- Training budget controls (`max_iter`, early stopping)

I ran a small 3-fold CV experiment on Functions 2, 5, and 8 with multiple MLP settings.

What I observed:
- **Function 5 (strongly non-linear, high dynamic range):**
  - `lbfgs`, hidden `(16,8)`, `alpha=1e-3` gave mean CV R2 about **0.963** (stable).
  - Similar larger model with stronger regularization (`(32,16)`, `alpha=1e-1`) was also strong at **0.964**.
  - `adam` settings were much less stable here (e.g., mean CV R2 around **-0.691** for `adam_mid_reg`).
- **Function 8 (8D):**
  - `lbfgs` improved clearly with more capacity/regularization: about **0.525** (`(8,4)`) to **0.837** (`(32,16)`, `alpha=1e-1`).
- **Function 2 (harder/low-signal regime):**
  - Most `lbfgs` settings remained negative R2.
  - `adam` with faster learning rate (`1e-2`) was the only setup with a positive mean CV R2 (about **0.113**), suggesting optimizer/step-size mattered more than raw model size there.

Takeaway: hyperparameters do not have one global “best” setting. They change by function landscape, noise, and dimensionality. Solver choice and regularization had the largest effect in my tests.

## 2) Discrete vs continuous hyperparameters
**Discrete hyperparameters (categorical/integer choices):**
- Number of layers
- Neurons per layer
- Solver type (`lbfgs`, `adam`)
- Activation function
- Early stopping on/off

**Continuous hyperparameters:**
- `alpha` (L2 regularization)
- `learning_rate_init`
- Momentum/beta-style optimizer terms (when used)

Why type matters for tuning:
- Discrete variables are better handled by categorical search (grid/random search, tree-based Bayesian search, bandit-style methods).
- Continuous variables benefit from log-scale sampling and smooth surrogate-based optimization.
- Mixed spaces (my case) are best handled by hybrid Bayesian optimization or staged tuning (coarse discrete choice first, then continuous fine-tuning).

## 3) Application to the capstone
If I use neural networks as surrogates in upcoming BBO rounds, these insights change my decisions in three ways:

1. **Function-specific defaults:**
   I will not force one MLP setup across all functions. I will start from robust defaults (`lbfgs`, medium width, moderate regularization), then adapt when diagnostics indicate mismatch.

2. **Stability-first selection:**
   I will compare candidate hyperparameter settings with cross-validation variance, not just mean score. If variance is high or gradients are erratic, I will prefer more regularized settings.

3. **Use BBO to tune the NN itself:**
   Yes, I can apply the same BBO logic to neural hyperparameters directly. Objective could be surrogate validation R2 (or prediction error) under a fixed compute budget, with variables like layer width, `alpha`, solver, and learning rate. This turns surrogate tuning into its own black-box optimization loop and should improve downstream query quality.

Overall, this component made me more deliberate: good neural performance in BBO is not only about model flexibility, but about controlled hyperparameter choices that balance fit quality, stability, and generalization.

---

## Optional short self-reply draft
After this reflection, my next improvement is to run a small automatic hyperparameter search per function before generating each weekly query. I expect this to reduce unstable surrogate behavior in harder functions (especially lower-signal cases) while keeping exploration decisions evidence-based.
