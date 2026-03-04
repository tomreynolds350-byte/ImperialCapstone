# Round 04 Reflection (Stage 2 - Component 15.1, Section A)

This reflection documents the final Round 4 strategy I actually submitted: an exploration-heavy policy.

## 1) Support-vector-like points and how they guided query selection
I reframed each function as a binary task (`good` vs `bad`) using a top-30% output threshold and trained an RBF SVM. I treated points near the SVM decision boundary as support-vector-like points because they are most informative for uncertain transitions between promising and non-promising regions.

I used these points to steer sampling toward under-explored boundary regions, not only around incumbent best points. This improved coverage and reduced over-commitment to one basin.

## 2) Neural-network/surrogate gradients and directions for better outputs
I trained an MLP regressor (backpropagation-based) as a non-linear surrogate and used local input gradients to refine candidate directions.

Even in exploration mode, gradients were useful for directional nudges after broad candidate generation. They helped identify which coordinates to move most strongly while keeping novelty constraints active.

## 3) Framing BBO as classification and trade-offs (misclassification vs exploration)
As a classification framing (`good` vs `bad`):
- Logistic regression gave stable, interpretable baseline boundaries.
- SVM gave flexible, margin-based boundaries for uncertain transitions.
- Neural models offered additional flexibility but higher tuning complexity.

Key trade-off:
- If I optimise classification confidence too hard, I risk exploitation and false negatives for unseen good regions.
- If I prioritise uncertainty and novelty, I may accept short-term misclassification to gain long-term search coverage.

For Round 4, I intentionally chose the second option (exploration-heavy).

## 4) Model choice and interpretability vs flexibility
I used a hybrid model stack rather than one model:
- GP for uncertainty-aware exploration (UCB).
- MLP for non-linear response structure and gradient analysis.
- Logistic/SVM for boundary interpretation.

This preserved interpretability where possible while keeping flexibility for non-linear behavior in higher dimensions.

## 5) Variables with steepest gradients and how this guided priorities
In the Round 4 surrogate diagnostics, the strongest local gradient dimensions were:
- f1: x2
- f2: x1
- f3: x3
- f4: x3
- f5: x3
- f6: x4
- f7: x4
- f8: x3

I used this to prioritise which coordinates to perturb more in local refinements, while still enforcing novelty floors to keep global exploration active.

## 6) Neural-network boundary approximation and role of backpropagation
The neural network approximated non-linear boundaries better than linear baselines in the more complex functions, while being less stable in low-signal cases.

Backpropagation helped by:
- Fitting non-linear response surfaces.
- Enabling gradient-based directional analysis for candidate refinement.

I cross-checked with SVM margin behavior when NN confidence appeared fragile.

## 7) Neural networks vs simpler models: was flexibility worth complexity?
In this round, the added flexibility was useful, but only in a controlled setup:
- Neural and kernel methods captured non-linear structure that linear/logistic models missed.
- Simpler models still served as important sanity checks.
- Exploration policy choices (UCB + novelty floor + wide candidate sampling) mattered at least as much as model class choice.

Conclusion: NN flexibility was worth it when combined with uncertainty and novelty controls, not as a stand-alone exploitation tool.
