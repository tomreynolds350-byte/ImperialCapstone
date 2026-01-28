# Function Descriptions Directive

Goal
- Capture the official function descriptions, dimensions, and initial sample sizes for all eight black-box functions.
- Provide a stable reference when choosing strategies and writing reflections.

When to Use
- When verifying dimensionality, initial data size, or scenario context for any function.
- When drafting reflections that reference the real-world analogy.

Inputs
- Official function descriptions (screenshots in initial_data).

Outputs
- Function-by-function reference: input shape, output shape, optimization goal, and scenario notes.

Context
- Mini-lesson 12.8 introduces the eight synthetic black-box functions and their real-world analogies.
- Initial data is in the initial_data folder as .npy files; use np.load() to examine them.
- All tasks are framed as maximisation; natural minimisation objectives are transformed (for example, negation) so higher is better.

Terminology
- "1D" and "1D array" both mean a one-dimensional vector (shape (n, )).
- "2D/3D/4D/..." in this document refers to input dimensionality d, with inputs stored as 2D arrays of shape (n, d), not higher-order tensors.

Function Reference

Function 1
- Input: 2D array (10, 2)
- Output: 1D array (10, )
- Optimisation goal: Maximise
- Scenario: Detect likely contamination sources in a two-dimensional area, such as a radiation field, where only proximity yields a non-zero reading. The system uses Bayesian optimisation to tune detection parameters and reliably identify both strong and weak sources.

Function 2
- Input: 2D array (10, 2)
- Output: 1D array (10, )
- Optimisation goal: Maximise
- Scenario: A black-box ML model takes two numbers as input and returns a log-likelihood score. The goal is to maximise the score under noisy outputs and local optima. Bayesian optimisation is suggested to balance exploration and exploitation.

Function 3
- Input: 3D array (15, 3)
- Output: 1D array (15, )
- Optimisation goal: Maximise (via transformed output)
- Scenario: Drug discovery with three compounds. Each experiment is stored in initial_inputs.npy as a 3D array; adverse reactions are stored in initial_outputs.npy as a 1D array. The true objective is to minimise side effects, reframed as maximising a transformed output (for example, the negative of side effects).

Function 4
- Input: 4D array (30, 4)
- Output: 1D array (30, )
- Optimisation goal: Maximise
- Scenario: Place products across warehouses. Accurate calculations are expensive and infrequent, so an ML model approximates results. Four hyperparameters are tuned; the output reflects difference from an expensive baseline. The system is dynamic with local optima, requiring careful tuning and validation.

Function 5
- Input: 4D array (20, 4)
- Output: 1D array (20, )
- Optimisation goal: Maximise
- Scenario: Optimise a four-variable black-box function representing chemical process yield. The function is typically unimodal with a single peak. The goal is to find the optimal input combination using systematic exploration and optimisation.

Function 6
- Input: 5D array (20, 5)
- Output: 1D array (20, )
- Optimisation goal: Maximise
- Scenario: Optimise a cake recipe with five ingredient inputs (for example flour, sugar, eggs, butter, milk). Each recipe is evaluated by a combined score based on flavour, consistency, calories, waste, and cost. Scores are negative by design. Maximise by moving the score toward zero, or equivalently maximising the negative of the total sum.

Function 7
- Input: 6D array (30, 6)
- Output: 1D array (30, )
- Optimisation goal: Maximise
- Scenario: Tune six hyperparameters of an ML model (for example learning rate, regularisation strength, number of hidden layers). The performance score (such as accuracy or F1) is a black-box function. Literature can inform the initial search space. The goal is the highest performance.

Function 8
- Input: 8D array (40, 8)
- Output: 1D array (40, )
- Optimisation goal: Maximise
- Scenario: Optimise an eight-dimensional black-box function with unknown internal mechanics. The objective is to find a parameter combination that maximises output, such as performance or validation accuracy. Global optimisation is hard; strong local maxima are practical targets. Example hyperparameters include learning rate, batch size, number of layers, dropout rate, regularisation strength, activation function (numerically encoded), optimiser type (encoded), and initial weight range.

Change History

| Version | Date | Changes |
| --- | --- | --- |
| 1.0.0 | January 28, 2026 | Initial directive created from fn-scrn-1.jpg and fn-scrn-2.jpg |
| 1.0.1 | January 28, 2026 | Added mini-lesson context and terminology clarifications |

Last Updated: January 28, 2026
Status: Active
