# Round 07 Reflection (Stage 2 - Component 18.1, Section A)

In this round, I treated hyperparameter tuning as the main lever for refining the black-box optimisation policy rather than changing the model stack again.

## 1) Hyperparameters tuned and why
The main hyperparameters I tuned were the strategy profile, UCB `kappa`, the EI/UCB switch threshold (`z_best_threshold`), and the boundary margin. I prioritised these because they directly control explore/exploit behaviour. With 16 points now available, the biggest question was no longer "which model class should I add?" but "how strongly should I trust the regions that are already performing well?" Lowering `kappa` reduced unnecessary exploration, lowering the EI threshold allowed more functions to exploit clearer incumbents, and tightening the boundary margin let strong regions near the edge stay available.

## 2) How tuning changed the query strategy
Earlier rounds were broader and more uncertainty-heavy, with high-UCB pressure and more novelty weight. This round I moved to an exploit profile. That made the search more local around high-performing basins instead of repeatedly scanning the whole domain. Only functions 1 and 3 stayed UCB-led; the other six functions shifted to EI or close local refinement. So the strategy changed from "find promising regions" to "improve the regions that have now demonstrated repeat value."

## 3) Methods used and trade-offs
I used manual adjustment plus a small grid search over policy settings, while the underlying query engine remained Bayesian optimisation. The manual part helped because I already had strong evidence from Round 6 that exploitation was starting to pay off. The small grid search made the decision less subjective by comparing settings for `kappa`, `z_best_threshold`, and `boundary_margin` on a temporary 16-point dataset. The trade-off is that this is much cheaper than full grid search, but it can still miss a slightly better setting. I did not use Hyperband here because the capstone gives one expensive evaluation per function per round, so early stopping is less relevant than careful sequential choice.

## 4) Model limitations that tuning made clearer
Tuning exposed a few limitations more clearly. First, some functions still look non-stationary: functions 4 and 6 improved locally, but their all-time bests are still from Round 3, which suggests one global GP can still oversmooth or favour the wrong basin. Second, function 5 shows a diminishing-returns risk because the optimiser keeps returning to the same near-corner region. That is efficient now, but it could still miss another basin. Third, higher-dimensional cases remain sparse even at 16 points, so high confidence can still be misleading. Functions 1 and 3 especially remain weak-signal cases.

## 5) Future use on larger datasets or more complex models
For larger datasets or future ML projects, I would still use Bayesian optimisation for the highest-impact hyperparameters, but I would pair it with Hyperband or multi-fidelity methods so weak configurations can be stopped early. For more complex models, I would also look at local or trust-region BO methods such as TuRBO, because this module highlighted that strong NeurIPS approaches had to deal with non-stationarity and heteroscedasticity rather than assume one smooth global surface.

## 6) Professional ML/AI thinking under incomplete information
This black-box setup is useful preparation for real ML/AI practice because it forces evidence-based choices under uncertainty. I cannot inspect the true function directly, so I have to rely on diagnostics, uncertainty, backtesting, and budget-aware trade-offs. That is close to professional work: identify the most influential hyperparameters, tune them systematically, accept that the model is imperfect, and avoid confusing one local success with a globally reliable strategy.
