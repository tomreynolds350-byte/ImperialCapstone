# Stage 2 Component 16.2 Reflection Draft (Section A)

Over the first five rounds, my repository has evolved from a simple submission folder into a more structured optimisation workflow. I now organise work into four main areas: `initial_data/` for the append-only dataset per function, `execution/` for scripts that ingest feedback and generate the next candidates, `deliverables/submissions/` for canonical portal artifacts, and `deliverables/reflections/` plus `deliverables/notes/` for technical decisions and weekly reasoning. This separation helped me avoid mixing model code with submission files and made it easier to trace exactly what was submitted in each round.

To improve clarity, navigability, and reproducibility further, I am standardising the workflow around explicit round artifacts and naming conventions. For example, each round now has a predictable set of outputs (`round_XX_inputs.txt`, `round_XX_portal_strings.txt`, `round_XX_*_debug.json`) and a corresponding work log. I also keep ingestion explicit by writing a canonical output snapshot for the latest completed round before generating the next one. This gives me a cleaner audit trail and makes it easier for a collaborator to reproduce the same decision path from the same inputs.

The central libraries in my approach are NumPy, SciPy, and scikit-learn. NumPy is used for data handling and vector operations across all functions. SciPy contributes probability tools used in acquisition logic. scikit-learn is the core modeling framework in this project, including Gaussian Process regression for uncertainty-aware search, MLP regressors for nonlinear surrogate structure, and logistic/SVM classifiers for boundary-aware candidate scoring.

These choices are appropriate for my current problem because the data volume is still relatively small and iterative. Gaussian Processes are useful in this stage because they provide both a mean estimate and uncertainty, which directly supports exploration/exploitation decisions (UCB and EI style behavior). MLP and SVM components help capture nonlinear patterns and region boundaries when GP behavior is unstable or oversmooth.

The trade-offs are practical. GP models become harder to tune as dimensionality and noise increase. Neural models add flexibility but are less interpretable and can become sensitive to hyperparameters with small sample sizes. scikit-learn gives fast iteration and strong baseline reliability, but it is not as flexible as full deep learning frameworks for large-scale custom training loops. I considered PyTorch and TensorFlow conceptually in this module, but for this capstone stage I prioritised rapid experimentation, reproducibility, and low overhead over deep custom model engineering.

My documentation currently explains the challenge objective, input/output contract, and strategy evolution, but earlier drafts were too centered on round-specific choices. To communicate effectively to external audiences (collaborators or employers), the repository now needs to foreground software architecture: where data lives, how scripts transform it, which artifacts are generated, and how to rerun the process safely. It also needs explicit statements about model limitations and why specific libraries were chosen.

My update plan for README and related docs is:

1. Keep a concise top-level README with project purpose, architecture map, and quickstart commands.
2. Add a dedicated architecture document describing data flow from portal feedback to candidate generation and final submission strings.
3. Add a reproducibility guide showing the exact commands and expected artifact files.
4. Keep a lightweight experiment log summary that tracks strategy changes by round.
5. Ensure reflections are linked as supporting context, while keeping the README focused on operational clarity.

I have started implementing this plan by updating the public-facing repository draft with architecture-focused documentation and clearer reproducibility guidance aligned with the round-5 workflow. The main goal is that someone new to the project can understand what the repository does, why the modelling stack was chosen, and how to regenerate the submission artifacts without guessing hidden steps.
