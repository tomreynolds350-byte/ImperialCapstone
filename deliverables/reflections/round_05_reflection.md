# Round 05 Reflection (Stage 2 - Component 16.1, Section A)

In this round, I kept an exploration-heavy policy while using advanced neural-network ideas from Module 16 to structure my decisions more deliberately.

## 1) Hierarchical feature learning and strategy refinement
Hierarchical feature learning changed how I think about search structure. Instead of assuming all coordinates matter equally at every step, I treated the process as layered: first identify broad promising regions (coarse structure), then refine local directions using surrogate gradients (fine structure). In practice, this meant using global candidate pools for coverage, then local gradient-based nudges from the MLP only after a candidate already passed novelty and uncertainty checks.

## 2) AlexNet/ImageNet breakthroughs vs incremental capstone gains
AlexNet/ImageNet showed that major jumps often come from better architecture plus scale, not just tiny parameter tweaks. In my capstone work, I see a parallel at smaller scale: each weekly submission is incremental, but occasional jumps came when I changed the strategy stack itself (for example, moving from simple heuristics to GP+NN+boundary models). So the lesson is that stable small gains matter, but step-changes come from better system design.

## 3) Trade-offs: depth/complexity/efficiency vs explore/exploit
I faced the same trade-off profile as neural training: more model complexity can capture richer patterns but increases instability and compute. Exploration vs exploitation had the same tension. Over-exploitation gave short-term score gains in a few functions but risked missing other basins; pure exploration improved coverage but hurt immediate weekly outputs. For Round 5 I intentionally stayed exploration-leaning (high-UCB, novelty floors) because Stage 2 still benefits from reducing structural uncertainty.

## 4) Neural-network building blocks and what changed in my thinking
The most useful building-block analogy was:
- Inputs: treat candidate design as representation quality, not just random numbers.
- Activations: non-linear surrogates can model interactions linear baselines miss.
- Loss: optimize not only predicted value, but a composite objective (value + uncertainty + novelty).
- Gradients: use local gradients for directional refinement, not as the sole decision rule.
- Weight updates: each new round is effectively a model update step with new supervision.

This helped me treat the capstone as an iterative learning system rather than a one-shot optimizer.

## 5) Framework lens: prototyping flexibility vs production structure
My current approach is closer to rapid prototyping with guardrails. It is flexible (easy to adjust acquisition weights, novelty floors, and candidate mix), but still has reproducible outputs and consistent artifacts. I am not yet at a fully production-ready architecture because the policy still changes round-to-round based on diagnostics, but the pipeline is becoming more structured each week.

## 6) Real-world deep learning use cases and benchmarking success
Giovanni Liotta's sport examples reinforce that success is not only one scalar metric. In real systems, you also track robustness, adaptation speed, and decision quality under uncertainty. I applied that here by evaluating both outcome and process: immediate score changes, diversity of sampled regions, and how much uncertainty was reduced for later rounds. This gives a better benchmark than short-term score alone, especially in black-box settings where delayed gains are common.

Overall, Module 16 pushed me to think in systems terms: architecture choices, learning dynamics, and evaluation criteria all matter. That perspective supports my current plan to stay exploration-forward while tightening model reliability as data volume grows.
