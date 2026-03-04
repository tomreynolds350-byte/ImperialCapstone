# Round 06 Reflection (Stage 2 - Component 17.1, Section A)

In this round, I shifted from exploration-heavy behavior toward controlled exploitation, while using Module 17 CNN concepts to structure that decision more rigorously.

## 1) Progressive feature extraction and BBO refinement
The CNN idea of progressive feature extraction (edges to textures to objects) changed how I framed my optimization updates. Early rounds acted like low-level feature learning: map broad response patterns, identify useful dimensions, and reduce uncertainty. By Round 6, I treated the process more like higher-level feature composition: combine these earlier signals and focus on regions that repeatedly showed stronger outputs. Practically, I reduced exploration pressure and used more local refinement around proven regions, instead of re-scanning the full space as aggressively.

## 2) LeNet/CNN breakthroughs vs incremental capstone gains
LeNet and later CNNs were not one single random improvement; they came from architecture choices that compounded over time (convolution, pooling, weight sharing, better training practice). My capstone progress feels similar at smaller scale. Single rounds can look noisy, but the methodology has improved step by step: first simple heuristics, then GP-driven acquisition, then hybrid GP+MLP+boundary models, and now a deliberate exploit transition. The parallel is that real gains come from cumulative system design changes, not just one lucky query.

## 3) Trade-offs: depth/cost/overfitting vs explore/exploit
CNN training balances representational depth against compute budget and overfitting risk. I faced the same pattern in query strategy. Exploration gave wider coverage and reduced structural uncertainty, but often hurt immediate scores. Exploitation improved short-term expected value, but can overfit to a local basin if done too early. For Round 6 I moved to a balanced mode with lower UCB pressure and EI in high-confidence functions, which is analogous to reducing unnecessary model complexity once enough signal has been learned. The goal is better sample efficiency without collapsing diversity completely.

## 4) CNN building blocks that changed my optimization thinking
The most useful concepts were:
- Convolution: local pattern extraction maps to local surrogate refinement around promising regions.
- Pooling: dimensional summarization maps to compressing noisy candidate pools into robust shortlist candidates.
- Activation/non-linearity: non-linear surrogates (MLP/SVM) capture interactions linear trends miss.
- Loss function: instead of optimizing only predicted value, my practical objective is composite (value + uncertainty + novelty + boundary confidence).

This made me treat optimization as a learning pipeline with multiple transformations, not a single score-maximization step.

## 5) Andrea Dunbar interview and benchmarking choices
Andrea Dunbar's edge-AI discussion highlighted real deployment constraints: low latency, low power, memory limits, privacy-preserving local processing, and hierarchical decision pipelines (simple low-power filter first, deeper model only when needed). That directly informs my benchmarking. Success in this capstone should not be only "did this week score increase?" It should also include:
- efficiency: how quickly strong regions are found with limited queries,
- stability: whether improvements persist across rounds,
- robustness: avoiding brittle one-function gains that fail elsewhere,
- decision quality under constraints: choosing when to spend exploration budget versus exploit known regions.

So my Round 6 benchmark is multi-criteria: immediate output movement plus whether the strategy is becoming more reliable and resource-aware, similar to how a deployable CNN system is judged in practice.
