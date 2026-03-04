# Black-Box Optimization Capstone (Stage 2)

## Project purpose
This project optimizes eight unknown black-box functions under a strict sequential feedback loop. Each round submits one candidate vector per function and receives one scalar output. The goal is to maximize each function while balancing exploration and exploitation with limited data.

## Current status
- Strategy has moved from exploration-heavy into controlled exploitation (Round 6): balanced mode with lower UCB pressure and selective EI where incumbents are strong.
- Modeling stack: Gaussian Process + MLP + logistic/SVM boundary models.
- Workflow is append-only and reproducible: ingest portal feedback, update data store, generate next-round candidates, export portal strings and diagnostics.

## Repository architecture (working repo)
- `initial_data/function_*/`: canonical append-only dataset (`initial_inputs.npy`, `initial_outputs.npy`).
- `execution/`: scripts for ingestion, candidate generation, diagnostics, and plotting.
- `deliverables/submissions/`: round artifacts (`round_XX_inputs.txt`, portal strings, debug JSON).
- `deliverables/notes/`: work logs with method and validation details.
- `deliverables/reflections/`: module reflections and discussion-board drafts.

## Key library choices and trade-offs
- `numpy` / `scipy`: numerical operations and acquisition-function math.
- `scikit-learn`: GP surrogate modeling, MLP regression, logistic/SVM classification.
- Why this stack: fast iteration, strong baselines for small-to-mid data, easy reproducibility.
- Trade-off: less flexible than full deep-learning frameworks for custom large-scale training loops, but better aligned with this capstone's iterative, low-data setting.

## Input-output contract
- Function dimensions: `f1=2D`, `f2=2D`, `f3=3D`, `f4=4D`, `f5=4D`, `f6=5D`, `f7=6D`, `f8=8D`.
- Candidate bounds used in this workflow: `[0.001, 0.98]`.
- Portal format: hyphen-separated decimals (6 d.p.), e.g. `0.123456-0.654321`.

## Reproducibility quickstart
1. Ensure latest round input/output batch files are available.
2. Run the round generator script with explicit paths and strategy settings.
3. Submit `round_XX_portal_strings.txt`.
4. Keep `round_XX_hybrid_debug.json` and work logs for traceability.

## Technical foundations
This project's BBO choices are grounded in BO literature and NeurIPS challenge solutions (HEBO, NVIDIA RAPIDS ensemble, JetBrains SPBOpt paper). See the dedicated documentation below.

Detailed docs:
- [Architecture](docs/ARCHITECTURE.md)
- [Reproducibility Guide](docs/REPRODUCIBILITY.md)
- [Technical Foundations](docs/TECHNICAL_FOUNDATIONS.md)
- [Stage 2 Component 16.2 Reflection](docs/stage2_component_16_2_reflection.md)
- [Stage 2 Component 17.2 Reflection](docs/stage2_component_17_2_reflection.md)
