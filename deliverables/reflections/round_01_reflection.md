This is my first formal submission for the Stage 2 BBO challenge, and I am deliberately keeping the method simple and exploration-first. As a novice, the standard baseline that makes sense to me is: generate a broad spread of candidate points (random sampling plus Latin hypercube for coverage), then pick the one that is furthest away from existing data (a max-distance rule). The logic is straightforward: in week one, we simply do not know the function shape, so it is more valuable to expand the map of the space than to over-exploit a noisy local pattern.

Per-function principle (what guided each query):
- Function 1 (2D): exploration first, choose a point maximising distance from existing samples.
- Function 2 (2D, noisy by description): exploration first, because early exploitation is likely to be brittle.
- Function 3 (3D): exploration first with broad coverage, as the data is still sparse.
- Function 4 (4D): exploration first; higher dimension means I prioritise coverage over local tuning.
- Function 5 (4D, likely unimodal by description): still exploration-leaning in week one, but within a broad coverage pattern.
- Function 6 (5D): exploration first because the transformed objective can hide gradients early on.
- Function 7 (6D): strong exploration bias because the dimensionality is high relative to sample count.
- Function 8 (8D): strongest exploration bias; the space is under-sampled and I am building baseline coverage.

Most challenging functions and why:
- Functions 7 and 8 are the hardest because the dimensionality is high and the sample count is still tiny; any early surrogate is likely to be unstable.
- Function 2 is also challenging because it is explicitly noisy and can have multiple local optima.
Additional information that would help: explicit input bounds per dimension, any known noise level (repeat-evaluation variance), and confirmation of whether the functions are smooth or highly rugged.

How I will adjust in future rounds:
- Once a few more points are collected, I will introduce a light surrogate (e.g., a GP from scikit-optimize) to balance exploration with cautious exploitation.
- I will tighten exploration in dimensions that show weak influence in the plots and summaries, while still probing uncertain regions.
- I will keep a simple, explainable process so that any improvement can be clearly linked to the data we have observed.

I will post this reflection to the discussion board after submitting the queries and will engage with peers as requested.
