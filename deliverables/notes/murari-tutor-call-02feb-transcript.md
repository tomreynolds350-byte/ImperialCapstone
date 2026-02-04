# Murari Tutor Call — 02 Feb 2026 — Transcript / Notes (dictated)

I didn’t capture an audio transcript; the call was short, so this is a dictated summary of the key points and advice.

## Round 02 strategy (weeks + optimisation approach)
- Keep going with mostly exploration (but guided) until about **week 5 or week 6**.
- Around **week 5/6**, start shifting hard into **exploitation** of the areas of interest you’ve uncovered for each function.
- For **Round 2**, run another iteration and keep trying to get **maximum outputs**.
- Exploration early on is important to avoid getting stuck at **local maxima** instead of the true maximum for each function.
- The leaderboard revealed at the end is based on the **final output in the final week** — the goal is the **highest number possible** as the output from **all 8 functions**.

## Notes on specific functions
- **Function 1** is not straightforward.
- My **post Round 1** output for Function 1 was **0.0**, so I have not gotten closer to the maximisation region yet.

## Tools / techniques to use
- Use a **Gaussian process (GP)** surrogate with **acquisition functions** to search for maxima for each function (Bayesian optimisation style).
- **NumPy** should be the primary method of data handling; **DataFrames** can also be useful.
- “Go big bang”: use whatever techniques you want in these initial weeks and keep exploring.
- Different Bayesian optimisation methods for different functions may be appropriate — try a variety.

## Reference repos Murari shared (for GP-based examples)
- `https://github.com/jdchen5/machinelearninglabs/blob/main/CCompetition/old_codes/function_1.ipynb`
- `https://github.com/jdchen5/machinelearninglabs`
