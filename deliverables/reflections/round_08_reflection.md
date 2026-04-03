# Round 08 Reflection (Stage 2 - Component 19.1, Section A)

In this round, I thought about the capstone workflow as an LLM-assisted reasoning loop: retrieve the right prior artifacts, issue a structured prompt, inspect the output, then tighten the instructions and rerun. That is similar in spirit to the autoresearch-style loop being discussed in the WhatsApp group, but I kept it small, explicit, and audit-friendly.

## 1) Prompt patterns used and why
I mostly used structured zero-shot prompts for statistics, candidate generation, and verification, then added a light few-shot element by reusing previous work logs and portal-string formats as examples of the expected output. Simplified prompts were faster, but they were more likely to miss naming conventions, mix up submitted-best versus all-time-best, or skip validation. Structured prompts with exact file paths, round numbers, and explicit output schemas produced more reliable results.

## 2) Decoding settings and trade-offs
For the LLM-assisted parts of the workflow, I chose low-diversity decoding: roughly `temperature 0.2`, `top-p 0.9`, `top-k 40`, and conservative `max_tokens` caps. For short formatting tasks, the token budget stayed tight; for reflection writing, it was higher but still bounded. The trade-off was deliberate: low temperature improved coherence, reproducibility, and compliance with strict output formats, while slightly broader diversity was only useful during brief brainstorming. In other words, I wanted the LLM to behave more like a careful analyst than a creative generator.

## 3) Token boundaries, unusual strings, and truncation
The riskiest strings were not natural language but artifacts such as hyphen-delimited portal strings and scientific notation in the batch files (for example `np.float64(...)` values near zero). Those are exactly the kinds of strings where token boundaries can cause formatting mistakes or misreading. I checked for this by re-reading the generated files after creation, comparing portal strings against saved JSON/debug artifacts, and keeping machine-readable outputs short and schema-like. I did not observe hard truncation this round, and I checked by keeping retrieval targeted and by measuring the final reflection length so it stayed comfortably under the word limit.

## 4) Limitations that became clearer at 17 data points
The main limitation was prompt overfitting to the most recent narrative. After Round 6, the story became "exploit harder," and Round 7 showed that this was too strong for some functions, especially f2. A second limitation was attention drift: if too much prior context is included, the model can latch onto details that are no longer decision-relevant. At 17 points, longer inputs also showed diminishing returns. The best results came from retrieving only the latest outputs, the current debug file, and the last work log instead of pasting the whole history.

## 5) Reducing hallucinations
The main anti-hallucination tools were tighter instructions, retrieval of specific prior artifacts, explicit output formats, and verification after generation. I also used guardrails inside the optimisation workflow itself, such as manually overriding the f5 proposal when the surrogate tried to leave a basin that the data had already supported twice. That is effectively the same principle as hallucination control in LLMs: do not let a plausible-looking output override strong retrieved evidence.

## 6) Scaling to larger datasets or more complex LLMs
For larger datasets, I would rely even more on retrieval instead of long prompts, keep separate prompts for planning versus formatting, and tune decoding settings by task. I would allow slightly higher diversity only for idea generation, then switch back to low-diversity decoding for canonical outputs. With stronger LLMs, I would also keep summaries of prior rounds rather than full transcripts so the model sees decision-relevant context rather than raw volume.

## 7) Practitioner mindset under uncertainty
These prompt and decoding choices helped me think like a practitioner because they forced the same trade-off I face in the black-box challenge itself: exploration is useful, but uncontrolled diversity creates risk; structure improves reliability, but too much context can distract the model. The practical goal is not maximum creativity. It is dependable decisions under incomplete information, limited budget, and real downstream consequences.
