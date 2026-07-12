# armsmith demo — voiceover script

Video: `video/armsmith-demo.mp4` (2:34, 1080p30, rendered from `video/remotion/`).
Record each block as its own take (any order, re-take freely). Speak a beat
slower than feels natural. I will place each block at its timestamp and mux,
same workflow as the flightrec video — you do NOT need to read against the
picture. Numbers below are from the committed run artifacts; if we re-render
from the full-workload run, I will update this file to match before you record.

| # | At | Scene | Line |
|---|------|----------|------|
| 1 | 0:00 | title | This is armsmith. An autonomous agent that optimizes LLM inference on AWS Graviton, using Arm's own tooling. |
| 2 | 0:08 | problem | Out of the box, most LLMs on Graviton ship at about twenty-two tokens a second. The same silicon can do forty-five. Closing that gap is expert work, so most developers never do it. |
| 3 | 0:20 | solution | armsmith runs that loop autonomously. A deterministic autotuner sweeps build flags, kernels, quantisation, threading and KV-cache. A Claude analyst reads Arm Performix counters, proposes the search order, and narrates every decision. The tuner decides. The model only advises. |
| 4 | 0:34 | baseline | Every run starts honest. The baseline is the portable build a developer would actually ship. And before discovery begins, an expert hand-tuned config is pinned as the hundred-percent mark, so the target can't be gamed afterwards. |
| 5 | 0:48 | prior | Step one. The analyst has no counter data yet, so it falls back on the datasheet: try KleidiAI first. |
| 6 | 1:01 | refute | Measured on real silicon, KleidiAI is slower on this core and this model. The prior was wrong. The tuner reverts it. Measurement outranks the datasheet. |
| 7 | 1:13 | pivot | Next lever: the native build. Decode more than doubles, confirmed across repeats with non-overlapping confidence intervals. Kept. |
| 8 | 1:27 | sweep | Then the tuner grinds through the rest of the space: threads, KV-cache types, KleidiAI again in new combinations. Sixteen candidates, and every keep and revert is logged with its evidence and its rationale. |
| 9 | 1:43 | quality | Every candidate also passes a quality guard. KL divergence against the baseline: effectively zero. Same model, double the speed. A config can never win by quietly breaking the model. |
| 10 | 1:55 | victory | The result: the autonomous run beat the expert config we pinned before it started, closing over a hundred percent of the naive-to-expert gap. |
| 11 | 2:10 | repro | And it's not a one-off. The winning config is saved as a recipe, and armsmith repro replays it on a fresh instance, with no LLM in the loop. If it's not replayable, it's not a result. |
| 12 | 2:24 | close | armsmith. Pip-installable, Apache-2 licensed, built on Arm Performix. The tuner decides, the analyst explains, and the recipe replays. |

Recording notes
- Mic close, quiet room, one block per take; leave ~1s of silence at each end.
- Save as one file or twelve; either works (`OneDrive/Documents/Sound Recordings/` as before).
- Blocks 6, 7 and 10 are the spine of the pitch; give those the most energy.
- Total speaking time ~2:05 against a 2:34 picture, so there is slack everywhere.
