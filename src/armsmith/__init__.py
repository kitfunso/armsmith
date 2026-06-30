"""armsmith - a reproducible Graviton LLM-optimization lab.

A deterministic autotuner explores llama.cpp inference configs on AWS Graviton;
an LLM analyst reads Arm Performix counters to prioritize the search and narrate
each keep/revert; the winning config replays from a saved recipe.
"""

__version__ = "0.0.1"
