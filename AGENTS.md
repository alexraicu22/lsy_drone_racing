# Competition simulator protection

`scripts/sim.py` is upstream-owned. The competition anti-cheat check requires it
to match the original upstream version exactly.

- Do not modify, reformat, or instrument `scripts/sim.py` in future development.
- Put reproducible seed control, timing diagnostics, logging, rendering helpers,
  and other experiments in separate development scripts.
- Preserve this restriction when running formatters or automated code fixes.
