# Prediction-before-measurement record (experiment 006, Part C)

Recorded 2026-07-18, before `results006c.jsonl` was produced.

Basis: the three (delta_bar, MC) points then available (chi_x = 16,
chi_w = 8) fit MC ~ 1.07/delta_bar + 3.5.  Predictions for the
verification runs:

| configuration | delta_bar (measured) | MC predicted |
|---|---|---|
| beta = 0.90, chi_x = 32 | 0.120 | ~12.4 |
| beta = 0.95, chi_x = 32 | 0.079 | ~17.0 |
| beta = 1.00, chi_x = 16 | 0.078 | trend value 17.2, predicted to fall well below trend (degenerate base dynamics) |

Outcome (results006c.jsonl): 45.5, 64.8, 14.1 respectively — the first
two predictions FAILED (3-4x too low), exposing the chi_x^2 dependence;
the third (trend break at beta = 1) held.  The failure led to the
two-resource law.

# Prediction record (experiment 006, Part C2)

Recorded 2026-07-18, before `results006d.jsonl` was produced.
Basis: MC ~ 0.060 * chi_x^2 at beta = 0.95.

| chi_x | 8 | 24 | 48 |
|---|---|---|---|
| MC predicted | 3.8 | 34.6 | 138 |

Outcome (results006d.jsonl): 0.80, 45.9, 91.3 — the quadratic law holds
only in the mid range; threshold breakdown at low chi_x and
noise-limited flattening at high chi_x.  This defined the three-regime
form of the two-resource law.
