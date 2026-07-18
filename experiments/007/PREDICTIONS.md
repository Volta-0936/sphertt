# Prediction-before-measurement record (experiment 007)

Recorded 2026-07-18, before `results007.jsonl` was produced.  Basis: the
decomposition law delta^2 = (c(1-beta))^2 + delta_stock^2 fitted in
experiment 006, and the two-resource law.

**A (lifting the noise limit, N = 65536, chi_x = 48):**
- delta_bar(beta = 0.98) predicted 0.038-0.042 -> measured 0.047 (close)
- delta_bar(beta = 0.99) predicted 0.032-0.038 -> measured 0.043 (close)
- MC(beta = 0.98) predicted 138-184 if the chi^2 series is rejoined
  -> measured 111.3 (partial lift only; a third constraint at high beta)

**B (delta_bar vs tensor order d at beta = 0.99, chi_x = 48):**
- hypothesis delta ~ sqrt(d-1) predicted 0.049 / 0.054 / 0.061 at
  d = 10 / 12 / 15 -> measured 0.078 / 0.095 / 0.105 (grows faster;
  open problem)

**C (headline, d = 15, N > 1e9):**
- predicted: if delta_bar stays below ~0.06, MC >= 100 survives beyond
  the state wall -> measured delta_bar = 0.116 and MC = 73.7 (seed 0;
  30.4 / 41.4 on seeds 1 / 2): memory survives, at a lower level than
  the optimistic branch.
