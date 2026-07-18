# Experiment scripts and logs

Reproducibility artifacts for the two sphertt papers.  Each directory
contains the exact scripts that were run, the JSONL result logs they
produced (fixed schema, seeds recorded), and the figures generated from
those logs only.  Scripts reference the library via a hard-coded
`sys.path` insert; replace it with `pip install sphertt` / your local
path to re-run.  All experiments run on a single plain CPU.

| dir | contents | backs |
|---|---|---|
| `005` | native TT-state engine: chi_x sweep at N=1024, N-scaling to 4^15, delta probes, beta rescue | Paper II Fig. 2 (fig7); Paper II Secs. 2-3 |
| `006` | exact-attractor compressibility + Schmidt spectra; decomposition-law grid; twin-trajectory dynamics; MC verification runs | Paper II Figs. 3-5 (fig10-12) |
| `007` | two-resource decisive test (noise-limit lift, delta vs tensor order, beyond-the-wall MC); trajectory participation-ratio check | Paper II Fig. 6 (fig13) |
| `008` | seed-robustness batch (decomposition grid, MC crossover, headline; seeds 1-2) | Paper II seed ranges |
| `009` | task validation: dense-state N sweep (delay-k, NARMA10) and TT-state at N > 1e9. `exp009_tasks2.py` / `results009b.jsonl` is the definitive protocol; `exp009_tasks.py` / `results009.jsonl` is the superseded first pass (kept for the record) | Paper I Table 1; Paper II Table 1 |
| `010` | self-contained re-validation of the amplification law (45 configs) | Paper I Fig. 5 (validation) |

Experiments 006 and 007 contain prediction-before-measurement records
(`PREDICTIONS.md` in each directory): the model predictions were written
down before the verification runs, and the records keep both the misses
and the hits.

The earlier baseline experiments of Paper I (its Figs. 1-4 and 6:
baselines, connectivity-rank sweep, truncation-error recursion, beta
trade-off, connectivity scaling) were run in an earlier research
environment; their protocols and full numbers are stated in the paper.
Everything specific to the truncation-error theory and the TT-state
engine is reproducible from this directory.
