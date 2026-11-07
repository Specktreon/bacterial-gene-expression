# Local eval scores (mode=train, mean NB-NLL; challenge = mean of A, B, C)

| version | A | B | C | D | ABC | fit+predict | notes |
|---|---|---|---|---|---|---|---|
| v0 baseline (repo solution) | 7.733 | 7.851 | 10.150 | 11.766 | 8.578 | - | reproduced on cached codes, same math |
| v1 additive + A block KRR/ridge + B metadata-kNN + C gene GBM, internal-holdout calibration | 7.263 | 7.639 | 7.524 | 8.022 | 7.475 | 745s | pseudo holdouts easier than real regions -> sigma overconfident |
| v1 (file) | 7.263 | 7.639 | 7.524 | 8.022 | 7.475 | - | saved as submission_v1.py (internal-holdout calibration) |
| v2 coarse constants fitted on eval (per region, smooth log-mu term), B seen-project shrink 0.58, C word encodings + strand-aware neighbours + neighbour/gene-name run profile | 7.071 | 7.420 | 7.450 | 7.929 | 7.313 | 552s benchopt (fit 394s) | benchopt run; constants in-sample. Honest 2-fold-by-genome CV: A 7.094, B 7.441, C 7.457, D 7.990, ABC 7.331 |
| v2 honest CV, same folds as v3 (rng over genome ids) | 7.096 | 7.480 | 7.458 | 7.966 | 7.345 | - | reference for v3 CV; differs from v2 row only by fold assignment |
| lab A: hole-aware block (local test-A holes kept out of R1; v2 zero-filled them) | 6.976 | - | - | - | - | blocks 159s | dev/exp_a4.py, A only, own 2-fold genome CV calibration, 0.5 blend (v2 same setup 7.094) |
| lab A: + internal-CV proxies, per-run adaptive blend | 6.788 | - | - | - | - | blocks 231s | dev/exp_a5.py + exp_a6.py, in-sample; transfers to server-like replays (mixed 6.51, random 6.25 when fit on local) |
| lab C: operon-mate gene features (op_te, op_n) + operon-mate run profile | - | - | 7.379 | - | - | gbm ~300s | dev/exp_c1.py/exp_c2.py, CV (v2 7.457) |
| lab B: separate seen/unseen-project calibration | - | 7.481 | - | - | - | - | dev/exp_b1.py, CV worse than global 7.440: dropped |
| lab C: + no-operon-mate dispersion proxy | - | - | 7.369 | - | - | - | dev/exp_c3.py, CV |
| lab A: krr_lam 0.1, ridge_lam 0.3, npc 240 (was 0.3/1.0/20) | - | - | - | - | - | - | dev/exp_a7*.py, rmse of linear blend: local 0.784 -> 0.746, replay mixed 0.606 -> ~0.57 |
| v3a (first build): hole-aware A + internal-CV proxies/per-run blend, C operon-mate features, B constants shipped | 6.754 | 7.420 | 7.362 | 7.939 | 7.179 | 382s benchopt (fit 195s) | benchopt run. Honest 2-fold-by-genome CV: A 6.759, B 7.480, C 7.379, D 7.982, ABC 7.206 (v2 same folds 7.345). A constants fitted on local + server-like replay (sim mixed: A 6.452) |
| v3 (final, submission_v3.py): v3a + A krr_lam 0.1 / ridge_lam 0.3 / npc 240, C no-operon-mate dispersion proxy | 6.699 | 7.420 | 7.348 | 7.939 | 7.156 | 386s benchopt (fit 194s); server scale 668s total (fit 198s, predict 457s, 4 threads) | benchopt run. Honest 2-fold-by-genome CV: A 6.703, B 7.480, C 7.369, D 7.982, ABC 7.184 (v2 same folds 7.345, -0.161). A constants fitted on local + server-like replay (replay mixed A 6.376; v2's server behaviour there 6.812) |
