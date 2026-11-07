# Bacterial gene expression prediction

Solution to the **Datacamp 2026 – Bacterial Gene Expression** challenge (Université Paris-Saclay, data curated at the Institut Pasteur), hosted on [Codabench](https://www.codabench.org/competitions/18481/).

## The task

Predict how strongly a bacterial gene is expressed in a given RNA-seq experiment. For each (gene, run) pair the model outputs a mean count `mu` and a Negative-Binomial dispersion `sigma`. Submissions are ranked by the mean Negative-Binomial negative log-likelihood (lower is better).

- ~78 million gene × experiment observations
- 88 bacterial species, 243 reference genomes, ~19,500 RNA-seq runs
- Inputs: gene table (coordinates, length, homology), GenBank genomes, curated experimental conditions (medium, temperature, pH, oxygen, growth phase, perturbation)

The test set is split into four regions:

| Region | Gene seen in training | Run seen in training |
|---|---|---|
| A | yes | yes (never together) |
| B | yes | no (new experiment) |
| C | no (new gene) | yes |
| D | no | no |

The ranking averages A, B and C.

## Approach

On the log scale, with `z = log(y + 0.5) - offset` (offset = gene length ratio + library size factor):

```
z[g, r] ≈ m + a[g] + b[r] + interaction[g, r]
```

Each region gets its own predictor for the part it cannot observe:

- **A, unseen pair.** For each genome, kernel ridge regression predicts the held-out block of runs × genes from the observed cells of the same runs and genes (run-space and gene-space kernels), blended with a per-run ridge on gene representations (240 principal components). Both predictors are re-run on internal pseudo-blocks; their errors set the blend weight per run and feed the dispersion model.
- **B, new run.** The run's profile is a similarity-weighted average of the other runs of the same genome, with weights learned from run metadata (project, medium, growth phase, perturbation, ...).
- **C, new gene.** Gradient boosting predicts the gene effect from features extracted from the GenBank genomes: codon usage, GC content, promoter motifs, operon context, effects of known operon-mates and neighbouring genes, and cross-species target encodings of product words, gene names and UniRef clusters. The run profile adds the residuals of operon-mates, neighbours and same-name genes.
- **Calibration.** Mean bias and NB dispersion use a few coarse constants per region, with a smooth term in the predicted level and variance proxies from the steps above.

## Results

| Version | A | B | C | D | Challenge (A, B, C) |
|---|---|---|---|---|---|
| Baseline (organisers), local | 7.733 | 7.851 | 10.150 | 11.766 | 8.578 |
| v2, local | 7.071 | 7.420 | 7.450 | 7.929 | 7.313 |
| v3, local | 6.699 | 7.420 | 7.348 | 7.939 | 7.156 |
| v2, public leaderboard | 6.69 | 6.71 | 7.40 | 7.03 | 6.93 |
| **v3, public leaderboard** | **6.45** | **6.70** | **7.26** | **7.02** | **6.80** |

Local scores use the public eval split. The platform trains on train + eval, which explains why its scores are lower. The full log of experiments is in [SCORES.md](SCORES.md).

Training and inference on the full data (73.6M rows) take about 11 minutes on 8 vCPUs, within the 1-hour budget.

## Running it

The code follows the competition's [benchopt](https://benchopt.github.io/) benchmark contract: `submission.py` defines `class Solver(GeneCompetSolver)`.

```bash
git clone https://github.com/x-datascience-datacamp/codabench_virtual_bacteria
cd codabench_virtual_bacteria
pip install -r requirements.txt
benchopt prepare benchmark -d "BacterialExpression[mode=train]"
benchopt run benchmark -d "BacterialExpression[mode=train]" -s /path/to/submission.py
```

Dependencies: numpy, scipy, pandas, scikit-learn, PyTorch. The competition data is not included in this repository; it is downloaded by the benchmark.
