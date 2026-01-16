# ALM Optimizer Bench
**Augmented Lagrangian constrained training vs. standard optimizers (image classification).**

This repository contains the full experimental code, logged results, and paper-ready artifacts for an **Augmented Lagrangian Method (ALM)** training approach that enforces a **global constraint** during deep network optimization and compares it against standard baselines (AdamW, SGD).

Source of truth: `notebooks/alm_experiments.ipynb`  
Exported code for review/search: `src/alm_experiments.py`

---
## 1. Problem Setting (Constrained Learning)

We consider empirical risk minimization with an additional constraint:

\[
\min_{\theta \in \mathbb{R}^d} \; f(\theta)
\quad \text{s.t.} \quad g(\theta) \le 0.
\]

- \(f(\theta)\): training objective (cross-entropy + weight decay).
- \(g(\theta)\): constraint measuring a global budget/statistic of the model.

In this benchmark we instantiate the constraint via a log-budget surrogate \(\log K(\theta)\) with target \(B\):

\[
g(\theta) = \log K(\theta) - B.
\]

---
## 2. ALM Updates and KKT Connection

The augmented Lagrangian is:

\[
\mathcal{L}_A(\theta,\lambda,\rho)
= f(\theta) + \lambda g(\theta) + \frac{\rho}{2} g(\theta)^2,
\]
with \(\lambda \ge 0\) and \(\rho>0\).

Classical KKT conditions for a (local) optimum \(\theta^\star\) include:
- Primal feasibility: \(g(\theta^\star)\le 0\)
- Dual feasibility: \(\lambda^\star \ge 0\)
- Complementary slackness: \(\lambda^\star g(\theta^\star)=0\)
- Stationarity:
\[
\nabla f(\theta^\star) + \lambda^\star \nabla g(\theta^\star)=0.
\]

Deep nets are stochastic and non-convex, so this repo uses a practical stochastic ALM variant:
- minibatch gradients for \(f\),
- periodic evaluation of \(g(\theta)\),
- bounded penalty growth and projected dual ascent for stability.

Primal step (implemented on top of AdamW):
\[
\theta_{t+1} \leftarrow \theta_t - \eta \, \widehat{\nabla_\theta \mathcal{L}_A(\theta_t,\lambda_t,\rho_t)}.
\]

Projected dual step:
\[
\lambda_{t+1} \leftarrow \max\{0,\; \lambda_t + \alpha\, g(\theta_t)\}.
\]

Penalty adaptation (bounded):
\[
\rho_{t+1} \leftarrow \min(\rho_{\max},\gamma\rho_t)
\quad \text{only under persistent violation.}
\]

---
## 3. Repository Contents

- Notebook (source of truth): `notebooks/alm_experiments.ipynb`
- Exported script (auto-generated): `src/alm_experiments.py`
- Results tables:
  - `results/tables/c10_r18_paired_seed_results.csv`
  - `results/tables/c10_r18_stats.txt`
  - `results/tables/c10_r18_stats_recomputed.txt`
- Figures:
  - `results/figures/c10_r18_test_acc_by_seed.png`
  - `results/figures/c10_r18_paired_diff.png`
- Utilities:
  - `scripts/make_plots.py`
  - `scripts/recompute_stats.py`

## 4. CIFAR-10 / ResNet-18 (paired-seed evaluation)

We report paired-seed evaluation for AdamW vs ALM-on-AdamW.  
See the raw table in `results/tables/c10_r18_paired_seed_results.csv`.

Summary statistics (paired bootstrap CI + paired permutation test):
- `results/tables/c10_r18_stats.txt` (original run output)
- `results/tables/c10_r18_stats_recomputed.txt` (recomputed from CSV)

Interpretation guideline:
- Overlapping CIs and a high paired permutation p-value indicate **no statistically significant improvement** under this specific recipe.
- The ALM method still demonstrates a clean, end-to-end constrained-training implementation and stable runs with tracked constraint diagnostics.

---
