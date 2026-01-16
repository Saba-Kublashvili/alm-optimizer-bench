# ALM Optimizer Bench
**KKT-guided Augmented Lagrangian constrained training on top of AdamW (image classification).**

This repository contains a **from-scratch** implementation of a practical **Augmented Lagrangian Method (ALM)** controller integrated into modern deep-learning training (PyTorch + AMP). The goal is to translate constrained optimization theory (**KKT / ALM**) into a **plug-and-play training mechanism** that enforces a **global constraint** during deep network optimization while remaining competitive with strong baselines.

**Source of truth:** `notebooks/alm_experiments.ipynb`  
**Exported code (auto-generated):** `src/alm_experiments.py`  
**Paper-ready artifacts:** `results/tables/*`, `results/figures/*`

---

## What this is (in one paragraph)
We study empirical risk minimization with an additional inequality constraint (a “budget” over a global statistic). We implement a **stochastic ALM wrapper** that:
1) adds ALM penalty/dual terms to the primal gradient (optimized with AdamW),  
2) updates the dual variable by projected ascent,  
3) adapts the penalty weight with bounded growth/shrink for stability under stochastic gradients.  

This is an **early-stage research system**, but it already demonstrates **stable constrained training** and **baseline-competitive accuracy** on CIFAR-10/ResNet-18 under paired-seed evaluation, with a clean research pipeline and paper-ready artifacts committed.

---

## 1) Constrained learning formulation
We consider:
$$
\min_{\theta \in \mathbb{R}^d} f(\theta)
\quad \text{s.t.} \quad g(\theta) \le 0,
$$
where:
- $f(\theta)$ is the standard training objective (cross-entropy + weight decay),
- $g(\theta)$ is a scalar constraint measuring a global model/training statistic.

### Budget instantiation used in this benchmark
We use a log-budget surrogate:
$$
g(\theta) = \log K(\theta) - B,
$$
where $B$ is a chosen budget (in log-space). The implementation logs:
- `logK` (the statistic),
- `B` (the target budget),
- `g_ema` (EMA-smoothed violation),
- controller state (`lam`, `rho`).

> The ALM machinery is agnostic to the specific definition of $K(\theta)$ as long as it is differentiable / autograd-friendly.

---

## 2) KKT connection and ALM objective
For inequality constraints, the KKT conditions (informally) require:
- primal feasibility: $g(\theta^\star)\le 0$,
- dual feasibility: $\lambda^\star \ge 0$,
- complementary slackness: $\lambda^\star g(\theta^\star)=0$,
- stationarity: $\nabla f(\theta^\star) + \lambda^\star \nabla g(\theta^\star)=0$.

We implement an augmented Lagrangian:
$$
\mathcal{L}_A(\theta,\lambda,\rho)
= f(\theta) + \lambda\,g(\theta) + \frac{\rho}{2}\,[g(\theta)]_+^2,
\quad \lambda\ge 0,\ \rho>0,
$$
where $[x]_+ = \max(x,0)$.

### Practical stochastic updates (what the code does)
**Primal step (AdamW on augmented gradient):**
$$
\theta_{t+1} = \mathrm{AdamW}\!\left(\theta_t,\;
\nabla f(\theta_t)
+ \lambda_t \nabla g(\theta_t)
+ \rho_t [g(\theta_t)]_+\, \nabla g(\theta_t)
\right).
$$

**Dual ascent (projected):**
$$
\lambda_{t+1} =
\Pi_{[0,\lambda_{\max}]}\!\left(\lambda_t + \alpha\, \tilde g_t\right),
\qquad
\tilde g_t = \beta \tilde g_{t-1} + (1-\beta) g(\theta_t).
$$

**Penalty control (bounded):**
$$
\rho \leftarrow \mathrm{clip}(\rho,\rho_{\min},\rho_{\max}),
$$
with an adaptive schedule that increases penalty under persistent violation and relaxes under sustained feasibility (details in the notebook).

---

## 3) Controller architecture (diagram)
```mermaid
flowchart LR
  A[Minibatch loss f(theta)] --> G[Compute grad_f]
  B[Constraint statistic logK(theta)] --> V[Violation g(theta)=logK(theta)-B]
  V --> E[EMA violation g_ema]
  E --> L[Dual update: lambda <- proj(lambda + alpha*g_ema)]
  E --> R[Penalty update: rho schedule + clipping]
  L --> P[Augmented term: lambda*grad_g]
  R --> Q[Penalty term: rho*[g]_+*grad_g]
  G --> S[Total grad = grad_f + lambda*grad_g + rho*[g]_+*grad_g]
  P --> S
  Q --> S
  S --> U[AdamW step]
  U --> B
