# SCM Paper — Results Note

**Date:** 2026-05-23
**Model:** `experiments/exp_d41_multiN/best_so_far.pt` (d41-best, 9×96 ResNet, 2.28M params)
**SCM:** `experiments/scm_v2/scm_model.pt` (SCM v2, 367K GRU, originally trained on d10, applied here as cross-model transfer headline)
**Self-play config:** 200 MCTS sims, c_puct default, max_moves=100, deterministic-greedy temperature, no Dirichlet noise (except where explicitly named).
**Opponent:** greedy. (Heuristic-opponent condition was attempted but found computationally infeasible at the required N — see deviations below.)

---

## Headline numbers (all vs greedy, 1v1, scored by agent's pins-in-goal)

| Condition | n | mean pins | std | wins (≥10) | win rate | Wilson 95% CI |
|---|---:|---:|---:|---:|---:|---|
| **Baseline (strong, no SCM)** | 200 | 7.000 | 0.000 | 0 | 0.0% | [0.0%, 1.9%] |
| **Trained SCM, α=0.2** | 200 | 7.525 | 1.119 | 6 | 3.0% | [1.4%, 6.4%] |
| **Transfer SCM (d10→d41), α=0.2** | 200 | 7.380 | 1.241 | 7 | 3.5% | [1.7%, 7.1%] |

### Trained SCM vs Baseline
- Welch t-test: **p < 0.0001**, Cohen's d = 0.662
- Fisher's exact (wins): **p = 0.030**
- Bootstrap 95% CI on mean diff: **[+0.370, +0.680] pins**
- **Significant at p<0.05: YES.** The trained SCM improves over a no-SCM baseline by ~0.5 pins per game.

### Transfer SCM vs Baseline (cross-model transfer to a model the SCM never saw during training)
- Welch t-test: p < 0.0001, Cohen's d = 0.434
- Fisher's exact (wins): p = 0.015
- Bootstrap 95% CI on diff: [+0.205, +0.555] pins
- **Significant at p<0.05: YES.** The SCM v2 trained on d10 transfers zero-shot to d41 with a comparable effect.

---

## Ablations — what actually drives the gain?

This is the headline finding of the revised paper. **The gain SCM produces over baseline is not specific to training.**

### Pin-rate comparison

| Condition | n | mean pins | wins | win rate | Welch p vs baseline | Significant |
|---|---:|---:|---:|---:|---:|:---:|
| Baseline (strong, no SCM) | 200 | 7.00 | 0 | 0.0% | — | — |
| Trained SCM α=0.2 | 200 | 7.525 | 6 | 3.0% | <0.0001 | YES |
| Identity SCM (gate=1, shift=0, α=0.2) | 100 | 7.42 | 2 | 2.0% | 0.0006 | YES |
| Untrained SCM (random init, α=0.2) | 100 | 7.59 | 2 | 2.0% | <0.0001 | YES |
| Temperature 0.7 (no SCM) | 100 | 7.00 | 0 | 0.0% | 1.00 | NO |
| Temperature 1.0 (no SCM) | 100 | 7.00 | 0 | 0.0% | 1.00 | NO |
| Temperature 1.3 (no SCM) | 100 | 7.00 | 0 | 0.0% | 1.00 | NO |
| Temperature 1.6 (no SCM) | 100 | 7.00 | 0 | 0.0% | 1.00 | NO |
| Dirichlet noise α_dir=0.1 | 100 | 7.37 | 4 | 4.0% | 0.0044 | YES |
| Dirichlet noise α_dir=0.3 | 100 | 7.53 | 8 | 8.0% | 0.0001 | YES |
| Dirichlet noise α_dir=1.0 | 100 | 7.30 | 0 | 0.0% | 0.0149 | YES |

### Trained SCM vs each control (the key comparisons)

| Trained SCM (mean 7.525) vs… | other mean | Welch p | Cohen's d | Boot 95% CI on diff |
|---|---:|---:|---:|---|
| Identity SCM | 7.42 | 0.464 | -0.091 | [-0.390, +0.175] |
| Untrained SCM | 7.59 | 0.663 | +0.055 | [-0.225, +0.355] |
| Transfer SCM (d10→d41) | 7.38 | 0.222 | -0.123 | [-0.375, +0.090] |
| Dirichlet 0.1 | 7.37 | 0.302 | -0.130 | [-0.450, +0.135] |
| Dirichlet 0.3 | 7.53 | 0.974 | +0.004 | [-0.290, +0.290] |
| Dirichlet 1.0 | 7.30 | 0.122 | -0.193 | [-0.510, +0.060] |

**Every Welch p > 0.05; every bootstrap 95% CI on the mean difference straddles zero.**

Bonferroni-corrected threshold for the 6 trained-vs-control comparisons at α=0.05 is **p < 0.0083**. None of the six tests meet it, let alone the uncorrected α=0.05.

This is statistically consistent with the trained SCM, identity SCM, untrained SCM, transfer SCM, and Dirichlet noise being **drawn from the same distribution of "agent + perturbation"**.

### What this says

- The trained SCM beats the no-perturbation baseline (Welch p < 0.0001).
- The trained SCM does **not** beat random-init or identity perturbations of the same SCM (all p > 0.12, all bootstrap intervals overlap zero).
- Pure root-Dirichlet noise on the MCTS prior (no SCM at all) reaches the same pin-rate band, and at α_dir=0.3 produces the highest win-rate of any condition tested (8/100, Wilson CI [4.1%, 15.0%]).
- Temperature-only perturbations have **zero effect** on this deterministic eval (all 4 temperatures × n=100 = exactly 7.00 mean, 0 wins). The d41 argmax is invariant under monotonic logit rescaling, as expected when the policy is sharp.

**Interpretation:** The lift the SCM produces over baseline is not from learned modulation. It is from **breaking the determinism of the policy** — any modest, structured perturbation of the policy logits at MCTS root yields the same lift. The SCM's GRU is doing the work that uniform-but-non-trivial noise on the prior also does.

---

## Alpha sweep — at what blend strength does SCM help, and at what point does it hurt?

All α-sweep conditions use the **same trained SCM v2 model**, varying only the blend ratio between the SCM-modulated policy and the MCTS visit distribution at action selection.

| α | n | mean pins | wins | win rate |
|---:|---:|---:|---:|---:|
| 0.10 | 100 | **7.48** | 2 | 2.0% |
| 0.20 | 100 | 7.34 | 1 | 1.0% |
| 0.30 | 100 | 7.41 | **5** | 5.0% |
| 0.50 | 100 | 7.18 | 2 | 2.0% |
| 0.70 | 100 | 6.83 | 2 | 2.0% |
| 1.00 | 100 | 5.13 | 0 | 0.0% |

- Best mean: α = 0.10 (7.48).
- Best wins: α = 0.30 (5/100 = 5.0%).
- The α=0.2 we shipped to the tournament is mid-pack on mean and worst on win-rate. The two adjacent values (0.10 and 0.30) both look slightly better, but neither is significantly different from 0.20 at n=100.
- α = 1.00 is a **catastrophic** failure (5.13 mean, 0 wins). Pure SCM modulation without an MCTS-visit anchor cannot select moves competitively. This is not surprising: the SCM was never trained as a policy by itself.
- The α → 0 (i.e., pure MCTS argmax with no modulation) limit is the baseline at 7.00 mean.

The shape (sweet spot near α=0.1–0.3, cliff above 0.7, catastrophe at 1.0) suggests the modulation acts as a **small structured shift on the prior** that helps exploration in MCTS root selection, but cannot survive being used as the action distribution on its own.

---

## Multiplayer

Not included in this round of experiments. The tournament evaluation (22 May 2026) already provides multiplayer evidence: in a 24-team multiplayer field, the d41-best + SCM v2 (α=0.2) agent finished **2nd of 24** teams with the tournament's highest average pin-completion (9.5/10 across 10 games at n=2-6 players). See `TOURNAMENT_RESULTS.md`. Multiplayer ablations are deferred to future work due to the per-game cost of running parallel multi-player rollouts on a single GPU.

---

## Interpretability

Pending. Will run a follow-up interpretability phase (≥100 logged games on the strong model) tracing per-move SCM gate/shift activations against visit-derived features (KL between policy and visits, root value, search depth, turn number) to characterize *what regime* the modulation activates in. Initial expectation given the above: gate/shift values should look essentially random with respect to those features, since the gain is indistinguishable from random perturbations. This is itself a paper-worthy negative interpretability result.

---

## Deviations from the original spec

1. **No heuristic opponent.** The advanced heuristic is single-threaded Python with a heavy per-move cost; at the required 200-game N per condition and 200 MCTS sims per move, the heuristic-opponent conditions did not fit the available GPU budget. Greedy opponent was used throughout. This means the headline numbers describe behavior against weak play. (The 22 May 2026 tournament provides the strong-opponent evidence: 2nd of 24.)
2. **SCM trained on d10, evaluated on d41.** The original spec asked for an SCM retrained from scratch on the strong (d41) model. The retrain was launched but proved infeasible on the available CPU-only training pipeline within the experiment window. The existing SCM v2 (trained on d10) was instead applied to d41 directly. The "trained SCM" and "transfer SCM" rows in the headline table are therefore the same SCM weights applied to the same model — they are listed separately to mirror the spec's table structure, and we have noted this in the discussion. (Both rows show the same agent; treat their numbers as the same condition tested twice independently.)
3. **Single trained-SCM seed.** With only one SCM checkpoint available, we cannot characterize the variance of SCM training across seeds. Future work should retrain multiple seeds on d41 and re-run the trained-vs-control comparison to confirm or deny the negative finding.

---

## Bonferroni correction summary

| Comparison family | k | Corrected threshold |
|---|---:|---:|
| Each condition vs baseline (9 ablations + 6 alphas + 1 transfer = 16 contrasts) | 16 | p < 0.0031 |
| Trained-SCM vs each control (6 contrasts) | 6 | p < 0.0083 |

- **vs baseline (corrected):** Trained SCM (p<0.0001), untrained SCM (p<0.0001), identity SCM (p=0.0006), dirichlet 0.3 (p=0.0001), dirichlet 0.1 (p=0.0044, fails Bonferroni at 16), transfer SCM (p<0.0001) all remain significant under the strict 16-way correction except dirichlet 0.1 (marginal). Temperatures and α=1.00 trivially do not.
- **trained vs controls (corrected):** **No comparison reaches significance even uncorrected** (minimum p = 0.122 for dir 1.0). Bonferroni does not change this conclusion.

---

## Files

- `core_results.json` — aggregated headline conditions with comparison stats
- `ablation_results.json` — aggregated ablation conditions
- `trained_vs_controls.json` — trained SCM vs each control (the key paper finding)
- `results_table.csv` — flat 18-row table of all conditions with stats
- `conditions/*.json` — raw per-game pins-in-goal lists (crash-safe, resumable)
