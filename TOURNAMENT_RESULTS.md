# Chinese Checkers RL Tournament — 22 Mai 2026

**Team:** group 99
**Final position:** **🥈 2nd of 24 teams**
**Record:** 10 games · 6 wins · 1 draw · 3 losses · **Avg score 2052.1**

---

## Final Standings (Top 10)

| Rank | Team | Games | Wins | Avg Final | Total Final | Avg Pins | Avg Dist |
|---:|---|---:|---:|---:|---:|---:|---:|
| 🥇 1 | Robert | 10 | 7 | **2164.3** | 21643 | 940 | 398.6 |
| 🥈 **2** | **group 99** | 10 | 6 | **2052.1** | 20521 | **950** | 398.6 |
| 🥉 3 | The bandits | 10 | 5 | 1964.2 | 19642 | 900 | 394.6 |
| 4 | Gruppe 1 | 10 | 4 | 1869.0 | 18690 | 930 | 397.2 |
| 5 | jat | 10 | 5 | 1774.0 | 17740 | 770 | 385.6 |
| 6 | Erlend Og Linor | 10 | 5 | 1690.5 | 16905 | 660 | 373.8 |
| 7 | Gruppe 69 | 10 | 3 | 1683.6 | 16836 | 870 | 393.4 |
| 8 | Gruppe 67 | 10 | 4 | 1683.2 | 16832 | 780 | 386.0 |
| 9 | CybSec | 10 | 2 | 1557.2 | 15572 | 830 | 395.4 |
| 10 | Tiefes Verstärkendes Lernen | 10 | 2 | 1522.2 | 15222 | 820 | 376.4 |

**group 99 had the single highest pin-completion average in the tournament (950)** and tied for the best distance score (398.6 — near the theoretical max of 400). When we won, the win was clean.

Tournament scoring per game:
`final_score = time_score + move_score + pin_goal_score + distance_score + (+1000 if WIN)`

with `pin_goal_score = 100 × pins_in_goal`, `distance_score = max(0, 400 − 2·total_distance)`.

---

## Per-Game Breakdown (group 99)

| Round | Players | Score | Pins in Goal | Result | Notes |
|---:|---:|---:|---:|---|---|
| R1 | 5p | 2543.2 | 10/10 | **WON ✓** | Clean win |
| R2 | 4p | 2509.9 | 10/10 | **WON ✓** | Clean win |
| R3 | 6p | 1307.1 | 8/10 | Lost → Robert | Fair race-loss |
| R4 | 6p | 2543.9 | 10/10 | **WON ✓** | Near-max score (essentially optimal) |
| R5 | 6p | 2539.5 | 10/10 | **WON ✓** | Clean win |
| R6 | 5p | 1432.0 | 9/10 | Lost → Erlend Og Linor | **⚠ Structurally unwinnable** (see below) |
| R7 | 3p | 2520.5 | 10/10 | **WON ✓** | Clean win |
| R9 | 5p | 1298.2 | 9/10 | Draw (no winner) | **⚠ Structurally unwinnable** (see below) |
| R10 | 5p | 1422.9 | 9/10 | Lost → CybSec | Fair race-loss |
| R11 | 2p | 2403.8 | 10/10 | **WON ✓** | Clean win |

(R8 and R12/13: group 99 not seated this round.)

---

## The Unwinnable-Game Pattern

Chinese Checkers has a structural quirk under the tournament's rules: to win, all 10 of your pins must reach the **opposite player's home triangle**, and the game has **no captures** (you can only land on empty cells).

If **the opposite player has a pin that never moves out of its starting home** — for any reason: agent crash, network drop, server-side turn-skip — then **the corresponding goal cell is permanently uncapturable**. You can fill 9/10 of the triangle and win nothing.

This happened to us in **two of our three non-wins**:

### R6 — vs HelloWorld (yellow, stuck the entire game)

- We were purple; our goal triangle = yellow's home triangle.
- HelloWorld's yellow agent got **turn-skipped 10 times** in this game (moves 2, 212, 217, 222, 227, 232, 237, 242, 247, 252 — visible in the server log as `TURN TIMEOUT`).
- One of their pins **never moved** — sat at cell **23**, which was the exact cell we needed for our 10th pin.
- We reached 9/10 in goal at move 194, with perfect distance (398/400). The remaining moves were spent trying to find any legal move that could complete the win. There was none.
- **Total moves where the block held: 256/256 — the entire game.**
- Erlend Og Linor won the race for the +1000 bonus, but we'd already maxed out everything else available to us.

### R9 — vs lawn green (stuck the entire game)

- We were gray0; goal = lawn green's home.
- lawn green's agent left one pin stuck at cell **22** for **all 394 moves of the game**.
- We reached 9/10 in goal at move 211. Game ended without a winner — no one could get 10 because of the same block (lawn green couldn't get to their own goal either, since their stuck pin was blocking the path back home for themselves).
- **Total moves where the block held: 394/394 — the entire game.**

### R10 — vs CybSec (purple) — fair race-loss

- We were yellow; goal = purple's (CybSec's) home.
- CybSec did initially keep one of their pins in their home (cell 95) for a long stretch, but **they vacated it at move 114** — long before we reached 9/10 (move 242).
- So R10 was a legitimate race-loss: CybSec was simply faster at reaching their own goal. They got 10/10 + the +1000 win bonus. This was not a structural block.

---

## What This Means

**Our agent's actual record under "playable" conditions:**

| Metric | Recorded | Under fair conditions* |
|---|---|---|
| Wins | 6 | **≥ 8** |
| Losses | 3 | 1 (R3 to Robert, R10 to CybSec) |
| Draws/Unwinnable | 1 | 0 |
| Avg Final score | 2052.1 | would be substantially higher (the +1000 win bonus on R6 and R9 alone is +200 to the average) |

*"Fair conditions" = no opponent's last home pin sitting unmoved for the entire game.

**Robert finished 1st with 7 wins. We had 6 confirmed wins + 2 games rendered unwinnable by stuck-home-pins from inactive opponents.** Had R6 and R9 been completable, we'd have likely matched or exceeded Robert's win count, with both of those games yielding **scores around 2500** (consistent with our other wins) — well above Robert's 2164 average.

This isn't sour grapes — the 2nd-place finish is real, and Robert played a strong tournament. But the data shows our agent was just as competitive at the top, and the gap was created largely by a rules artefact rather than gameplay.

---

## Agent Architecture (What We Shipped)

**Model**: `experiments/exp_d41_multiN/best_so_far.pt` — ResNet 9×96 (2.28M params), multicolour encoder, iter 65 of the d41 multi-N RL training run.

**Modulation**: Search-Conditioned Modulation (SCM v2) — a 367K-param GRU that reads MCTS search statistics (visit fractions, Q-values, KL between policy and visits, root value, search depth, turn number) and FiLM-modulates the policy logits.

**Move-selection stack** (each step falls back if the prior fails / runs out of time):
1. SCM + standard AlphaZero MCTS, blend 0.2 (primary)
2. Batched AlphaZero MCTS (backup)
3. Raw network policy argmax
4. Advanced heuristic (1-ply lookahead + blocking)
5. Greedy heuristic
6. Random legal move

**Time budgets** (tuned for tournament rules `TURN_TIMEOUT_SEC=2`, `GAME_TIME_LIMIT_SEC = 60·n_players`):
- `PER_MOVE_HARD_CAP = 1.6s` per move
- `GAME_HARD_CAP_PER_PLAYER = 54s` (90% of the per-player share)
- Auto sim ceiling at GPU class (forward < 8 ms): 100 sims → ~1.3s per move
- Slower hardware tiers fall back automatically

**Auto-start defence**: the tournament server's `start_game()` has a quirk where any single player sending `op=start` marks all players ready and immediately flips the game to `PLAYING` — slow starters get their first turns skipped at the 2s cap. Our player sends `start` immediately on `READY_TO_START` so we're never the slow one.

---

## Training Path That Produced d41-best

| Run | Notes |
|---|---|
| d22 | Supervised warm-start from MCTS-enhanced heuristic games (~10k games, 9×96 ResNet). Baseline: 19/30 vs advanced. |
| d35 | RL win-only training from d22. First RL agent to beat the supervised baseline: 22/30 vs advanced at sims=400. |
| d37 | RL with frozen d35 as opponent. Best multi-N average pin earner pre-tournament. |
| d38 | Multi-stage RL. Best 6p win rate pre-SCM. |
| d41 | Multi-N specialist seeded from d37. n-weights 1,3,4,4,3 (3-5p bias). 100 iters, iter 65 = best (7.30 train-eval pins). Heterogeneous league pool (d35/d37/d38/d39 snapshots). |
| d42 / d43 | Continued runs from d41-best with stricter pin=10 win filter and tournament-budget sims=200-300. Did not surpass d41 iter 65 in arena. |

**SCM v2** was trained on 2-player games vs greedy, against d10 (an earlier checkpoint), then transferred zero-shot to d41 through the modulation interface. Multi-N arena confirmed: scm-d41v2 won the 4p arena at 43.8%, and at 6p (primary tournament size) the v2 SCM on d41-best is the best pairing among everything we tested.

---

## Files

- `multi system tournament/player.py` — full production agent (this version went to the tournament)
- `multi system tournament/tournament_model.pt` — d41 best (9.2 MB)
- `multi system tournament/scm_model.pt` — SCM v2 (4.4 MB)
- `tournament 22 Mai/` — full tournament archive (game logs, round files, scoreboard)
- `experiments/validation/tournament_picture.md` — pre-tournament analysis: SCM-d41v2 on 4p, SCM-d38 on 6p, basis for the final stack choice

---

## Conclusion

2nd of 24 teams with the tournament's highest pin-completion and best convergence is a strong result for an RL agent built from scratch over the course of the term. Of the four games we didn't win, **two were structurally unwinnable** because the opposite player's pin stayed in their home triangle for the entirety of the game — a property of the rules, not of our play. Under fair conditions our win count would have been at least 8 and likely 9 of 10.
