"""d34: Self-play training with replay buffer + warmstart reservoir.

Diagnosis of d27/d29/d33 stagnation (May 2026):
  1. d27/d29 train each iter on only that iter's ~900 samples and discard
     them. No replay buffer. Catastrophic forgetting.
  2. The 1.35M-sample warmstart_data.npz is never mixed back during
     self-play, so gradient descent overwrites the warm-start.
  3. Adam optimizer was reconstructed every iter, wiping moments.
  4. value_target_lambda=0.8 in d33 means 80% of the value target is the
     game outcome — but with max_moves=80 almost every game truncates to
     v_outcome ~ 0, so the value head gets no signal.
  5. d29 snapshot gate (line 313-322) skipped a snapshot whenever the iter
     wasn't an eval iter (avg_pins=None < gate). With snapshot_every=5 +
     eval_every=3, iters 5/10/20/25 never snapshot.
  6. With --heuristic-opponent advanced and league_fraction=1.0, "current"
     loses 15-16 of 16 league games per iter, so all ~900 samples are from
     losing trajectories. Without a buffer, the model is asked to imitate
     its own losing distribution every iter.
  7. Default temperature_moves=15 / temperature_low=0.3 made post-15 moves
     near-greedy. With Dirichlet only at root, exploration was very narrow.

This module fixes (1)-(7):

  - ReplayBuffer with main pool (40k circular) + reservoir from warmstart
    (20k samples, never evicted). Each training step samples a full batch
    from the buffer with main:reservoir ratio.
  - Adam optimizer is created once and persisted across iters; checkpointed
    in {output_dir}/optimizer.pt for resume.
  - value_target_lambda=0.5 by default (more weight on per-step MCTS
    value, less on flat truncated outcomes).
  - max_moves bumped to 120 (configurable).
  - Snapshot gate uses last_eval_pins (most recent eval) instead of None
    when the current iter has no eval.
  - Opponent mix: 30% advanced heuristic, 30% league snapshot, 40% true
    self-play. Configurable.
  - KL filter: drop samples where KL(MCTS || raw_policy) < threshold.
    Train only on positions where MCTS disagrees with the policy.
  - Optional: freeze BatchNorm running stats during fine-tune to stop
    drift on small batches.

Usage:
    bash scripts/launch_d34.sh
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "2"

import io
import sys
import time
import json
import math
import random
import argparse
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
try:
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
except Exception:
    pass

from src.network.alphazero_net import AlphaZeroNet, NetworkConfig
from src.training.alphazero_self_play import SelfPlayConfig
from src.training.parallel_self_play import run_iteration_parallel
from src.training.d27_league_train import (
    _save_snapshot, _list_snapshots, quick_eval_vs_greedy,
)


# --------------------------------------------------------------------------- #
# Replay buffer with reservoir
# --------------------------------------------------------------------------- #
class ReplayBuffer:
    """Two-pool replay buffer.

      - main: circular FIFO of recent self-play samples (max_size).
      - reservoir: fixed warmstart samples, never evicted. Sampled with
        probability `reservoir_ratio` per drawn item.

    Stored as 4 numpy arrays per pool for fast batch sampling. We keep the
    arrays pre-allocated where possible to avoid Python overhead.
    """

    def __init__(self, max_size: int, reservoir_ratio: float):
        self.max_size = int(max_size)
        self.reservoir_ratio = float(reservoir_ratio)

        # main pool — lazily grown to max_size, then circular
        self.main_obs = None
        self.main_mask = None
        self.main_pol = None
        self.main_val = None
        self.main_n = 0
        self.main_write_idx = 0

        # reservoir — fixed-size, set once
        self.res_obs = None
        self.res_mask = None
        self.res_pol = None
        self.res_val = None
        self.res_n = 0

    def seed_reservoir(self, npz_path: str, max_samples: int = 20000,
                       seed: int = 0):
        """Load up to max_samples random rows from a warmstart_data.npz file."""
        d = np.load(npz_path)
        n_total = d["obs"].shape[0]
        rng = np.random.default_rng(seed)
        if n_total > max_samples:
            idx = rng.choice(n_total, size=max_samples, replace=False)
            idx.sort()  # I/O-friendlier read pattern
        else:
            idx = np.arange(n_total)
        self.res_obs = d["obs"][idx].astype(np.float32, copy=False)
        self.res_mask = d["action_masks"][idx].astype(np.bool_, copy=False)
        self.res_pol = d["policies"][idx].astype(np.float32, copy=False)
        self.res_val = d["values"][idx].astype(np.float32, copy=False)
        self.res_n = self.res_obs.shape[0]
        d.close()
        print(f"  Reservoir: {self.res_n} warmstart samples loaded from "
              f"{npz_path}", flush=True)

    def add(self, obs: np.ndarray, mask: np.ndarray, pol: np.ndarray,
            val: np.ndarray):
        """Append a chunk of samples to the main pool (circular)."""
        n_new = obs.shape[0]
        if n_new == 0:
            return
        if self.main_obs is None:
            # Allocate once at max_size; reuse memory across iters.
            C, H, W = obs.shape[1:]
            A = mask.shape[1]
            self.main_obs = np.zeros((self.max_size, C, H, W), dtype=np.float32)
            self.main_mask = np.zeros((self.max_size, A), dtype=np.bool_)
            self.main_pol = np.zeros((self.max_size, A), dtype=np.float32)
            self.main_val = np.zeros((self.max_size,), dtype=np.float32)

        # Write n_new samples, possibly wrapping
        for i in range(n_new):
            j = self.main_write_idx
            self.main_obs[j] = obs[i]
            self.main_mask[j] = mask[i]
            self.main_pol[j] = pol[i]
            self.main_val[j] = val[i]
            self.main_write_idx = (self.main_write_idx + 1) % self.max_size
            if self.main_n < self.max_size:
                self.main_n += 1

    def sample_batch(self, batch_size: int, rng: np.random.Generator):
        """Sample a mixed batch. Returns (obs, mask, pol, val) numpy arrays."""
        if self.main_n == 0 and self.res_n == 0:
            raise RuntimeError("Replay buffer is empty")
        if self.main_n == 0:
            n_res = batch_size
            n_main = 0
        elif self.res_n == 0:
            n_res = 0
            n_main = batch_size
        else:
            n_res = int(round(batch_size * self.reservoir_ratio))
            n_main = batch_size - n_res

        parts_obs, parts_mask, parts_pol, parts_val = [], [], [], []
        if n_main > 0:
            idx_m = rng.integers(0, self.main_n, size=n_main)
            parts_obs.append(self.main_obs[idx_m])
            parts_mask.append(self.main_mask[idx_m])
            parts_pol.append(self.main_pol[idx_m])
            parts_val.append(self.main_val[idx_m])
        if n_res > 0:
            idx_r = rng.integers(0, self.res_n, size=n_res)
            parts_obs.append(self.res_obs[idx_r])
            parts_mask.append(self.res_mask[idx_r])
            parts_pol.append(self.res_pol[idx_r])
            parts_val.append(self.res_val[idx_r])

        obs = np.concatenate(parts_obs, axis=0)
        mask = np.concatenate(parts_mask, axis=0)
        pol = np.concatenate(parts_pol, axis=0)
        val = np.concatenate(parts_val, axis=0)
        # Shuffle so res/main aren't in fixed order inside each batch
        order = rng.permutation(obs.shape[0])
        return obs[order], mask[order], pol[order], val[order]

    def __len__(self):
        return self.main_n + self.res_n


# --------------------------------------------------------------------------- #
# KL filter
# --------------------------------------------------------------------------- #
def kl_filter(network: AlphaZeroNet, obs: np.ndarray, mask: np.ndarray,
              pol: np.ndarray, val: np.ndarray, threshold: float,
              batch_size: int = 256):
    """Keep only samples where KL(pol || raw_policy) >= threshold.

    raw_policy = the network's current masked-softmax. If MCTS visit
    distribution barely differs from raw_policy, training on that sample
    can't improve the policy. Threshold ~ 0.05 nats works in practice.
    Disabled when threshold <= 0.
    """
    if threshold <= 0 or obs.shape[0] == 0:
        return obs, mask, pol, val, obs.shape[0], obs.shape[0]
    n = obs.shape[0]
    keep = np.zeros(n, dtype=np.bool_)
    model = network.model
    device = network.device
    model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            o = torch.from_numpy(obs[start:end]).to(device)
            m = torch.from_numpy(mask[start:end]).to(device)
            p = torch.from_numpy(pol[start:end]).to(device)
            logits, _ = model(o)
            logits = logits.masked_fill(~m, -1e9)
            log_q = torch.nn.functional.log_softmax(logits, dim=-1)
            # KL(p||q) = sum p * (log p - log q). Add eps to avoid log(0).
            log_p = torch.log(p.clamp_min(1e-12))
            kl = (p * (log_p - log_q)).sum(dim=-1)
            kl_np = kl.detach().cpu().numpy()
            keep[start:end] = kl_np >= threshold
    n_kept = int(keep.sum())
    return obs[keep], mask[keep], pol[keep], val[keep], n_kept, n


# --------------------------------------------------------------------------- #
# Trainer with persistent optimizer
# --------------------------------------------------------------------------- #
class PersistentTrainer:
    """Persistent Adam optimizer; train on samples drawn from a buffer."""

    def __init__(self, network: AlphaZeroNet, lr: float,
                 weight_decay: float = 1e-4, value_loss_weight: float = 1.0,
                 freeze_bn_stats: bool = True, outcome_weight: float = 0.0):
        self.network = network
        self.opt = torch.optim.Adam(network.model.parameters(),
                                    lr=lr, weight_decay=weight_decay)
        self.value_loss_weight = value_loss_weight
        self.freeze_bn_stats = freeze_bn_stats
        self.outcome_weight = float(outcome_weight)

    def state_dict(self):
        return {"opt": self.opt.state_dict(),
                "value_loss_weight": self.value_loss_weight}

    def load_state_dict(self, sd):
        self.opt.load_state_dict(sd["opt"])
        self.value_loss_weight = float(sd.get("value_loss_weight",
                                              self.value_loss_weight))

    def _set_bn_track(self, track: bool):
        for m in self.network.model.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.track_running_stats = track

    def train_steps(self, buffer: ReplayBuffer, num_steps: int,
                    batch_size: int, rng: np.random.Generator):
        model = self.network.model
        device = self.network.device
        model.train()
        if self.freeze_bn_stats:
            self._set_bn_track(False)
        ep_total = 0.0; ep_pi = 0.0; ep_v = 0.0; nb = 0
        for _ in range(num_steps):
            obs, mask, pol, val = buffer.sample_batch(batch_size, rng)
            o = torch.from_numpy(obs).to(device)
            m = torch.from_numpy(mask).to(device)
            p = torch.from_numpy(pol).to(device)
            v = torch.from_numpy(val).to(device)
            self.opt.zero_grad()
            logits, value = model(o)
            logits = logits.masked_fill(~m, -1e9)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            # Per-sample policy loss; -(p * log_probs).sum(-1) is shape (B,).
            sample_policy_loss = -(p * log_probs).sum(dim=-1)
            if self.outcome_weight > 0:
                # Up-weight winning samples (v close to +1), down-weight losing
                # ones (v close to -1). Clip to keep gradients bounded.
                w = (1.0 + self.outcome_weight * v).clamp(0.2, 2.0)
                # Normalize so the mean weight is 1 — keeps loss magnitude
                # roughly comparable across iters.
                w = w / w.mean().clamp_min(1e-6)
                policy_loss = (sample_policy_loss * w).mean()
            else:
                policy_loss = sample_policy_loss.mean()
            value_loss = torch.nn.functional.mse_loss(value.squeeze(-1), v)
            loss = policy_loss + self.value_loss_weight * value_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            self.opt.step()
            ep_total += float(loss.item()); ep_pi += float(policy_loss.item())
            ep_v += float(value_loss.item()); nb += 1
        if self.freeze_bn_stats:
            self._set_bn_track(True)
        model.eval()
        return {
            "steps": nb,
            "loss": ep_total / max(nb, 1),
            "pi_loss": ep_pi / max(nb, 1),
            "v_loss": ep_v / max(nb, 1),
        }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--iterations", type=int, default=30)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--games-per-worker", type=int, default=4)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--snapshot-every", type=int, default=5)
    ap.add_argument("--eval-every", type=int, default=3)
    ap.add_argument("--league-fraction", type=float, default=0.6,
                    help="Total fraction of games using league/heuristic. "
                          "league_heuristic_share of those use heuristic, "
                          "the rest use league snapshots.")
    ap.add_argument("--league-heuristic-share", type=float, default=0.5,
                    help="Within league games, fraction using heuristic vs "
                          "snapshot. 0.5 = half/half.")
    ap.add_argument("--league-delay", type=int, default=0)
    ap.add_argument("--snapshot-gate-pins", type=float, default=-1.0,
                    help="If >=0, only snapshot when last eval avg_pins >= "
                          "this. Uses LAST eval, not just current iter.")
    ap.add_argument("--revert-pins", type=float, default=-1.0)
    ap.add_argument("--num-blocks", type=int, default=9)
    ap.add_argument("--num-filters", type=int, default=96)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--steps-per-iter", type=int, default=200,
                    help="Gradient steps per iter from the buffer.")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--value-loss-weight", type=float, default=1.0)
    ap.add_argument("--mcts-batch-size", type=int, default=32)
    ap.add_argument("--max-moves", type=int, default=120)
    ap.add_argument("--temperature-moves", type=int, default=30,
                    help="Use temperature=1 for first N moves of generation. "
                          "Default 30 (was 15 in SelfPlayConfig — too short).")
    ap.add_argument("--temperature-low", type=float, default=0.5,
                    help="Temperature after the first N moves. 0.5 keeps "
                          "exploration; was 0.3 (too greedy).")
    ap.add_argument("--value-target-lambda", type=float, default=0.5,
                    help="Per-step value target = lambda*outcome + "
                          "(1-lambda)*MCTS_root_value. Lower lambda gives "
                          "the value head dense per-position signal even "
                          "when games truncate. Was 0.8 in d33.")
    ap.add_argument("--use-gumbel", action="store_true")
    ap.add_argument("--num-considered-actions", type=int, default=16)
    ap.add_argument("--use-heuristic-value", action="store_true")
    ap.add_argument("--win-filter-min-pins", type=int, default=0,
                    help="Game-level filter: drop the whole game if neither "
                          "player reached this many pins. 0 disables.")
    ap.add_argument("--max-attempts-per-game", type=int, default=1)
    ap.add_argument("--per-colour-min-pins", type=int, default=0,
                    help="Per-colour filter: only emit training samples "
                          "from colour c when c won OR c.pins_in_goal >= "
                          "this. Drops losing-trajectory data — the d34 run "
                          "showed that training on losing trajectories "
                          "teaches the model to lose efficiently rather "
                          "than win. 7 is a strong setting; 5 is moderate. "
                          "0 disables.")
    ap.add_argument("--outcome-weight", type=float, default=0.0,
                    help="Per-sample outcome-weighted policy loss. Each "
                          "sample's policy loss is scaled by clip(1 + α * "
                          "value_target, 0.2, 2.0). With α=0.7, winners "
                          "(v=+1) contribute 1.7×, losers (v=-1) contribute "
                          "0.3×. 0 disables (uniform weighting).")
    ap.add_argument("--reservoir-anneal-per-iter", type=float, default=0.0,
                    help="Each iter, decrease reservoir_ratio by this much "
                          "(floor at --reservoir-floor). Lets self-play data "
                          "dominate later iters once the buffer is full of "
                          "good data. 0.01 = drop ratio by 1%% per iter.")
    ap.add_argument("--reservoir-floor", type=float, default=0.1,
                    help="Minimum reservoir_ratio when annealing. 0.1 keeps "
                          "10%% warmstart anchor at all times.")
    ap.add_argument("--n-weights", type=str, default="3,2,2,2,1",
                    help="Comma-separated weights for N=2,3,4,5,6 player games "
                          "during generation. Default '3,2,2,2,1' (favours 2p). "
                          "Multi-N specialist: '1,3,4,4,3' (favours 3-5p).")
    ap.add_argument("--greedy-share", type=float, default=0.5,
                    help="Within heuristic-mode iters, fraction that uses "
                          "greedy (vs advanced). 0.5 = half/half. greedy "
                          "teaches breaking truncations; advanced teaches "
                          "winning head-to-head.")
    ap.add_argument("--league-recency-bias", type=int, default=1,
                    help="Multiplier for the most-recent snapshot in the "
                          "league pool. With value k, the latest snapshot "
                          "appears k times, second-latest (k-1)x ... oldest "
                          "1x. k=1 is uniform (current d34 behavior). k=4 "
                          "biases toward strong recent opponents.")
    ap.add_argument("--seed", type=int, default=34_343)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--eval-games", type=int, default=20)
    ap.add_argument("--buffer-size", type=int, default=40000,
                    help="Max main-pool size in samples.")
    ap.add_argument("--reservoir-path", default=None,
                    help="Path to warmstart_data.npz for the reservoir. "
                          "If unset, no reservoir is used.")
    ap.add_argument("--reservoir-samples", type=int, default=20000)
    ap.add_argument("--reservoir-ratio", type=float, default=0.3,
                    help="Fraction of each batch drawn from the reservoir. "
                          "0.3 means 30%% warmstart, 70%% recent self-play.")
    ap.add_argument("--kl-threshold", type=float, default=0.0,
                    help="Drop samples where KL(MCTS || raw policy) < this. "
                          "0 disables. ~0.05 nats keeps only positions "
                          "where MCTS disagrees with the network.")
    ap.add_argument("--freeze-bn-stats", action="store_true",
                    help="Freeze BatchNorm running_mean/var during fine-tune. "
                          "Avoids drift on small batches.")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    snapshots_dir = os.path.join(args.output_dir, "snapshots")
    os.makedirs(snapshots_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "train_log.jsonl")

    net_cfg = NetworkConfig(num_blocks=args.num_blocks,
                             num_filters=args.num_filters)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    network = AlphaZeroNet(net_cfg, device=device)

    start_iter = 1
    latest_path = os.path.join(args.output_dir, "latest.pt")
    optimizer_path = os.path.join(args.output_dir, "optimizer.pt")
    if args.resume and os.path.exists(latest_path):
        ckpt = network.load_checkpoint(latest_path)
        start_iter = int(ckpt.get("iteration", 0)) + 1
        print(f"RESUMING from {latest_path} at iter {start_iter}", flush=True)
    elif args.start is not None:
        network.load_checkpoint(args.start)
        print(f"Loaded {args.start} (params={network.parameter_count()}, "
              f"device={device})", flush=True)
    else:
        print(f"FRESH random init (params={network.parameter_count()})",
              flush=True)

    sp_cfg = SelfPlayConfig(
        num_simulations=args.sims,
        mcts_batch_size=args.mcts_batch_size,
        use_heuristic_value=bool(args.use_heuristic_value),
        use_batched_mcts=True,
        max_moves=args.max_moves,
        min_pins_to_keep=2,
        temperature_moves=args.temperature_moves,
        temperature_low=args.temperature_low,
        value_target_lambda=args.value_target_lambda,
    )
    sp_cfg_dict = asdict(sp_cfg)
    net_cfg_dict = asdict(net_cfg)

    # Trainer with persistent optimizer
    trainer = PersistentTrainer(
        network=network, lr=args.lr,
        value_loss_weight=args.value_loss_weight,
        freeze_bn_stats=bool(args.freeze_bn_stats),
        outcome_weight=float(args.outcome_weight),
    )
    if args.resume and os.path.exists(optimizer_path):
        try:
            trainer.load_state_dict(
                torch.load(optimizer_path, map_location=device,
                           weights_only=False))
            print(f"  Optimizer state restored from {optimizer_path}",
                  flush=True)
        except Exception as e:
            print(f"  ⚠ Optimizer restore failed ({e}); starting fresh.",
                  flush=True)

    # Replay buffer
    buf = ReplayBuffer(max_size=args.buffer_size,
                       reservoir_ratio=args.reservoir_ratio)
    if args.reservoir_path and os.path.exists(args.reservoir_path):
        buf.seed_reservoir(args.reservoir_path,
                           max_samples=args.reservoir_samples,
                           seed=args.seed)
    else:
        if args.reservoir_path:
            print(f"  ⚠ Reservoir path {args.reservoir_path} not found",
                  flush=True)

    # Pre-load main pool from disk on resume (if previous iters cached)
    main_cache = os.path.join(args.output_dir, "buffer_main.npz")
    if args.resume and os.path.exists(main_cache):
        try:
            d = np.load(main_cache)
            if d["obs"].shape[0] > 0:
                buf.add(d["obs"], d["mask"], d["pol"], d["val"])
                print(f"  Buffer main pool restored: {buf.main_n} samples",
                      flush=True)
            d.close()
        except Exception as e:
            print(f"  ⚠ Main pool restore failed ({e})", flush=True)

    best_path = os.path.join(args.output_dir, "best_so_far.pt")
    best_pins = -1.0
    if os.path.exists(best_path):
        try:
            ck = torch.load(best_path, map_location="cpu", weights_only=False)
            best_pins = float(ck.get("avg_pins", -1.0))
            print(f"Existing best_so_far: {best_pins:.2f} pins", flush=True)
        except Exception:
            pass

    # Initial snapshot + baseline eval
    if start_iter == 1:
        init_snap_path = os.path.join(snapshots_dir, "snap_iter000.pt")
        if os.path.exists(init_snap_path):
            print(f"Keeping existing initial snapshot at {init_snap_path}",
                  flush=True)
        else:
            init_snap = _save_snapshot(network, snapshots_dir, 0)
            print(f"Initial snapshot → {init_snap}", flush=True)
        if best_pins < 0 and args.start is not None:
            print("Establishing baseline eval on seed model...", flush=True)
            t0 = time.time()
            base = quick_eval_vs_greedy(network, sp_cfg,
                                        num_games=args.eval_games,
                                        n_players=2)
            best_pins = float(base["avg_pins"])
            network.save_checkpoint(best_path, iteration=0,
                                    extra={"encoder_mode": "multicolour",
                                            "avg_pins": best_pins})
            print(f"Seed baseline: {best_pins:.2f} pins in "
                  f"{time.time()-t0:.0f}s", flush=True)

    # last-known eval, used for the snapshot gate when current iter has no eval
    last_eval_pins = best_pins if best_pins >= 0 else None

    log_f = open(log_path, "a")

    # The d33 generator splits "league" between snapshots and heuristic via
    # heuristic_opponent. Here we want a 3-way mix. We approximate it by
    # alternating per iter: even iters use heuristic opponent, odd iters
    # use snapshot opponents. The fraction is set so the average matches
    # league_fraction. With a buffer this stochastic mixing is fine.
    for it in range(start_iter, args.iterations + 1):
        t_iter = time.time()

        # Anneal reservoir ratio. Lets self-play data dominate later iters.
        if args.reservoir_anneal_per_iter > 0:
            iters_done = it - start_iter  # 0 on first iter of this run
            new_ratio = max(args.reservoir_floor,
                            args.reservoir_ratio
                            - args.reservoir_anneal_per_iter * iters_done)
            if abs(new_ratio - buf.reservoir_ratio) > 1e-6:
                buf.reservoir_ratio = new_ratio

        cur_path = os.path.join(args.output_dir, "current_for_workers.pt")
        network.save_checkpoint(cur_path, iteration=it,
                                extra={"encoder_mode": "multicolour"})

        snap_paths = _list_snapshots(snapshots_dir)

        # Choose this iter's opponent flavor.
        eff_league_frac = (
            0.0 if it <= args.league_delay else args.league_fraction
        )
        if eff_league_frac > 0 and rng.random() < args.league_heuristic_share:
            # Alternate advanced and greedy as heuristic opponent.
            # advanced teaches "win head-to-head vs strong opponent"
            # greedy teaches "break truncations vs obstructor"
            # Both signals matter. Coin flip, biased toward greedy_share.
            if rng.random() < args.greedy_share:
                heur_opp = "greedy"
                mode_label = "heuristic_greedy"
            else:
                heur_opp = "advanced"
                mode_label = "heuristic_advanced"
            opp_ckpts = []
        elif eff_league_frac > 0 and snap_paths:
            heur_opp = None
            # Recency-biased league pool: each snapshot appears k_i times,
            # where k_i = max(1, league_recency_bias - (n_total - 1 - i))
            # so the latest snapshot has weight=league_recency_bias, the
            # next-latest league_recency_bias-1, ..., older ones weight=1.
            k_max = max(1, int(args.league_recency_bias))
            n_total = len(snap_paths)
            opp_ckpts = []
            for i, p in enumerate(snap_paths):
                # i counts oldest→newest. weight increases toward newest.
                rank_from_newest = (n_total - 1) - i
                weight = max(1, k_max - rank_from_newest)
                opp_ckpts.extend([p] * weight)
            mode_label = "league"
        else:
            heur_opp = None
            opp_ckpts = []
            mode_label = "self"
            eff_league_frac = 0.0  # pure self-play

        print(f"\n=== Iteration {it}/{args.iterations} ===", flush=True)
        print(f"  workers={args.num_workers}, games/worker={args.games_per_worker} "
              f"(total={args.num_workers * args.games_per_worker}), "
              f"sims={args.sims}, opp={mode_label}, "
              f"league_frac={eff_league_frac:.2f}, "
              f"snaps={len(snap_paths)} (pool_size={len(opp_ckpts)}), "
              f"res_ratio={buf.reservoir_ratio:.2f}", flush=True)

        chunks_dir = os.path.join(args.output_dir, f"iter_{it:03d}_chunks")
        if os.path.exists(chunks_dir):
            shutil.rmtree(chunks_dir)

        t0 = time.time()
        data, stats = run_iteration_parallel(
            current_ckpt=cur_path,
            opponent_ckpts=opp_ckpts,
            net_cfg_dict=net_cfg_dict,
            sp_cfg_dict=sp_cfg_dict,
            output_dir=chunks_dir,
            num_workers=args.num_workers,
            games_per_worker=args.games_per_worker,
            league_fraction=eff_league_frac,
            max_moves=args.max_moves,
            base_seed=args.seed + it * 10000,
            use_gumbel=bool(args.use_gumbel),
            num_considered_actions=args.num_considered_actions,
            win_filter_min_pins=args.win_filter_min_pins,
            max_attempts_per_game=args.max_attempts_per_game,
            heuristic_opponent=heur_opp,
            per_colour_min_pins=args.per_colour_min_pins,
            n_weights=tuple(int(w) for w in args.n_weights.split(",")),
        )
        gen_time = time.time() - t0
        n_samples = int(data["obs"].shape[0])
        print(f"  gen done: {n_samples} samples in {gen_time:.0f}s "
              f"({stats['completed']} games, {stats['self_play']} SP, "
              f"{stats['league_play']} league, "
              f"current_wins_in_league={stats['current_wins_in_league']}/"
              f"{stats['league_play']})", flush=True)
        print(f"  per-N: {stats['per_n']}", flush=True)
        if args.per_colour_min_pins > 0:
            pcd = stats.get("per_colour_dropped", 0)
            pck = stats.get("per_colour_kept", 0)
            tot = pcd + pck
            ratio = (pck / tot) if tot > 0 else 0.0
            print(f"  per-colour-filter (≥{args.per_colour_min_pins} pins): "
                  f"{pck}/{tot} kept ({ratio:.0%})", flush=True)

        try:
            shutil.rmtree(chunks_dir)
        except Exception:
            pass

        if n_samples == 0:
            print("  ⚠ no samples this iteration", flush=True)
            continue

        # KL filter (optional)
        if args.kl_threshold > 0:
            kf_obs, kf_mask, kf_pol, kf_val, n_kept, n_total = kl_filter(
                network, data["obs"], data["action_masks"],
                data["policies"], data["values"],
                threshold=args.kl_threshold,
                batch_size=args.batch_size,
            )
            print(f"  kl-filter: {n_kept}/{n_total} kept "
                  f"(threshold={args.kl_threshold})", flush=True)
        else:
            kf_obs = data["obs"]
            kf_mask = data["action_masks"]
            kf_pol = data["policies"]
            kf_val = data["values"]
            n_kept = n_samples

        # Add to buffer
        if n_kept > 0:
            buf.add(kf_obs, kf_mask, kf_pol, kf_val)
        del data

        print(f"  buffer: main={buf.main_n} reservoir={buf.res_n}",
              flush=True)

        if len(buf) == 0:
            print("  ⚠ empty buffer, skipping training", flush=True)
            continue

        t0 = time.time()
        train_stats = trainer.train_steps(
            buf, num_steps=args.steps_per_iter,
            batch_size=args.batch_size, rng=rng,
        )
        train_time = time.time() - t0
        print(f"  train: {train_stats['steps']} steps in {train_time:.0f}s; "
              f"loss={train_stats['loss']:.4f} "
              f"pi={train_stats['pi_loss']:.4f} "
              f"v={train_stats['v_loss']:.4f}", flush=True)

        eval_result = None
        if it % args.eval_every == 0:
            t0 = time.time()
            eval_result = quick_eval_vs_greedy(network, sp_cfg,
                                                num_games=args.eval_games,
                                                n_players=2)
            print(f"  eval vs greedy (n=2, {args.eval_games} games): "
                  f"{eval_result['avg_pins']:.2f} pins, "
                  f"{eval_result['wins']}/{args.eval_games} wins, "
                  f"{eval_result['truncated']}/{args.eval_games} trunc "
                  f"(in {time.time()-t0:.0f}s)", flush=True)
            last_eval_pins = float(eval_result["avg_pins"])

            if last_eval_pins > best_pins:
                best_pins = last_eval_pins
                network.save_checkpoint(
                    best_path, iteration=it,
                    extra={"encoder_mode": "multicolour",
                            "avg_pins": best_pins},
                )
                print(f"  new best: {best_pins:.2f} pins → {best_path}",
                      flush=True)

            if args.revert_pins >= 0 and last_eval_pins < args.revert_pins:
                print(f"  ⚠ avg_pins {last_eval_pins:.2f} < "
                      f"{args.revert_pins:.2f}; REVERTING to best_so_far.pt",
                      flush=True)
                if os.path.exists(best_path):
                    network.load_checkpoint(best_path)

        # Snapshot — uses LAST eval, not just current iter (fixes d29 bug)
        if it % args.snapshot_every == 0:
            gate = args.snapshot_gate_pins
            ref = last_eval_pins
            should_snap = (gate < 0) or (ref is not None and ref >= gate)
            if should_snap:
                sp = _save_snapshot(network, snapshots_dir, it)
                print(f"  snapshot → {sp} (last_eval={ref})", flush=True)
            else:
                print(f"  skip snapshot (last_eval={ref} < gate={gate:.2f})",
                      flush=True)

        # Persist latest model + optimizer + buffer main pool
        network.save_checkpoint(latest_path, iteration=it,
                                extra={"encoder_mode": "multicolour"})
        try:
            torch.save(trainer.state_dict(), optimizer_path)
        except Exception as e:
            print(f"  ⚠ optimizer save failed: {e}", flush=True)
        # Buffer cache (slice the active main pool only)
        try:
            if buf.main_n > 0:
                np.savez_compressed(
                    main_cache,
                    obs=buf.main_obs[:buf.main_n],
                    mask=buf.main_mask[:buf.main_n],
                    pol=buf.main_pol[:buf.main_n],
                    val=buf.main_val[:buf.main_n],
                )
        except Exception as e:
            print(f"  ⚠ buffer save failed: {e}", flush=True)

        log_f.write(json.dumps({
            "iteration": it, "samples": n_samples, "kept": int(n_kept),
            "buffer_main": buf.main_n, "buffer_res": buf.res_n,
            "gen_time": gen_time, "train_time": train_time,
            "stats": stats, "train": train_stats, "eval": eval_result,
            "iter_time": time.time() - t_iter,
            "opp_mode": mode_label,
        }) + "\n")
        log_f.flush()
        print(f"  iter total: {time.time() - t_iter:.0f}s", flush=True)

    log_f.close()
    final = os.path.join(args.output_dir, "final.pt")
    network.save_checkpoint(final, iteration=args.iterations,
                            extra={"encoder_mode": "multicolour"})
    print(f"\nFinal model → {final}", flush=True)


if __name__ == "__main__":
    main()
