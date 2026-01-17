# -*- coding: utf-8 -*-
"""rl_adjuster_new_path_metrics_H5_schemeA_minstart2_max10.py

Exports:
  - RLPathOptimizer
  - evaluate_policy

Fixed-horizon (H=5) open-loop path planning with PPO.

Scheme A (recommended):
  - Terminal(final) quality reward dominates.
  - Small step shaping reward helps exploration but has much smaller scale.

Critical semantics alignment:
  - For an episode starting at time index t0, the generated sequence **includes all
    history up to t0** (inclusive) and then appends the H recommended items.
  - When computing yt_after (KT mastery on generated sequence), we run the base
    model on:  [all history up to t0] + [recommended path], with future original
    interactions masked as PAD. This matches your requirement.

Ablation support (terminal reward only):
  - You can turn off terminal reward components during training (so they are NOT
    computed nor used as reward), while evaluation still computes and prints all
    components for fair comparison.

NOTE: This module intentionally keeps the API used by train_rl_new.py:
  - RLPathOptimizer
  - evaluate_policy
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------- utils -------------------------

def _ensure_2d(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(0) if x.dim() == 1 else x


def _cosine_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a_n = a / (a.norm(dim=-1, keepdim=True) + eps)
    b_n = b / (b.norm(dim=-1, keepdim=True) + eps)
    return (a_n * b_n).sum(dim=-1)


# ------------------------- Difficulty mapping -------------------------

@dataclass
class DifficultyMapping:
    # idx2u: internal_idx -> original_uid
    idx2u: object
    # difficulty_by_uid: uid -> raw difficulty level (usually 1/2/3)
    difficulty_by_uid: Dict[int, int]

    @staticmethod
    def load_from_options(data_name: str) -> Optional["DifficultyMapping"]:
        """Load idx2u and difficulty file paths from your repo's dataLoader.Options.

        If Options or files are unavailable, returns None (metrics fall back safely).
        """
        try:
            from dataLoader import Options
        except Exception:
            return None

        try:
            opt = Options(data_name)
            with open(opt.idx2u_dict, "rb") as f:
                idx2u = pickle.load(f)

            difficulty_by_uid: Dict[int, int] = {}
            with open(opt.difficult_file, "r", encoding="utf-8") as f:
                # expect header: problem_id,level (or similar)
                try:
                    next(f)
                except StopIteration:
                    pass
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) < 2:
                        continue
                    try:
                        uid = int(parts[0].strip())
                        d = int(parts[1].strip())
                        difficulty_by_uid[uid] = d
                    except Exception:
                        continue

            return DifficultyMapping(idx2u=idx2u, difficulty_by_uid=difficulty_by_uid)
        except Exception:
            return None

    def _idx_to_uid(self, idx: int) -> Optional[int]:
        try:
            if hasattr(self.idx2u, "get"):
                return self.idx2u.get(int(idx), None)
            i = int(idx)
            if 0 <= i < len(self.idx2u):
                return int(self.idx2u[i])
        except Exception:
            return None
        return None

    def get_difficulty_raw(self, internal_idx: int, default: int = 2) -> int:
        uid = self._idx_to_uid(internal_idx)
        if uid is None:
            return default
        return int(self.difficulty_by_uid.get(int(uid), default))

    def get_difficulty_norm(self, internal_idx: int, default: int = 2) -> float:
        # raw 1/2/3 -> 0/0.5/1
        d = int(self.get_difficulty_raw(internal_idx, default=default))
        d = max(1, min(3, d))
        return (d - 1) / 2.0


# ------------------------- Policy / Value net -------------------------

class PolicyValueNet(nn.Module):
    """Candidate-wise policy + pooled value.

    Input: cand_feat [B, K, F]
    Output: logits [B, K], value [B]
    """

    def __init__(self, feat_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.logit_head = nn.Linear(hidden_dim, 1)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, cand_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.mlp(cand_feat)                  # [B,K,H]
        logits = self.logit_head(h).squeeze(-1)  # [B,K]
        att = torch.softmax(logits, dim=-1).unsqueeze(-1)  # [B,K,1]
        pooled = (h * att).sum(dim=1)            # [B,H]
        value = self.value_head(pooled).squeeze(-1)  # [B]
        return logits, value


# ------------------------- PPO trainer -------------------------

@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    minibatch_size: int = 256


class PPOTrainer:
    def __init__(self, policy: PolicyValueNet, lr: float = 3e-4, cfg: Optional[PPOConfig] = None):
        self.policy = policy
        self.cfg = cfg or PPOConfig()
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=lr)

    @torch.no_grad()
    def compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """rewards/values/dones: [T,B]

        dones[t,b]=1 means terminal at t.
        """
        T, B = rewards.shape
        adv = torch.zeros_like(rewards)
        last_gae = torch.zeros((B,), device=rewards.device)
        last_value = torch.zeros((B,), device=rewards.device)

        for t in reversed(range(T)):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.cfg.gamma * last_value * mask - values[t]
            last_gae = delta + self.cfg.gamma * self.cfg.gae_lambda * mask * last_gae
            adv[t] = last_gae
            last_value = values[t]

        returns = adv + values
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, returns

    def update(
        self,
        cand_feat: torch.Tensor,     # [N,K,F]
        actions: torch.Tensor,       # [N]
        old_logp: torch.Tensor,      # [N]
        old_values: torch.Tensor,    # [N]
        advantages: torch.Tensor,    # [N]
        returns: torch.Tensor,       # [N]
    ) -> Dict[str, float]:
        cfg = self.cfg
        N = cand_feat.size(0)
        idx = torch.randperm(N, device=cand_feat.device)

        losses = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "total_loss": 0.0}

        for _ in range(cfg.ppo_epochs):
            for s in range(0, N, cfg.minibatch_size):
                mb = idx[s : s + cfg.minibatch_size]
                mb_feat = cand_feat[mb]
                mb_act = actions[mb]
                mb_old_logp = old_logp[mb]
                mb_adv = advantages[mb]
                mb_ret = returns[mb]

                logits, value = self.policy(mb_feat)
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(mb_act)
                entropy = dist.entropy().mean()

                ratio = torch.exp(logp - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(value, mb_ret)

                loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.opt.step()

                losses["policy_loss"] += float(policy_loss.detach().cpu())
                losses["value_loss"] += float(value_loss.detach().cpu())
                losses["entropy"] += float(entropy.detach().cpu())
                losses["total_loss"] += float(loss.detach().cpu())

        denom = max(1, cfg.ppo_epochs * math.ceil(N / cfg.minibatch_size))
        for k in losses:
            losses[k] /= denom
        return losses


# ------------------------- Environment -------------------------

class OnlineLearningPathEnv:
    """Fixed-horizon planning environment.

    For each episode b:
      - Start from history prefix [0..start_t[b]] (inclusive).
      - Generate exactly H next items: positions start_t[b]+1 ... start_t[b]+H.
      - Simulation answer: from pre-step KT prob at chosen item.

    IMPORTANT:
      - gen_seq begins with full history, and masks original future with PAD.
      - yt_after is computed by running base_model on gen_seq (history + path).
    """

    def __init__(
        self,
        base_model: nn.Module,
        num_items: int,
        data_name: str,
        device: torch.device,
        pad_val: int = 0,
        topk: int = 10,
        cand_k: int = 50,
        history_window_T: int = 10,
        horizon_H: int = 5,
        target_future_M: int = 1,
        step_reward_scale: float = 0.1,
        epsilon: float = 1e-5,
        w_step: Optional[Dict[str, float]] = None,
    ):
        self.base_model = base_model
        self.num_items = int(num_items)
        self.data_name = data_name
        self.device = device
        self.pad_val = int(pad_val)
        self.topk = int(topk)
        self.cand_k = int(cand_k)
        self.T = int(history_window_T)
        self.H = int(horizon_H)
        self.future_M = int(target_future_M)
        self.step_reward_scale = float(step_reward_scale)
        self.eps = float(epsilon)

        self.w_step = w_step or {"preference": 1.0, "adaptivity": 1.0, "novelty": 0.2}

        self.diff_map = DifficultyMapping.load_from_options(data_name)

        # episode buffers
        self.graph = None
        self.hypergraph_list = None

        self.orig_seq = None
        self.orig_ts = None
        self.orig_idx = None
        self.orig_ans = None

        self.valid_lens = None
        self.start_t = None  # [B]

        self.gen_seq = None
        self.gen_ans = None

        self.hidden_item = None  # [N,d]

        # records per episode
        self.chosen_items: List[List[int]] = []
        self.topk_recs_policy: List[List[List[int]]] = []

        # rollout step counter (0..H)
        self.s = 0

        # pre-step cache
        self._pre_base_probs = None  # [B,N]
        self._pre_yt = None          # [B,N]
        self._pre_cand_ids = None    # [B,K]
        self._pre_cand_feat = None   # [B,K,F]
        self._pre_delta = None       # [B]
        self._pre_last_emb = None    # [B,d]
        self._pre_path_mean_emb = None

    @torch.no_grad()
    def _forward_base(self, seq: torch.Tensor, ts: torch.Tensor, idx: torch.Tensor, ans: torch.Tensor, trans_idx: torch.Tensor):
        """Run base_model and extract per-sample next-item distribution at a given transition index.

        seq: [B, L]
        pred_flat expected shape:
          - [B*(L-1), N]  OR  [B, L-1, N]
        trans_idx: [B] in [0, L-2]
        """
        out = self.base_model(seq, ts, idx, ans, self.graph, self.hypergraph_list)

        if isinstance(out, dict):
            pred_flat = out.get("pred_flat", out.get("pred", None))
            yt = out.get("yt", None)
            hidden = out.get("hidden", out.get("item_emb", None))
        else:
            pred_flat = out[0] if len(out) > 0 else None
            yt = out[3] if len(out) > 3 else None
            hidden = out[4] if len(out) > 4 else None

        if pred_flat is None:
            raise RuntimeError("base_model forward must provide pred logits (pred_flat/pred).")

        B, L = seq.shape
        if pred_flat.dim() == 2 and pred_flat.size(0) == B * (L - 1):
            pred_all = pred_flat.view(B, L - 1, -1)  # [B,L-1,N]
        elif pred_flat.dim() == 3 and pred_flat.size(0) == B:
            pred_all = pred_flat
        else:
            raise RuntimeError(f"Unsupported pred_flat shape: {tuple(pred_flat.shape)}")

        # gather per sample
        trans_idx = trans_idx.clamp(0, pred_all.size(1) - 1)
        pred_logits = pred_all[torch.arange(B, device=seq.device), trans_idx]  # [B,N]
        probs_last = torch.softmax(pred_logits, dim=-1)

        yt_last = None
        if yt is not None:
            if yt.dim() == 3 and yt.size(0) == B:
                yt_last = yt[torch.arange(B, device=seq.device), trans_idx]  # [B,S]
            elif yt.dim() == 2 and yt.size(0) == B:
                yt_last = yt

        if yt_last is None:
            yt_last = probs_last

        hidden_item = None
        if hidden is not None:
            if hidden.dim() == 3:
                hidden_item = hidden[0]
            else:
                hidden_item = hidden

        return probs_last, yt_last, hidden_item

    def _difficulty_norm(self, ids: torch.Tensor) -> torch.Tensor:
        # ids: [B,K]
        if self.diff_map is None:
            return torch.full_like(ids, 0.5, dtype=torch.float32, device=ids.device)
        ids_np = ids.detach().cpu().numpy()
        out = torch.zeros((ids_np.shape[0], ids_np.shape[1]), device=ids.device, dtype=torch.float32)
        for b in range(ids_np.shape[0]):
            for k in range(ids_np.shape[1]):
                out[b, k] = float(self.diff_map.get_difficulty_norm(int(ids_np[b, k]), default=2))
        return out

    def _compute_delta_t(self, hist_items: torch.Tensor, hist_ans: torch.Tensor) -> torch.Tensor:
        """Eq.(19) delta_t computed on a window of recent history.

        hist_items/hist_ans: [B, Lhist]
        """
        B, L = hist_items.shape
        delta = torch.ones((B,), device=hist_items.device, dtype=torch.float32)
        if self.diff_map is None:
            return delta

        items_np = hist_items.detach().cpu().numpy()
        ans_np = hist_ans.detach().cpu().numpy()
        for b in range(B):
            valid = [(int(items_np[b, t]), float(ans_np[b, t])) for t in range(L) if int(items_np[b, t]) != self.pad_val]
            if len(valid) < max(1, self.T // 2):
                delta[b] = 1.0
                continue
            window = valid[max(0, len(valid) - self.T):]
            num = 0.0
            den = 0.0
            for it, r in window:
                d = float(self.diff_map.get_difficulty_norm(it, default=2))
                num += d * r
                den += r
            delta[b] = float(num / (den + self.eps))
        return delta

    def reset(
        self,
        tgt: torch.Tensor,
        tgt_timestamp: torch.Tensor,
        tgt_idx: torch.Tensor,
        ans: torch.Tensor,
        start_t: torch.Tensor,
        graph=None,
        hypergraph_list=None,
    ) -> Dict[str, torch.Tensor]:
        self.graph = graph
        self.hypergraph_list = hypergraph_list

        self.orig_seq = _ensure_2d(tgt).to(self.device)
        self.orig_ts = _ensure_2d(tgt_timestamp).to(self.device)
        self.orig_idx = tgt_idx.to(self.device)
        if self.orig_idx.dim() == 2 and self.orig_idx.size(1) == 1:
            self.orig_idx = self.orig_idx.squeeze(1)
        self.orig_ans = _ensure_2d(ans).to(self.device)

        B, L = self.orig_seq.shape
        self.valid_lens = (self.orig_seq != self.pad_val).sum(dim=1).clamp_min(1)

        # clamp start_t so prefix length >=2 for model and enough space for H steps
        start_t = start_t.to(self.device).long().clamp_min(1)
        max_start = (torch.full((B,), L - self.H - 1, device=self.device, dtype=torch.long))
        start_t = torch.minimum(start_t, max_start)
        start_t = torch.minimum(start_t, self.valid_lens - 1)
        self.start_t = start_t

        # build generated sequence: keep history up to start_t, mask future to PAD
        self.gen_seq = torch.full_like(self.orig_seq, self.pad_val)
        self.gen_ans = torch.full_like(self.orig_ans, 0)
        for b in range(B):
            t0 = int(self.start_t[b].item())
            self.gen_seq[b, : t0 + 1] = self.orig_seq[b, : t0 + 1]
            self.gen_ans[b, : t0 + 1] = self.orig_ans[b, : t0 + 1]

        self.chosen_items = [[] for _ in range(B)]
        self.topk_recs_policy = [[] for _ in range(B)]
        self.s = 0

        self._update_pre_step_cache()
        return {"candidate_ids": self._pre_cand_ids, "candidate_features": self._pre_cand_feat}

    @torch.no_grad()
    def _update_pre_step_cache(self):
        B, L = self.gen_seq.shape

        # current prefix length per sample
        cur_len = (self.start_t + 1 + self.s).clamp(min=2, max=L)  # [B]
        # we run base_model on full length L with PAD future (safe) and gather per-sample transition index
        trans_idx = (cur_len - 2).clamp(min=0, max=L - 2)  # [B]

        seq = self.gen_seq
        ts = self.orig_ts
        ans = self.gen_ans
        idx = self.orig_idx

        probs_last, yt_last, hidden_item = self._forward_base(seq, ts, idx, ans, trans_idx=trans_idx)
        self.hidden_item = hidden_item

        Kc = min(self.cand_k, probs_last.size(-1))
        cand_ids = torch.topk(probs_last, k=Kc, dim=-1).indices  # [B,Kc]
        cand_probs = probs_last.gather(1, cand_ids)              # [B,Kc]
        cand_ability = yt_last.gather(1, cand_ids.clamp(0, yt_last.size(1) - 1))

        cand_diff = self._difficulty_norm(cand_ids)

        # delta_t uses a window (can be shorter than full history)
        # history for delta_t is from gen prefix up to cur_len
        delta_t = torch.ones((B,), device=self.device)
        recent_corr = torch.zeros((B,), device=self.device)
        recent_davg = torch.zeros((B,), device=self.device)

        for b in range(B):
            l = int(cur_len[b].item())
            hist_items = self.gen_seq[b : b + 1, :l]
            hist_ans = self.gen_ans[b : b + 1, :l]
            delta_t[b] = self._compute_delta_t(hist_items, hist_ans)[0]

            valid_mask = (hist_items[0] != self.pad_val)
            valid_items = hist_items[0][valid_mask]
            valid_ans = hist_ans[0][valid_mask].float()
            if valid_items.numel() == 0:
                recent_corr[b] = 0.0
                recent_davg[b] = 0.5
            else:
                w = min(int(valid_items.numel()), self.T)
                recent_corr[b] = valid_ans[-w:].mean()
                if self.diff_map is None:
                    recent_davg[b] = 0.5
                else:
                    items_np = valid_items[-w:].detach().cpu().numpy()
                    recent_davg[b] = float(np.mean([self.diff_map.get_difficulty_norm(int(it), default=2) for it in items_np]))

        self._pre_delta = delta_t

        # similarity features
        if hidden_item is not None:
            d = hidden_item.size(-1)
            path_mean = torch.zeros((B, d), device=self.device)
            last_emb = torch.zeros((B, d), device=self.device)
            for b in range(B):
                l = int(cur_len[b].item())
                valid_items = self.gen_seq[b, :l]
                valid_items = valid_items[valid_items != self.pad_val]
                if valid_items.numel() == 0:
                    continue
                emb = hidden_item[valid_items]
                path_mean[b] = emb.mean(dim=0)
                last_emb[b] = emb[-1]
            self._pre_path_mean_emb = path_mean
            self._pre_last_emb = last_emb

            cand_emb = hidden_item[cand_ids]  # [B,Kc,d]
            sim_path = _cosine_sim(cand_emb, path_mean.unsqueeze(1))
            sim_last = _cosine_sim(cand_emb, last_emb.unsqueeze(1))
        else:
            sim_path = torch.zeros_like(cand_probs)
            sim_last = torch.zeros_like(cand_probs)

        hist_feat = torch.stack([recent_corr, recent_davg, delta_t], dim=-1)  # [B,3]
        hist_feat = hist_feat.unsqueeze(1).expand(-1, cand_ids.size(1), -1)

        cand_feat = torch.cat(
            [
                cand_probs.unsqueeze(-1),
                cand_ability.unsqueeze(-1),
                cand_diff.unsqueeze(-1),
                sim_path.unsqueeze(-1),
                sim_last.unsqueeze(-1),
                hist_feat,
            ],
            dim=-1,
        ).float()  # [B,Kc,F]

        self._pre_base_probs = probs_last
        self._pre_yt = yt_last
        self._pre_cand_ids = cand_ids
        self._pre_cand_feat = cand_feat

    @torch.no_grad()
    def record_policy_topk(self, logits: torch.Tensor):
        """Record per-step policy topK ids (for debugging only)."""
        B, Kc = logits.shape
        K = min(self.topk, Kc)
        topk_idx = torch.topk(logits, k=K, dim=-1).indices  # indices within candidate set
        cand_ids = self._pre_cand_ids
        for b in range(B):
            ids = cand_ids[b, topk_idx[b]].detach().cpu().tolist()
            self.topk_recs_policy[b].append([int(x) for x in ids])

    @torch.no_grad()
    def step_env(self, action_idx: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict]:
        """Take one action for each episode in the expanded batch.

        action_idx: [B] index within candidate set.
        returns: next_state, reward [B], done [B]
        """
        B, L = self.gen_seq.shape
        Kc = self._pre_cand_ids.size(1)

        action_idx = action_idx.clamp(0, Kc - 1)
        chosen = self._pre_cand_ids.gather(1, action_idx.view(-1, 1)).squeeze(1)  # [B]

        for b in range(B):
            self.chosen_items[b].append(int(chosen[b].item()))

        # ----- step shaping reward (small scale) -----
        pref = self._pre_base_probs.gather(1, chosen.view(-1, 1)).squeeze(1)

        if self.diff_map is None:
            chosen_diff = torch.full((B,), 0.5, device=self.device)
        else:
            chosen_np = chosen.detach().cpu().numpy()
            chosen_diff = torch.tensor(
                [self.diff_map.get_difficulty_norm(int(it), default=2) for it in chosen_np],
                device=self.device,
                dtype=torch.float32,
            )

        adapt = 1.0 - torch.abs(self._pre_delta - chosen_diff)

        if self.hidden_item is not None and self._pre_last_emb is not None:
            chosen_emb = self.hidden_item[chosen]
            sim_last = _cosine_sim(chosen_emb, self._pre_last_emb)
            novelty = 1.0 - sim_last
        else:
            novelty = torch.zeros((B,), device=self.device)

        reward_step = (
            self.w_step.get("preference", 0.0) * pref
            + self.w_step.get("adaptivity", 0.0) * adapt
            + self.w_step.get("novelty", 0.0) * novelty
        ).float()
        reward = self.step_reward_scale * reward_step

        # ----- append chosen item at per-episode next position -----
        next_pos = (self.start_t + 1 + self.s).long()  # [B]
        done = torch.zeros((B,), device=self.device, dtype=torch.float32)

        for b in range(B):
            p = int(next_pos[b].item())
            if p < 0 or p >= L:
                done[b] = 1.0
                continue
            # write chosen
            self.gen_seq[b, p] = chosen[b]
            # simulate answer from pre-step mastery prob (item-level)
            p_correct = float(self._pre_yt[b, int(chosen[b].item())].item()) if int(chosen[b].item()) < self._pre_yt.size(1) else float(pref[b].item())
            self.gen_ans[b, p] = 1 if p_correct >= 0.5 else 0

        # advance step counter
        self.s += 1
        if self.s >= self.H:
            done[:] = 1.0

        if float(done.min().item()) < 1.0:
            self._update_pre_step_cache()

        next_state = {"candidate_ids": self._pre_cand_ids, "candidate_features": self._pre_cand_feat}
        info = {}
        return next_state, reward, done, info

    @torch.no_grad()
    def compute_final_quality(
        self,
        reward_weights: Optional[Dict[str, float]] = None,
        reward_components: Optional[Dict[str, bool]] = None,
        compute_all: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Terminal metrics computed on the **actual RL chosen path**.

        - If compute_all=False: only compute metrics that are enabled in reward_components.
          Disabled ones return 0 (and are not used in reward).
        - If compute_all=True: always compute all 3 metrics (for evaluation output).

        Returns tensors of shape [B] for each metric.
        """
        w = reward_weights or {"effectiveness": 1.0, "adaptivity": 1.0, "diversity": 1.0}
        rc = reward_components or {"effectiveness": True, "adaptivity": True, "diversity": True}

        B, L = self.orig_seq.shape
        H = int(self.H)

        # build RL-path tensor
        path_tensor = torch.full((B, H), self.pad_val, device=self.device, dtype=torch.long)
        for b in range(B):
            steps = min(len(self.chosen_items[b]), H)
            for s in range(steps):
                path_tensor[b, s] = int(self.chosen_items[b][s])

        # flags
        need_eff = compute_all or rc.get("effectiveness", True)
        need_adp = compute_all or rc.get("adaptivity", True)
        need_div = compute_all or rc.get("diversity", True)

        eff_scores = torch.zeros((B,), device=self.device)
        adapt_scores = torch.zeros((B,), device=self.device)
        div_scores = torch.zeros((B,), device=self.device)

        # ---- adaptivity Eq.(19) ----
        if need_adp and self.diff_map is not None:
            for b in range(B):
                total = 0.0
                cnt = 0
                for s in range(H):
                    rec = int(path_tensor[b, s].item())
                    if rec == self.pad_val or rec <= 1:
                        continue
                    prefix_len = int(self.start_t[b].item()) + 1 + s
                    prefix_len = max(1, min(prefix_len, L))
                    hist_items = self.gen_seq[b : b + 1, :prefix_len]
                    hist_ans = self.gen_ans[b : b + 1, :prefix_len]
                    delta = float(self._compute_delta_t(hist_items, hist_ans)[0].item())
                    rec_diff = float(self.diff_map.get_difficulty_norm(rec, default=2))
                    total += (1.0 - abs(delta - rec_diff))
                    cnt += 1
                adapt_scores[b] = total / max(1, cnt)

        # ---- effectiveness Eq.(20) ----
        # A-scheme target set (offline): use the *future ground-truth items* after start_t as targets.
        #
        # For each episode b:
        #   - targets T_b = { orig_seq[b, start_t+1 ... start_t+M] } (filter PAD/EOS)
        #   - pb: mastery on targets at the *pre-path* slice t0 = start_t
        #   - pa: mastery on targets at the *post-path* slice t1 = start_t + H - 1
        #   - effectiveness = mean_{q in T_b} (pa - pb) / (1 - pb)
        #
        # IMPORTANT: yt_before must be computed on *history prefix only* (mask out original future),
        # otherwise it leaks future interactions.

        def _run_kt_like(seq, ts, idx, ans):
            out = self.base_model(seq, ts, idx, ans, self.graph, self.hypergraph_list)
            if isinstance(out, dict):
                yt = out.get("yt", None)
            else:
                yt = out[3] if len(out) > 3 else None
            if yt is None:
                return None
            if yt.dim() == 3:
                return yt
            return None

        if need_eff:
            # ---- build BEFORE inputs: keep only prefix [0..start_t] ----
            seq_before = self.orig_seq.clone()
            ans_before = self.orig_ans.clone()
            ts_before = self.orig_ts.clone()
            idx_before = self.orig_idx.clone()

            for b in range(B):
                cut = int(self.start_t[b].item()) + 1  # keep inclusive prefix
                cut = max(1, min(cut, L))
                if cut < L:
                    seq_before[b, cut:] = self.pad_val
                    ans_before[b, cut:] = 0
                    ts_before[b, cut:] = 0
                    idx_before[b, cut:] = 0

            # ---- build AFTER inputs: use history+RL path, but zero ts/idx on PAD positions ----
            seq_after = self.gen_seq
            ans_after = self.gen_ans
            ts_after = self.orig_ts.clone()
            idx_after = self.orig_idx.clone()
            pad_mask_after = (seq_after == self.pad_val)
            ts_after[pad_mask_after] = 0
            idx_after[pad_mask_after] = 0

            yt_before = _run_kt_like(seq_before, ts_before, idx_before, ans_before)
            yt_after = _run_kt_like(seq_after, ts_after, idx_after, ans_after)

            if yt_before is not None and yt_after is not None:
                Tm = min(yt_before.size(1), yt_after.size(1), L - 1)
                yt_before = yt_before[:, :Tm]
                yt_after = yt_after[:, :Tm]
                S = yt_before.size(-1)

                M = int(getattr(self, "future_M", 1))

                for b in range(B):
                    # time slices
                    t0 = int(self.start_t[b].item())
                    t0 = max(0, min(t0, Tm - 1))
                    t1 = int(self.start_t[b].item()) + H - 1
                    t1 = max(0, min(t1, Tm - 1))

                    # collect future targets from ORIGINAL sequence
                    targets = []
                    for j in range(1, M + 1):
                        pos = int(self.start_t[b].item()) + j
                        if pos < 0 or pos >= L:
                            continue
                        item = int(self.orig_seq[b, pos].item())
                        if item in (self.pad_val, 1) or item <= 1:
                            continue
                        targets.append(item)

                    if not targets:
                        eff_scores[b] = 0.0
                        continue

                    total = 0.0
                    cnt = 0
                    for r in targets:
                        if 0 <= r < S:
                            pb = float(yt_before[b, t0, r].item())
                            pa = float(yt_after[b, t1, r].item())
                            if pb < 0.9 and pa > 0:
                                total += (pa - pb) / (1.0 - pb)
                                cnt += 1
                    eff_scores[b] = total / max(1, cnt)

        # ---- diversity Eq.(21) ----
        if need_div and self.hidden_item is not None:
            emb = self.hidden_item
            for b in range(B):
                items = [int(path_tensor[b, s].item()) for s in range(H) if int(path_tensor[b, s].item()) not in (self.pad_val, 1) and int(path_tensor[b, s].item()) > 1]
                if len(items) < 2:
                    div_scores[b] = 0.0
                    continue
                e = emb[torch.tensor(items, device=self.device)]
                e = e / (e.norm(dim=-1, keepdim=True) + 1e-8)
                sim = e @ e.t()
                triu = torch.triu(sim, diagonal=1)
                vals = triu[triu != 0]
                div_scores[b] = 0.0 if vals.numel() == 0 else (1.0 - vals).mean()

        # reward only uses enabled components
        final_quality = (
            (w["effectiveness"] * eff_scores if rc.get("effectiveness", True) else 0.0)
            + (w["adaptivity"] * adapt_scores if rc.get("adaptivity", True) else 0.0)
            + (w["diversity"] * div_scores if rc.get("diversity", True) else 0.0)
        )

        return {
            "effectiveness": eff_scores,
            "adaptivity": adapt_scores,
            "diversity": div_scores,
            "final_quality": final_quality,
        }


# ------------------------- RL wrapper -------------------------

class RLPathOptimizer:
    """Wraps env + policy/value + PPO.

    Includes:
      - multi-start episode sampling per sequence
      - terminal reward ablation toggles
    """

    def __init__(
        self,
        base_model: nn.Module,
        num_items: int,
        data_name: str,
        device: torch.device,
        pad_val: int = 0,
        topk: int = 10,
        cand_k: int = 50,
        history_window_T: int = 10,
        rl_lr: float = 3e-4,
        target_future_M: int = 1,
        policy_hidden: int = 128,
        ppo_config: Optional[PPOConfig] = None,
        step_reward_weights: Optional[Dict[str, float]] = None,
        final_reward_weights: Optional[Dict[str, float]] = None,
        terminal_reward_scale: float = 1.0,
        # fixed-horizon + multi-start
        horizon_H: int = 5,
        min_start: int = 2,
        max_starts_per_seq: int = 10,
        # terminal reward ablation (TRAINING only)
        terminal_reward_components: Optional[Dict[str, bool]] = None,
        train_compute_all_terminal_metrics: bool = False,
    ):
        self.device = device
        self.env = OnlineLearningPathEnv(
            base_model=base_model,
            num_items=num_items,
            data_name=data_name,
            device=device,
            pad_val=pad_val,
            topk=topk,
            cand_k=cand_k,
            history_window_T=history_window_T,
            horizon_H=horizon_H,
            target_future_M=target_future_M,
            step_reward_scale=0.1,
            w_step=step_reward_weights,
        )

        self.policy: Optional[PolicyValueNet] = None
        self.trainer: Optional[PPOTrainer] = None

        self.rl_lr = float(rl_lr)
        self.policy_hidden = int(policy_hidden)
        self.ppo_config = ppo_config or PPOConfig()

        self.final_reward_weights = final_reward_weights or {"effectiveness": 1.0, "adaptivity": 1.0, "diversity": 1.0}
        self.terminal_reward_scale = float(terminal_reward_scale)

        self.horizon_H = int(horizon_H)
        self.min_start = int(min_start)
        self.max_starts_per_seq = int(max_starts_per_seq)

        self.terminal_reward_components = terminal_reward_components or {"effectiveness": True, "adaptivity": True, "diversity": True}
        self.train_compute_all_terminal_metrics = bool(train_compute_all_terminal_metrics)

    def _lazy_init(self, feat_dim: int):
        if self.policy is None:
            self.policy = PolicyValueNet(feat_dim=int(feat_dim), hidden_dim=self.policy_hidden).to(self.device)
            self.trainer = PPOTrainer(self.policy, lr=self.rl_lr, cfg=self.ppo_config)

    @torch.no_grad()
    def ensure_initialized(self, tgt, tgt_timestamp, tgt_idx, ans, graph=None, hypergraph_list=None) -> None:
        if self.policy is not None and self.trainer is not None:
            return
        # choose a safe start_t for lazy init: clamp to min_start
        B, L = tgt.shape
        start_t = torch.full((B,), self.min_start, device=tgt.device, dtype=torch.long)
        state = self.env.reset(tgt, tgt_timestamp, tgt_idx, ans, start_t=start_t, graph=graph, hypergraph_list=hypergraph_list)
        self._lazy_init(int(state["candidate_features"].size(-1)))

    def _sample_starts(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (episode_src_indices, episode_start_t) for the given batch.

        seq: [B,L]
        """
        B, L = seq.shape
        valid_lens = (seq != self.env.pad_val).sum(dim=1).clamp_min(1)
        max_start_global = L - self.horizon_H - 1
        ep_src: List[int] = []
        ep_t: List[int] = []

        for b in range(B):
            vlen = int(valid_lens[b].item())
            max_t = min(vlen - 1, max_start_global)
            if max_t < self.min_start:
                continue
            candidates = list(range(self.min_start, max_t + 1))
            if len(candidates) > self.max_starts_per_seq:
                # random sample
                perm = torch.randperm(len(candidates))[: self.max_starts_per_seq].tolist()
                candidates = [candidates[i] for i in perm]
            for t0 in candidates:
                ep_src.append(b)
                ep_t.append(t0)

        if len(ep_src) == 0:
            # fallback: at least one episode from the first sequence if possible
            b = 0
            vlen = int(valid_lens[b].item())
            max_t = min(vlen - 1, max_start_global)
            t0 = min(max(max_t, self.min_start), max_t) if max_t >= self.min_start else 1
            ep_src = [b]
            ep_t = [t0]

        return torch.tensor(ep_src, dtype=torch.long, device=seq.device), torch.tensor(ep_t, dtype=torch.long, device=seq.device)

    def collect_trajectory(self, tgt, tgt_timestamp, tgt_idx, ans, graph=None, hypergraph_list=None) -> Dict[str, torch.Tensor]:
        """Collect one PPO batch from multi-start episodes."""
        # expand batch by sampling multiple start points per sequence
        src_idx, start_t = self._sample_starts(tgt)

        tgt_e = tgt.index_select(0, src_idx)
        ts_e = tgt_timestamp.index_select(0, src_idx)
        ans_e = ans.index_select(0, src_idx)
        if tgt_idx.dim() == 1:
            idx_e = tgt_idx.index_select(0, src_idx)
        else:
            idx_e = tgt_idx.index_select(0, src_idx)

        state = self.env.reset(tgt_e, ts_e, idx_e, ans_e, start_t=start_t, graph=graph, hypergraph_list=hypergraph_list)
        cand_feat = state["candidate_features"]
        self._lazy_init(int(cand_feat.size(-1)))

        B = cand_feat.size(0)
        T = self.horizon_H

        all_cand_feat: List[torch.Tensor] = []
        all_actions: List[torch.Tensor] = []
        all_logp: List[torch.Tensor] = []
        all_values: List[torch.Tensor] = []
        all_rewards: List[torch.Tensor] = []
        all_dones: List[torch.Tensor] = []

        done = torch.zeros((B,), device=self.device, dtype=torch.float32)

        for t in range(T):
            cand_feat = state["candidate_features"]
            logits, value = self.policy(cand_feat)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action)

            # optional record
            self.env.record_policy_topk(logits)

            next_state, reward, step_done, _ = self.env.step_env(action)
            done = torch.maximum(done, step_done)

            all_cand_feat.append(cand_feat)
            all_actions.append(action)
            all_logp.append(logp)
            all_values.append(value)
            all_rewards.append(reward)
            all_dones.append(done.clone())

            state = next_state

        # terminal metrics (reward uses enabled components; training may skip disabled computations)
        final_metrics = self.env.compute_final_quality(
            reward_weights=self.final_reward_weights,
            reward_components=self.terminal_reward_components,
            compute_all=self.train_compute_all_terminal_metrics,
        )
        terminal_r = final_metrics["final_quality"] * self.terminal_reward_scale
        all_rewards[-1] = all_rewards[-1] + terminal_r

        rewards = torch.stack(all_rewards, dim=0)  # [T,B]
        values = torch.stack(all_values, dim=0)   # [T,B]
        # dones: terminal at last step
        dones = torch.zeros_like(rewards)
        dones[-1] = 1.0

        adv, rets = self.trainer.compute_gae(rewards, values, dones)

        cand_feat_t = torch.stack(all_cand_feat, dim=0)  # [T,B,K,F]
        actions_t = torch.stack(all_actions, dim=0)      # [T,B]
        logp_t = torch.stack(all_logp, dim=0)            # [T,B]

        # flatten all (no padding steps in fixed horizon)
        T, B = rewards.shape
        cand_feat_flat = cand_feat_t.reshape(T * B, cand_feat_t.size(2), cand_feat_t.size(3))
        actions_flat = actions_t.reshape(-1)
        logp_flat = logp_t.reshape(-1)
        values_flat = values.reshape(-1)
        adv_flat = adv.reshape(-1)
        rets_flat = rets.reshape(-1)

        return {
            "cand_feat": cand_feat_flat,
            "actions": actions_flat,
            "old_logp": logp_flat,
            "old_values": values_flat,
            "advantages": adv_flat,
            "returns": rets_flat,
            "final_metrics": final_metrics,
        }

    def update_policy(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        return self.trainer.update(
            cand_feat=batch["cand_feat"],
            actions=batch["actions"],
            old_logp=batch["old_logp"],
            old_values=batch["old_values"],
            advantages=batch["advantages"],
            returns=batch["returns"],
        )


@torch.no_grad()
def evaluate_policy(
    rl: RLPathOptimizer,
    data_loader,
    graph,
    hypergraph_list,
    device: torch.device,
    max_batches: int = 50,
) -> Dict[str, float]:
    """Compute mean terminal metrics over a few batches.

    IMPORTANT for your ablations:
      - Evaluation ALWAYS computes all 3 metrics, even if training ablated some.
      - Evaluation does NOT change reward settings; it just reports.
    """
    if rl.policy is None:
        first = next(iter(data_loader))
        tgt, ts, idx, ans = first[0].to(device), first[1].to(device), first[2].to(device), first[3].to(device)
        rl.ensure_initialized(tgt, ts, idx, ans, graph=graph, hypergraph_list=hypergraph_list)

    rl.policy.eval()

    agg = {"effectiveness": 0.0, "adaptivity": 0.0, "diversity": 0.0, "final_quality": 0.0}
    n = 0

    for i, batch in enumerate(data_loader):
        if i >= max_batches:
            break
        tgt, ts, idx, ans = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)

        # collect one rollout to build env.gen_seq/ans, then compute full metrics
        src_idx, start_t = rl._sample_starts(tgt)
        tgt_e = tgt.index_select(0, src_idx)
        ts_e = ts.index_select(0, src_idx)
        ans_e = ans.index_select(0, src_idx)
        idx_e = idx.index_select(0, src_idx) if idx.dim() == 1 else idx.index_select(0, src_idx)

        state = rl.env.reset(tgt_e, ts_e, idx_e, ans_e, start_t=start_t, graph=graph, hypergraph_list=hypergraph_list)
        cand_feat = state["candidate_features"]

        # rollout policy for H steps
        for _ in range(rl.horizon_H):
            logits, _ = rl.policy(cand_feat)
            action = torch.distributions.Categorical(logits=logits).sample()
            state, _, _, _ = rl.env.step_env(action)
            cand_feat = state["candidate_features"]

        metrics = rl.env.compute_final_quality(
            reward_weights=rl.final_reward_weights,
            reward_components=rl.terminal_reward_components,
            compute_all=True,
        )

        for k in agg:
            agg[k] += float(metrics[k].mean().detach().cpu())
        n += 1

    if n == 0:
        return {k: 0.0 for k in agg}
    return {k: v / n for k, v in agg.items()}
