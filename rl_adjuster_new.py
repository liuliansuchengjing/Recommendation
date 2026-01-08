# -*- coding: utf-8 -*-
"""
rl_adjuster.py
强化学习路径优化：候选集合内的策略分布 top-k + Metrics 口径的终端指标（按样本计算）
- 保留对 train_rl.py / 旧接口的兼容：PolicyNetwork / LearningPathEnv / PPOTrainer / RLPathOptimizer / evaluate_policy
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import pickle

import Constants
from Metrics import Metrics

try:
    from dataLoader import Options
except Exception:
    Options = None

try:
    # HGAT.py 里定义了 KTOnlyModel
    from HGAT import KTOnlyModel
except Exception:
    KTOnlyModel = None


def _ensure_2d(x: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is [B, L]."""
    if x.dim() == 1:
        return x.unsqueeze(0)
    return x


def _pad_and_concat_like(original_2d: torch.Tensor, ext_len: int, pad_value: int = 0) -> torch.Tensor:
    """
    original_2d: [B, L]
    return: [B, L + ext_len] where the new tail is pad_value
    """
    original_2d = _ensure_2d(original_2d)
    B = original_2d.size(0)
    pad = torch.full((B, ext_len), pad_value, device=original_2d.device, dtype=original_2d.dtype)
    return torch.cat([original_2d, pad], dim=1)


class PolicyNetwork(nn.Module):
    """
    简单 policy：对候选集合内每个 item 输出一个 logit
    输入:
      knowledge_state: [B, N]  (N=num_skills/items)
      candidate_features: [B, K, 1]  (这里仅用 cand_probs)
    输出:
      logits: [B, K]
    """

    def __init__(self, knowledge_dim: int, candidate_feature_dim: int = 1, hidden_dim: int = 128, topk: int = 10):
        super().__init__()
        self.knowledge_dim = int(knowledge_dim)
        self.topk = int(topk)

        self.know_proj = nn.Linear(self.knowledge_dim, hidden_dim)
        self.cand_proj = nn.Linear(candidate_feature_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, knowledge_state: torch.Tensor, candidate_features: torch.Tensor) -> torch.Tensor:
        # knowledge_state: [B,N]
        # candidate_features: [B,K,1]
        B, K, _ = candidate_features.shape
        h = torch.tanh(self.know_proj(knowledge_state))  # [B,H]
        c = torch.tanh(self.cand_proj(candidate_features))  # [B,K,H]
        h = h.unsqueeze(1).expand(-1, K, -1)  # [B,K,H]
        logits = self.out(h + c).squeeze(-1)  # [B,K]
        return logits


class LearningPathEnv:
    """
    环境：给定学生历史序列（tgt/...），每一步在候选集合上选择一个 item 追加到路径末端。
    关键点（已按你们讨论的方案落地）：
    1) step-level 奖励中的 preference/difficulty **基于 step 前（旧 state）的输出**。
    2) 记录 step_topk_items：候选集合内的“策略分布 top-k”（由 policy logits 取 top-k）。
    3) 终端指标 compute_final_metrics：按样本计算（不做 global mean 再复制）。
    4) diversity：严格按 Metrics 口径（使用 base_model.gnn 的 hidden 作 item embedding，计算 1-cosine 相似度的均值）。
    5) adaptivity：默认用 yt_before 在“相关 item 子集”（本条推荐路径）上的能力聚合 + 真实难度聚合。
       （保留 simulated-answer 版本的备注入口，但默认不启用）
    """

    def __init__(
            self,
            batch_size: int,
            base_model,
            recommendation_length: int,
            data_name: str,
            graph=None,
            hypergraph_list=None,
            policy_topk: int = 10,
            metrics_topnum: Optional[int] = None,
            metrics_T: int = 5,
            final_weights: Optional[Dict[str, float]] = None,
    ):
        self.batch_size = int(batch_size)
        self.base_model = base_model
        self.recommendation_length = int(recommendation_length)
        self.data_name = data_name
        self.graph = graph
        self.hypergraph_list = hypergraph_list

        self.policy_topk = int(policy_topk)
        self.metrics_topnum = int(metrics_topnum if metrics_topnum is not None else policy_topk)
        self.metrics_topnum = max(1, min(self.metrics_topnum, self.policy_topk))
        self.metrics_T = int(metrics_T)

        self.metric = Metrics()
        self.final_weights = final_weights or {
            "effectiveness": 0.4,
            "adaptivity": 0.3,
            "diversity": 0.2,
            "preference": 0.1,
        }

        # difficulty cache (idx -> diff_int in {1,2,3})
        self.idx2u = None
        self.difficulty_data: Dict[int, int] = {}
        self._load_difficulty_data()

        # KT-only model for effectiveness simulation
        self.kt_model = None
        if KTOnlyModel is not None:
            try:
                self.kt_model = KTOnlyModel(self.base_model)
            except Exception:
                self.kt_model = None

        self._reset_episode()

    def _load_difficulty_data(self):
        if Options is None:
            return
        try:
            opt = Options(self.data_name)
            with open(opt.idx2u_dict, "rb") as f:
                self.idx2u = pickle.load(f)
            diff = {}
            with open(opt.difficult_file, "r") as f:
                next(f)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(",")
                    if len(parts) < 2:
                        continue
                    try:
                        cid = int(parts[0].strip())
                        d = int(parts[1].strip())
                        diff[cid] = d
                    except Exception:
                        continue
            self.difficulty_data = diff
        except Exception:
            # fall back to empty
            self.idx2u = None
            self.difficulty_data = {}

    def _diff_norm(self, item_ids: torch.Tensor) -> torch.Tensor:
        """
        item_ids: [B] long
        return diff_norm: [B] float in {0.0,0.5,1.0}
        """
        item_ids = item_ids.long()
        out = torch.zeros(item_ids.size(0), device=item_ids.device, dtype=torch.float)
        if self.idx2u is None or not self.difficulty_data:
            return out
        # idx2u maps idx -> original_id (challenge_id)
        for i in range(item_ids.size(0)):
            idx = int(item_ids[i].item())
            try:
                oid = int(self.idx2u[idx])
            except Exception:
                oid = None
            d = self.difficulty_data.get(oid, 1) if oid is not None else 1
            # normalize 1/2/3 -> 0/0.5/1
            out[i] = float(max(1, min(3, int(d))) - 1) / 2.0
        return out

    def _reset_episode(self):
        self.original_tgt = None
        self.original_tgt_timestamp = None
        self.original_tgt_idx = None
        self.original_ans = None

        self.paths: List[List[int]] = []
        self.topk_recs: List[List[List[int]]] = []  # per sample: list of steps, each step list of topn item ids

        self.current_step = 0

        # caches from last forward
        self._last_pred_probs_full = None  # [B, L, N]
        self._last_yt_full = None  # [B, L, N]
        self._last_hidden = None  # [N, D]

        # caches at reset
        self._yt_before = None  # [B, N]

        # step-level logs for preference
        self._pref_hist: List[List[float]] = []
        self._repeat_hist: List[List[int]] = []

    def reset(self, tgt, tgt_timestamp, tgt_idx, ans, graph=None, hypergraph_list=None):
        self._reset_episode()
        self.original_tgt = _ensure_2d(tgt)
        self.original_tgt_timestamp = _ensure_2d(tgt_timestamp)
        self.original_tgt_idx = _ensure_2d(tgt_idx)
        self.original_ans = _ensure_2d(ans)

        if graph is not None:
            self.graph = graph
        if hypergraph_list is not None:
            self.hypergraph_list = hypergraph_list

        B = self.original_tgt.size(0)
        self.paths = [[] for _ in range(B)]
        self.topk_recs = [[] for _ in range(B)]
        self._pref_hist = [[] for _ in range(B)]
        self._repeat_hist = [[] for _ in range(B)]
        self.current_step = 0

        self._forward_base(self.original_tgt, self.original_tgt_timestamp, self.original_tgt_idx, self.original_ans)
        self._yt_before = self._last_yt_full[:, -1, :].detach()
        return self._make_state_from_last()

    def _forward_base(self, seq, ts, idx, ans):
        """
        Forward the pretrained/base model (MSHGAT) to obtain:
          - next-item logits over all items (flattened) -> reshape to [B, T, N]
          - KT outputs yt -> [B, T, N]
          - item graph embeddings hidden -> [N, D] (for diversity)

        IMPORTANT (robustness):
        - Do NOT assume T == seq_len-1 because your dataloader pads to max_len (e.g. 200)
          but the model may internally drop padded steps and output a shorter flattened tensor.
        - Therefore infer T from pred_flat.shape[0] // B when possible.
        """
        self.base_model.eval()
        with torch.no_grad():
            outs = self.base_model(seq, ts, idx, ans, self.graph, self.hypergraph_list)

        # Your calculate_muti_obj.py uses: pred, pred_res, kt_mask, yt_before, hidden = model(...)
        # Some versions may return extra tensors; we only rely on the first 5.
        if isinstance(outs, (list, tuple)):
            if len(outs) < 5:
                raise RuntimeError(f"base_model should return at least 5 tensors, got {len(outs)}")
            pred_flat, pred_res, kt_mask, yt, hidden = outs[:5]
        else:
            raise RuntimeError("base_model forward must return a tuple/list")

        B = int(seq.size(0))

        # pred_flat must be [B*T, N]
        if pred_flat.dim() != 2:
            raise RuntimeError(f"pred_flat must be 2D [B*T, N], got {tuple(pred_flat.shape)}")

        N = int(pred_flat.size(-1))
        T = int(pred_flat.size(0) // B) if pred_flat.size(0) % B == 0 else max(int(seq.size(1)) - 1, 1)

        pred_flat = pred_flat[:B * T]  # safe truncate
        pred_probs = torch.softmax(pred_flat.view(B, T, N), dim=-1)  # [B,T,N]
        self._last_pred_probs_full = pred_probs

        # yt should be [B,T,N] (DKT sigmoid probabilities). Truncate/pad to match T.
        if yt.dim() == 2:
            # occasionally returned flattened -> reshape like pred_flat with unknown N'
            ytN = int(yt.size(-1))
            ytT = int(yt.size(0) // B) if yt.size(0) % B == 0 else T
            yt = yt[:B * ytT].view(B, ytT, ytN)
        elif yt.dim() == 3:
            pass
        else:
            raise RuntimeError(f"yt must be 2D or 3D, got {tuple(yt.shape)}")

        # Align length to T
        if yt.size(1) >= T:
            yt = yt[:, :T, :]
        else:
            # pad time dimension if needed
            pad_t = T - yt.size(1)
            yt = torch.cat([yt, yt[:, -1:, :].repeat(1, pad_t, 1)], dim=1)

        self._last_yt_full = yt.detach()  # keep as probs
        self._last_hidden = hidden.detach() if isinstance(hidden, torch.Tensor) else hidden

    def _make_state_from_last(self):
        pred_last = self._last_pred_probs_full[:, -1, :]  # [B,N]
        yt_last = self._last_yt_full[:, -1, :]  # [B,N]

        topk = self.policy_topk
        K = min(topk, pred_last.size(1))
        cand_probs, cand_ids = torch.topk(pred_last, k=K, dim=-1)  # [B,K]
        return {
            "knowledge_state": yt_last,
            "cand_probs": cand_probs,
            "cand_ids": cand_ids,
        }

    def step(self, chosen_item_ids: torch.Tensor, step_topk_items: torch.Tensor):
        """
        chosen_item_ids: [B] item ids (NOT indices in candidate list)
        step_topk_items: [B, topn] item ids computed from policy distribution top-k within candidate space
        """
        B = chosen_item_ids.size(0)
        device = chosen_item_ids.device

        # -------- cache OLD outputs (step-timing correctness) --------
        old_pred_last = self._last_pred_probs_full[:, -1, :].detach()  # [B,N]
        old_yt_last = self._last_yt_full[:, -1, :].detach()  # [B,N]

        # record path and topk
        for i in range(B):
            self.paths[i].append(int(chosen_item_ids[i].item()))
            self.topk_recs[i].append([int(x) for x in step_topk_items[i].tolist()])

        # step reward
        step_reward = self._step_reward(chosen_item_ids, old_pred_last, old_yt_last)

        # build extended inputs (history + appended path so far)
        ext_len = len(self.paths[0])
        path_tensor = torch.tensor(self.paths, device=device, dtype=self.original_tgt.dtype)  # [B, ext_len]
        ext_tgt = torch.cat([self.original_tgt, path_tensor], dim=1)
        ext_timestamp = _pad_and_concat_like(self.original_tgt_timestamp, ext_len, pad_value=0)
        ext_idx = _pad_and_concat_like(self.original_tgt_idx, ext_len, pad_value=0)
        pad_ans = torch.zeros(B, ext_len, device=device, dtype=self.original_ans.dtype)
        ext_ans = torch.cat([self.original_ans, pad_ans], dim=1)

        self._forward_base(ext_tgt, ext_timestamp, ext_idx, ext_ans)

        self.current_step += 1
        done = torch.zeros(B, dtype=torch.bool, device=device)
        if self.current_step >= self.recommendation_length:
            done[:] = True

        return self._make_state_from_last(), step_reward, done

    def _step_reward(self, chosen_item_ids, pred_last, yt_last):
        """
        即时奖励（按旧 state 计算）：
          preference: pred_last[item]
          difficulty match: 1 - | ability(item) - diff_norm(item) |
          diversity: repeat penalty
        """
        B = chosen_item_ids.size(0)
        device = chosen_item_ids.device
        r = torch.zeros(B, device=device)

        # weights (保持你原来口径，可在外部调)
        w_pref = 0.10
        w_diff = 0.10
        w_div = 0.05

        diff_norm = self._diff_norm(chosen_item_ids)  # [B]

        for i in range(B):
            item = int(chosen_item_ids[i].item())

            # preference
            if 0 <= item < pred_last.size(1):
                pref = pred_last[i, item]
                r[i] += w_pref * pref
                self._pref_hist[i].append(float(pref.item()))
            else:
                self._pref_hist[i].append(0.0)

            # diversity: repeat penalty
            is_repeat = 1 if item in self.paths[i][:-1] else 0
            self._repeat_hist[i].append(is_repeat)
            if is_repeat:
                r[i] -= w_div

            # difficulty match (ability is yt probability on this item)
            if 0 <= item < yt_last.size(1):
                ability = yt_last[i, item]
                r[i] += w_diff * (1.0 - torch.abs(ability - diff_norm[i]))

        return r

    # ---------------- final metrics ----------------

    def _pairwise_diversity(self, emb: torch.Tensor) -> torch.Tensor:
        """emb: [M,D]; return mean(1 - cosine_sim) over unordered pairs (upper triangle excl diag)."""
        M = emb.size(0)
        if M < 2:
            return torch.tensor(0.0, device=emb.device)
        emb = F.normalize(emb, p=2, dim=-1)
        sim = emb @ emb.t()  # [M,M], cosine sim in [-1,1]
        idx = torch.triu_indices(M, M, offset=1, device=emb.device)
        sim_pairs = sim[idx[0], idx[1]]  # [M*(M-1)/2]
        div = (1.0 - sim_pairs).mean()
        return div

    def _path_level_adaptivity(self) -> torch.Tensor:
        """
        per-sample adaptivity:
          ability_path = mean( yt_before[item] for item in path )
          diff_path = mean( diff_norm(item) for item in path )
          score = 1 - |ability_path - diff_path|
        """
        B = self.batch_size
        device = self.original_tgt.device
        out = torch.zeros(B, device=device)
        if self._yt_before is None:
            return out

        for i in range(B):
            path = [int(x) for x in self.paths[i] if int(x) > 1]
            if len(path) == 0:
                continue
            items = torch.tensor(path, device=device, dtype=torch.long)
            items = items.clamp(0, self._yt_before.size(1) - 1)
            ability = self._yt_before[i].gather(0, items).mean()
            diff = self._diff_norm(items).mean()
            out[i] = 1.0 - torch.abs(ability - diff)
        return out

    @torch.no_grad()
    def _simulate_yt_after(self, full_seq: torch.Tensor, full_ans: torch.Tensor) -> torch.Tensor:
        """
        Run KTOnlyModel to obtain yt_all: [B, L, N]
        """
        if self.kt_model is None:
            # fallback: use last cached yt
            return self._last_yt_full
        return self.kt_model(full_seq, full_ans, self.graph)

    def compute_final_metrics(self) -> Dict[str, torch.Tensor]:
        """Compute per-sample metrics on the *full* trajectory (history + recommended path).

        IMPORTANT (matches user's original setup):
        - topk_indices exists at EVERY time step (t = 0..L_full-2). For the history part we use
          base-model top-k; for the appended RL steps we overwrite with policy top-k.
        - preference follows Metrics.combined_metrics definition: mean predicted probability of the recommended items.
        - diversity uses 1 - cosine similarity on embeddings from the base model's graph hidden.
        """
        if self.original_tgt is None:
            z = torch.zeros(self.batch_size)
            return {"effectiveness": z, "adaptivity": z, "diversity": z, "preference": z, "final_quality": z}

        device = self.original_tgt.device
        B = int(self.batch_size)

        # -------------------------
        # 1) Build full sequence: history + path
        # -------------------------
        ext_len = len(self.paths[0]) if (self.paths is not None and len(self.paths) > 0) else 0
        if ext_len <= 0:
            z = torch.zeros(B, device=device)
            return {"effectiveness": z, "adaptivity": z, "diversity": z, "preference": z, "final_quality": z}

        path_tensor = torch.tensor(self.paths, device=device, dtype=self.original_tgt.dtype)  # [B, ext_len]
        full_seq = torch.cat([self.original_tgt, path_tensor], dim=1)  # [B, L_hist + ext_len]
        L_hist = int(self.original_tgt.size(1))
        L_full = int(full_seq.size(1))

        # -------------------------
        # 2) Build full answers: history ans + simulated answers for appended items
        # -------------------------
        full_ans = torch.zeros_like(full_seq, dtype=torch.float32, device=device)
        full_ans[:, :L_hist] = self.original_ans.float()

        # Simulated answers for appended items: use yt_before (history last state) probability threshold 0.5
        yt_hist_all = self._kt_forward(self.original_tgt, self.original_ans)  # [B, L_hist, num_skills] (yt_all)
        if yt_hist_all.dim() == 3 and yt_hist_all.size(1) == L_hist:
            yt_hist = yt_hist_all[:, :-1, :]  # [B, L_hist-1, num_skills]
        else:
            # fallback: treat as already [B, L_hist-1, K]
            yt_hist = yt_hist_all

        last_state = yt_hist[:, -1, :]  # [B, num_skills]
        for t in range(ext_len):
            items = path_tensor[:, t].long()  # [B]
            prob = last_state.gather(1, items.unsqueeze(1)).squeeze(1)  # [B]
            full_ans[:, L_hist + t] = (prob > 0.5).float()

        # -------------------------
        # 3) Get yt_before_all and yt_after_all aligned to L_full-1
        # -------------------------
        yt_before_all = torch.cat([yt_hist, last_state.unsqueeze(1).repeat(1, ext_len, 1)], dim=1)  # [B, L_full-1, K]

        yt_after_all = self._kt_forward(full_seq, full_ans)  # yt_all: [B, L_full, K]
        if yt_after_all.size(1) == L_full:
            yt_after = yt_after_all[:, :-1, :]  # [B, L_full-1, K]
        else:
            yt_after = yt_after_all
            if yt_after.size(1) > (L_full - 1):
                yt_after = yt_after[:, :L_full - 1, :]

        # -------------------------
        # 4) Build topk_indices for EVERY time step (history: base top-k; appended: policy top-k)
        # -------------------------
        # Forward base model on full_seq to obtain pred logits for each time step (flattened)
        pred_flat, _, kt_mask, _, hidden, _ = self._base_forward(full_seq, full_ans)  # pred_flat: [B*T, N]
        # Infer T from pred_flat length; we expect T == L_full-1, but be robust.
        N = int(pred_flat.size(-1))
        T = int(pred_flat.size(0) // B)
        pred_logits = pred_flat.view(B, T, N)  # [B, T, N]
        pred_probs = torch.softmax(pred_logits, dim=-1)

        # Create a dense [B, L_full-1, N] by placing steps according to kt_mask when needed.
        # In your DKT, kt_mask = (questions[:, 1:] >= 2).float(), so it is [B, L_full-1].
        # If T == L_full-1 we can use directly; otherwise we scatter by mask order.
        if T == (L_full - 1):
            pred_probs_full = pred_probs  # [B, L_full-1, N]
        else:
            pred_probs_full = torch.zeros(B, L_full - 1, N, device=device)
            if kt_mask is not None and kt_mask.dim() == 2 and kt_mask.size(1) == (L_full - 1):
                # Fill valid positions sequentially from pred_probs' time axis.
                for b in range(B):
                    idx = (kt_mask[b] > 0).nonzero(as_tuple=False).squeeze(1).tolist()
                    take = min(len(idx), T)
                    if take > 0:
                        pred_probs_full[b, idx[:take], :] = pred_probs[b, :take, :]
            else:
                # Last resort: align to the tail
                take = min(T, L_full - 1)
                pred_probs_full[:, :take, :] = pred_probs[:, :take, :]

        K = int(self.metrics_topnum)
        base_topk = pred_probs_full.topk(K, dim=-1).indices.long()  # [B, L_full-1, K]
        topk_indices = base_topk.clone()

        # Overwrite appended steps with POLICY top-k (candidate-space policy distribution top-k already stored in self.topk_recs)
        # Mapping: recommendation step r=0 predicts item at position L_hist (t = L_hist-1 in metrics loop).
        for b in range(B):
            for r, recs in enumerate(self.topk_recs[b]):
                pos = (L_hist - 1) + r
                if 0 <= pos < (L_full - 1):
                    rr = recs[:K] + [0] * max(0, K - len(recs))
                    topk_indices[b, pos, :] = torch.tensor(rr, device=device, dtype=torch.long)

        # -------------------------
        # 5) Effectiveness (per sample) using Metrics.compute_effectiveness_tensor (preferred) to avoid global mean
        # -------------------------
        try:
            gain_tensor, _ = self.metric.compute_effectiveness_tensor(full_seq, yt_before_all, yt_after, topk_indices)
            # per-sample effectiveness: average non-zero gains over time & K
            eff = torch.zeros(B, device=device)
            for b in range(B):
                g = gain_tensor[b]  # [L_full-1, K]
                mask_g = (g != 0)
                if mask_g.any():
                    eff[b] = g[mask_g].mean()
        except Exception:
            eff = torch.zeros(B, device=device)

        # -------------------------
        # 6) Preference (per sample) - EXACTLY like Metrics.combined_metrics but per sample (no global mean)
        # -------------------------
        pref = torch.zeros(B, device=device)
        for b in range(B):
            total = 0.0
            cnt = 0
            for t in range(L_full - 1):
                if int(full_seq[b, t].item()) == 0:
                    continue
                recs = topk_indices[b, t].tolist()
                for r in recs:
                    if r != 0:
                        total += float(pred_probs_full[b, t, r].item()) if r < N else 0.0
                        cnt += 1
            pref[b] = (total / cnt) if cnt > 0 else 0.0

        # -------------------------
        # 7) Adaptivity (per sample) - path-level: ability from yt_before on related items; difficulty from file (1/2/3 -> 0/0.5/1)
        # -------------------------
        ada = torch.zeros(B, device=device)
        for b in range(B):
            items = self.paths[b]
            if len(items) == 0:
                continue
            # ability_path = mean(yt_before_last[item])
            ab = []
            df = []
            for it in items:
                if it == 0:
                    continue
                ab.append(float(last_state[b, it].item()) if it < last_state.size(1) else 0.0)
                df.append(float(self._diff_norm(it)))
            if len(ab) > 0 and len(df) > 0:
                ability = sum(ab) / len(ab)
                diffm = sum(df) / len(df)
                ada[b] = 1.0 - abs(ability - diffm)

        # -------------------------
        # 8) Diversity (per sample) - 1 - cosine similarity on base-model hidden embeddings (graph hidden)
        # -------------------------
        div = torch.zeros(B, device=device)
        if hidden is not None and hidden.dim() == 2:
            # hidden: [num_items, dim]
            for b in range(B):
                items = [it for it in self.paths[b] if it != 0 and it < hidden.size(0)]
                M = len(items)
                if M < 2:
                    div[b] = 0.0
                    continue
                emb = hidden[torch.tensor(items, device=device, dtype=torch.long)]  # [M, D]
                emb = torch.nn.functional.normalize(emb, dim=-1)
                sim = emb @ emb.t()  # [M, M]
                idx = torch.triu_indices(M, M, offset=1, device=device)
                sim_pairs = sim[idx[0], idx[1]]
                div[b] = (1.0 - sim_pairs).mean()

        # -------------------------
        # 9) Final quality (per sample)
        # -------------------------
        final_quality = (
                self.w_eff * eff +
                self.w_ada * ada +
                self.w_div * div +
                self.w_pre * pref
        )

        return {
            "effectiveness": eff,
            "adaptivity": ada,
            "diversity": div,
            "preference": pref,
            "final_quality": final_quality
        }

    def compute_final_reward(self) -> torch.Tensor:
        """Final scalar reward per sample (used by trainer)."""
        return self.compute_final_metrics()["final_quality"]


class PPOTrainer:
    """
    轻量 PG (REINFORCE)：
    - 每条轨迹 T 步收集 rewards/log_probs/entropy
    - 最后一步 rewards += final_reward
    - returns = discounted sum
    """

    def __init__(self, policy_net: PolicyNetwork, env: LearningPathEnv, lr=3e-4, gamma=0.99, entropy_coef=1e-4):
        self.policy_net = policy_net
        self.env = env
        self.gamma = float(gamma)
        self.entropy_coef = float(entropy_coef)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

    def collect_trajectory(self, tgt, tgt_timestamp, tgt_idx, ans, graph=None, hypergraph_list=None,
                           deterministic: bool = False):
        state = self.env.reset(tgt, tgt_timestamp, tgt_idx, ans, graph=graph, hypergraph_list=hypergraph_list)

        rewards, log_probs, entropies = [], [], []
        B = tgt.size(0)
        device = tgt.device

        for _ in range(self.env.recommendation_length):
            cand_probs = state["cand_probs"]  # [B,K]
            cand_ids = state["cand_ids"]  # [B,K]
            cand_feat = cand_probs.unsqueeze(-1)  # [B,K,1]

            logits = self.policy_net(state["knowledge_state"], cand_feat)  # [B,K]
            dist = Categorical(logits=logits)

            if deterministic:
                action_index = torch.argmax(logits, dim=-1)
            else:
                action_index = dist.sample()

            log_prob = dist.log_prob(action_index)
            entropy = dist.entropy()

            chosen_item = cand_ids.gather(1, action_index.view(-1, 1)).squeeze(1)  # [B]

            # policy-distribution top-k within candidate space (用于 Metrics 顶层指标计算)
            topn = min(self.env.metrics_topnum, cand_ids.size(1))
            topn_idx = torch.topk(logits, k=topn, dim=-1).indices
            step_topk = cand_ids.gather(1, topn_idx)

            next_state, step_reward, done = self.env.step(chosen_item, step_topk)

            rewards.append(step_reward)
            log_probs.append(log_prob)
            entropies.append(entropy)

            state = next_state
            if torch.all(done):
                break

        final_reward = self.env.compute_final_reward()  # [B]
        rewards[-1] = rewards[-1] + final_reward

        return rewards, log_probs, entropies, final_reward

    def update_policy(self, rewards, log_probs, entropies):
        rewards_t = torch.stack(rewards, dim=0)  # [T,B]
        logp_t = torch.stack(log_probs, dim=0)  # [T,B]
        ent_t = torch.stack(entropies, dim=0)  # [T,B]

        T = rewards_t.size(0)
        returns = torch.zeros_like(rewards_t)
        running = torch.zeros_like(rewards_t[0])
        for t in reversed(range(T)):
            running = rewards_t[t] + self.gamma * running
            returns[t] = running

        baseline = returns.mean(dim=0, keepdim=True)
        adv = returns - baseline

        adv_std = adv.detach().std(dim=0, keepdim=True).clamp_min(1e-6)
        adv = adv / adv_std

        pg_loss = -(logp_t * adv.detach()).mean()
        ent_loss = -ent_t.mean()
        loss = pg_loss + self.entropy_coef * ent_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        print(
            f"pg_loss={pg_loss.item():.6f}  entropy={ent_t.mean().item():.6f}  adv_std={adv.detach().std().item():.6f}")
        return float(loss.item())


class RLPathOptimizer:
    """
    兼容旧接口：封装 env + policy + trainer
    """

    def __init__(
            self,
            pretrained_model,
            num_skills: int,
            batch_size: int,
            recommendation_length: int,
            topk: int,
            data_name: str,
            graph=None,
            hypergraph_list=None,
            hidden_dim=128,
            lr=3e-4,
            gamma=0.99,
            entropy_coef=1e-4,
            metrics_topnum: Optional[int] = None,
            metrics_T: int = 5,
            device=None,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        if metrics_topnum is None:
            metrics_topnum = topk
        else:
            metrics_topnum = min(int(metrics_topnum), int(topk))

        self.env = LearningPathEnv(
            batch_size=batch_size,
            base_model=pretrained_model,
            recommendation_length=recommendation_length,
            data_name=data_name,
            graph=graph,
            hypergraph_list=hypergraph_list,
            policy_topk=topk,
            metrics_topnum=metrics_topnum,
            metrics_T=metrics_T,
        )

        self.policy_net = PolicyNetwork(
            knowledge_dim=num_skills,
            candidate_feature_dim=1,
            hidden_dim=hidden_dim,
            topk=topk,
        ).to(device)

        self.trainer = PPOTrainer(
            policy_net=self.policy_net,
            env=self.env,
            lr=lr,
            gamma=gamma,
            entropy_coef=entropy_coef,
        )


@torch.no_grad()
def evaluate_policy(env: LearningPathEnv, policy_net: PolicyNetwork, data_loader, relation_graph=None,
                    hypergraph_list=None, num_episodes: int = 5):
    """
    用确定性策略评估：返回 batch-mean 的 validity(effectiveness)/diversity/adaptivity
    """
    effs, divs, adas, prefs, fqs = [], [], [], [], []
    device = next(policy_net.parameters()).device

    for _ in range(num_episodes):
        batch_count = 0
        for batch in data_loader:
            if batch_count >= 1:
                break
            tgt, tgt_timestamp, tgt_idx, ans = batch
            tgt = tgt.to(device);
            tgt_timestamp = tgt_timestamp.to(device);
            tgt_idx = tgt_idx.to(device);
            ans = ans.to(device)

            state = env.reset(tgt, tgt_timestamp, tgt_idx, ans, graph=relation_graph, hypergraph_list=hypergraph_list)
            for _ in range(env.recommendation_length):
                cand_probs = state["cand_probs"]
                cand_ids = state["cand_ids"]
                logits = policy_net(state["knowledge_state"], cand_probs.unsqueeze(-1))
                action_index = torch.argmax(logits, dim=-1)
                chosen_item = cand_ids.gather(1, action_index.view(-1, 1)).squeeze(1)
                topn = min(env.metrics_topnum, cand_ids.size(1))
                topn_idx = torch.topk(logits, k=topn, dim=-1).indices
                step_topk = cand_ids.gather(1, topn_idx)
                state, _, done = env.step(chosen_item, step_topk)
                if torch.all(done):
                    break

            m = env.compute_final_metrics()
            effs.append(m["effectiveness"].mean().item())
            divs.append(m["diversity"].mean().item())
            adas.append(m["adaptivity"].mean().item())
            prefs.append(m["preference"].mean().item())
            fqs.append(m["final_quality"].mean().item())

            batch_count += 1

    return float(sum(effs) / len(effs)), float(sum(divs) / len(divs)), float(sum(adas) / len(adas)), float(
        sum(prefs) / len(prefs)), float(sum(fqs) / len(fqs))
