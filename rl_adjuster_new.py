
# -*- coding: utf-8 -*-
"""
rl_adjuster_new.py

修正版要点（按你确认的口径）：
1) Metrics 口径：历史每个时间步都有 topk
   - history: 使用 base model (MSHGAT) 在全资源 softmax 后的 top-k
   - appended(RL): 使用 policy 分布 top-k 覆盖对应时间步
2) step reward 的 preference / difficulty：使用 step 前旧 state（旧 forward）计算
3) 不修改 Metrics.py：在本文件内实现 per-sample wrapper（避免 global mean / 除零）
4) diversity：严格 1 - cosine，相同公式；只对真实 pair 统计（避免下三角 0 误计）
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import Constants
from Metrics import Metrics

try:
    from HGAT import KTOnlyModel  # 你的 HGAT.py 里定义了 KTOnlyModel
except Exception:
    KTOnlyModel = None


def _ensure_2d(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(0) if x.dim() == 1 else x


def _pad_and_concat_like(original_2d: torch.Tensor, ext_len: int, pad_value: int = 0) -> torch.Tensor:
    """original_2d: [B,L] -> [B,L+ext_len]"""
    original_2d = _ensure_2d(original_2d)
    if ext_len <= 0:
        return original_2d
    B = original_2d.size(0)
    pad = torch.full((B, ext_len), pad_value, device=original_2d.device, dtype=original_2d.dtype)
    return torch.cat([original_2d, pad], dim=1)


class PolicyNetwork(nn.Module):
    """
    输入：候选集合的特征（例如 base 模型 embedding/状态拼接后的向量）
    输出：候选集合上的 logits（用于 Categorical）
    """
    def __init__(self, in_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # per-candidate logit
        )

    def forward(self, cand_feat: torch.Tensor) -> torch.Tensor:
        """
        cand_feat: [B, Kcand, Din]
        return logits: [B, Kcand]
        """
        B, K, D = cand_feat.shape
        x = self.mlp(cand_feat).view(B, K)
        return x


class LearningPathEnv:
    """
    环境封装：
    - reset() 传入一个 batch 的历史序列与图结构
    - step(action_idx_in_candidate) 扩展路径一步
    - 记录：
        self.paths[b] : RL 选中的 item id 序列
        self.topk_recs[b][t] : 每一步 policy 分布 top-k 的 item id（动作空间内映射到全局 id）
    - 最终：
        compute_final_metrics()：按样本输出 effectiveness/adaptivity/diversity/preference/final_quality
    """
    def __init__(
        self,
        pretrained_model: nn.Module,
        num_skills: int,
        recommendation_length: int = 5,
        policy_topk: int = 20,
        metrics_topnum: int = 20,
        final_weights: Optional[Dict[str, float]] = None,
        step_weights: Optional[Dict[str, float]] = None,
        device: Optional[torch.device] = None,
    ):
        self.base_model = pretrained_model
        self.num_skills = int(num_skills)
        self.recommendation_length = int(recommendation_length)
        self.policy_topk = int(policy_topk)
        self.metrics_topnum = int(metrics_topnum)
        self.final_weights = final_weights or {"effectiveness": 0.4, "adaptivity": 0.3, "diversity": 0.2, "preference": 0.1}
        self.step_weights = step_weights or {"preference": 0.5, "difficulty": 0.5}
        self.device = device or next(pretrained_model.parameters()).device

        # Metrics evaluator (不改动 Metrics.py)
        self.metric = Metrics()

        # KTOnlyModel for effectiveness (不改动 Metrics.py)
        self.kt_model = KTOnlyModel(pretrained_model).to(self.device) if KTOnlyModel is not None else None
        if self.kt_model is not None:
            self.kt_model.eval()

        # runtime buffers
        self.graph = None
        self.hypergraph_list = None

        self.original_tgt = None
        self.original_tgt_timestamp = None
        self.original_tgt_idx = None
        self.original_ans = None

        self.batch_size = 0
        self.t = 0

        # cached from last base forward on current extended seq
        self._last_pred_probs_full: Optional[torch.Tensor] = None  # [B, T, N]
        self._last_yt_full: Optional[torch.Tensor] = None          # [B, T, S]
        self._last_hidden: Optional[torch.Tensor] = None           # [N, d] or [B,...] depending on model

        # step-time caches (old state)
        self._old_pred_last: Optional[torch.Tensor] = None  # [B, N]
        self._old_yt_last: Optional[torch.Tensor] = None    # [B, S]

        # episode records
        self.paths: List[List[int]] = []
        self.topk_recs: List[List[List[int]]] = []
        self.chosen_actions: List[List[int]] = []

        # difficulty mapping (1/2/3) -> normalized to [0,1]
        # 如果你后面要从文件读取真实难度，可在这里替换
        self.item_difficulty = None  # Optional[torch.Tensor] shape [N] values in {1,2,3}

    @torch.no_grad()
    def _forward_base(self, seq: torch.Tensor, ts: torch.Tensor, idx: torch.Tensor, ans: torch.Tensor):
        """
        MSHGAT(seq, ts, idx, ans, graph, hypergraph_list) -> (pred_flat, pred_res, kt_mask, yt, hidden, status_emb)
        我们只依赖：
          pred_flat: [B*T, N] logits(+mask)
          yt:        [B, T, S] or [B, T, S]（来自 ktmodel）
          hidden:    item embedding [N, d]
        """
        self.base_model.eval()
        out = self.base_model(seq, ts, idx, ans, self.graph, self.hypergraph_list)

        # 兼容返回值数量不同
        if isinstance(out, (tuple, list)):
            pred_flat = out[0]
            yt = out[3] if len(out) > 3 else None
            hidden = out[4] if len(out) > 4 else None
        else:
            raise RuntimeError("base_model forward must return tuple/list")

        B = seq.size(0)
        T = pred_flat.size(0) // B
        pred_logits = pred_flat.view(B, T, -1)               # [B,T,N]
        pred_probs = F.softmax(pred_logits, dim=-1)          # [B,T,N]

        self._last_pred_probs_full = pred_probs
        self._last_yt_full = yt
        self._last_hidden = hidden

        return pred_probs, yt, hidden

    @torch.no_grad()
    def _kt_forward(self, seq: torch.Tensor, ans: torch.Tensor) -> torch.Tensor:
        """
        返回 yt_all: [B, (L-1), S]，与 Metrics.compute_effectiveness 口径一致
        """
        if self.kt_model is None:
            # fallback：如果没法单独跑 KT，就用 base forward 的 yt（需已 forward）
            if self._last_yt_full is None:
                raise AttributeError("kt_model is None and _last_yt_full is None")
            return self._last_yt_full
        yt_all = self.kt_model(seq, ans, self.graph)  # HGAT.KTOnlyModel: yt_all
        return yt_all

    def reset(self, tgt, tgt_timestamp, tgt_idx, ans, graph=None, hypergraph_list=None) -> Dict[str, torch.Tensor]:
        self.graph = graph
        self.hypergraph_list = hypergraph_list

        self.original_tgt = _ensure_2d(tgt).to(self.device)
        self.original_tgt_timestamp = _ensure_2d(tgt_timestamp).to(self.device)
        self.original_tgt_idx = _ensure_2d(tgt_idx).to(self.device)
        self.original_ans = _ensure_2d(ans).to(self.device)

        self.batch_size = self.original_tgt.size(0)
        self.t = 0

        self.paths = [[] for _ in range(self.batch_size)]
        self.topk_recs = [[] for _ in range(self.batch_size)]
        self.chosen_actions = [[] for _ in range(self.batch_size)]

        # base forward on history (初始化 old state)
        pred_probs, yt, hidden = self._forward_base(self.original_tgt, self.original_tgt_timestamp, self.original_tgt_idx, self.original_ans)

        # old state is last time-step of history prediction horizon
        self._old_pred_last = pred_probs[:, -1, :]           # [B,N]
        self._old_yt_last = yt[:, -1, :] if yt is not None else None

        return self._build_state()

    def _build_state(self) -> Dict[str, torch.Tensor]:
        """
        state for policy:
          - candidate_ids: [B, Kcand]
          - candidate_features: [B, Kcand, Din]
        """
        assert self._old_pred_last is not None
        B = self.batch_size
        Kcand = self.policy_topk

        # candidates from old_pred_last (base model distribution over ALL resources)
        cand_probs, cand_ids = torch.topk(self._old_pred_last, k=Kcand, dim=-1)  # [B,Kcand]
        # candidate features: [prob, (optional) ability_on_item]
        if self._old_yt_last is not None:
            ability = self._old_yt_last.gather(1, cand_ids.clamp(0, self._old_yt_last.size(1)-1))
            cand_feat = torch.stack([cand_probs, ability], dim=-1)  # [B,K,2]
        else:
            cand_feat = cand_probs.unsqueeze(-1)  # [B,K,1]

        return {
            "candidate_ids": cand_ids,
            "candidate_features": cand_feat,
        }

    def _diff_norm(self, item_ids: torch.Tensor) -> torch.Tensor:
        """
        item difficulty in {1,2,3} -> [0,1]
        若未提供真实难度，则用 2 作为中等难度占位
        """
        if self.item_difficulty is None:
            d = torch.full_like(item_ids, 2, dtype=torch.float32)
        else:
            d = self.item_difficulty[item_ids.clamp(0, self.item_difficulty.numel()-1)].float()
        return (d - 1.0) / 2.0

    def _step_reward(self, chosen_global_ids: torch.Tensor, old_pred_last: torch.Tensor, old_yt_last: Optional[torch.Tensor]) -> torch.Tensor:
        """
        即时奖励：基于旧 state（旧 forward）
        - preference: old_pred_last[b, chosen]
        - difficulty: 1 - | ability - diff |
        """
        device = old_pred_last.device
        B = old_pred_last.size(0)

        # preference (full softmax prob)
        pref = old_pred_last.gather(1, chosen_global_ids.view(B, 1)).squeeze(1).clamp_min(0.0)

        # difficulty match
        if old_yt_last is None:
            diff_reward = torch.zeros(B, device=device)
        else:
            ability = old_yt_last.gather(1, chosen_global_ids.view(B, 1)).squeeze(1)  # yt is probability
            diff = self._diff_norm(chosen_global_ids)
            diff_reward = (1.0 - (ability - diff).abs()).clamp(0.0, 1.0)

        w_p = float(self.step_weights.get("preference", 0.5))
        w_d = float(self.step_weights.get("difficulty", 0.5))
        return w_p * pref + w_d * diff_reward

    def step(self, action_idx: torch.Tensor, policy_topk_items: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, bool]:
        """
        action_idx: [B] index in candidate set
        policy_topk_items: [B, K] global item ids for this step (策略分布 top-k)
        """
        action_idx = action_idx.to(self.device).long()
        policy_topk_items = policy_topk_items.to(self.device).long()

        state = self._build_state()
        cand_ids = state["candidate_ids"]  # [B,Kcand]
        B = cand_ids.size(0)

        # map to global id
        chosen_global = cand_ids.gather(1, action_idx.view(B, 1)).squeeze(1)  # [B]

        # record
        for b in range(B):
            self.paths[b].append(int(chosen_global[b].item()))
            self.topk_recs[b].append([int(x) for x in policy_topk_items[b].tolist()])
            self.chosen_actions[b].append(int(action_idx[b].item()))

        # step reward uses OLD state cached
        assert self._old_pred_last is not None
        step_reward = self._step_reward(chosen_global, self._old_pred_last, self._old_yt_last)

        # extend inputs
        ext_len = len(self.paths[0])
        path_tensor = torch.tensor(self.paths, device=self.device, dtype=self.original_tgt.dtype)  # [B, ext_len]
        ext_seq = torch.cat([self.original_tgt, path_tensor], dim=1)

        ext_ts = _pad_and_concat_like(self.original_tgt_timestamp, ext_len, pad_value=0)
        ext_idx = _pad_and_concat_like(self.original_tgt_idx, ext_len, pad_value=0)

        # simulate answers for appended part using old_yt_last on chosen item (simple)
        pad_ans = torch.zeros(B, ext_len, device=self.device, dtype=self.original_ans.dtype)
        if self._old_yt_last is not None:
            prob = self._old_yt_last.gather(1, chosen_global.view(B, 1)).squeeze(1)
            pad_ans[:, -1] = (prob > 0.5).long().to(pad_ans.dtype)
        ext_ans = torch.cat([self.original_ans, pad_ans], dim=1)

        # base forward to update state for next step
        pred_probs, yt, _ = self._forward_base(ext_seq, ext_ts, ext_idx, ext_ans)
        self._old_pred_last = pred_probs[:, -1, :]
        self._old_yt_last = yt[:, -1, :] if yt is not None else None

        self.t += 1
        done = (self.t >= self.recommendation_length)
        return self._build_state(), step_reward, done

    # ---------- per-sample wrappers (对齐 Metrics 口径，但避免 batch mean) ----------

    @torch.no_grad()
    def _per_sample_effectiveness(self, original_seqs: torch.Tensor, yt_before: torch.Tensor, yt_after: torch.Tensor, topk_indices: torch.Tensor) -> torch.Tensor:
        """
        按 Metrics.compute_effectiveness 的公式逐样本计算：
        - 对每个 t，取 valid_rec
        - gain_step = mean( yt_after - yt_before on valid_rec )
        - effectiveness = mean over valid t
        """
        B, T, K = topk_indices.shape
        eff = torch.zeros(B, device=topk_indices.device)
        for b in range(B):
            total = 0.0
            cnt = 0
            for t in range(T):
                if int(original_seqs[b, t].item()) == self.metric.PAD:
                    continue
                recs = topk_indices[b, t].tolist()
                valid = [r for r in recs if 0 <= r < yt_before.size(-1)]
                if not valid:
                    continue
                gain = 0.0
                for r in valid:
                    gain += float(yt_after[b, t, r].item() - yt_before[b, t, r].item())
                total += gain / max(1, len(valid))
                cnt += 1
            eff[b] = total / max(1, cnt)
        return eff

    @torch.no_grad()
    def _per_sample_preference(self, pred_probs: torch.Tensor, original_seqs: torch.Tensor, topk_indices: torch.Tensor) -> torch.Tensor:
        """
        对齐 Metrics.combined_metrics 的 preference：
          per step: mean(pred_probs[t][r] for r in valid_rec)
          then mean over valid steps
        pred_probs: [B, T, N]
        topk_indices: [B, T, K]
        """
        B, T, K = topk_indices.shape
        pref = torch.zeros(B, device=pred_probs.device)
        for b in range(B):
            total = 0.0
            cnt = 0
            for t in range(T):
                if int(original_seqs[b, t].item()) == self.metric.PAD:
                    continue
                recs = topk_indices[b, t].tolist()
                valid = [r for r in recs if 0 <= r < pred_probs.size(-1)]
                if not valid:
                    continue
                psum = 0.0
                for r in valid:
                    psum += float(pred_probs[b, t, r].item())
                total += psum / max(1, len(valid))
                cnt += 1
            pref[b] = total / max(1, cnt)
        return pref

    @torch.no_grad()
    def _per_sample_diversity(self, hidden_item: torch.Tensor, original_seqs: torch.Tensor, topk_indices: torch.Tensor) -> torch.Tensor:
        """
        对齐 Metrics 的 diversity（同公式）：
          per step: mean_{i<j}(1 - cos(e_i, e_j))
          then mean over valid steps
        hidden_item: [N, d]
        """
        B, T, K = topk_indices.shape
        div = torch.zeros(B, device=topk_indices.device)
        for b in range(B):
            total = 0.0
            cnt = 0
            for t in range(T):
                if int(original_seqs[b, t].item()) == self.metric.PAD:
                    continue
                recs = topk_indices[b, t].tolist()
                valid = [r for r in recs if 0 <= r < hidden_item.size(0)]
                if len(valid) < 2:
                    continue
                emb = hidden_item[torch.tensor(valid, device=hidden_item.device)]
                emb = F.normalize(emb, dim=-1)
                sim = emb @ emb.t()
                idx = torch.triu_indices(sim.size(0), sim.size(1), offset=1, device=sim.device)
                sim_pairs = sim[idx[0], idx[1]]
                d = (1.0 - sim_pairs).mean().item()
                total += d
                cnt += 1
            div[b] = total / max(1, cnt)
        return div

    @torch.no_grad()
    def _per_sample_adaptivity(self, original_seqs: torch.Tensor, yt_before: torch.Tensor, topk_indices: torch.Tensor) -> torch.Tensor:
        """
        你确认的 A 方案（path-level / step-level 都可），这里按 Metrics style：
          ability_t = mean( yt_before[b,t, r] ) over rec
          diff_t    = mean( diff_norm(r) ) over rec
          ada_step  = 1 - |ability_t - diff_t|
          adaptivity = mean over valid steps
        """
        B, T, K = topk_indices.shape
        ada = torch.zeros(B, device=topk_indices.device)
        for b in range(B):
            total = 0.0
            cnt = 0
            for t in range(T):
                if int(original_seqs[b, t].item()) == self.metric.PAD:
                    continue
                recs = topk_indices[b, t].tolist()
                valid = [r for r in recs if 0 <= r < yt_before.size(-1)]
                if not valid:
                    continue
                ab = 0.0
                df = 0.0
                for r in valid:
                    ab += float(yt_before[b, t, r].item())
                    df += float(self._diff_norm(torch.tensor(r, device=topk_indices.device)).item())
                ab /= max(1, len(valid))
                df /= max(1, len(valid))
                total += max(0.0, 1.0 - abs(ab - df))
                cnt += 1
            ada[b] = total / max(1, cnt)
        return ada

    @torch.no_grad()
    def compute_final_metrics(self) -> Dict[str, torch.Tensor]:
        if self.original_tgt is None:
            z = torch.zeros(self.batch_size, device=self.device)
            return {"effectiveness": z, "adaptivity": z, "diversity": z, "preference": z, "final_quality": z}

        device = self.device
        B = self.batch_size
        ext_len = len(self.paths[0])

        # full sequence
        if ext_len > 0:
            path_tensor = torch.tensor(self.paths, device=device, dtype=self.original_tgt.dtype)  # [B,ext]
            full_seq = torch.cat([self.original_tgt, path_tensor], dim=1)
        else:
            full_seq = self.original_tgt

        L_hist = self.original_tgt.size(1)
        L_full = full_seq.size(1)
        T_full = L_full - 1

        # full answers: history + simulated (simple, based on yt_before last state for that appended item)
        full_ans = self.original_ans
        if ext_len > 0:
            sim_ans = torch.zeros(B, ext_len, device=device, dtype=self.original_ans.dtype)
            # 用当前 episode 中每一步选中 item 对应的旧能力来模拟（简单版本）
            # 这里不影响“口径正确性”，只影响数值质量；后续可升级为滚动模拟
            if self._last_yt_full is not None:
                last_hist_yt = self._kt_forward(self.original_tgt, self.original_ans)[:, -1, :]  # [B,S]
                for t in range(ext_len):
                    items_t = path_tensor[:, t].long()
                    prob = last_hist_yt.gather(1, items_t.view(B, 1)).squeeze(1)
                    sim_ans[:, t] = (prob > 0.5).long().to(sim_ans.dtype)
            full_ans = torch.cat([self.original_ans, sim_ans], dim=1)

        # yt_after on full_seq
        yt_after = self._kt_forward(full_seq, full_ans)  # [B, T_full, S]  (KTOnlyModel: seq_len-1)
        yt_hist = self._kt_forward(self.original_tgt, self.original_ans)  # [B, L_hist-1, S]

        # yt_before_all: extend history last state to full length-1
        if yt_hist.size(1) < T_full:
            pad_steps = T_full - yt_hist.size(1)
            last = yt_hist[:, -1:, :].repeat(1, pad_steps, 1)
            yt_before = torch.cat([yt_hist, last], dim=1)
        else:
            yt_before = yt_hist[:, :T_full, :]

        # ensure pred_probs for full_seq exists (after last step it should already be for full_seq)
        if self._last_pred_probs_full is None or self._last_pred_probs_full.size(1) != T_full:
            # build dummy ts/idx for appended
            ext_ts = _pad_and_concat_like(self.original_tgt_timestamp, ext_len, pad_value=0)
            ext_idx = _pad_and_concat_like(self.original_tgt_idx, ext_len, pad_value=0)
            self._forward_base(full_seq, ext_ts, ext_idx, full_ans)

        pred_probs = self._last_pred_probs_full  # [B,T_full,N]
        hidden_item = self._last_hidden  # [N,d]

        K = self.metrics_topnum
        # history topk from base model for ALL steps
        pad_id = int(getattr(Constants, "PAD", 0))
        pred_probs = pred_probs.clone()
        pred_probs[..., pad_id] = -1e9  # 或者 0.0 后再用 topk 前改成 -inf；这里用 -1e9 最稳

        _, base_topk = torch.topk(pred_probs, k=K, dim=-1)  # [B,T_full,K]
        topk_indices = base_topk.clone()

        # override appended steps with policy topk (策略分布 top-k)
        # appended step 0 corresponds to predicting item at index L_hist (t = L_hist-1)
        for b in range(B):
            for t, recs in enumerate(self.topk_recs[b]):
                pos = (L_hist - 1) + t
                if 0 <= pos < T_full:
                    pad_id = int(getattr(Constants, "PAD", 0))
                    rr = recs[:K] + [pad_id] * max(0, K - len(recs))

                    topk_indices[b, pos] = torch.tensor(rr, device=device, dtype=torch.long)

        # original_seqs for Metrics loop is the "sequence used to decide valid t"
        # Metrics uses original_seqs[b][t] != PAD for t in [0, T_full-1]
        original_for_metrics = full_seq[:, :T_full]

        eff = self._per_sample_effectiveness(original_for_metrics, yt_before, yt_after, topk_indices)
        pref = self._per_sample_preference(pred_probs, original_for_metrics, topk_indices)

        if hidden_item is None or (isinstance(hidden_item, torch.Tensor) and hidden_item.dim() < 2):
            div = torch.zeros(B, device=device)
        else:
            div = self._per_sample_diversity(hidden_item, original_for_metrics, topk_indices)

        ada = self._per_sample_adaptivity(original_for_metrics, yt_before, topk_indices)

        # fq = (float(self.final_weights.get("effectiveness", 0.4)) * eff
        #       + float(self.final_weights.get("adaptivity", 0.3)) * ada
        #       + float(self.final_weights.get("diversity", 0.2)) * div
        #       + float(self.final_weights.get("preference", 0.1)) * pref)
        fq = (float(self.final_weights.get("effectiveness", 0.4)) * eff
              + float(self.final_weights.get("diversity", 0.2)) * div
              + float(self.final_weights.get("preference", 0.1)) * pref)

        return {"effectiveness": eff, "adaptivity": ada, "diversity": div, "preference": pref, "final_quality": fq}

    def compute_final_reward(self) -> torch.Tensor:
        return self.compute_final_metrics()["final_quality"]


class PPOTrainer:
    """
    轻量 Policy Gradient（REINFORCE + entropy）：
      - 轨迹 rewards（step）+ final_reward（加到最后一步）
      - returns = discounted sum
      - advantage = (returns - returns.mean)/std
    """
    def __init__(self, policy_net: PolicyNetwork, env: LearningPathEnv, lr=3e-4, gamma=0.99, entropy_coef=1e-3):
        self.policy_net = policy_net
        self.env = env
        self.gamma = float(gamma)
        self.entropy_coef = float(entropy_coef)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=float(lr))

        self._traj_rewards = []
        self._traj_logp = []
        self._traj_ent = []

    def collect_trajectory(
            self,
            tgt,
            tgt_timestamp,
            tgt_idx,
            ans,
            graph=None,
            hypergraph_list=None,
            deterministic: bool = False,
            **kwargs
    ):

        state = self.env.reset(tgt, tgt_timestamp, tgt_idx, ans, graph=graph, hypergraph_list=hypergraph_list)

        rewards = []
        logps = []
        ents = []

        done = False
        while not done:
            cand_feat = state["candidate_features"]  # [B,Kcand,D]
            cand_ids = state["candidate_ids"]        # [B,Kcand]

            logits = self.policy_net(cand_feat)  # [B, Kcand]
            dist = Categorical(logits=logits)

            if deterministic:
                action = torch.argmax(logits, dim=-1)  # [B] 贪心
            else:
                action = dist.sample()  # [B] 采样探索

            logp = dist.log_prob(action)  # [B]
            ent = dist.entropy()  # [B]


            # 策略分布 top-k（用 logits 排序）
            K = min(self.env.metrics_topnum, logits.size(1))
            _, topk_pos = torch.topk(logits, k=K, dim=-1)
            step_topk_items = cand_ids.gather(1, topk_pos)  # [B,K] global ids

            next_state, step_reward, done = self.env.step(action, step_topk_items)

            rewards.append(step_reward)
            logps.append(logp)
            ents.append(ent)

            state = next_state

        # final reward
        final_reward = self.env.compute_final_reward()  # [B]
        rewards[-1] = rewards[-1] + final_reward

        rewards_t = torch.stack(rewards, dim=0)   # [T,B]
        logps_t = torch.stack(logps, dim=0)       # [T,B]
        ents_t = torch.stack(ents, dim=0)         # [T,B]

        self._traj_rewards = rewards_t
        self._traj_logp = logps_t
        self._traj_ent = ents_t

        return rewards_t, logps_t, ents_t, final_reward

    def update_policy(self):
        rewards = self._traj_rewards      # [T,B]
        logps = self._traj_logp           # [T,B]
        ents = self._traj_ent             # [T,B]
        T, B = rewards.shape

        # returns
        returns = torch.zeros_like(rewards)
        running = torch.zeros(B, device=rewards.device)
        for t in reversed(range(T)):
            running = rewards[t] + self.gamma * running
            returns[t] = running

        # advantage normalize
        adv = returns - returns.mean(dim=0, keepdim=True)
        adv_std = adv.std().clamp_min(1e-6)
        adv = adv / adv_std

        pg_loss = -(logps * adv.detach()).mean()
        ent_loss = -ents.mean()
        loss = pg_loss + self.entropy_coef * ent_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
        self.optimizer.step()

        # 打印诊断（与你之前日志一致）
        print(f"pg_loss={pg_loss.item():.6f}  entropy={ents.mean().item():.6f}  adv_std={adv_std.item():.6f}")
        return float(loss.item())


class RLPathOptimizer:
    """
    兼容你现有 train_rl.py 的外层封装
    """
    def __init__(
        self,
        pretrained_model,
        num_skills,
        batch_size,
        recommendation_length=5,
        topk=20,
        data_name="",
        graph=None,
        hypergraph_list=None,
        policy_hidden_dim=128,
        lr=3e-4,
        gamma=0.99,
        entropy_coef=1e-3,
        metrics_topnum=None,
        device=None,
    ):
        self.device = device or next(pretrained_model.parameters()).device
        self.env = LearningPathEnv(
            pretrained_model=pretrained_model,
            num_skills=num_skills,
            recommendation_length=recommendation_length,
            policy_topk=topk,
            metrics_topnum=metrics_topnum or topk,
            device=self.device,
        )

        # candidate_features dim = 2 (prob + ability) or 1 if no yt
        in_dim = 2
        self.policy_net = PolicyNetwork(in_dim=in_dim, hidden_dim=policy_hidden_dim).to(self.device)
        self.trainer = PPOTrainer(self.policy_net, self.env, lr=lr, gamma=gamma, entropy_coef=entropy_coef)

        self.graph = graph
        self.hypergraph_list = hypergraph_list


@torch.no_grad()
def evaluate_policy(env: LearningPathEnv, policy_net: PolicyNetwork, data_loader, relation_graph=None, hypergraph_list=None, num_episodes=5):
    """
    简化评估：跑若干 batch，输出平均指标（标量）
    """
    device = env.device
    metrics_sum = {"effectiveness": 0.0, "adaptivity": 0.0, "diversity": 0.0, "preference": 0.0, "final_quality": 0.0}
    n = 0

    for batch_idx, batch in enumerate(data_loader):
        if batch_idx >= num_episodes:
            break
        tgt, tgt_ts, tgt_idx, ans = batch
        tgt = tgt.to(device); tgt_ts = tgt_ts.to(device); tgt_idx = tgt_idx.to(device); ans = ans.to(device)

        state = env.reset(tgt, tgt_ts, tgt_idx, ans, graph=relation_graph, hypergraph_list=hypergraph_list)
        done = False
        while not done:
            logits = policy_net(state["candidate_features"])
            dist = Categorical(logits=logits)
            action = torch.argmax(logits, dim=-1)  # greedy eval
            K = min(env.metrics_topnum, logits.size(1))
            _, topk_pos = torch.topk(logits, k=K, dim=-1)
            step_topk_items = state["candidate_ids"].gather(1, topk_pos)
            state, _, done = env.step(action, step_topk_items)

        m = env.compute_final_metrics()
        for k in metrics_sum:
            metrics_sum[k] += float(m[k].mean().item())
        n += 1

    if n == 0:
        return 0.0, 0.0, 0.0
    avg = {k: v / n for k, v in metrics_sum.items()}
    return avg["effectiveness"], avg["diversity"], avg["adaptivity"]
