
# -*- coding: utf-8 -*-
"""
rl_adjuster_new.py (reworked)

This version aligns with your confirmed requirements:

1) RL decision at EVERY valid time step (mask PAD=0 by default).
   - Episode length = valid_len-1 for each sample (predict next-step recommendation repeatedly).
2) Richer state for the policy:
   - Candidate features: base_prob, kt_mastery(prob), difficulty_norm, sim_to_path_mean, sim_to_last
   - History features (broadcast to candidates): recent_correct_rate, recent_avg_difficulty, delta_t (Eq.19)
3) Step reward uses ONLY "pre-step" info (no leakage) + optional novelty term.
4) Terminal(final) quality reward STRICTLY follows your provided formulas:
   - Adaptivity Eq.(19): uses delta_t computed from REAL answers (here: simulated answers on generated path),
     and Dif_i queried from difficulty file via idx2u mapping.
   - Effectiveness Eq.(20): Gain = (pa - pb)/(1 - pb) with pb<0.9; uses KT outputs
     before/after (original vs generated).
   - Diversity Eq.(21): mean_{pairs} (1 - cosine_sim(emb_i, emb_j)) over recommended items.
5) Trainer is true PPO (actor-critic, GAE, clipped surrogate, value loss, entropy bonus).

NOTE:
- We deliberately keep the external API used by train_rl_new.py:
    * RLPathOptimizer
    * evaluate_policy

张量维度说明：
- [B]: Batch size (批处理大小)
- [N]: Number of items/questions (物品/问题总数)
- [L]: Sequence length (序列长度)
- [d]: Embedding dimension (嵌入维度)
- [K]/[Kc]: Number of candidates (候选项目数)
- [F]: Feature dimension (特征维度)
- [T]: Time steps (时间步数)
- [topk]: Top-k推荐数
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


# ------------------------- small utils -------------------------

def _ensure_2d(x: torch.Tensor) -> torch.Tensor:
    """
    确保张量为二维
    
    Args:
        x: 输入张量 [任意维度]
        
    Returns:
        确保为二维的张量 [N, M] 或 [1, N]
    """
    if x.dim() == 1:
        return x.unsqueeze(0)  # [N] -> [1, N]
    return x

def _to_device(x, device):
    """
    将张量或对象移动到指定设备
    
    Args:
        x: 张量或None或其他对象
        device: 目标设备
        
    Returns:
        移动到目标设备的对象
    """
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return x

def _cosine_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    计算余弦相似度
    
    Args:
        a: 第一个张量 [..., d] - 任意前导维度，最后一维为嵌入维度
        b: 第二个张量 [..., d] - 与a形状兼容
        eps: 防止除零的小值
        
    Returns:
        余弦相似度张量 [...] - 与a和b广播后的形状相同
    """
    # 标准化张量 [..., d]
    a_n = a / (a.norm(dim=-1, keepdim=True) + eps)  # [..., d]
    b_n = b / (b.norm(dim=-1, keepdim=True) + eps)  # [..., d]
    # 计算点积得到余弦相似度
    return (a_n * b_n).sum(dim=-1)  # [...] - 除了最后一个维度外的广播结果

def _masked_mean(x: torch.Tensor, mask: torch.Tensor, dim=None, eps: float = 1e-8) -> torch.Tensor:
    """
    基于掩码计算均值
    
    Args:
        x: 输入张量 [任意维度]
        mask: 掩码张量 [与x广播兼容的布尔张量]
        dim: 要计算均值的维度
        eps: 防止除零的小值
        
    Returns:
        掩码均值张量
    """
    # mask: 可广播的布尔张量
    x = x * mask.to(x.dtype)  # 将被掩码部分置为0
    # 计算非掩码元素的数量，防止除零
    denom = mask.to(x.dtype).sum(dim=dim, keepdim=True).clamp_min(eps)  # [广播后的形状]
    return (x.sum(dim=dim, keepdim=True) / denom).squeeze(dim if dim is not None else -1)


# ------------------------- Difficulty & Mapping loader -------------------------

@dataclass
class DifficultyMapping:
    idx2u: Dict[int, int]
    difficulty_by_uid: Dict[int, int]  # uid -> raw difficulty level (e.g., 1/2/3)

    @staticmethod
    def load_from_options(data_name: str) -> Optional["DifficultyMapping"]:
        """
        Try to load idx2u and difficulty file using dataLoader.Options (your repo).
        If unavailable in the current runtime, return None.
        """
        try:
            from dataLoader import Options  # your project file
        except Exception:
            return None

        try:
            opt = Options(data_name)
            with open(opt.idx2u_dict, "rb") as f:
                idx2u = pickle.load(f)

            difficulty_by_uid: Dict[int, int] = {}
            with open(opt.difficult_file, "r", encoding="utf-8") as f:
                next(f)
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

    def get_difficulty_raw(self, idx: int, default: int = 2) -> int:
        # idx is internal u2idx-mapped item id.
        # idx2u may be a dict {idx -> uid} or a list where idx2u[idx] = uid.
        uid = None
        try:
            if hasattr(self.idx2u, "get"):
                uid = self.idx2u.get(int(idx), None)
            else:
                i = int(idx)
                if 0 <= i < len(self.idx2u):
                    uid = self.idx2u[i]
        except Exception:
            uid = None

        if uid is None:
            return default

        # difficulty_by_uid is typically keyed by original uid (challenge_id)
        try:
            return int(self.difficulty_by_uid.get(int(uid), default))
        except Exception:
            return int(self.difficulty_by_uid.get(uid, default))

    def get_difficulty_norm(self, idx: int, default: int = 2) -> float:
        # map raw 1/2/3 -> 0/0.5/1
        d = self.get_difficulty_raw(idx, default=default)
        d = max(1, min(3, int(d)))
        return (d - 1) / 2.0


# ------------------------- Policy / Value net -------------------------

class PolicyValueNet(nn.Module):
    """
    策略价值网络
    
    输入:
        cand_feat: [B, K, F] - 候选特征，B为批量大小，K为候选数，F为特征维度
    输出:
        logits: [B, K] - 动作logits，用于选择下一个项目
        value: [B] - 状态价值估计
    """
    def __init__(self, feat_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.feat_dim = feat_dim      # 特征维度 [F]
        self.hidden_dim = hidden_dim  # 隐藏层维度 [H]

        # 候选项目MLP编码器
        # 输入: [B, K, F] -> 输出: [B, K, H]
        self.cand_mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),  # [F, H] - 特征到隐藏层映射
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),  # [H, H] - 隐藏层内部映射
            nn.Tanh(),
        )
        # Logits头 - 为每个候选项目生成一个logit值
        # 输入: [B, K, H] -> 输出: [B, K, 1] -> [B, K]
        self.logit_head = nn.Linear(hidden_dim, 1)  # [H, 1] - 隐藏层到单个logit

        # 价值头池化候选表示
        # 使用注意力池化将候选表示聚合为单一状态表示
        self.value_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  # [H, H] - 状态价值编码
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)  # [H, 1] - 最终价值输出
        )

    def forward(self, cand_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            cand_feat: 候选特征 [B, K, F] - B: 批量大小, K: 候选数, F: 特征维度
            
        Returns:
            logits: 动作logits [B, K] - 每个候选项目的logit值
            value: 状态价值 [B] - 当前状态的价值估计
        """
        # 编码候选特征 [B, K, F] -> [B, K, H]
        h = self.cand_mlp(cand_feat)            # [B, K, H]
        # 生成动作logits [B, K, H] -> [B, K]
        logits = self.logit_head(h).squeeze(-1) # [B, K]

        # 注意力池化计算价值
        # 使用logits作为注意力权重对候选表示进行加权求和
        att = torch.softmax(logits, dim=-1).unsqueeze(-1)  # [B, K, 1] - 注意力权重
        pooled = (h * att).sum(dim=1)                      # [B, H] - 加权池化
        value = self.value_mlp(pooled).squeeze(-1)         # [B] - 价值估计
        return logits, value


# ------------------------- PPO Trainer -------------------------

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
    """
    PPO (Proximal Policy Optimization) 训练器
    """
    def __init__(self, policy: PolicyValueNet, lr: float = 3e-4, config: Optional[PPOConfig] = None):
        self.policy = policy
        self.config = config or PPOConfig()
        # PPO优化器，用于更新策略和价值网络参数
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=lr)

    @torch.no_grad()
    def compute_gae(
        self,
        rewards: torch.Tensor,      # [T, B] - T时间步，B批量大小的奖励
        values: torch.Tensor,       # [T, B] - T时间步，B批量大小的状态价值
        dones: torch.Tensor         # [T, B] - T时间步，B批量大小的终止标志 (1表示该时间步结束，否则为0)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算广义优势估计 (Generalized Advantage Estimation)
        
        Args:
            rewards: 奖励张量 [T, B] - T时间步，B样本数
            values: 价值张量 [T, B] - T时间步，B样本数的状态价值估计
            dones: 终止标志 [T, B] - T时间步，B样本数的终止状态
            
        Returns:
            advantages: 优势 [T, B] - 每个时间步的优势值
            returns: 回报 [T, B] - 每个时间步的累积回报
        """
        T, B = rewards.shape  # T: 时间步数, B: 批量大小
        adv = torch.zeros_like(rewards)  # [T, B] - 优势张量
        last_gae = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)  # [B] - 上一时间步的GAE
        last_value = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)  # [B] - 上一时间步的价值

        for t in reversed(range(T)):  # 从后往前计算GAE
            mask = 1.0 - dones[t]  # [B] - 掩码，0表示已终止，1表示继续
            # 计算TD误差: r_t + γ*V(s_{t+1}) - V(s_t)
            delta = rewards[t] + self.config.gamma * last_value * mask - values[t]  # [B]
            # 计算GAE: δ_t + γλ(mask)*last_gae
            last_gae = delta + self.config.gamma * self.config.gae_lambda * mask * last_gae  # [B]
            adv[t] = last_gae  # [B] - 当前时间步的GAE
            last_value = values[t]  # [B] - 更新上一时间步价值

        returns = adv + values  # [T, B] - 回报 = 优势 + 价值
        return adv, returns

    def update(
        self,
        cand_feat: torch.Tensor,      # [N, K, F] - 展平的候选特征，N为(时间步,批量)的总数
        actions: torch.Tensor,        # [N] - 动作索引
        old_logp: torch.Tensor,       # [N] - 旧策略的对数概率
        old_values: torch.Tensor,     # [N] - 旧价值估计
        advantages: torch.Tensor,     # [N] - 优势值
        returns: torch.Tensor         # [N] - 回报值
    ) -> Dict[str, float]:
        """
        PPO策略更新
        
        Args:
            cand_feat: 候选特征 [N, K, F] - N: 总样本数(时间步*批量)，K: 候选数，F: 特征维
            actions: 动作 [N] - 选择的动作索引
            old_logp: 旧对数概率 [N] - 采样策略下的对数概率
            old_values: 旧价值 [N] - 旧策略的价值估计
            advantages: 优势 [N] - 优势函数值
            returns: 回报 [N] - 累积回报值
            
        Returns:
            损失字典，包含policy_loss, value_loss, entropy, total_loss等标量值
        """
        cfg = self.config
        # 分离回放数据张量：PPO应将收集的数据视为常数
        # 这可以防止在多次PPO周期/小批量处理时出现"通过图第二次反向传播"错误
        cand_feat = cand_feat.detach()      # [N, K, F] - 候选特征
        actions = actions.detach()          # [N] - 动作
        old_logp = old_logp.detach()        # [N] - 旧对数概率
        old_values = old_values.detach()    # [N] - 旧价值
        advantages = advantages.detach()    # [N] - 优势
        returns = returns.detach()          # [N] - 回报

        # 标准化优势值
        advantages = (advantages - advantages.mean()) / (advantages.std().clamp_min(1e-8))  # [N]

        N = cand_feat.size(0)  # 总样本数
        idx = torch.randperm(N, device=cand_feat.device)  # [N] - 随机排列的索引

        # 初始化损失统计
        losses = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "total_loss": 0.0}

        # PPO多轮更新
        for _ in range(cfg.ppo_epochs):  # cfg.ppo_epochs轮更新
            for start in range(0, N, cfg.minibatch_size):  # 按小批量处理
                mb = idx[start:start + cfg.minibatch_size]  # [minibatch_size] - 小批量索引
                mb_feat = cand_feat[mb]         # [minibatch_size, K, F] - 小批量特征
                mb_act = actions[mb]            # [minibatch_size] - 小批量动作
                mb_old_logp = old_logp[mb]      # [minibatch_size] - 小批量旧对数概率
                mb_old_val = old_values[mb]     # [minibatch_size] - 小批量旧价值
                mb_adv = advantages[mb]         # [minibatch_size] - 小批量优势
                mb_ret = returns[mb]            # [minibatch_size] - 小批量回报

                # 前向传播获取新策略输出
                logits, value = self.policy(mb_feat)  # [minibatch_size, K], [minibatch_size]
                
                # 创建分类分布
                dist = torch.distributions.Categorical(logits=logits)  # [minibatch_size]
                logp = dist.log_prob(mb_act)    # [minibatch_size] - 新策略下动作的对数概率
                entropy = dist.entropy().mean() # 标量 - 策略熵

                # 计算PPO比率
                ratio = torch.exp(logp - mb_old_logp)  # [minibatch_size] - 概率比

                # PPO截断策略梯度
                surr1 = ratio * mb_adv  # [minibatch_size] - 原始策略梯度
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * mb_adv  # [minibatch_size] - 截断策略梯度
                policy_loss = -torch.min(surr1, surr2).mean()  # 标量 - 策略损失

                # 价值函数损失
                value_loss = F.mse_loss(value, mb_ret)  # 标量 - 价值损失

                # 总损失
                loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy  # 标量

                # 反向传播和参数更新
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.opt.step()

                # 累积损失值
                losses["policy_loss"] += float(policy_loss.detach().cpu())
                losses["value_loss"] += float(value_loss.detach().cpu())
                losses["entropy"] += float(entropy.detach().cpu())
                losses["total_loss"] += float(loss.detach().cpu())

        # 计算平均损失
        denom = max(1, cfg.ppo_epochs * math.ceil(N / cfg.minibatch_size))
        for k in losses:
            losses[k] /= denom  # 平均每个epoch每个minibatch的损失
        return losses


# ------------------------- Environment -------------------------

class OnlineLearningPathEnv:
    """
    Online RL over the whole sequence (every valid time step).

    At each step t:
      - We build candidate set from base model distribution at current prefix.
      - Policy chooses one candidate as next item.
      - We simulate the answer for the chosen item using KT probability (>=0.5 => correct).
      - Append chosen item/answer to the generated prefix.
      - Move to next step.

    We record:
      - topk_recs_policy[b][t] : policy distribution top-K mapped to global item ids
      - chosen_items[b][t]     : chosen item id
      - generated_ans[b][t]    : simulated answers (for generated items beyond the seed prefix)
    """
    def __init__(
        self,
        base_model: nn.Module,
        num_items: int,
        data_name: str,
        device: torch.device,
        pad_val: int = 0,
        topk: int = 5,
        cand_k: int = 5,
        history_window_T: int = 5,
        epsilon: float = 1e-5,
        w_step: Optional[Dict[str, float]] = None,
    ):
        self.base_model = base_model
        self.num_items = num_items
        self.data_name = data_name
        self.device = device
        self.pad_val = pad_val
        self.topk = topk
        self.cand_k = cand_k
        self.T = history_window_T
        self.eps = epsilon

        # weights for step reward
        self.w_step = w_step or {
            "preference": 2.0,
            "adaptivity": 1.0,
            "novelty": 0.2,
        }

        # try load mapping+difficulty (for Eq.19)
        self.diff_map = DifficultyMapping.load_from_options(data_name)

        # buffers per episode
        self.graph = None
        self.hypergraph_list = None

        self.orig_seq = None
        self.orig_ts = None
        self.orig_idx = None
        self.orig_ans = None

        self.valid_lens = None
        self.max_steps = None

        # generated prefix starts with first valid item (seed length = 1)
        self.gen_seq = None
        self.gen_ans = None

        self.hidden_item = None  # [N,d] embeddings for diversity and similarity

        # records
        self.topk_recs_policy: List[List[List[int]]] = []  # B x steps x topk
        self.chosen_items: List[List[int]] = []            # B x steps
        self.step = 0

        # caches at "pre-step"
        self._pre_base_probs = None   # [B,N] prob at current step (for preference)
        self._pre_yt = None           # [B,N] mastery prob (item-level)
        self._pre_cand_ids = None     # [B,Kcand]
        self._pre_cand_feat = None    # [B,Kcand,F]
        self._pre_delta = None        # [B] delta_t (Eq.19)
        self._pre_recent_corr = None  # [B]
        self._pre_recent_davg = None  # [B]
        self._pre_path_mean_emb = None# [B,d]
        self._pre_last_emb = None     # [B,d]

    @torch.no_grad()
    def _forward_base(self, seq: torch.Tensor, ts: torch.Tensor, idx: torch.Tensor, ans: torch.Tensor):
        """
        Returns:
            probs_last: [B,N] probability distribution for next item
            yt_last:    [B,N] mastery estimate (item-level). Uses base_model output `yt` if available.
            hidden:     [N,d] item embeddings for similarity/diversity
        """
        out = self.base_model(seq, ts, idx, ans, self.graph, self.hypergraph_list)
        # compatible unpacking: allow tuple/list/dict
        if isinstance(out, dict):
            pred_flat = out.get("pred_flat", out.get("pred", None))
            yt = out.get("yt", None)
            hidden = out.get("hidden", out.get("item_emb", None))
        else:
            # common: (pred_flat, yt, hidden, ...)
            pred_flat = out[0] if len(out) > 0 else None
            yt = out[3] if len(out) > 3 else None
            hidden = out[4] if len(out) > 4 else None

        if pred_flat is None:
            raise RuntimeError("base_model forward must provide pred logits (pred_flat/pred).")

        B, L = seq.shape
        # pred_flat often [B*(L-1), N]; we need last-step distribution
        if pred_flat.dim() == 2 and pred_flat.size(0) == B * (L - 1):
            pred_logits = pred_flat.view(B, L - 1, -1)[:, -1]   # [B,N]
        elif pred_flat.dim() == 3:
            pred_logits = pred_flat[:, -1]                      # [B,N]
        else:
            raise RuntimeError(f"Unsupported pred_flat shape: {tuple(pred_flat.shape)}")

        probs_last = torch.softmax(pred_logits, dim=-1)

        # yt: expect [B, L-1, N] or [B, L-1, S] with S==N (item-level KT)
        yt_last = None
        if yt is not None:
            if yt.dim() == 3 and yt.size(0) == B:
                yt_last = yt[:, -1]  # [B, S]
            elif yt.dim() == 2 and yt.size(0) == B:
                yt_last = yt
        if yt_last is None:
            # fallback: use probs as mastery proxy (not ideal, but keeps running)
            yt_last = probs_last

        if hidden is not None:
            if hidden.dim() == 3:
                # some models output [B, N, d]
                hidden_item = hidden[0]
            else:
                hidden_item = hidden
        else:
            hidden_item = None
        # Sanity: hidden_item must be [N,d] with N==num_items, otherwise disable similarity features
        if hidden_item is not None:
            if (hidden_item.dim() != 2) or (hidden_item.size(0) != self.num_items):
                # Wrong tensor (e.g., kt_mask) would crash when indexed by item ids
                hidden_item = None

        return probs_last, yt_last, hidden_item

    def _difficulty_norm(self, idx_tensor: torch.Tensor) -> torch.Tensor:
        # idx_tensor: [B,K]
        if self.diff_map is None:
            # default difficulty=2 -> norm 0.5
            return torch.full_like(idx_tensor, 0.5, dtype=torch.float32)
        # loop in python (Kcand is small)
        B, K = idx_tensor.shape
        out = torch.zeros((B, K), device=idx_tensor.device, dtype=torch.float32)
        idx_np = idx_tensor.detach().cpu().numpy()
        for b in range(B):
            for k in range(K):
                out[b, k] = float(self.diff_map.get_difficulty_norm(int(idx_np[b, k]), default=2))
        return out

    def _compute_delta_t(self, hist_items: torch.Tensor, hist_ans: torch.Tensor) -> torch.Tensor:
        """
        Eq.(19) delta_t using recent T window:
            delta_t = sum(Dif_i * r_i) / (sum(r_i) + eps)
        Using REAL answers r_i (here: generated answers for generated items; original answers for seed prefix if you choose).
        hist_items, hist_ans: [B, Lprefix]
        """
        B, L = hist_items.shape
        delta = torch.ones((B,), device=hist_items.device, dtype=torch.float32)

        if self.diff_map is None:
            return delta  # fallback (same as Metrics default path)

        items_np = hist_items.detach().cpu().numpy()
        ans_np = hist_ans.detach().cpu().numpy()
        for b in range(B):
            # collect valid history (exclude PAD)
            valid = [(int(items_np[b, t]), float(ans_np[b, t])) for t in range(L) if int(items_np[b, t]) != self.pad_val]
            if len(valid) < max(1, self.T // 2):
                delta[b] = 1.0
                continue
            # last window
            window = valid[max(0, len(valid) - self.T):]
            num = 0.0
            den = 0.0
            for it, r in window:
                d = float(self.diff_map.get_difficulty_norm(it, default=2))
                num += d * r
                den += r
            delta[b] = float(num / (den + self.eps))
        return delta

    def reset(self, tgt, tgt_timestamp, tgt_idx, ans, graph=None, hypergraph_list=None) -> Dict[str, torch.Tensor]:
        self.graph = graph
        self.hypergraph_list = hypergraph_list

        self.orig_seq = _ensure_2d(tgt).to(self.device)
        self.orig_ts = _ensure_2d(tgt_timestamp).to(self.device)
        self.orig_idx = tgt_idx.to(self.device)
        # DataLoader in this repo yields tgt_idx as shape [B] (cascade id per sequence), not [B,L].
        # Keep it 1D to match MSHGAT.forward expectations.
        if self.orig_idx.dim() == 2 and self.orig_idx.size(1) == 1:
            self.orig_idx = self.orig_idx.squeeze(1)
        self.orig_ans = _ensure_2d(ans).to(self.device)

        # ===== DEBUG CHECK (CPU-side) =====
        # If CUDA device-side assert happens, it is almost always due to illegal indices used in embedding/gather.
        # We check question/item ids (tgt) and correctness labels (ans) BEFORE any model forward.
        try:
            seq_cpu = self.orig_seq.detach().cpu()
            ans_cpu = self.orig_ans.detach().cpu()
            n_items = int(self.num_items)
            q_min = int(seq_cpu.min().item())
            q_max = int(seq_cpu.max().item())
            if q_min < 0 or q_max >= n_items:
                bad = (seq_cpu < 0) | (seq_cpu >= n_items)
                bad_pos = bad.nonzero(as_tuple=False)
                print(f"[RL DEBUG][OOB][questions] num_items={n_items} q_min={q_min} q_max={q_max} bad_cnt={int(bad.sum().item())}")
                for p in bad_pos[:10]:
                    b, t = int(p[0]), int(p[1])
                    print("  bad@", (b, t), "val=", int(seq_cpu[b, t].item()))
                raise RuntimeError("[RL DEBUG] Out-of-bound question/item ids in tgt.")

            # correct_embed vocab size (usually 2)
            n_correct = 2
            try:
                ce = getattr(getattr(getattr(self.base_model, "ktmodel", None), "correct_embed", None), "num_embeddings", None)
                if ce is not None:
                    n_correct = int(ce)
            except Exception:
                pass
            a_min = float(ans_cpu.min().item())
            a_max = float(ans_cpu.max().item())
            if a_min < 0 or a_max > (n_correct - 1):
                uniq = ans_cpu.unique()
                uniq = uniq[:20].tolist() if uniq.numel() > 0 else []
                print(f"[RL DEBUG][OOB][answers] correct_embed_size={n_correct} ans_min={a_min} ans_max={a_max} uniq_head={uniq}")
                raise RuntimeError("[RL DEBUG] Out-of-range correctness labels in ans.")
        except Exception as e:
            # Re-raise so you get a clear Python error instead of CUDA async assert.
            raise
        # ===== END DEBUG CHECK =====

        B, L = self.orig_seq.shape
        self.valid_lens = (self.orig_seq != self.pad_val).sum(dim=1).clamp_min(1)  # [B]
        self.max_steps = int(self.valid_lens.max().item()) - 1  # predict next for each step
        self.max_steps = max(0, self.max_steps)

        # seed prefix: first 2 valid items per sample (need len>=2 to get a next-step prediction from base_model)
        self.gen_seq = torch.full_like(self.orig_seq, self.pad_val)
        self.gen_ans = torch.full_like(self.orig_ans, 0)

        # seed_len per sample: 2 if available else 1 (samples with <2 valid interactions will have zero RL steps)
        self.seed_len = 2

        for b in range(B):
            vlen = int(self.valid_lens[b].item())
            if vlen >= 1:
                self.gen_seq[b, 0] = self.orig_seq[b, 0]
                self.gen_ans[b, 0] = self.orig_ans[b, 0]
            if vlen >= 2:
                self.gen_seq[b, 1] = self.orig_seq[b, 1]
                self.gen_ans[b, 1] = self.orig_ans[b, 1]

        # start step at last seeded index (seed_len-1), so the first RL action predicts position seed_len
        self.step = 1  # corresponds to having a prefix length of 2
        self.start_t = self.step  # time index in [0, L-2] where we start recording policy topK

        self.topk_recs_policy = [[] for _ in range(B)]
        self.chosen_items = [[] for _ in range(B)]

        # build initial cache using prefix length = step+1 = 2
        self._update_pre_step_cache()
        return {"candidate_ids": self._pre_cand_ids, "candidate_features": self._pre_cand_feat}

    @torch.no_grad()
    def _update_pre_step_cache(self):
        """
        Build state for current step based on current generated prefix.
        """
        B, L = self.gen_seq.shape

        # build current prefix length = step+1 (seed=1)
        cur_len = self.step + 1
        cur_len = min(cur_len, L)
        seq = self.gen_seq[:, :cur_len]
        ts = self.orig_ts[:, :cur_len]  # keep original timestamps for aligned length
        idx = self.orig_idx  # [B] cascade id per sequence
        ans = self.gen_ans[:, :cur_len]

        probs_last, yt_last, hidden_item = self._forward_base(seq, ts, idx, ans)
        self.hidden_item = hidden_item

        # candidate ids by base probs
        Kc = min(self.cand_k, probs_last.size(-1))
        cand_ids = torch.topk(probs_last, k=Kc, dim=-1).indices  # [B,Kc]
        cand_probs = probs_last.gather(1, cand_ids)              # [B,Kc]
        cand_ability = yt_last.gather(1, cand_ids.clamp(0, yt_last.size(1)-1))  # [B,Kc]

        cand_diff = self._difficulty_norm(cand_ids)              # [B,Kc]

        # history stats for Eq.(19)
        delta_t = self._compute_delta_t(seq, ans)                # [B]
        self._pre_delta = delta_t

        # recent correctness rate, recent avg difficulty
        recent_corr = torch.zeros((B,), device=self.device)
        recent_davg = torch.zeros((B,), device=self.device)
        for b in range(B):
            # valid history
            valid_items = seq[b][seq[b] != self.pad_val]
            valid_ans = ans[b][seq[b] != self.pad_val].float()
            if valid_items.numel() == 0:
                recent_corr[b] = 0.0
                recent_davg[b] = 0.5
                continue
            w = min(int(valid_items.numel()), self.T)
            recent_corr[b] = valid_ans[-w:].mean()
            if self.diff_map is None:
                recent_davg[b] = 0.5
            else:
                items_np = valid_items[-w:].detach().cpu().numpy()
                recent_davg[b] = float(np.mean([self.diff_map.get_difficulty_norm(int(it), default=2) for it in items_np]))

        self._pre_recent_corr = recent_corr
        self._pre_recent_davg = recent_davg

        # similarity to path mean / last
        if hidden_item is not None:
            d = hidden_item.size(-1)
            path_mean = torch.zeros((B, d), device=self.device)
            last_emb = torch.zeros((B, d), device=self.device)
            for b in range(B):
                valid_items = seq[b][seq[b] != self.pad_val]
                if valid_items.numel() == 0:
                    continue
                emb = hidden_item[valid_items]         # [len,d]
                path_mean[b] = emb.mean(dim=0)
                last_emb[b] = emb[-1]
            self._pre_path_mean_emb = path_mean
            self._pre_last_emb = last_emb

            cand_emb = hidden_item[cand_ids]           # [B,Kc,d]
            sim_path = _cosine_sim(cand_emb, path_mean.unsqueeze(1))  # [B,Kc]
            sim_last = _cosine_sim(cand_emb, last_emb.unsqueeze(1))   # [B,Kc]
        else:
            sim_path = torch.zeros_like(cand_probs)
            sim_last = torch.zeros_like(cand_probs)

        # broadcast history features to candidates
        hist_feat = torch.stack([recent_corr, recent_davg, delta_t], dim=-1)  # [B,3]
        hist_feat = hist_feat.unsqueeze(1).expand(-1, cand_ids.size(1), -1)   # [B,Kc,3]

        # candidate features: [prob, ability, diff, sim_path, sim_last, hist...]
        cand_feat = torch.cat([
            cand_probs.unsqueeze(-1),
            cand_ability.unsqueeze(-1),
            cand_diff.unsqueeze(-1),
            sim_path.unsqueeze(-1),
            sim_last.unsqueeze(-1),
            hist_feat
        ], dim=-1).float()  # [B,Kc,F]

        self._pre_base_probs = probs_last
        self._pre_yt = yt_last
        self._pre_cand_ids = cand_ids
        self._pre_cand_feat = cand_feat

    @torch.no_grad()
    def step_env(self, action_idx: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict]:
        """
        action_idx: [B] index in candidate set.
        Returns:
            next_state dict
            reward [B]
            done   [B] (1 if terminal at this step for the sample)
        """
        B, L = self.gen_seq.shape
        Kc = self._pre_cand_ids.size(1)

        action_idx = action_idx.clamp(0, Kc-1)
        chosen = self._pre_cand_ids.gather(1, action_idx.view(-1,1)).squeeze(1)  # [B]

        # record policy topK for this time step (mapped to global ids)
        # policy topK will be filled by caller (needs policy logits). Here keep chosen.
        for b in range(B):
            self.chosen_items[b].append(int(chosen[b].item()))

        # ------- step reward using PRE-STEP info only -------
        # preference: base probability of chosen item
        pref = self._pre_base_probs.gather(1, chosen.view(-1,1)).squeeze(1)  # [B]

        # adaptivity at this time step: 1 - |delta_t - Dif(chosen)|
        if self.diff_map is None:
            chosen_diff = torch.full((B,), 0.5, device=self.device)
        else:
            chosen_np = chosen.detach().cpu().numpy()
            chosen_diff = torch.tensor(
                [self.diff_map.get_difficulty_norm(int(it), default=2) for it in chosen_np],
                device=self.device, dtype=torch.float32
            )
        adapt = 1.0 - torch.abs(self._pre_delta - chosen_diff)

        # novelty: 1 - sim_to_last (encourage less repetitive)
        if self.hidden_item is not None and self._pre_last_emb is not None:
            chosen_emb = self.hidden_item[chosen]  # [B,d]
            sim_last = _cosine_sim(chosen_emb, self._pre_last_emb)
            novelty = 1.0 - sim_last
        else:
            novelty = torch.zeros((B,), device=self.device)

        # reward = (
        #     self.w_step["preference"] * pref +
        #     self.w_step["adaptivity"] * adapt +
        #     self.w_step.get("novelty", 0.0) * novelty
        # ).float()
        reward = (
                self.w_step["preference"] * pref +
                self.w_step["adaptivity"] * adapt +  # 注释掉适应性奖励
                self.w_step.get("novelty", 0.0) * novelty
        ).float()

        # ------- transition: append chosen item & simulated answer -------
        next_pos = self.step + 1
        done = torch.zeros((B,), device=self.device, dtype=torch.float32)

        for b in range(B):
            # if this sample already finished (next_pos >= valid_len), mark done and skip
            if next_pos >= int(self.valid_lens[b].item()):
                done[b] = 1.0
                continue
            # set generated item
            self.gen_seq[b, next_pos] = chosen[b]

            # simulate answer using PRE-STEP mastery of chosen (no leakage)
            p_correct = self._pre_yt.gather(1, chosen.view(-1,1)).squeeze(1)[b].item()
            self.gen_ans[b, next_pos] = 1 if p_correct >= 0 else 0

        # advance step
        self.step += 1

        # update next state's cache if not finished for all
        if self.step < self.max_steps:
            self._update_pre_step_cache()

        next_state = {"candidate_ids": self._pre_cand_ids, "candidate_features": self._pre_cand_feat}
        info = {"chosen_items": chosen.detach().cpu().tolist()}
        return next_state, reward, done, info

    # -------- Terminal reward metrics (Eq 19/20/21) --------

    @torch.no_grad()
    def _compute_topk_from_policy_logits(self, cand_ids: torch.Tensor, policy_logits: torch.Tensor, k: int) -> torch.Tensor:
        """
        cand_ids: [B,Kc] global ids
        policy_logits: [B,Kc]
        returns global topk ids [B,k]
        """
        k = min(k, cand_ids.size(1))
        topk_idx = torch.topk(policy_logits, k=k, dim=-1).indices
        return cand_ids.gather(1, topk_idx)

    @torch.no_grad()
    def record_policy_topk(self, policy_logits: torch.Tensor):
        """
        Called by rollout loop at each step to record policy topk recommendations (global ids),
        aligned to "current step".
        """
        B = self._pre_cand_ids.size(0)
        topk_global = self._compute_topk_from_policy_logits(self._pre_cand_ids, policy_logits, self.topk)  # [B,topk]
        for b in range(B):
            self.topk_recs_policy[b].append([int(x) for x in topk_global[b].detach().cpu().tolist()])

    @torch.no_grad()
    def compute_final_quality(self, policy_weight: Optional[Dict[str,float]] = None, compute_all: bool = False) -> Dict[str, torch.Tensor]:
        """
        Return per-sample metrics + final_quality = weighted sum.
        """
        w = policy_weight or {"effectiveness": 1.0, "adaptivity": 1.0, "diversity": 1.0}
        need_eff = compute_all or (w.get("effectiveness", 0.0) != 0.0)
        need_adp = compute_all or (w.get("adaptivity", 0.0) != 0.0)
        need_div = compute_all or (w.get("diversity", 0.0) != 0.0)

        B, L = self.orig_seq.shape
        K = self.topk

        eff_scores = torch.zeros((B,), device=self.device)
        adapt_scores = torch.zeros((B,), device=self.device)
        div_scores = torch.zeros((B,), device=self.device)

        # build topk tensor [B, L-1, K] (pad with pad_val)
        topk_tensor = torch.full((B, L-1, K), self.pad_val, device=self.device, dtype=torch.long)
        for b in range(B):
            steps = min(len(self.topk_recs_policy[b]), L-1)
            start_t = int(getattr(self, "start_t", 0))
            for s in range(steps):
                t = start_t + s
                if t >= (L - 1):
                    break
                recs = self.topk_recs_policy[b][s][:K]
                if len(recs) < K:
                    recs = recs + [self.pad_val] * (K - len(recs))
                topk_tensor[b, t] = torch.tensor(recs, device=self.device, dtype=torch.long)

        if need_adp:
            # -------- adaptivity Eq.(19) over all recs --------
            # adapt_scores = torch.zeros((B,), device=self.device)
            if self.diff_map is None:
                adapt_scores[:] = 0.0
            else:
                for b in range(B):
                    valid_len = int(self.valid_lens[b].item())
                    if valid_len <= 1:
                        continue

                    # history diffs/results from generated sequence up to valid_len
                    hist_items = self.gen_seq[b, :valid_len].detach().cpu().numpy().tolist()
                    hist_ans = self.gen_ans[b, :valid_len].detach().cpu().numpy().tolist()

                    # pre-compute per time step delta_t (as in Metrics.calculate_adaptivity_tensor)
                    history_diffs = []
                    history_results = []
                    for t in range(valid_len - 1):
                        it = hist_items[t]
                        if it != self.pad_val and it > 1:
                            history_diffs.append(self.diff_map.get_difficulty_norm(it, default=2))
                            history_results.append(float(hist_ans[t]))

                    total = 0.0
                    cnt = 0
                    for t in range(min(valid_len - 1, topk_tensor.size(1))):
                        if len(history_diffs[:t]) < self.T // 2:
                            delta = 1.0
                        else:
                            start = max(0, t - self.T)
                            recent_diffs = history_diffs[start:t]
                            recent_res = history_results[start:t]
                            if len(recent_diffs) > 0:
                                num = sum(d * r for d, r in zip(recent_diffs, recent_res))
                                den = sum(recent_res) + self.eps
                                delta = num / den
                            else:
                                delta = 1.0

                        for k in range(K):
                            rec = int(topk_tensor[b, t, k].item())
                            if rec != self.pad_val and rec > 1:
                                rec_diff = self.diff_map.get_difficulty_norm(rec, default=2)
                                val = 1.0 - abs(delta - rec_diff)
                                total += val
                                cnt += 1
                    adapt_scores[b] = total / max(1, cnt)

        if need_eff:
            # -------- effectiveness Eq.(20) over all recs --------
            # pb: KT on original, pa: KT on generated
            # We need yt tensors [B, L-1, N]; try to get from base_model if it provides yt for full sequence.
            def _run_kt_like(seq, ts, idx, ans):
                out = self.base_model(seq, ts, idx, ans, self.graph, self.hypergraph_list)
                if isinstance(out, dict):
                    yt = out.get("yt", None)
                else:
                    yt = out[3] if len(out) > 3 else None
                if yt is None:
                    return None
                if yt.dim() == 3 and yt.size(1) >= 1:
                    return yt  # [B, L-1, S]
                return None

            yt_before = _run_kt_like(self.orig_seq, self.orig_ts, self.orig_idx, self.orig_ans)
            yt_after = _run_kt_like(self.gen_seq, self.orig_ts, self.orig_idx, self.gen_ans)
            # eff_scores = torch.zeros((B,), device=self.device)

            if yt_before is None or yt_after is None:
                eff_scores[:] = 0.0
            else:
                # ensure same time dim
                Tm = min(yt_before.size(1), yt_after.size(1), L - 1)
                yt_before = yt_before[:, :Tm]
                yt_after = yt_after[:, :Tm]
                S = yt_before.size(-1)

                for b in range(B):
                    valid_len = int(self.valid_lens[b].item())
                    if valid_len <= 1:
                        continue
                    total = 0.0
                    cnt = 0
                    for t in range(min(valid_len - 1, Tm)):
                        if int(self.orig_seq[b, t].item()) == self.pad_val:
                            continue
                        recs = topk_tensor[b, t]  # [K]
                        for k in range(K):
                            r = int(recs[k].item())
                            if r == self.pad_val:
                                continue
                            if 0 <= r < S:
                                pb = float(yt_before[b, t, r].item())
                                pa = float(yt_after[b, t, r].item())
                                if pb < 0.9 and pa > 0:
                                    gain = (pa - pb) / (1.0 - pb)
                                    total += gain
                                    cnt += 1
                    eff_scores[b] = total / max(1, cnt)

        if need_div:
            # -------- diversity Eq.(21) over all recs (per-sample) --------
            # div_scores = torch.zeros((B,), device=self.device)
            if self.hidden_item is None:
                div_scores[:] = 0.0
            else:
                emb = self.hidden_item  # [N,d]
                for b in range(B):
                    valid_len = int(self.valid_lens[b].item())
                    if valid_len <= 1:
                        continue
                    # collect all recommended items in valid time steps
                    items: List[int] = []
                    for t in range(min(valid_len - 1, topk_tensor.size(1))):
                        for k in range(K):
                            r = int(topk_tensor[b, t, k].item())
                            if r != self.pad_val and r > 1:
                                items.append(r)
                    if len(items) < 2:
                        div_scores[b] = 0.0
                        continue
                    e = emb[torch.tensor(items, device=self.device)]  # [M,d]
                    e = e / (e.norm(dim=-1, keepdim=True) + 1e-8)
                    sim = e @ e.t()  # [M,M]
                    # upper triangle without diagonal
                    M = sim.size(0)
                    triu = torch.triu(sim, diagonal=1)
                    vals = triu[triu != 0]
                    if vals.numel() == 0:
                        div_scores[b] = 0.0
                    else:
                        div_scores[b] = (1.0 - vals).mean()

        final_quality = (
            w["effectiveness"] * eff_scores +
            w["adaptivity"] * adapt_scores +
            w["diversity"] * div_scores
        )

        return {
            "effectiveness": eff_scores,
            "adaptivity": adapt_scores,
            "diversity": div_scores,
            "final_quality": final_quality
        }


# ------------------------- Optimizer wrapper -------------------------

class RLPathOptimizer:
    """
    Wraps:
      - OnlineLearningPathEnv
      - PolicyValueNet
      - PPOTrainer
    """
    def __init__(
        self,
        base_model: nn.Module,
        num_items: int,
        data_name: str,
        device: torch.device,
        pad_val: int = 0,
        topk: int = 5,
        cand_k: int = 5,
        history_window_T: int = 10,
        rl_lr: float = 3e-4,
        policy_hidden: int = 128,
        ppo_config: Optional[PPOConfig] = None,
        step_reward_weights: Optional[Dict[str,float]] = None,
        final_reward_weights: Optional[Dict[str,float]] = None,
        terminal_reward_scale: float = 1.0,
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
            w_step=step_reward_weights
        )
        # feature dim is determined at runtime after reset; we will init lazily
        self.policy: Optional[PolicyValueNet] = None
        self.trainer: Optional[PPOTrainer] = None
        self.rl_lr = rl_lr
        self.policy_hidden = policy_hidden
        self.ppo_config = ppo_config or PPOConfig()
        self.final_reward_weights = final_reward_weights or {"effectiveness": 1.0, "adaptivity": 1.0, "diversity": 1.0}
        self.terminal_reward_scale = terminal_reward_scale

    def _lazy_init(self, feat_dim: int):
        if self.policy is None:
            self.policy = PolicyValueNet(feat_dim=feat_dim, hidden_dim=self.policy_hidden).to(self.device)
            self.trainer = PPOTrainer(self.policy, lr=self.rl_lr, config=self.ppo_config)


    @torch.no_grad()
    def ensure_initialized(
        self,
        tgt: torch.Tensor,
        tgt_timestamp: torch.Tensor,
        tgt_idx: torch.Tensor,
        ans: torch.Tensor,
        graph=None,
        hypergraph_list=None,
    ) -> None:
        """Initialize policy/trainer lazily using a real batch (no rollout).
        Prevents calling rl.policy.train()/eval() when policy is still None.
        """
        if self.policy is not None and self.trainer is not None:
            return
        state = self.env.reset(tgt, tgt_timestamp, tgt_idx, ans, graph=graph, hypergraph_list=hypergraph_list)
        cand_feat = state["candidate_features"]
        self._lazy_init(int(cand_feat.size(-1)))

    def collect_trajectory(
        self,
        tgt, tgt_timestamp, tgt_idx, ans,
        graph=None, hypergraph_list=None,
        compute_all=False
    ) -> Dict[str, torch.Tensor]:
        """
        Rollout policy for one batch, over all valid time steps.

        Returns a dict containing flattened tensors for PPO update.
        """
        state = self.env.reset(tgt, tgt_timestamp, tgt_idx, ans, graph=graph, hypergraph_list=hypergraph_list)
        cand_feat = state["candidate_features"]  # [B,K,F]
        self._lazy_init(cand_feat.size(-1))

        B = cand_feat.size(0)
        max_steps = self.env.max_steps

        # lists over time
        all_cand_feat: List[torch.Tensor] = []
        all_actions: List[torch.Tensor] = []
        all_logp: List[torch.Tensor] = []
        all_values: List[torch.Tensor] = []
        all_rewards: List[torch.Tensor] = []
        all_dones: List[torch.Tensor] = []

        done = torch.zeros((B,), device=self.device, dtype=torch.float32)

        for t in range(max_steps):
            cand_feat = state["candidate_features"]  # [B,K,F]
            logits, value = self.policy(cand_feat)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()              # [B]
            logp = dist.log_prob(action)        # [B]
            entropy = dist.entropy()            # [B] (unused here; PPO uses mean entropy in update)

            # record policy topK BEFORE stepping (aligned with current step)
            self.env.record_policy_topk(logits)

            next_state, reward, step_done, _ = self.env.step_env(action)

            # update done mask: once done, remain done
            done = torch.maximum(done, step_done)

            all_cand_feat.append(cand_feat)
            all_actions.append(action)
            all_logp.append(logp)
            all_values.append(value)
            all_rewards.append(reward)
            all_dones.append(done.clone())

            state = next_state

            # if all done, break
            if float(done.min().item()) >= 1.0:
                break

        # terminal reward (final quality)
        final_metrics = self.env.compute_final_quality(self.final_reward_weights, compute_all)
        terminal_r = final_metrics["final_quality"] * self.terminal_reward_scale  # [B]
        # add terminal reward to last collected reward step (for each sample that had at least 1 step)
        if len(all_rewards) > 0:
            all_rewards[-1] = all_rewards[-1] + terminal_r

        # stack to [T,B,...]
        rewards = torch.stack(all_rewards, dim=0)  # [T,B]
        values = torch.stack(all_values, dim=0)   # [T,B]
        dones = torch.stack(all_dones, dim=0)     # [T,B]

        # compute GAE
        adv, rets = self.trainer.compute_gae(rewards, values, dones)

        # flatten valid steps: we keep all steps, but mask out already-done steps
        T, B = rewards.shape
        valid_mask = (1.0 - dones)  # [T,B]  1 means still active at that step
        # include the step where it becomes done? in our env done is cumulative; last step for a sample has done=1,
        # so mask would drop it. We want to keep steps where action was taken. Use per-step active before update:
        # approximate by shifting:
        active = torch.ones_like(dones)
        active[1:] = 1.0 - dones[:-1]
        active[0] = 1.0  # first step always active if rollout happened
        active = active.clamp(0,1)

        cand_feat_t = torch.stack(all_cand_feat, dim=0)  # [T,B,K,F]
        actions_t = torch.stack(all_actions, dim=0)      # [T,B]
        logp_t = torch.stack(all_logp, dim=0)            # [T,B]
        values_t = values                                # [T,B]
        adv_t = adv                                      # [T,B]
        rets_t = rets                                    # [T,B]

        # flatten
        active_flat = active.reshape(-1).bool()
        cand_feat_flat = cand_feat_t.reshape(T*B, cand_feat_t.size(2), cand_feat_t.size(3))[active_flat]
        actions_flat = actions_t.reshape(-1)[active_flat]
        logp_flat = logp_t.reshape(-1)[active_flat]
        values_flat = values_t.reshape(-1)[active_flat]
        adv_flat = adv_t.reshape(-1)[active_flat]
        rets_flat = rets_t.reshape(-1)[active_flat]

        return {
            "cand_feat": cand_feat_flat,
            "actions": actions_flat,
            "old_logp": logp_flat,
            "old_values": values_flat,
            "advantages": adv_flat,
            "returns": rets_flat,
            "final_metrics": final_metrics,  # per-sample tensor
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
    max_batches: int = 20,
    compute_all=True
) -> Dict[str, float]:
    """
    Evaluation over a few batches: report mean final metrics.
    """
    # policy is lazily initialized; init once using the first batch before calling eval()
    if rl.policy is None:
        first = next(iter(data_loader))
        tgt, tgt_timestamp, tgt_idx, ans = first[0], first[1], first[2], first[3]
        tgt = tgt.to(device); tgt_timestamp = tgt_timestamp.to(device); tgt_idx = tgt_idx.to(device); ans = ans.to(device)
        rl.ensure_initialized(tgt, tgt_timestamp, tgt_idx, ans, graph=graph, hypergraph_list=hypergraph_list)
    rl.policy.eval()
    agg = {"effectiveness": 0.0, "adaptivity": 0.0, "diversity": 0.0, "final_quality": 0.0}
    n = 0

    for i, batch in enumerate(data_loader):
        if i >= max_batches:
            break
        # batch structure follows your DataLoader: (tgt, tgt_timestamp, tgt_idx, ans, ...)
        tgt, tgt_timestamp, tgt_idx, ans = batch[0], batch[1], batch[2], batch[3]
        tgt = tgt.to(device); tgt_timestamp = tgt_timestamp.to(device); tgt_idx = tgt_idx.to(device); ans = ans.to(device)

        rollout = rl.collect_trajectory(tgt, tgt_timestamp, tgt_idx, ans, graph=graph, hypergraph_list=hypergraph_list, compute_all=compute_all)
        fm = rollout["final_metrics"]
        for k in agg:
            agg[k] += float(fm[k].mean().detach().cpu())
        n += 1

    if n == 0:
        return {k: 0.0 for k in agg}
    return {k: v / n for k, v in agg.items()}
