rl_adjuster_new.py
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


# ------------------------- 小工具函数 -------------------------

def _ensure_2d(x: torch.Tensor) -> torch.Tensor:
    """
    确保张量是二维的
    
    Args:
        x: 输入张量
        
    Returns:
        二维张量
    """
    if x.dim() == 1:  # 如果是一维张量
        return x.unsqueeze(0)  # 增加一个维度变为二维 [1, original_size]
    return x  # 否则返回原张量


def _to_device(x, device):
    """
    将数据移动到指定设备
    
    Args:
        x: 数据（张量或其他类型）
        device: 目标设备
        
    Returns:
        移动到目标设备的数据
    """
    if x is None:  # 如果输入为空
        return None
    if isinstance(x, torch.Tensor):  # 如果是张量
        return x.to(device)  # 移动到指定设备
    return x  # 否则返回原数据


def _cosine_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    计算余弦相似度
    
    Args:
        a: 张量a，形状 [..., d]
        b: 张量b，形状 [..., d]
        eps: 防止除零的小常数
        
    Returns:
        余弦相似度张量
    """
    # 归一化张量a
    a_n = a / (a.norm(dim=-1, keepdim=True) + eps)  # [ ..., d ] -> [ ..., d ]
    # 归一化张量b
    b_n = b / (b.norm(dim=-1, keepdim=True) + eps)  # [ ..., d ] -> [ ..., d ]
    # 计算点积得到余弦相似度
    return (a_n * b_n).sum(dim=-1)  # [ ... ]


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, dim=None, eps: float = 1e-8) -> torch.Tensor:
    """
    计算掩码平均值
    
    Args:
        x: 输入张量
        mask: 掩码张量（布尔型，可广播）
        dim: 求平均的维度
        eps: 防止除零的小常数
        
    Returns:
        掩码平均值张量
    """
    # mask: 可广播的布尔张量
    x = x * mask.to(x.dtype)  # 将掩码之外的值设为0
    # 计算掩码中True的数量，限制最小值为eps防止除零
    denom = mask.to(x.dtype).sum(dim=dim, keepdim=True).clamp_min(eps)
    # 计算平均值并去除指定维度
    return (x.sum(dim=dim, keepdim=True) / denom).squeeze(dim if dim is not None else -1)


# ------------------------- 难度和映射加载器 -------------------------

@dataclass
class DifficultyMapping:
    """难度映射数据类"""
    idx2u: Dict[int, int]  # 内部索引到用户ID的映射
    difficulty_by_uid: Dict[int, int]  # 用户ID到难度等级的映射（例如1/2/3）

    @staticmethod
    def load_from_options(data_name: str) -> Optional["DifficultyMapping"]:
        """
        从dataLoader.Options加载idx2u和difficulty文件（您的仓库）
        如果当前运行时不可用，则返回None
        
        Args:
            data_name: 数据集名称
            
        Returns:
            难度映射对象或None
        """
        try:
            from dataLoader import Options  # 导入您项目中的Options类
        except Exception:
            return None

        try:
            opt = Options(data_name)  # 创建Options实例
            # 加载idx2u映射
            with open(opt.idx2u_dict, "rb") as f:
                idx2u = pickle.load(f)

            difficulty_by_uid: Dict[int, int] = {}  # 初始化难度映射字典
            # 读取难度文件
            with open(opt.difficult_file, "r", encoding="utf-8") as f:
                next(f)  # 跳过标题行
                for line in f:
                    parts = line.strip().split(",")  # 按逗号分割
                    if len(parts) < 2:  # 如果列数不够则跳过
                        continue
                    try:
                        uid = int(parts[0].strip())  # 用户ID
                        d = int(parts[1].strip())    # 难度等级
                        difficulty_by_uid[uid] = d   # 添加到映射
                    except Exception:
                        continue

            return DifficultyMapping(idx2u=idx2u, difficulty_by_uid=difficulty_by_uid)
        except Exception:
            return None

    def get_difficulty_raw(self, idx: int, default: int = 2) -> int:
        """
        获取原始难度等级
        
        Args:
            idx: 内部u2idx映射的项目ID
            default: 默认难度等级
            
        Returns:
            原始难度等级
        """
        # idx是内部u2idx映射的项目ID
        # idx2u可能是字典{idx -> uid}或列表，其中idx2u[idx] = uid
        uid = None
        try:
            if hasattr(self.idx2u, "get"):  # 如果idx2u是字典
                uid = self.idx2u.get(int(idx), None)
            else:  # 如果idx2u是列表
                i = int(idx)
                if 0 <= i < len(self.idx2u):  # 检查索引是否有效
                    uid = self.idx2u[i]
        except Exception:
            uid = None

        if uid is None:  # 如果找不到uid则返回默认值
            return default

        # difficulty_by_uid通常以原始uid（challenge_id）为键
        try:
            return int(self.difficulty_by_uid.get(int(uid), default))
        except Exception:
            return int(self.difficulty_by_uid.get(uid, default))

    def get_difficulty_norm(self, idx: int, default: int = 2) -> float:
        """
        获取归一化难度（1/2/3 -> 0/0.5/1）
        
        Args:
            idx: 内部索引
            default: 默认难度
            
        Returns:
            归一化难度值（0-1之间）
        """
        # 映射原始难度1/2/3到0/0.5/1
        d = self.get_difficulty_raw(idx, default=default)  # 获取原始难度
        d = max(1, min(3, int(d)))  # 限制在1-3范围内
        return (d - 1) / 2.0  # 归一化到0-1范围


# ------------------------- 策略/价值网络 -------------------------

class PolicyValueNet(nn.Module):
    """
    策略价值网络
    
    输入:
        cand_feat: [B, K, F]  (F >= 1) - 候选特征
    输出:
        logits: [B, K] - 逻辑回归值
        value: [B] - 价值估计
    """

    def __init__(self, feat_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        """
        初始化策略价值网络
        
        Args:
            feat_dim: 特征维度
            hidden_dim: 隐藏层维度
            dropout: Dropout比率
        """
        super().__init__()  # 调用父类构造函数
        self.feat_dim = feat_dim      # 特征维度
        self.hidden_dim = hidden_dim  # 隐藏层维度

        # 候选项目MLP：处理每个候选的特征
        self.cand_mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),  # [F] -> [H]
            nn.Tanh(),                       # 激活函数
            nn.Dropout(dropout),             # Dropout层
            nn.Linear(hidden_dim, hidden_dim), # [H] -> [H]
            nn.Tanh(),                       # 激活函数
        )
        # 逻辑回归头：为每个候选生成logit
        self.logit_head = nn.Linear(hidden_dim, 1)  # [H] -> [1]

        # 价值头：聚合候选表示来预测价值
        self.value_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  # [H] -> [H]
            nn.Tanh(),                         # 激活函数
            nn.Dropout(dropout),               # Dropout层
            nn.Linear(hidden_dim, 1)           # [H] -> [1]
        )

    def forward(self, cand_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            cand_feat: 候选特征 [B, K, F]
            
        Returns:
            logits: 逻辑回归值 [B, K]
            value: 价值估计 [B]
        """
        h = self.cand_mlp(cand_feat)  # [B, K, F] -> [B, K, H] - 处理候选特征
        logits = self.logit_head(h).squeeze(-1)  # [B, K, H] -> [B, K] - 生成逻辑回归值

        # 注意力池化用于价值预测
        att = torch.softmax(logits, dim=-1).unsqueeze(-1)  # [B, K] -> [B, K, 1] - 注意力权重
        pooled = (h * att).sum(dim=1)  # [B, K, H] * [B, K, 1] -> [B, H] - 加权求和
        value = self.value_mlp(pooled).squeeze(-1)  # [B, H] -> [B] - 价值估计
        return logits, value


# ------------------------- PPO训练器 -------------------------

@dataclass
class PPOConfig:
    """PPO配置参数"""
    gamma: float = 0.99          # 折扣因子
    gae_lambda: float = 0.95     # GAE lambda参数
    clip_eps: float = 0.2        # PPO裁剪参数
    vf_coef: float = 0.5         # 价值函数系数
    ent_coef: float = 0.01       # 熵系数
    max_grad_norm: float = 0.5   # 最大梯度范数
    ppo_epochs: int = 4          # PPO训练轮数
    minibatch_size: int = 256    # 小批次大小


class PPOTrainer:
    """PPO训练器"""
    def __init__(self, policy: PolicyValueNet, lr: float = 3e-4, config: Optional[PPOConfig] = None):
        """
        初始化PPO训练器
        
        Args:
            policy: 策略网络
            lr: 学习率
            config: PPO配置
        """
        self.policy = policy  # 策略网络
        self.config = config or PPOConfig()  # PPO配置
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=lr)  # Adam优化器

    @torch.no_grad()
    def compute_gae(
            self,
            rewards: torch.Tensor,  # [T, B] - 奖励
            values: torch.Tensor,   # [T, B] - 价值
            dones: torch.Tensor     # [T, B] - 结束标志（如果在t时刻结束则为1，否则为0）
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算广义优势估计（GAE）
        
        Args:
            rewards: 奖励张量 [T, B]
            values: 价值张量 [T, B]
            dones: 结束标志张量 [T, B]
            
        Returns:
            advantages: 优势 [T, B]
            returns: 回报 [T, B]
        """
        T, B = rewards.shape  # 获取时间步T和批次B
        adv = torch.zeros_like(rewards)  # [T, B] - 初始化优势张量
        last_gae = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)  # [B] - 上一时间步的GAE
        last_value = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)  # [B] - 上一时间步的价值

        for t in reversed(range(T)):  # 从最后时间步向前遍历
            mask = 1.0 - dones[t]  # [B] - 掩码（如果done则为0，否则为1）
            # 计算TD误差
            delta = rewards[t] + self.config.gamma * last_value * mask - values[t]
            # 更新GAE
            last_gae = delta + self.config.gamma * self.config.gae_lambda * mask * last_gae
            adv[t] = last_gae  # 存储当前时间步的优势
            last_value = values[t]  # 更新上一时间步的价值

        returns = adv + values  # [T, B] - 计算回报
        return adv, returns

    def update(
            self,
            cand_feat: torch.Tensor,  # [N, K, F] - 在(t,b)有效步骤上展开的候选特征
            actions: torch.Tensor,    # [N] - 动作
            old_logp: torch.Tensor,   # [N] - 旧的对数概率
            old_values: torch.Tensor, # [N] - 旧的价值
            advantages: torch.Tensor, # [N] - 优势
            returns: torch.Tensor     # [N] - 回报
    ) -> Dict[str, float]:
        """
        更新策略
        
        Args:
            cand_feat: 候选特征 [N, K, F]
            actions: 动作 [N]
            old_logp: 旧的对数概率 [N]
            old_values: 旧的价值 [N]
            advantages: 优势 [N]
            returns: 回报 [N]
            
        Returns:
            损失字典
        """
        cfg = self.config  # 获取配置
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std().clamp_min(1e-8))

        N = cand_feat.size(0)  # 获取样本数N
        idx = torch.randperm(N, device=cand_feat.device)  # [N] - 随机排列的索引

        # 初始化损失统计
        losses = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "total_loss": 0.0}

        for _ in range(cfg.ppo_epochs):  # 遍历PPO训练轮数
            for start in range(0, N, cfg.minibatch_size):  # 遍历小批次
                mb = idx[start:start + cfg.minibatch_size]  # 获取小批次索引
                mb_feat = cand_feat[mb]      # [minibatch_size, K, F] - 小批次候选特征
                mb_act = actions[mb]         # [minibatch_size] - 小批次动作
                mb_old_logp = old_logp[mb]   # [minibatch_size] - 小批次旧对数概率
                mb_old_val = old_values[mb]  # [minibatch_size] - 小批次旧价值
                mb_adv = advantages[mb]      # [minibatch_size] - 小批次优势
                mb_ret = returns[mb]         # [minibatch_size] - 小批次回报

                # 通过策略网络获取新逻辑回归值和价值
                logits, value = self.policy(mb_feat)  # logits: [minibatch_size, K], value: [minibatch_size]
                
                # 创建分类分布
                dist = torch.distributions.Categorical(logits=logits)  # 分类分布
                logp = dist.log_prob(mb_act)  # [minibatch_size] - 新对数概率
                entropy = dist.entropy().mean()  # 标量 - 平均熵

                # 计算重要性采样比率
                ratio = torch.exp(logp - mb_old_logp)  # [minibatch_size]
                
                # 计算PPO裁剪的目标
                surr1 = ratio * mb_adv  # [minibatch_size]
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * mb_adv  # [minibatch_size]
                policy_loss = -torch.min(surr1, surr2).mean()  # 标量 - 策略损失

                # 价值损失
                value_loss = F.mse_loss(value, mb_ret)  # 标量

                # 总损失
                loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy  # 标量

                # 反向传播和参数更新
                self.opt.zero_grad(set_to_none=True)  # 清空梯度
                loss.backward()  # 反向传播
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)  # 梯度裁剪
                self.opt.step()  # 更新参数

                # 累积损失值
                losses["policy_loss"] += float(policy_loss.detach().cpu())  # 策略损失
                losses["value_loss"] += float(value_loss.detach().cpu())     # 价值损失
                losses["entropy"] += float(entropy.detach().cpu())           # 熵
                losses["total_loss"] += float(loss.detach().cpu())           # 总损失

        # 计算平均损失
        denom = max(1, cfg.ppo_epochs * math.ceil(N / cfg.minibatch_size))  # 分母
        for k in losses:
            losses[k] /= denom  # 计算平均值
        return losses


# ------------------------- 环境 -------------------------

class OnlineLearningPathEnv:
    """
    整个序列上的在线RL（每个有效时间步）。

    在每个时间步t:
      - 我们从基础模型分布在当前前缀中构建候选集。
      - 策略选择一个候选作为下一个项目。
      - 我们使用KT概率模拟所选项目的答案（>=0.5 => 正确）。
      - 将所选项目/答案追加到生成的前缀。
      - 移动到下一步。

    我们记录:
      - topk_recs_policy[b][t] : 策略分布top-K映射到全局项目ID
      - chosen_items[b][t]     : 选择的项目ID
      - generated_ans[b][t]    : 模拟答案（对于超出种子前缀的生成项目）
    """

    def __init__(
            self,
            base_model: nn.Module,          # 基础模型
            num_items: int,                 # 项目数量
            data_name: str,                 # 数据集名称
            device: torch.device,           # 设备
            pad_val: int = 0,               # 填充值
            topk: int = 10,                 # Top-K数量
            cand_k: int = 50,               # 候选数量
            history_window_T: int = 10,     # 历史窗口大小
            epsilon: float = 1e-5,          # 小常数
            w_step: Optional[Dict[str, float]] = None,  # 步骤权重
    ):
        """
        初始化在线学习路径环境
        
        Args:
            base_model: 基础模型
            num_items: 项目数量
            data_name: 数据集名称
            device: 计算设备
            pad_val: 填充值
            topk: Top-K数量
            cand_k: 候选数量
            history_window_T: 历史窗口大小
            epsilon: 小常数
            w_step: 步骤权重
        """
        self.base_model = base_model  # 基础模型
        self.num_items = num_items    # 项目数量
        self.data_name = data_name    # 数据集名称
        self.device = device          # 设备
        self.pad_val = pad_val        # 填充值
        self.topk = topk              # Top-K数量
        self.cand_k = cand_k          # 候选数量
        self.T = history_window_T     # 历史窗口大小
        self.eps = epsilon            # 小常数

        # 步骤奖励权重
        self.w_step = w_step or {
            "preference": 1.0,  # 偏好权重
            "adaptivity": 1.0,  # 适应性权重
            "novelty": 0.2,     # 新颖性权重
        }

        # 尝试加载映射+难度（用于Eq.19）
        self.diff_map = DifficultyMapping.load_from_options(data_name)

        # 每个episode的缓冲区
        self.graph = None             # 关系图
        self.hypergraph_list = None   # 超图列表

        self.orig_seq = None          # 原始序列
        self.orig_ts = None           # 原始时间戳
        self.orig_idx = None          # 原始索引
        self.orig_ans = None          # 原始答案

        self.valid_lens = None        # 有效长度
        self.max_steps = None         # 最大步数

        # 生成的前缀从第一个有效项目开始（种子长度=1）
        self.gen_seq = None           # 生成的序列
        self.gen_ans = None           # 生成的答案

        self.hidden_item = None       # [N, d] 用于多样性和相似性的嵌入

        # 记录
        self.topk_recs_policy: List[List[List[int]]] = []  # B x steps x topk - 策略top-K记录
        self.chosen_items: List[List[int]] = []            # B x steps - 选择项目记录
        self.step = 0                                      # 当前步数

        # "前一步"的缓存
        self._pre_base_probs = None   # [B, N] 当前步骤的概率（用于偏好）
        self._pre_yt = None           # [B, N] 掌握概率（项目级别）
        self._pre_cand_ids = None     # [B, Kcand] 候选ID
        self._pre_cand_feat = None    # [B, Kcand, F] 候选特征
        self._pre_delta = None        # [B] delta_t (Eq.19)
        self._pre_recent_corr = None  # [B] 最近正确率
        self._pre_recent_davg = None  # [B] 最近平均难度
        self._pre_path_mean_emb = None # [B, d] 路径平均嵌入
        self._pre_last_emb = None     # [B, d] 最后嵌入


    @torch.no_grad()
    def _sanitize_seq_indices(self, seq: torch.Tensor) -> torch.Tensor:
        """
        防止CUDA设备端断言，确保所有索引都在[-N, N-1]范围内。
        在此仓库中，有效项目ID为0..(num_items-1)。超出范围的值被替换为PAD。
        
        Args:
            seq: 序列张量
            
        Returns:
            清理后的序列张量
        """
        if seq.dtype != torch.long:  # 如果不是长整型则转换
            seq = seq.long()
        # 允许范围[-num_items, -1]内的负索引（PyTorch支持），但其他任何值都是无效的
        invalid = (seq >= self.num_items) | (seq < -self.num_items)  # 标记无效索引
        if invalid.any():  # 如果有无效索引
            # 仅在每个进程第一次时打印，避免日志刷屏
            if not hasattr(self, "_printed_oob_warning"):
                self._printed_oob_warning = True
                mx = int(seq.max().detach().cpu())  # 最大值
                mn = int(seq.min().detach().cpu())  # 最小值
                cnt = int(invalid.sum().detach().cpu())  # 无效数量
                print(
                    f"[WARN][RL Env] Found {cnt} out-of-range item indices in batch (min={mn}, max={mx}, num_items={self.num_items}). Replacing with PAD={self.pad_val}.")
            seq = seq.clone()  # 克隆张量
            seq[invalid] = int(self.pad_val)  # 将无效索引替换为填充值
        return seq  # 返回清理后的序列


    @torch.no_grad()
    def _forward_base(self, seq: torch.Tensor, ts: torch.Tensor, idx: torch.Tensor, ans: torch.Tensor):
        """
        基础模型前向传播
        
        Args:
            seq: 序列 [B, L]
            ts: 时间戳 [B, L]
            idx: 索引 [B] (每个序列一个级联ID)
            ans: 答案 [B, L]
            
        Returns:
            probs_last: [B, N] 下一项的概率分布
            yt_last: [B, N] 掌握估计（项目级别）
            hidden: [N, d] 用于相似性/多样性的项目嵌入
        """
        # 调用基础模型
        out = self.base_model(seq, ts, idx, ans, self.graph, self.hypergraph_list)
        # out通常是元组 (pred_flat, pred_res, kt_mask, yt, hidden, status_emb, ...)

        # 兼容解包：允许元组/列表/字典
        if isinstance(out, dict):
            pred_flat = out.get("pred_flat", out.get("pred", None))  # 预测平面
            yt = out.get("yt", None)      # 掌握概率
            hidden = out.get("hidden", out.get("item_emb", None))  # 隐藏状态/项目嵌入
        else:
            # 常见：(pred_flat, yt, hidden, ...)
            pred_flat = out[0] if len(out) > 0 else None  # pred_flat: [B*(L-1), N] 或 [B, L-1, N]
            yt = out[3] if len(out) > 3 else None         # yt: [B, L-1, S] 或 [B, L-1, N]
            hidden = out[2] if len(out) > 2 else None     # hidden: [N, d] 或 [B, N, d]

        if pred_flat is None:  # 如果没有预测平面则报错
            raise RuntimeError("base_model forward must provide pred logits (pred_flat/pred).")

        B, L = seq.shape  # 获取批次B和序列长度L

        # pred_flat通常是[B*(L-1), N]；我们需要最后一步的分布
        if pred_flat.dim() == 2 and pred_flat.size(0) == B * (L - 1):  # 如果是扁平化形式
            pred_logits = pred_flat.view(B, L - 1, -1)[:, -1]  # [B*(L-1), N] -> [B, L-1, N] -> [B, N]
        elif pred_flat.dim() == 3:  # 如果已经是3维形式
            pred_logits = pred_flat[:, -1]  # [B, L-1, N] -> [B, N]
        else:
            raise RuntimeError(f"Unsupported pred_flat shape: {tuple(pred_flat.shape)}")

        probs_last = torch.softmax(pred_logits, dim=-1)  # [B, N] -> [B, N] - 计算概率

        # yt: 期望[B, L-1, N]或[B, L-1, S]，其中S==N（项目级别的KT）
        yt_last = None  # 初始化掌握概率
        if yt is not None:  # 如果yt存在
            if yt.dim() == 3 and yt.size(0) == B:  # 如果是3维且批次匹配
                yt_last = yt[:, -1]  # [B, L-1, S] -> [B, S] - 获取最后一步
            elif yt.dim() == 2 and yt.size(0) == B:  # 如果是2维且批次匹配
                yt_last = yt  # [B, S] - 直接使用
        if yt_last is None:  # 如果yt_last仍为空
            # 后备方案：使用概率作为掌握代理（不是理想的，但可以运行）
            yt_last = probs_last  # [B, N] - 使用概率作为掌握概率

        if hidden is not None:  # 如果隐藏状态存在
            if hidden.dim() == 3:  # 如果是3维 [B, N, d]
                # 一些模型输出[B, N, d]
                hidden_item = hidden[0]  # [N, d] - 取第一个批次的嵌入
            else:
                hidden_item = hidden  # [N, d] - 直接使用
        else:
            hidden_item = None  # 没有隐藏项目嵌入

        return probs_last, yt_last, hidden_item  # 返回最后一步的概率、掌握概率和项目嵌入


    def _difficulty_norm(self, idx_tensor: torch.Tensor) -> torch.Tensor:
        """
        归一化难度
        
        Args:
            idx_tensor: 索引张量 [B, K]
            
        Returns:
            归一化难度张量 [B, K]
        """
        # idx_tensor: [B, K] - 索引张量
        if self.diff_map is None:  # 如果没有难度映射
            # 默认难度=2 -> 归一化值0.5
            return torch.full_like(idx_tensor, 0.5, dtype=torch.float32)  # [B, K] - 全0.5张量
        # 在Python中循环（Kcand较小）
        B, K = idx_tensor.shape  # 获取批次B和候选K
        out = torch.zeros((B, K), device=idx_tensor.device, dtype=torch.float32)  # [B, K] - 初始化输出
        idx_np = idx_tensor.detach().cpu().numpy()  # [B, K] - 转换为numpy数组
        for b in range(B):  # 遍历批次
            for k in range(K):  # 遍历候选
                out[b, k] = float(self.diff_map.get_difficulty_norm(int(idx_np[b, k]), default=2))  # 获取归一化难度
        return out  # [B, K] - 返回归一化难度


    def _compute_delta_t(self, hist_items: torch.Tensor, hist_ans: torch.Tensor) -> torch.Tensor:
        """
        计算Eq.(19) delta_t使用最近T窗口：
            delta_t = sum(Dif_i * r_i) / (sum(r_i) + eps)
        使用真实答案r_i（这里：为生成项目生成的答案；如果你选择，为种子前缀使用原始答案）。
        hist_items, hist_ans: [B, Lprefix]
        
        Args:
            hist_items: 历史项目 [B, Lprefix]
            hist_ans: 历史答案 [B, Lprefix]
            
        Returns:
            delta_t张量 [B]
        """
        B, L = hist_items.shape  # 获取批次B和长度L
        delta = torch.ones((B,), device=hist_items.device, dtype=torch.float32)  # [B] - 初始化为1

        if self.diff_map is None:  # 如果没有难度映射
            return delta  # 返回默认值（与Metrics默认路径相同）

        items_np = hist_items.detach().cpu().numpy()  # [B, L] - 项目转numpy
        ans_np = hist_ans.detach().cpu().numpy()      # [B, L] - 答案转numpy
        for b in range(B):  # 遍历批次
            # 收集有效历史（排除PAD）
            valid = [(int(items_np[b, t]), float(ans_np[b, t])) for t in range(L) if
                     int(items_np[b, t]) != self.pad_val]  # 有效项目-答案对
            if len(valid) < max(1, self.T // 2):  # 如果有效历史太短
                delta[b] = 1.0  # 设置为1.0
                continue
            # 最后窗口
            window = valid[max(0, len(valid) - self.T):]  # 获取最后T个项目
            num = 0.0  # 分子
            den = 0.0  # 分母
            for it, r in window:  # 遍历窗口中的项目-答案对
                d = float(self.diff_map.get_difficulty_norm(it, default=2))  # 获取难度
                num += d * r  # 累加分子
                den += r      # 累加分母
            delta[b] = float(num / (den + self.eps))  # 计算delta_t
        return delta  # [B] - 返回delta_t


    def _debug_check_inputs(self):
        """
        CPU端健全性检查，确定哪个张量导致CUDA索引越界。
        在可能导致设备端断言的任何模型前向/收集之前调用。
        """
        try:
            seq_cpu = self.orig_seq.detach().cpu()  # 将原始序列移到CPU
            ans_cpu = self.orig_ans.detach().cpu()  # 将原始答案移到CPU
        except Exception as e:
            print("[RL DEBUG] failed to move tensors to cpu:", e)  # 移动失败
            return

        # --- 检查问题/项目索引 ---
        q_min = int(seq_cpu.min().item())  # 最小问题ID
        q_max = int(seq_cpu.max().item())  # 最大问题ID
        n_items = int(self.num_items)      # 项目总数
        if q_min < 0 or q_max >= n_items:  # 如果超出范围
            bad = (seq_cpu < 0) | (seq_cpu >= n_items)  # 标记坏索引
            bad_pos = bad.nonzero(as_tuple=False)       # 获取坏索引位置
            print(
                f"[RL DEBUG][OOB][questions] num_items={n_items} q_min={q_min} q_max={q_max} bad_cnt={int(bad.sum().item())}")
            for p in bad_pos[:10]:  # 显示前10个坏索引
                b, t = int(p[0]), int(p[1])  # 批次、时间步
                print("  bad@", (b, t), "val=", int(seq_cpu[b, t].item()))  # 打印坏索引信息
            raise RuntimeError("[RL DEBUG] Found out-of-bound question/item ids in sequence.")  # 抛出错误

        # --- 检查由DKT.correct_embed使用的正确性索引 ---
        n_correct = 2  # 默认正确性数量
        try:
            # 尝试获取correct_embed的嵌入数量
            ce = getattr(getattr(getattr(self.base_model, "ktmodel", None), "correct_embed", None), "num_embeddings",
                         None)
            if ce is not None:
                n_correct = int(ce)  # 更新正确性数量
        except Exception:
            pass

        a_min = float(ans_cpu.min().item())  # 最小答案值
        a_max = float(ans_cpu.max().item())  # 最大答案值
        if a_min < 0 or a_max > (n_correct - 1):  # 如果超出范围
            # 显示一些唯一值
            uniq = ans_cpu.unique()  # 获取唯一值
            uniq = uniq[:20].tolist() if uniq.numel() > 0 else []  # 获取前20个
            print(
                f"[RL DEBUG][OOB][answers] correct_embed_size={n_correct} ans_min={a_min} ans_max={a_max} uniq_head={uniq}")
            raise RuntimeError("[RL DEBUG] Found out-of-range correctness labels for correct_embed.")  # 抛出错误

        # --- 检查级联索引范围（应在批次长度列表内） ---
        try:
            idx_cpu = self.orig_idx.detach().cpu()  # 将索引移到CPU
            i_min = int(idx_cpu.min().item())  # 最小索引
            i_max = int(idx_cpu.max().item())  # 最大索引
            if i_min < 0:  # 如果最小索引小于0
                print(f"[RL DEBUG][OOB][idx] idx_min={i_min} idx_max={i_max}")
            # 检查超图列表长度（如果提供）
            if self.hypergraph_list is not None:
                # hypergraph_list是时间切片列表，所以idx不用于在MSHGAT中索引它，
                # 但我们仍然为了可见性打印。
                print(f"[RL DEBUG] idx range: [{i_min}, {i_max}] (hypergraph_list_len={len(self.hypergraph_list)})")
        except Exception:
            pass


    def reset(self, tgt, tgt_timestamp, tgt_idx, ans, graph=None, hypergraph_list=None) -> Dict[str, torch.Tensor]:
        """
        重置环境
        
        Args:
            tgt: 目标序列
            tgt_timestamp: 目标时间戳
            tgt_idx: 目标索引
            ans: 答案
            graph: 关系图
            hypergraph_list: 超图列表
            
        Returns:
            状态字典
        """
        self.graph = graph  # 设置关系图
        self.hypergraph_list = hypergraph_list  # 设置超图列表

        self.orig_seq = _ensure_2d(tgt).to(self.device)  # [B, L] - 确保目标序列是2D并移到设备
        self.orig_seq = self._sanitize_seq_indices(self.orig_seq)  # 清理序列索引
        self.orig_ts = _ensure_2d(tgt_timestamp).to(self.device)  # [B, L] - 确保时间戳是2D并移到设备
        self.orig_idx = tgt_idx.to(self.device)  # [B] - 将索引移到设备
        # 此仓库中的DataLoader产生tgt_idx形状为[B]（每个序列一个级联ID），而不是[B,L]。
        # 保持1D以匹配MSHGAT.forward期望。
        if self.orig_idx.dim() == 2 and self.orig_idx.size(1) == 1:  # 如果是2D但第二维为1
            self.orig_idx = self.orig_idx.squeeze(1)  # [B, 1] -> [B] - 压缩维度
        self.orig_ans = _ensure_2d(ans).to(self.device)  # [B, L] - 确保答案是2D并移到设备

        # --- DEBUG: 尽早确定索引越界（CPU端） ---
        self._debug_check_inputs()  # 调试输入检查

        B, L = self.orig_seq.shape  # 获取批次B和长度L
        self.valid_lens = (self.orig_seq != self.pad_val).sum(dim=1).clamp_min(1)  # [B] - 有效长度（至少为1）
        self.max_steps = int(self.valid_lens.max().item()) - 1  # 预测每个步骤的下一个
        self.max_steps = max(0, self.max_steps)  # 确保非负

        # 种子前缀：每个样本的前2个有效项目（需要长度>=2才能从base_model获得下一步预测）
        self.gen_seq = torch.full_like(self.orig_seq, self.pad_val)  # [B, L] - 初始化为填充值
        self.gen_ans = torch.full_like(self.orig_ans, 0)  # [B, L] - 初始化为0

        # 种子长度每样本：如果有>=2则为2，否则为1（<2个有效交互的样本将有零RL步骤）
        self.seed_len = 2  # 种子长度

        for b in range(B):  # 遍历批次
            vlen = int(self.valid_lens[b].item())  # 获取样本b的有效长度
            if vlen >= 1:  # 如果有效长度>=1
                self.gen_seq[b, 0] = self.orig_seq[b, 0]  # 设置第一个项目
                self.gen_ans[b, 0] = self.orig_ans[b, 0]  # 设置第一个答案
            if vlen >= 2:  # 如果有效长度>=2
                self.gen_seq[b, 1] = self.orig_seq[b, 1]  # 设置第二个项目
                self.gen_ans[b, 1] = self.orig_ans[b, 1]  # 设置第二个答案

        # 从最后种子索引开始步骤（seed_len-1），所以第一个RL动作预测位置seed_len
        self.step = 1  # 对应于具有长度为2的前缀
        self.start_t = self.step  # 时间索引在[0, L-2]中我们开始记录策略topK

        self.topk_recs_policy = [[] for _ in range(B)]  # 初始化策略topK记录
        self.chosen_items = [[] for _ in range(B)]      # 初始化选择项目记录

        # 使用前缀长度=step+1=2构建初始缓存
        self._update_pre_step_cache()  # 更新"前一步"缓存
        return {"candidate_ids": self._pre_cand_ids, "candidate_features": self._pre_cand_feat}  # 返回初始状态


    @torch.no_grad()
    def _update_pre_step_cache(self):
        """
        基于当前生成前缀为当前步骤构建状态。
        """
        B, L = self.gen_seq.shape  # 获取批次B和长度L

        # 构建当前前缀长度=step+1（种子=1）
        cur_len = self.step + 1  # 当前长度
        cur_len = min(cur_len, L)  # 确保不超过最大长度
        seq = self.gen_seq[:, :cur_len]  # [B, cur_len] - 当前序列
        ts = self.orig_ts[:, :cur_len]   # [B, cur_len] - 当前时间戳（保持原始时间戳用于对齐长度）
        idx = self.orig_idx  # [B] - 级联ID每序列
        ans = self.gen_ans[:, :cur_len]  # [B, cur_len] - 当前答案

        probs_last, yt_last, hidden_item = self._forward_base(seq, ts, idx, ans)  # 执行基础模型前向传播
        self.hidden_item = hidden_item  # 保存项目嵌入

        # 通过基础概率获取候选ID
        Kc = min(self.cand_k, probs_last.size(-1))  # 获取候选数量（取候选K和项目总数的最小值）
        cand_ids = torch.topk(probs_last, k=Kc, dim=-1).indices  # [B, Kc] - 获取Top-K候选ID
        cand_probs = probs_last.gather(1, cand_ids)  # [B, Kc] - 获取对应概率
        # 从yt_last获取候选能力，确保索引在范围内
        cand_ability = yt_last.gather(1, cand_ids.clamp(0, yt_last.size(1) - 1))  # [B, Kc] - 获取能力

        cand_diff = self._difficulty_norm(cand_ids)  # [B, Kc] - 获取难度

        # 历史统计用于Eq.(19)
        delta_t = self._compute_delta_t(seq, ans)  # [B] - 计算delta_t
        self._pre_delta = delta_t  # 保存delta_t

        # 最近正确率、最近平均难度
        recent_corr = torch.zeros((B,), device=self.device)  # [B] - 初始化最近正确率
        recent_davg = torch.zeros((B,), device=self.device)  # [B] - 初始化最近平均难度
        for b in range(B):  # 遍历批次
            # 有效历史
            valid_items = seq[b][seq[b] != self.pad_val]  # [valid_len] - 有效项目
            valid_ans = ans[b][seq[b] != self.pad_val].float()  # [valid_len] - 有效答案
            if valid_items.numel() == 0:  # 如果没有有效项目
                recent_corr[b] = 0.0  # 设置为0
                recent_davg[b] = 0.5  # 设置为0.5
                continue
            w = min(int(valid_items.numel()), self.T)  # 取有效项目数和历史窗口的最小值
            recent_corr[b] = valid_ans[-w:].mean()  # 计算最近正确率
            if self.diff_map is None:  # 如果没有难度映射
                recent_davg[b] = 0.5  # 设置为0.5
            else:  # 否则计算平均难度
                items_np = valid_items[-w:].detach().cpu().numpy()  # 获取最后w个项目
                # 计算这些项目的平均归一化难度
                recent_davg[b] = float(np.mean([self.diff_map.get_difficulty_norm(int(it), default=2) for it in items_np]))

        self._pre_recent_corr = recent_corr  # 保存最近正确率
        self._pre_recent_davg = recent_davg  # 保存最近平均难度

        # 与路径平均/最后的相似性
        if hidden_item is not None:  # 如果项目嵌入存在
            d = hidden_item.size(-1)  # 获取嵌入维度
            path_mean = torch.zeros((B, d), device=self.device)  # [B, d] - 路径平均嵌入
            last_emb = torch.zeros((B, d), device=self.device)   # [B, d] - 最后嵌入
            for b in range(B):  # 遍历批次
                valid_items = seq[b][seq[b] != self.pad_val]  # [valid_len] - 有效项目
                if valid_items.numel() == 0:  # 如果没有有效项目
                    continue
                emb = hidden_item[valid_items]  # [valid_len, d] - 获取嵌入
                path_mean[b] = emb.mean(dim=0)  # [d] - 计算平均嵌入
                last_emb[b] = emb[-1]  # [d] - 获取最后嵌入
            self._pre_path_mean_emb = path_mean  # 保存路径平均嵌入
            self._pre_last_emb = last_emb  # 保存最后嵌入

            cand_emb = hidden_item[cand_ids]  # [B, Kc, d] - 候选嵌入
            sim_path = _cosine_sim(cand_emb, path_mean.unsqueeze(1))  # [B, Kc, d] x [B, 1, d] -> [B, Kc] - 与路径平均的相似性
            sim_last = _cosine_sim(cand_emb, last_emb.unsqueeze(1))  # [B, Kc, d] x [B, 1, d] -> [B, Kc] - 与最后的相似性
        else:  # 如果没有项目嵌入
            sim_path = torch.zeros_like(cand_probs)  # [B, Kc] - 初始化为0
            sim_last = torch.zeros_like(cand_probs)  # [B, Kc] - 初始化为0

        # 将历史特征广播到候选
        hist_feat = torch.stack([recent_corr, recent_davg, delta_t], dim=-1)  # [B, 3] - 历史特征
        hist_feat = hist_feat.unsqueeze(1).expand(-1, cand_ids.size(1), -1)  # [B, 1, 3] -> [B, Kc, 3] - 广播到候选

        # 候选特征：[概率, 能力, 难度, 与路径相似, 与最后相似, 历史...]
        cand_feat = torch.cat([
            cand_probs.unsqueeze(-1),   # [B, Kc, 1] - 概率
            cand_ability.unsqueeze(-1), # [B, Kc, 1] - 能力
            cand_diff.unsqueeze(-1),    # [B, Kc, 1] - 难度
            sim_path.unsqueeze(-1),     # [B, Kc, 1] - 与路径相似
            sim_last.unsqueeze(-1),     # [B, Kc, 1] - 与最后相似
            hist_feat                   # [B, Kc, 3] - 历史特征
        ], dim=-1).float()  # [B, Kc, F] - 拼接所有特征并转换为float

        self._pre_base_probs = probs_last  # 保存基础概率
        self._pre_yt = yt_last             # 保存掌握概率
        self._pre_cand_ids = cand_ids      # 保存候选ID
        self._pre_cand_feat = cand_feat    # 保存候选特征


    @torch.no_grad()
    def step_env(self, action_idx: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict]:
        """
        环境执行一步
        
        Args:
            action_idx: [B] 候选集中的索引
            
        Returns:
            next_state: 下一状态字典
            reward: [B] 奖励
            done: [B] 完成标志（如果样本在此步骤结束则为1）
        """
        B, L = self.gen_seq.shape  # 获取批次B和长度L
        Kc = self._pre_cand_ids.size(1)  # 获取候选数量

        action_idx = action_idx.clamp(0, Kc - 1)  # [B] - 限制动作索引在有效范围内
        chosen = self._pre_cand_ids.gather(1, action_idx.view(-1, 1)).squeeze(1)  # [B] - 获取选择的项目ID

        # 记录此时间步的策略topK（映射到全局ID）
        # 策略topK将由调用者填写（需要策略逻辑）。这里保留选择。
        for b in range(B):  # 遍历批次
            self.chosen_items[b].append(int(chosen[b].item()))  # 记录选择的项目

        # ------- 使用仅"前一步"信息的步骤奖励 -------
        # 偏好：选择项目的基概率
        pref = self._pre_base_probs.gather(1, chosen.view(-1, 1)).squeeze(1)  # [B] - 获取选择项目的概率

        # 此时间步的适应性：1 - |delta_t - Dif(选择)|
        if self.diff_map is None:  # 如果没有难度映射
            chosen_diff = torch.full((B,), 0.5, device=self.device)  # [B] - 初始化为0.5
        else:  # 否则获取选择项目的难度
            chosen_np = chosen.detach().cpu().numpy()  # [B] - 转换为numpy
            chosen_diff = torch.tensor(
                [self.diff_map.get_difficulty_norm(int(it), default=2) for it in chosen_np],  # 计算归一化难度
                device=self.device, dtype=torch.float32
            )
        adapt = 1.0 - torch.abs(self._pre_delta - chosen_diff)  # [B] - 计算适应性

        # 新颖性：1 - 与最后的相似性（鼓励较少重复）
        if self.hidden_item is not None and self._pre_last_emb is not None:  # 如果嵌入存在
            chosen_emb = self.hidden_item[chosen]  # [B, d] - 选择项目的嵌入
            sim_last = _cosine_sim(chosen_emb, self._pre_last_emb)  # [B] - 与最后嵌入的相似性
            novelty = 1.0 - sim_last  # [B] - 计算新颖性
        else:  # 否则新颖性为0
            novelty = torch.zeros((B,), device=self.device)

        # 计算总奖励
        reward = (
                self.w_step["preference"] * pref +    # 偏好奖励
                self.w_step["adaptivity"] * adapt +   # 适应性奖励
                self.w_step.get("novelty", 0.0) * novelty  # 新颖性奖励
        ).float()  # [B] - 总奖励

        # ------- 转换：附加选择项目&模拟答案 -------
        next_pos = self.step + 1  # 下一位置
        done = torch.zeros((B,), device=self.device, dtype=torch.float32)  # [B] - 初始化完成标志

        for b in range(B):  # 遍历批次
            # 如果此样本已完成（next_pos >= 有效长度），标记完成并跳过
            if next_pos >= int(self.valid_lens[b].item()):  # 如果下一位置超出有效长度
                done[b] = 1.0  # 标记为完成
                continue
            # 设置生成项目
            self.gen_seq[b, next_pos] = chosen[b]  # 设置选择的项目

            # 使用"前一步"掌握模拟答案（无泄漏）
            p_correct = self._pre_yt.gather(1, chosen.view(-1, 1)).squeeze(1)[b].item()  # 获取正确概率
            self.gen_ans[b, next_pos] = 1 if p_correct >= 0.5 else 0  # 根据概率设置答案

        # 前进一步
        self.step += 1  # 增加步骤

        # 如果未对所有样本完成，则更新下一状态的缓存
        if self.step < self.max_steps:  # 如果未达到最大步数
            self._update_pre_step_cache()  # 更新缓存

        next_state = {"candidate_ids": self._pre_cand_ids, "candidate_features": self._pre_cand_feat}  # 下一状态
        info = {"chosen_items": chosen.detach().cpu().tolist()}  # 信息字典
        return next_state, reward, done, info  # 返回下一状态、奖励、完成标志和信息


# -------- 终端奖励指标（Eq 19/20/21）--------

    @torch.no_grad()
    def _compute_topk_from_policy_logits(self, cand_ids: torch.Tensor, policy_logits: torch.Tensor, k: int) -> torch.Tensor:
        """
        从策略逻辑回归计算topk
        
        Args:
            cand_ids: [B, Kc] 全局ID
            policy_logits: [B, Kc] 策略逻辑回归
            k: topk数量
            
        Returns:
            全局topk ID [B, k]
        """
        k = min(k, cand_ids.size(1))  # 确保k不超过候选数量
        topk_idx = torch.topk(policy_logits, k=k, dim=-1).indices  # [B, k] - topk索引
        return cand_ids.gather(1, topk_idx)  # [B, k] - 获取对应的全局ID


    @torch.no_grad()
    def record_policy_topk(self, policy_logits: torch.Tensor):
        """
        在每个步骤由回放循环调用以记录策略topk推荐（全局ID），
        与"当前步骤"对齐。
        
        Args:
            policy_logits: 策略逻辑回归
        """
        B = self._pre_cand_ids.size(0)  # 获取批次大小
        # 从策略逻辑回归计算topk全局ID
        topk_global = self._compute_topk_from_policy_logits(self._pre_cand_ids, policy_logits, self.topk)  # [B, topk]
        for b in range(B):  # 遍历批次
            # 记录策略topk推荐
            self.topk_recs_policy[b].append([int(x) for x in topk_global[b].detach().cpu().tolist()])


    @torch.no_grad()
    def compute_final_quality(self, policy_weight: Optional[Dict[str, float]] = None) -> Dict[str, torch.Tensor]:
        """
        返回每个样本的指标+最终质量=加权和。
        
        Args:
            policy_weight: 策略权重
            
        Returns:
            最终质量指标字典
        """
        w = policy_weight or {"effectiveness": 1.0, "adaptivity": 1.0, "diversity": 1.0}  # 默认权重

        B, L = self.orig_seq.shape  # 获取批次B和长度L
        K = self.topk  # Top-K数量

        # 构建topk张量[B, L-1, K]（用pad_val填充）
        topk_tensor = torch.full((B, L - 1, K), self.pad_val, device=self.device, dtype=torch.long)  # [B, L-1, K] - 初始化topk张量
        for b in range(B):  # 遍历批次
            steps = min(len(self.topk_recs_policy[b]), L - 1)  # 获取步骤数
            start_t = int(getattr(self, "start_t", 0))  # 获取起始时间
            for s in range(steps):  # 遍历步骤
                t = start_t + s  # 计算时间
                if t >= (L - 1):  # 如果超出范围
                    break
                recs = self.topk_recs_policy[b][s][:K]  # 获取推荐
                if len(recs) < K:  # 如果推荐数量不足
                    recs = recs + [self.pad_val] * (K - len(recs))  # 用填充值补齐
                topk_tensor[b, t] = torch.tensor(recs, device=self.device, dtype=torch.long)  # 设置topk张量

        # -------- Eq.(19)适应性 over all recs --------
        adapt_scores = torch.zeros((B,), device=self.device)  # [B] - 初始化适应性分数
        if self.diff_map is None:  # 如果没有难度映射
            adapt_scores[:] = 0.0  # 设置为0
        else:  # 否则计算适应性
            for b in range(B):  # 遍历批次
                valid_len = int(self.valid_lens[b].item())  # 获取有效长度
                if valid_len <= 1:  # 如果长度<=1
                    continue

                # 生成序列到有效长度的历史难度/结果
                hist_items = self.gen_seq[b, :valid_len].detach().cpu().numpy().tolist()  # 历史项目
                hist_ans = self.gen_ans[b, :valid_len].detach().cpu().numpy().tolist()    # 历史答案

                # 预计算每个时间步的delta_t（如Metrics.calculate_adaptivity_tensor中）
                history_diffs = []  # 历史难度
                history_results = []  # 历史结果
                for t in range(valid_len - 1):  # 遍历时间步
                    it = hist_items[t]  # 获取项目
                    if it != self.pad_val and it > 1:  # 如果不是填充值且>1
                        history_diffs.append(self.diff_map.get_difficulty_norm(it, default=2))  # 添加归一化难度
                        history_results.append(float(hist_ans[t]))  # 添加结果

                total = 0.0  # 总和
                cnt = 0      # 计数
                for t in range(min(valid_len - 1, topk_tensor.size(1))):  # 遍历时间步
                    if len(history_diffs[:t]) < self.T // 2:  # 如果历史太短
                        delta = 1.0  # 设置为1.0
                    else:  # 否则计算delta
                        start = max(0, t - self.T)  # 起始位置
                        recent_diffs = history_diffs[start:t]  # 最近难度
                        recent_res = history_results[start:t]  # 最近结果
                        if len(recent_diffs) > 0:  # 如果有最近难度
                            num = sum(d * r for d, r in zip(recent_diffs, recent_res))  # 计算分子
                            den = sum(recent_res) + self.eps  # 计算分母
                            delta = num / den  # 计算delta
                        else:  # 否则设置为1.0
                            delta = 1.0

                    for k in range(K):  # 遍历topk
                        rec = int(topk_tensor[b, t, k].item())  # 获取推荐
                        if rec != self.pad_val and rec > 1:  # 如果不是填充值且>1
                            rec_diff = self.diff_map.get_difficulty_norm(rec, default=2)  # 获取难度
                            val = 1.0 - abs(delta - rec_diff)  # 计算值
                            total += val  # 累加
                            cnt += 1   # 计数

                adapt_scores[b] = total / max(1, cnt)  # 计算平均适应性

        # -------- Eq.(20)效果 over all recs --------
        # pb: 原始KT, pa: 生成KT
        # 我们需要yt张量[B, L-1, N]；如果基础模型提供完整序列的yt，则尝试获取。
        def _run_kt_like(seq, ts, idx, ans):
            """运行KT模型"""
            out = self.base_model(seq, ts, idx, ans, self.graph, self.hypergraph_list)  # 基础模型输出
            if isinstance(out, dict):  # 如果是字典
                yt = out.get("yt", None)  # 获取yt
            else:  # 否则是元组
                yt = out[3] if len(out) > 3 else None  # 获取yt
            if yt is None:  # 如果yt为空
                return None
            if yt.dim() == 3 and yt.size(1) >= 1:  # 如果是3维且时间步>=1
                return yt  # [B, L-1, S] - 返回yt
            return None

        yt_before = _run_kt_like(self.orig_seq, self.orig_ts, self.orig_idx, self.orig_ans)  # 原始序列的yt
        yt_after = _run_kt_like(self.gen_seq, self.orig_ts, self.orig_idx, self.gen_ans)     # 生成序列的yt
        eff_scores = torch.zeros((B,), device=self.device)  # [B] - 初始化效果分数

        if yt_before is None or yt_after is None:  # 如果任一yt为空
            eff_scores[:] = 0.0  # 设置为0
        else:  # 否则计算效果
            # 确保相同时间维度
            Tm = min(yt_before.size(1), yt_after.size(1), L - 1)  # 获取最小时间维度
            yt_before = yt_before[:, :Tm]  # [B, Tm, S] - 调整yt_before
            yt_after = yt_after[:, :Tm]    # [B, Tm, S] - 调整yt_after
            S = yt_before.size(-1)         # 获取技能数量

            for b in range(B):  # 遍历批次
                valid_len = int(self.valid_lens[b].item())  # 获取有效长度
                if valid_len <= 1:  # 如果长度<=1
                    continue
                total = 0.0  # 总和
                cnt = 0      # 计数
                for t in range(min(valid_len - 1, Tm)):  # 遍历时间步
                    if int(self.orig_seq[b, t].item()) == self.pad_val:  # 如果是填充值
                        continue
                    recs = topk_tensor[b, t]  # [K] - 获取推荐
                    for k in range(K):  # 遍历topk
                        r = int(recs[k].item())  # 获取推荐
                        if r == self.pad_val:  # 如果是填充值
                            continue
                        if 0 <= r < S:  # 如果索引有效
                            pb = float(yt_before[b, t, r].item())  # 获取原始掌握概率
                            pa = float(yt_after[b, t, r].item())   # 获取生成掌握概率
                            if pb < 0.9 and pa > 0:  # 如果pb<0.9且pa>0
                                gain = (pa - pb) / (1.0 - pb)  # 计算增益
                                total += gain  # 累加
                                cnt += 1   # 计数
                eff_scores[b] = total / max(1, cnt)  # 计算平均效果

        # -------- Eq.(21)多样性 over all recs (per-sample) --------
        div_scores = torch.zeros((B,), device=self.device)  # [B] - 初始化多样性分数
        if self.hidden_item is None:  # 如果没有项目嵌入
            div_scores[:] = 0.0  # 设置为0
        else:  # 否则计算多样性
            emb = self.hidden_item  # [N, d] - 项目嵌入
            for b in range(B):  # 遍历批次
                valid_len = int(self.valid_lens[b].item())  # 获取有效长度
                if valid_len <= 1:  # 如果长度<=1
                    continue
                # 收集所有推荐项目在有效时间步
                items: List[int] = []  # 项目列表
                for t in range(min(valid_len - 1, topk_tensor.size(1))):  # 遍历时间步
                    for k in range(K):  # 遍历topk
                        r = int(topk_tensor[b, t, k].item())  # 获取推荐
                        if r != self.pad_val and r > 1:  # 如果不是填充值且>1
                            items.append(r)  # 添加到列表
                if len(items) < 2:  # 如果项目数<2
                    div_scores[b] = 0.0  # 设置为0
                    continue
                e = emb[torch.tensor(items, device=self.device)]  # [M, d] - 获取嵌入
                e = e / (e.norm(dim=-1, keepdim=True) + 1e-8)  # [M, d] - 归一化
                sim = e @ e.t()  # [M, M] - 计算相似度矩阵
                # 上三角（不含对角线）
                M = sim.size(0)  # 获取大小
                triu = torch.triu(sim, diagonal=1)  # [M, M] - 上三角矩阵
                vals = triu[triu != 0]  # [num_upper_tri_nonzero] - 非零值
                if vals.numel() == 0:  # 如果没有非零值
                    div_scores[b] = 0.0  # 设置为0
                else:
                    div_scores[b] = (1.0 - vals).mean()  # 计算平均多样性

        # 计算最终质量
        final_quality = (
                w["effectiveness"] * eff_scores +  # 效果
                w["adaptivity"] * adapt_scores +   # 适应性
                w["diversity"] * div_scores       # 多样性
        )

        return {
            "effectiveness": eff_scores,    # 效果
            "adaptivity": adapt_scores,     # 适应性
            "diversity": div_scores,        # 多样性
            "final_quality": final_quality  # 最终质量
        }


# ------------------------- 优化器包装器 -------------------------

class RLPathOptimizer:
    """
    包装:
      - OnlineLearningPathEnv
      - PolicyValueNet
      - PPOTrainer
    """

    def __init__(
            self,
            base_model: nn.Module,                    # 基础模型
            num_items: int,                           # 项目数量
            data_name: str,                           # 数据集名称
            device: torch.device,                     # 设备
            pad_val: int = 0,                         # 填充值
            topk: int = 10,                           # Top-K数量
            cand_k: int = 50,                         # 候选数量
            history_window_T: int = 10,               # 历史窗口大小
            rl_lr: float = 3e-4,                      # RL学习率
            policy_hidden: int = 128,                 # 策略隐藏层大小
            ppo_config: Optional[PPOConfig] = None,   # PPO配置
            step_reward_weights: Optional[Dict[str, float]] = None,  # 步骤奖励权重
            final_reward_weights: Optional[Dict[str, float]] = None, # 最终奖励权重
            terminal_reward_scale: float = 1.0,       # 终端奖励缩放
    ):
        """
        初始化RL路径优化器
        
        Args:
            base_model: 基础模型
            num_items: 项目数量
            data_name: 数据集名称
            device: 设备
            pad_val: 填充值
            topk: Top-K数量
            cand_k: 候选数量
            history_window_T: 历史窗口大小
            rl_lr: RL学习率
            policy_hidden: 策略隐藏层大小
            ppo_config: PPO配置
            step_reward_weights: 步骤奖励权重
            final_reward_weights: 最终奖励权重
            terminal_reward_scale: 终端奖励缩放
        """
        self.device = device  # 设备
        # 初始化环境
        self.env = OnlineLearningPathEnv(
            base_model=base_model,        # 基础模型
            num_items=num_items,          # 项目数量
            data_name=data_name,          # 数据集名称
            device=device,                # 设备
            pad_val=pad_val,              # 填充值
            topk=topk,                    # Top-K数量
            cand_k=cand_k,                # 候选数量
            history_window_T=history_window_T,  # 历史窗口大小
            w_step=step_reward_weights    # 步骤权重
        )
        # 特征维度在reset后确定；我们将延迟初始化
        self.policy: Optional[PolicyValueNet] = None  # 策略网络（可选）
        self.trainer: Optional[PPOTrainer] = None     # 训练器（可选）
        self.rl_lr = rl_lr                    # RL学习率
        self.policy_hidden = policy_hidden    # 策略隐藏层大小
        self.ppo_config = ppo_config or PPOConfig()  # PPO配置
        # 最终奖励权重
        self.final_reward_weights = final_reward_weights or {"effectiveness": 1.0, "adaptivity": 1.0, "diversity": 1.0}
        self.terminal_reward_scale = terminal_reward_scale  # 终端奖励缩放

    def _lazy_init(self, feat_dim: int):
        """
        延迟初始化策略网络和训练器
        
        Args:
            feat_dim: 特征维度
        """
        if self.policy is None:  # 如果策略网络未初始化
            self.policy = PolicyValueNet(feat_dim=feat_dim, hidden_dim=self.policy_hidden).to(self.device)  # 创建策略网络
            self.trainer = PPOTrainer(self.policy, lr=self.rl_lr, config=self.ppo_config)  # 创建训练器

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
        """
        使用真实批次延迟初始化策略/训练器（无回放）。
        防止在策略仍为None时调用rl.policy.train()/eval()。
        
        Args:
            tgt: 目标序列
            tgt_timestamp: 目标时间戳
            tgt_idx: 目标索引
            ans: 答案
            graph: 关系图
            hypergraph_list: 超图列表
        """
        if self.policy is not None and self.trainer is not None:  # 如果已初始化
            return
        state = self.env.reset(tgt, tgt_timestamp, tgt_idx, ans, graph=graph, hypergraph_list=hypergraph_list)  # 重置环境
        cand_feat = state["candidate_features"]  # [B, K, F] - 获取候选特征
        self._lazy_init(int(cand_feat.size(-1)))  # 延迟初始化

    def collect_trajectory(
            self,
            tgt, tgt_timestamp, tgt_idx, ans,
            graph=None, hypergraph_list=None
    ) -> Dict[str, torch.Tensor]:
        """
        为一批次回放策略，跨越所有有效时间步。

        返回一个字典，包含PPO更新的展平张量。

        Args:
            tgt: 目标序列
            tgt_timestamp: 目标时间戳
            tgt_idx: 目标索引
            ans: 答案
            graph: 关系图
            hypergraph_list: 超图列表

        Returns:
            包含轨迹数据的字典
        """
        # 重置环境并获取初始状态
        state = self.env.reset(tgt, tgt_timestamp, tgt_idx, ans, graph=graph, hypergraph_list=hypergraph_list)
        cand_feat = state["candidate_features"]  # [B, K, F] - 候选特征
        self._lazy_init(cand_feat.size(-1))  # 延迟初始化策略网络

        B = cand_feat.size(0)  # 获取批次大小
        max_steps = self.env.max_steps  # 获取最大步数

        # 时间步列表
        all_cand_feat: List[torch.Tensor] = []  # 候选特征列表
        all_actions: List[torch.Tensor] = []     # 动作列表
        all_logp: List[torch.Tensor] = []        # 对数概率列表
        all_values: List[torch.Tensor] = []      # 价值列表
        all_rewards: List[torch.Tensor] = []     # 奖励列表
        all_dones: List[torch.Tensor] = []       # 完成标志列表

        done = torch.zeros((B,), device=self.device, dtype=torch.float32)  # [B] - 初始化完成标志

        for t in range(max_steps):  # 遍历时间步
            cand_feat = state["candidate_features"]  # [B, K, F] - 当前候选特征
            logits, value = self.policy(cand_feat)   # 通过策略网络获取逻辑回归和价值
            dist = torch.distributions.Categorical(logits=logits)  # 创建分类分布
            action = dist.sample()  # [B] - 采样动作
            logp = dist.log_prob(action)  # [B] - 计算对数概率
            entropy = dist.entropy()  # [B] - 计算熵（在此处未使用；PPO在更新中使用平均熵）

            # 记录策略topK（在执行步骤之前，与当前步骤对齐）
            self.env.record_policy_topk(logits)

            # 执行环境步骤
            next_state, reward, step_done, _ = self.env.step_env(action)

            # 更新完成标志：一旦完成，保持完成
            done = torch.maximum(done, step_done)

            # 记录当前步骤的数据
            all_cand_feat.append(cand_feat)  # 添加候选特征
            all_actions.append(action)       # 添加动作
            all_logp.append(logp)            # 添加对数概率
            all_values.append(value)         # 添加价值
            all_rewards.append(reward)       # 添加奖励
            all_dones.append(done.clone())   # 添加完成标志副本

            state = next_state  # 更新状态

            # 如果全部完成，则退出
            if float(done.min().item()) >= 1.0:  # 如果所有样本都完成
                break

        # 终端奖励（最终质量）
        final_metrics = self.env.compute_final_quality(self.final_reward_weights)  # 计算最终指标
        terminal_r = final_metrics["final_quality"] * self.terminal_reward_scale  # [B] - 终端奖励
        # 将终端奖励添加到最后收集的奖励步骤（对每个至少有1步的样本）
        if len(all_rewards) > 0:  # 如果有奖励数据
            all_rewards[-1] = all_rewards[-1] + terminal_r  # 添加终端奖励

        # 堆叠到[T, B, ...]
        rewards = torch.stack(all_rewards, dim=0)  # [T, B] - 奖励
        values = torch.stack(all_values, dim=0)    # [T, B] - 价值
        dones = torch.stack(all_dones, dim=0)      # [T, B] - 完成标志

        # 计算GAE
        adv, rets = self.trainer.compute_gae(rewards, values, dones)  # 计算优势和回报

        # 展平有效步骤：我们保留所有步骤，但掩盖已经完成的步骤
        T, B = rewards.shape  # 获取时间T和批次B
        valid_mask = (1.0 - dones)  # [T, B]  1表示在该步骤仍活跃
        # 包含成为完成的最后步骤？在我们的环境中，done是累积的；最后步骤对于样本来说done=1，
        # 所以掩码会删除它。我们想要保留采取行动的步骤。使用更新前的每步活动：
        # 通过移位近似：
        active = torch.ones_like(dones)  # [T, B] - 初始化为全1
        active[1:] = 1.0 - dones[:-1]    # [T, B] - 设置活动状态
        active[0] = 1.0  # 第一步总是活动的（如果发生回放）
        active = active.clamp(0, 1)  # 限制在[0, 1]范围

        cand_feat_t = torch.stack(all_cand_feat, dim=0)  # [T, B, K, F] - 候选特征
        actions_t = torch.stack(all_actions, dim=0)      # [T, B] - 动作
        logp_t = torch.stack(all_logp, dim=0)            # [T, B] - 对数概率
        values_t = values  # [T, B] - 价值
        adv_t = adv  # [T, B] - 优势
        rets_t = rets  # [T, B] - 回报

        # 展平
        active_flat = active.reshape(-1).bool()  # [T*B] - 活跃掩码
        # [T*B, K, F] - 展平候选特征
        cand_feat_flat = cand_feat_t.reshape(T * B, cand_feat_t.size(2), cand_feat_t.size(3))[active_flat]
        actions_flat = actions_t.reshape(-1)[active_flat]  # [num_active] - 展平动作
        logp_flat = logp_t.reshape(-1)[active_flat]        # [num_active] - 展平对数概率
        values_flat = values_t.reshape(-1)[active_flat]    # [num_active] - 展平价值
        adv_flat = adv_t.reshape(-1)[active_flat]          # [num_active] - 展平优势
        rets_flat = rets_t.reshape(-1)[active_flat]        # [num_active] - 展平回报

        return {
            "cand_feat": cand_feat_flat,    # 候选特征
            "actions": actions_flat,        # 动作
            "old_logp": logp_flat,          # 旧对数概率
            "old_values": values_flat,      # 旧价值
            "advantages": adv_flat,         # 优势
            "returns": rets_flat,           # 回报
            "final_metrics": final_metrics, # 最终指标（每样本张量）
        }

    def update_policy(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        更新策略
        
        Args:
            batch: 包含批次数据的字典
            
        Returns:
            损失字典
        """
        return self.trainer.update(
            cand_feat=batch["cand_feat"],    # 候选特征
            actions=batch["actions"],        # 动作
            old_logp=batch["old_logp"],      # 旧对数概率
            old_values=batch["old_values"],  # 旧价值
            advantages=batch["advantages"],  # 优势
            returns=batch["returns"],        # 回报
        )


@torch.no_grad()
def evaluate_policy(
        rl: RLPathOptimizer,      # RL路径优化器
        data_loader,              # 数据加载器
        graph,                    # 关系图
        hypergraph_list,          # 超图列表
        device: torch.device,     # 设备
        max_batches: int = 50     # 最大批次数
) -> Dict[str, float]:
    """
    在几个批次上评估：报告平均最终指标。
    
    Args:
        rl: RL路径优化器
        data_loader: 数据加载器
        graph: 关系图
        hypergraph_list: 超图列表
        device: 设备
        max_batches: 最大批次数
        
    Returns:
        评估指标字典
    """
    # 策略是延迟初始化的；在第一个批次上使用一次初始化，然后调用eval()
    if rl.policy is None:  # 如果策略未初始化
        first = next(iter(data_loader))  # 获取第一个批次
        tgt, tgt_timestamp, tgt_idx, ans = first[0], first[1], first[2], first[3]  # 解包数据
        tgt = tgt.to(device); tgt_timestamp = tgt_timestamp.to(device); tgt_idx = tgt_idx.to(device); ans = ans.to(device)  # 移到设备
        # 使用第一个批次初始化
        rl.ensure_initialized(tgt, tgt_timestamp, tgt_idx, ans, graph=graph, hypergraph_list=hypergraph_list)
    rl.policy.eval()  # 设置策略为评估模式
    # 初始化聚合字典
    agg = {"effectiveness": 0.0, "adaptivity": 0.0, "diversity": 0.0, "final_quality": 0.0}
    n = 0  # 样本计数

    for i, batch in enumerate(data_loader):  # 遍历数据加载器
        if i >= max_batches:  # 如果超过最大批次
            break
        # 批次结构遵循您的DataLoader：(tgt, tgt_timestamp, tgt_idx, ans, ...)
        tgt, tgt_timestamp, tgt_idx, ans = batch[0], batch[1], batch[2], batch[3]  # 解包批次数据
        # 将数据移到设备
        tgt = tgt.to(device); tgt_timestamp = tgt_timestamp.to(device); tgt_idx = tgt_idx.to(device); ans = ans.to(device)

        # 收集轨迹
        rollout = rl.collect_trajectory(tgt, tgt_timestamp, tgt_idx, ans, graph=graph, hypergraph_list=hypergraph_list)
        fm = rollout["final_metrics"]  # 获取最终指标
        for k in agg:  # 遍历指标
            agg[k] += float(fm[k].mean().detach().cpu())  # 累加平均值
        n += 1  # 增加计数

    if n == 0:  # 如果没有样本
        return {k: 0.0 for k in agg}  # 返回0值
    return {k: v / n for k, v in agg.items()}  # 返回平均值