import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

import Constants
from Metrics import Metrics


# ======================================================
# Policy Network
# ======================================================
class PolicyNetwork(nn.Module):
    def __init__(self, knowledge_dim, candidate_feature_dim, hidden_dim, topk):
        super().__init__()
        self.fc1 = nn.Linear(knowledge_dim + candidate_feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.topk = topk

    def forward(self, knowledge_state, candidate_features):
        """
        knowledge_state: [B, K]
        candidate_features: [B, topk, F]
        """
        B, K = knowledge_state.shape
        x = knowledge_state.unsqueeze(1).repeat(1, self.topk, 1)
        x = torch.cat([x, candidate_features], dim=-1)
        h = F.relu(self.fc1(x))
        scores = self.fc2(h).squeeze(-1)  # [B, topk]
        return scores


# ======================================================
# Helpers
# ======================================================
def _pad_and_concat_like(original_2d: torch.Tensor, extra_len: int, pad_value: int = 0) -> torch.Tensor:
    """
    original_2d: [B, T]
    return: [B, T + extra_len]
    """
    if extra_len <= 0:
        return original_2d
    B = original_2d.size(0)
    pad = torch.full((B, extra_len), pad_value, device=original_2d.device, dtype=original_2d.dtype)
    return torch.cat([original_2d, pad], dim=1)


def _reshape_pred(pred_flat: torch.Tensor, batch_size: int, steps: int) -> torch.Tensor:
    """
    MSHGAT returns pred flattened: [B*steps, num_skills]
    reshape to [B, steps, num_skills]
    """
    return pred_flat.view(batch_size, steps, -1)


# ======================================================
# Learning Path Environment
# ======================================================
class LearningPathEnv:
    def __init__(
        self,
        batch_size,
        base_model,
        recommendation_length,
        data_name,
        graph=None,
        hypergraph_list=None,
        topnum_for_metrics=1,
        metrics_T=5,
    ):
        self.batch_size = batch_size
        self.base_model = base_model
        self.recommendation_length = recommendation_length
        self.data_name = data_name

        self.graph = graph
        self.hypergraph_list = hypergraph_list

        # step reward weights（轻量 shaping，可自行调）
        self.step_weights = {
            "preference": 0.1,
            "diversity": 0.05,
            "difficulty": 0.05,
        }

        # final reward weights（和 calculate_muti_obj 的多目标一致）
        self.final_weights = {
            "effectiveness": 0.4,
            "adaptivity": 0.3,
            "diversity": 0.2,
            "preference": 0.1,
        }

        self.metric = Metrics()
        self.topnum_for_metrics = int(topnum_for_metrics)  # RL 推荐每步只有 1 个动作，默认 topnum=1
        self.metrics_T = int(metrics_T)

        self._clear_episode_cache()

    def _clear_episode_cache(self):
        self.original_tgt = None
        self.original_tgt_timestamp = None
        self.original_tgt_idx = None
        self.original_ans = None

        self.paths = None
        self.all_knowledge_states = None  # per-batch list of tensors [K]
        self.all_predictions = None       # per-batch list of tensors [num_skills] prob

        self.current_step = 0

        # caches for final Metrics usage
        self._last_hidden = None
        self._last_pred_probs_full = None  # [B, steps, num_skills]
        self._last_yt_full = None          # [B, steps, num_skills]

    # ---------------- reset ----------------
    def reset(self, tgt, tgt_timestamp, tgt_idx, ans):
        """
        tgt: [B, T]
        ans: [B, T]
        """
        B = tgt.size(0)
        self.batch_size = B  # 防止 DataLoader 最后一批不足 batch_size 时出错

        self.original_tgt = tgt
        self.original_tgt_timestamp = tgt_timestamp
        self.original_tgt_idx = tgt_idx
        self.original_ans = ans

        self.paths = [[] for _ in range(B)]
        self.all_knowledge_states = [[] for _ in range(B)]
        self.all_predictions = [[] for _ in range(B)]

        self.current_step = 0

        # forward base model
        with torch.no_grad():
            # HGAT.MSHGAT forward returns:
            # pred_flat, pred_res, kt_mask, yt, hidden, status_emb
            pred_flat, _, _, yt, hidden, *_ = self.base_model(
                tgt,
                tgt_timestamp,
                tgt_idx,
                ans,
                self.graph,
                self.hypergraph_list,
            )

        steps = tgt.size(1) - 1  # model内部用 input[:, :-1]
        pred = _reshape_pred(pred_flat, B, steps)  # [B, steps, num_skills]
        pred_probs = torch.softmax(pred, dim=-1)   # [B, steps, num_skills]

        self._last_hidden = hidden
        self._last_pred_probs_full = pred_probs
        self._last_yt_full = yt

        state = {
            "knowledge_state": yt[:, -1, :],     # [B, K]
            "pred_probs": pred_probs[:, -1, :],  # [B, num_skills]
        }
        return state

    # ---------------- step ----------------
    def step(self, action):
        """
        action: LongTensor [B]
        """
        B = action.size(0)
        device = action.device

        # 1) update path
        for i in range(B):
            self.paths[i].append(int(action[i].item()))

        # 2) build extended sequence
        # current appended length = len(self.paths[i]) (all equal)
        ext_len = len(self.paths[0])
        path_tensor = torch.tensor(self.paths, device=device, dtype=torch.long)  # [B, ext_len]

        ext_tgt = torch.cat([self.original_tgt, path_tensor], dim=1)  # [B, T + ext_len]

        # timestamp / idx 对齐长度（MSHGAT 里基本没用 timestamp，但这里保证维度不出错）
        ext_timestamp = _pad_and_concat_like(self.original_tgt_timestamp, ext_len, pad_value=0)
        ext_idx = _pad_and_concat_like(self.original_tgt_idx, ext_len, pad_value=0)

        pad_ans = torch.zeros(B, ext_len, device=device, dtype=self.original_ans.dtype)
        ext_ans = torch.cat([self.original_ans, pad_ans], dim=1)

        # 3) forward model
        with torch.no_grad():
            pred_flat, _, _, yt, hidden, *_ = self.base_model(
                ext_tgt,
                ext_timestamp,
                ext_idx,
                ext_ans,
                self.graph,
                self.hypergraph_list,
            )

        steps = ext_tgt.size(1) - 1
        pred = _reshape_pred(pred_flat, B, steps)      # [B, steps, num_skills]
        pred_probs = torch.softmax(pred, dim=-1)       # [B, steps, num_skills]

        self._last_hidden = hidden
        self._last_pred_probs_full = pred_probs
        self._last_yt_full = yt

        # 4) cache per-step info (只记录推荐阶段的“最后一步状态/分布”)
        for i in range(B):
            self.all_knowledge_states[i].append(yt[i, -1, :].detach())
            self.all_predictions[i].append(pred_probs[i, -1, :].detach())

        # 5) step reward（shaping）
        step_reward = self._compute_step_reward(action, yt[:, -1, :], pred_probs[:, -1, :])

        # 6) done
        self.current_step += 1
        done = torch.zeros(B, dtype=torch.bool, device=device)
        if self.current_step >= self.recommendation_length:
            done[:] = True

        next_state = {
            "knowledge_state": yt[:, -1, :],
            "pred_probs": pred_probs[:, -1, :],
        }
        return next_state, step_reward, done

    # ---------------- step reward ----------------
    def _compute_step_reward(self, action, knowledge_state, pred_probs_last):
        B = action.size(0)
        reward = torch.zeros(B, device=action.device)

        for i in range(B):
            a = int(action[i].item())

            # preference: 模型对该资源的概率
            if a < pred_probs_last.size(1):
                reward[i] += self.step_weights["preference"] * pred_probs_last[i, a]

            # diversity: repeat penalty
            if a in self.paths[i][:-1]:
                reward[i] -= self.step_weights["diversity"]

            # difficulty proxy: knowledge close to 0.5 preferred
            if a < knowledge_state.size(1):
                k = knowledge_state[i, a]
                reward[i] += self.step_weights["difficulty"] * (1 - torch.abs(k - 0.5))

        return reward

    # ---------------- final reward (Metrics-compatible) ----------------
    def compute_final_reward(self):
        """
        使用 Metrics.py 的“批量/全步”计算逻辑来算最终多目标奖励（不改 Metrics）
        - adaptivity: metric.calculate_adaptivity
        - diversity: metric.calculate_diversity
        - effectiveness / preference: 采用与 Metrics combined_metrics 同构的定义（只对推荐阶段计）
        """
        B = self.batch_size
        device = self.original_tgt.device
        final_reward = torch.zeros(B, device=device, dtype=torch.float)

        # 组装 full_sequence / full_answers（历史 + 推荐）
        original_seqs = self.original_tgt.detach().cpu().tolist()
        original_ans = self.original_ans.detach().cpu().tolist()

        rec_len = self.recommendation_length
        hist_len = self.original_tgt.size(1)

        full_sequences = []
        full_answers = []
        for i in range(B):
            full_seq_i = original_seqs[i] + self.paths[i]
            # 推荐阶段答案未知，用 0 填（与 calculate_muti_obj.py 的拼接思路一致）
            full_ans_i = original_ans[i] + [0] * len(self.paths[i])
            full_sequences.append(full_seq_i)
            full_answers.append(full_ans_i)

        # 构造 topk_sequence: [B, seq_len-1, topnum]
        # 为了兼容 Metrics 的 “逐时间步” 形式：历史阶段填空列表，推荐阶段填 [action]
        seq_len_full = hist_len + rec_len
        topnum = self.topnum_for_metrics

        topk_sequence = []
        for i in range(B):
            seq_steps = seq_len_full - 1
            per_t = []
            for t in range(seq_steps):
                # 推荐阶段起点：t >= (hist_len - 1)
                rec_t = t - (hist_len - 1)
                if 0 <= rec_t < len(self.paths[i]):
                    per_t.append([self.paths[i][rec_t]][:topnum])
                else:
                    per_t.append([])  # 历史阶段 or 越界
            topk_sequence.append(per_t)

        # 构造 topk_indices tensor: [B, seq_len-1, topnum]
        topk_indices = torch.full(
            (B, seq_len_full - 1, topnum),
            fill_value=Constants.PAD,
            dtype=torch.long,
            device=device,
        )
        for b in range(B):
            for t in range(seq_len_full - 1):
                if len(topk_sequence[b][t]) > 0:
                    topk_indices[b, t, 0] = int(topk_sequence[b][t][0])

        # yt_before / yt_after: 这里只对推荐阶段严格有效
        # 我们用 episode 内缓存的 yt（最后一次 forward 的 yt 包含历史+推荐的知识状态序列）
        yt_full = self._last_yt_full  # [B, steps, K], steps = seq_len_full - 1
        if yt_full is None or yt_full.dim() != 3:
            return final_reward

        yt_before = yt_full  # 对齐 combined_metrics 的输入形式
        # 简化：yt_after 用“向后平移一位 + 最后一位复制”
        yt_after = torch.zeros_like(yt_before)
        yt_after[:, :-1, :] = yt_before[:, 1:, :]
        yt_after[:, -1, :] = yt_before[:, -1, :]

        # pred_probs_flat: [B*(seq_len_full-1), num_skills]
        pred_probs_full = self._last_pred_probs_full  # [B, steps, num_skills]
        if pred_probs_full is None:
            return final_reward
        pred_probs_flat = pred_probs_full.reshape(-1, pred_probs_full.size(-1)).detach().cpu().numpy()

        # difficulty/adaptivity/diversity/preference 用 Metrics 现有函数
        # 注意：Metrics 内部对空推荐会跳过/不计数，符合我们“只评价推荐阶段”的目的
        adaptivity = self.metric.calculate_adaptivity(full_sequences, full_answers, topk_sequence, self.data_name)
        diversity = self.metric.calculate_diversity(
            full_sequences,
            topk_indices,
            self._last_hidden,
            B,
            seq_len_full,
            topnum
        )

        # effectiveness & preference：按 combined_metrics 的思想，但只对推荐阶段有效步累积
        eff_sum = 0.0
        eff_cnt = 0
        pref_sum = 0.0
        pref_cnt = 0

        hist_offset = hist_len - 1
        for b in range(B):
            for j, rec in enumerate(self.paths[b]):
                t = hist_offset + j
                if t < 0 or t >= (seq_len_full - 1):
                    continue

                # effectiveness：使用 gain 公式
                pb = yt_before[b, t, rec].item() if rec < yt_before.size(2) else None
                pa = yt_after[b, t, rec].item() if rec < yt_after.size(2) else None
                if pb is not None and pa is not None and pb < 0.9 and pa > 0:
                    eff_sum += (pa - pb) / (1.0 - pb)
                    eff_cnt += 1

                # preference：使用该 time_step 对 rec 的预测概率
                flat_index = b * (seq_len_full - 1) + t
                if flat_index < pred_probs_flat.shape[0] and rec < pred_probs_flat.shape[1]:
                    pref_sum += float(pred_probs_flat[flat_index, rec])
                    pref_cnt += 1

        effectiveness = eff_sum / eff_cnt if eff_cnt > 0 else 0.0
        preference = pref_sum / pref_cnt if pref_cnt > 0 else 0.0

        # 将标量扩展为 batch（Metrics 的 calculate_* 返回标量，我们直接广播）
        effectiveness_t = torch.full((B,), float(effectiveness), device=device)
        adaptivity_t = torch.full((B,), float(adaptivity), device=device)
        diversity_t = torch.full((B,), float(diversity), device=device)
        preference_t = torch.full((B,), float(preference), device=device)

        final_reward = (
            self.final_weights["effectiveness"] * effectiveness_t
            + self.final_weights["adaptivity"] * adaptivity_t
            + self.final_weights["diversity"] * diversity_t
            + self.final_weights["preference"] * preference_t
        )

        return final_reward


# ======================================================
# PPO / Policy Gradient Trainer (可跑通版)
# ======================================================
class PPOTrainer:
    def __init__(self, policy_net, env, lr=3e-4, gamma=0.99, entropy_coef=0.01):
        self.policy_net = policy_net
        self.env = env
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)

    def collect_trajectory(self, tgt, tgt_timestamp, tgt_idx, ans):
        state = self.env.reset(tgt, tgt_timestamp, tgt_idx, ans)

        states_k = []      # [T, B, K]
        actions = []       # [T, B]
        rewards = []       # [T, B]
        log_probs = []     # [T, B]
        entropies = []     # [T, B]

        B = tgt.size(0)
        device = tgt.device

        for _ in range(self.env.recommendation_length):
            # candidate features 先用 0（不影响 reward 修正）
            cand_feat = torch.zeros(B, self.policy_net.topk, 5, device=device)

            scores = self.policy_net(state["knowledge_state"], cand_feat)  # [B, topk]
            dist = Categorical(logits=scores)

            action = dist.sample()           # [B]
            log_prob = dist.log_prob(action) # [B]
            entropy = dist.entropy()         # [B]

            next_state, step_reward, done = self.env.step(action)

            states_k.append(state["knowledge_state"])  # [B, K]
            actions.append(action)
            rewards.append(step_reward)
            log_probs.append(log_prob)
            entropies.append(entropy)

            state = next_state

        # episode final reward（broadcast to each step）
        final_reward = self.env.compute_final_reward()  # [B]
        for t in range(len(rewards)):
            rewards[t] = rewards[t] + final_reward

        return states_k, actions, rewards, log_probs, entropies

    def update_policy(self, states_k, actions, rewards, log_probs, entropies):
        """
        REINFORCE with discounted returns (先保证训练不报错、reward 语义正确)
        """
        T = len(rewards)
        if T == 0:
            return 0.0

        # stack: [T, B]
        rewards_t = torch.stack(rewards, dim=0)
        logp_t = torch.stack(log_probs, dim=0)
        ent_t = torch.stack(entropies, dim=0)

        # discounted returns
        returns = torch.zeros_like(rewards_t)
        running = torch.zeros_like(rewards_t[0])
        for t in reversed(range(T)):
            running = rewards_t[t] + self.gamma * running
            returns[t] = running

        # normalize returns for stability
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # loss
        pg_loss = -(logp_t * returns).mean()
        ent_loss = -ent_t.mean()
        loss = pg_loss + self.entropy_coef * ent_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        return float(loss.item())
