# rl_adjuster.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from calculate_muti_obj import simulate_learning
import Constants
from Metrics import Metrics

# 尝试导入 KTOnlyModel（在 HGAT.py 中定义）
try:
    from HGAT import KTOnlyModel
except Exception:
    KTOnlyModel = None


# ======================================================
# Policy Network
# ======================================================
class PolicyNetwork(nn.Module):
    """
    Policy outputs logits over candidate positions (0..topk-1).
    candidate_features: [B, topk, F]
    """
    def __init__(self, knowledge_dim: int, candidate_feature_dim: int, hidden_dim: int, topk: int):
        super().__init__()
        self.topk = int(topk)
        self.fc1 = nn.Linear(knowledge_dim + candidate_feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, knowledge_state: torch.Tensor, candidate_features: torch.Tensor) -> torch.Tensor:
        """
        knowledge_state: [B, K]
        candidate_features: [B, topk, F]
        return logits: [B, topk]
        """
        B, K = knowledge_state.shape
        x = knowledge_state.unsqueeze(1).expand(B, self.topk, K)
        x = torch.cat([x, candidate_features], dim=-1)
        h = F.relu(self.fc1(x))
        logits = self.fc2(h).squeeze(-1)
        return logits


# ======================================================
# Helpers
# ======================================================
def _pad_and_concat_like(original: torch.Tensor, extra_len: int, pad_value: int = 0) -> torch.Tensor:
    """
    Pad along time dim=1 for sequence tensors shaped like [B, T, ...] or [B, T].
    If original is 1D ([B]) or scalar, return as-is (do NOT pad).
    """
    if original is None or extra_len <= 0:
        return original
    if original.dim() < 2:
        return original

    pad_shape = list(original.shape)
    pad_shape[1] = extra_len
    pad = torch.full(pad_shape, pad_value, device=original.device, dtype=original.dtype)
    return torch.cat([original, pad], dim=1)


def _reshape_pred_flat(pred_flat: torch.Tensor, batch_size: int, steps: int) -> torch.Tensor:
    """
    HGAT.MSHGAT returns flattened pred: [B*steps, num_skills].
    Reshape to [B, steps, num_skills].
    """
    return pred_flat.view(batch_size, steps, -1)


def _safe_to_cpu_list_1d(x: torch.Tensor):
    return x.detach().cpu().tolist()


# ======================================================
# Learning Simulation (NO change to Metrics)
# ======================================================
@torch.no_grad()
def simulate_learning_for_metrics(
    kt_model,
    original_seqs,      # List[List[int]]  shape [B, seq_len]
    original_ans,       # List[List[int]]  shape [B, seq_len]
    topk_sequence,      # List[List[List[int]]] shape [B][seq_len-1][K]
    graph,
    yt_before: torch.Tensor,  # [B, seq_len-1, num_skills]
    batch_size: int,
    K: int,             # topnum
):
    """
    严格复刻 calculate_muti_obj.py 的 simulate_learning():
    - 对每个时间步 t：在 t+1 后插入 K 个推荐
    - 只保留到 t+K（max_len = insert_pos + K）
    - 用 KTOnlyModel 跑
    - 取最后一步 yt_after[:, -1, :] 作为该 t 的 after
    - 堆叠得到 [B, seq_len-1, num_skills]
    """
    device = yt_before.device
    seq_len_minus_1 = len(topk_sequence[0])  # = seq_len-1
    yt_after_list = []

    for t in range(seq_len_minus_1):
        extended_inputs = []
        extended_ans = []

        for b in range(batch_size):
            original_seq = list(original_seqs[b])
            original_an = list(original_ans[b])

            recs = topk_sequence[b][t] if t < len(topk_sequence[b]) else []
            recommended = [int(r) for r in recs[:K] if int(r) != int(Constants.PAD)]

            insert_pos = t + 1
            new_seq = original_seq[:insert_pos] + recommended
            # ✅ 与原版一致：用 yt_before 的符号生成伪答案（>=0）
            if len(recommended) > 0:
                rec_tensor = torch.tensor(recommended, device=device, dtype=torch.long)
                pred_answers = (yt_before[b, t, rec_tensor] >= 0).float().tolist()
            else:
                pred_answers = []

            new_ans = original_an[:insert_pos] + pred_answers

            # ✅ 与原版一致：max_len = insert_pos + K（只保留到 t+K）
            max_len = insert_pos + K
            new_seq = new_seq[:max_len]
            new_ans = new_ans[:max_len]

            extended_inputs.append(new_seq)
            extended_ans.append(new_ans)

        # padding 到 max_len（与原版一致：每个 t 的 max_len 不同）
        max_len = (t + 1) + K
        padded_inputs = torch.full((batch_size, max_len), Constants.PAD, dtype=torch.long, device=device)
        padded_ans = torch.zeros((batch_size, max_len), dtype=torch.float, device=device)

        for b in range(batch_size):
            seq = extended_inputs[b]
            ans = extended_ans[b]
            padded_inputs[b, :len(seq)] = torch.tensor(seq, device=device, dtype=torch.long)
            padded_ans[b, :len(ans)] = torch.tensor(ans, device=device, dtype=torch.float)

        yt_after = kt_model(padded_inputs, padded_ans, graph)  # [B, max_len, num_skills] 或 [B, max_len-1, num_skills]
        # ✅ 与原版一致：取最后一步
        p_after = yt_after[:, -1, :].detach()
        yt_after_list.append(p_after)

    return torch.stack(yt_after_list, dim=1)  # [B, seq_len-1, num_skills]


# ======================================================
# Learning Path Environment
# ======================================================
class LearningPathEnv:
    """
    Vectorized (batch) environment.
    - reset(...) takes a batch of student histories.
    - step(chosen_item_ids, step_topk_items) appends to each student's path.
    - compute_final_metrics() uses Metrics.combined_metrics with yt_after built by KTOnlyModel simulate_learning.
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
        metrics_topnum: int = 2,
        metrics_T: int = 5,
    ):
        self.batch_size = int(batch_size)
        self.base_model = base_model
        self.recommendation_length = int(recommendation_length)
        self.data_name = data_name
        self.graph = graph
        self.hypergraph_list = hypergraph_list

        self.policy_topk = int(policy_topk)

        # MUST be >=2 for diversity in Metrics.combined_metrics
        self.metrics_topnum = max(2, int(metrics_topnum))
        self.metrics_T = int(metrics_T)

        self.metric = Metrics()

        # step-level shaping weights (NOT Metrics)
        self.step_weights = {
            "preference": 0.05,
            "diversity": 0.02,
            "difficulty": 0.02,
        }

        # final reward weights (Metrics-aligned)
        # self.final_weights = {
        #     "effectiveness": 0.4,
        #     "adaptivity": 0.3,
        #     "diversity": 0.2,
        #     "preference": 0.1,
        # }
        self.final_weights = {
            "effectiveness": 0.0,
            "adaptivity": 0.3,
            "diversity": 0.2,
            "preference": 0.1,
        }

        # KT model for simulate_learning
        self.kt_model = None
        if KTOnlyModel is not None:
            try:
                self.kt_model = KTOnlyModel(self.base_model)
            except Exception:
                self.kt_model = None

        self._clear_episode_cache()

    def _clear_episode_cache(self):
        self.original_tgt = None
        self.original_tgt_timestamp = None
        self.original_tgt_idx = None
        self.original_ans = None

        self.paths = None               # chosen-only path: List[List[int]]
        self.topk_recs = None           # per-step Top-N: List[List[List[int]]]
        self.current_step = 0

        # caches from last forward
        self._last_pred_probs_full = None  # [B, steps, num_skills]
        self._last_yt_full = None          # [B, steps, num_skills]
        self._last_hidden = None
        # --------- for evaluation metrics (no Metrics.combined_metrics) ----------
        self._pref_hist = None      # List[List[float]] per sample per step
        self._repeat_hist = None    # List[List[int]] 1 if repeated else 0

    def _forward_base(self, seq, timestamp, idx, ans):
        """
        Run base model and reshape outputs into [B, steps, num_skills] where steps = seq_len-1.
        """
        B = seq.size(0)
        seq_len = seq.size(1)
        steps = seq_len - 1

        with torch.no_grad():
            pred_flat, _, _, yt, hidden, *_ = self.base_model(
                seq, timestamp, idx, ans, self.graph, self.hypergraph_list
            )

        pred = _reshape_pred_flat(pred_flat, B, steps)
        pred_probs = torch.softmax(pred, dim=-1)

        self._last_pred_probs_full = pred_probs
        self._last_yt_full = yt
        self._last_hidden = hidden
        return pred_probs, yt, hidden

    def _make_state_from_last(self):
        pred_probs = self._last_pred_probs_full
        yt = self._last_yt_full

        last_probs = pred_probs[:, -1, :]  # [B, num_skills]
        cand_probs, cand_ids = torch.topk(last_probs, k=self.policy_topk, dim=-1)  # [B, topk]

        return {
            "knowledge_state": yt[:, -1, :],  # [B, num_skills]
            "pred_probs": last_probs,         # [B, num_skills]
            "cand_ids": cand_ids,             # [B, topk]
            "cand_probs": cand_probs,         # [B, topk]
        }

    def reset(self, tgt, tgt_timestamp, tgt_idx, ans):
        B = tgt.size(0)
        self.batch_size = B

        self.original_tgt = tgt
        self.original_tgt_timestamp = tgt_timestamp
        self.original_tgt_idx = tgt_idx
        self.original_ans = ans

        self.paths = [[] for _ in range(B)]
        self.topk_recs = [[] for _ in range(B)]
        self.current_step = 0
        self._pref_hist = [[] for _ in range(B)]
        self._repeat_hist = [[] for _ in range(B)]

        self._forward_base(tgt, tgt_timestamp, tgt_idx, ans)
        return self._make_state_from_last()

    def step(self, chosen_item_ids: torch.Tensor, step_topk_items: torch.Tensor):
        """
        chosen_item_ids: LongTensor [B] real item ids.
        step_topk_items: LongTensor [B, metrics_topnum] for Metrics (Top-N at this step).
        """
        B = chosen_item_ids.size(0)
        device = chosen_item_ids.device

        # ====== (A) cache OLD state outputs BEFORE transition ======
        # old_pred_probs_last: [B, num_skills]
        # old_knowledge_last:  [B, num_skills]
        old_pred_probs_last = self._last_pred_probs_full[:, -1, :].detach()
        old_knowledge_last = self._last_yt_full[:, -1, :].detach()

        # 1) update chosen paths
        for i in range(B):
            self.paths[i].append(int(chosen_item_ids[i].item()))

        # 2) record step Top-N recs for Metrics
        for i in range(B):
            recs = step_topk_items[i].tolist()
            self.topk_recs[i].append([int(x) for x in recs])

        # ====== (B) compute step reward using OLD state (correct timing) ======
        step_reward = self._compute_step_reward_from_cached(
            chosen_item_ids,
            old_pred_probs_last,
            old_knowledge_last,
        )

        # 3) build extended inputs
        ext_len = len(self.paths[0])
        path_tensor = torch.tensor(self.paths, device=device, dtype=self.original_tgt.dtype)  # [B, ext_len]
        ext_tgt = torch.cat([self.original_tgt, path_tensor], dim=1)

        ext_timestamp = _pad_and_concat_like(self.original_tgt_timestamp, ext_len, pad_value=0)
        ext_idx = _pad_and_concat_like(self.original_tgt_idx, ext_len, pad_value=0)

        pad_ans = torch.zeros(B, ext_len, device=device, dtype=self.original_ans.dtype)
        ext_ans = torch.cat([self.original_ans, pad_ans], dim=1)

        # 4) forward base model for NEW state
        self._forward_base(ext_tgt, ext_timestamp, ext_idx, ext_ans)

        # 5) done
        self.current_step += 1
        done = torch.zeros(B, dtype=torch.bool, device=device)
        if self.current_step >= self.recommendation_length:
            done[:] = True

        return self._make_state_from_last(), step_reward, done

    def _compute_step_reward(self, chosen_item_ids: torch.Tensor) -> torch.Tensor:
        B = chosen_item_ids.size(0)
        device = chosen_item_ids.device
        reward = torch.zeros(B, device=device)

        pred_probs_last = self._last_pred_probs_full[:, -1, :]  # [B, num_skills]
        knowledge_last = self._last_yt_full[:, -1, :]           # [B, num_skills]

        for i in range(B):
            item = int(chosen_item_ids[i].item())

            # preference
            pref_val = 0.0
            if 0 <= item < pred_probs_last.size(1):
                pref_val = float(pred_probs_last[i, item].item())
                reward[i] += self.step_weights["preference"] * pred_probs_last[i, item]

            # repeat penalty
            is_repeat = 1 if item in self.paths[i][:-1] else 0
            if is_repeat:
                reward[i] -= self.step_weights["diversity"]

            # -------- record for evaluation --------
            if self._pref_hist is not None:
                self._pref_hist[i].append(pref_val)
            if self._repeat_hist is not None:
                self._repeat_hist[i].append(is_repeat)

            # difficulty proxy
            if 0 <= item < knowledge_last.size(1):
                k = knowledge_last[i, item]
                reward[i] += self.step_weights["difficulty"] * (1 - torch.abs(k - 0.5))

        return reward

    def _compute_step_reward_from_cached(
            self,
            chosen_item_ids: torch.Tensor,
            pred_probs_last: torch.Tensor,  # OLD [B, num_skills]
            knowledge_last: torch.Tensor,  # OLD [B, num_skills]
    ) -> torch.Tensor:
        B = chosen_item_ids.size(0)
        device = chosen_item_ids.device
        reward = torch.zeros(B, device=device)

        for i in range(B):
            item = int(chosen_item_ids[i].item())

            # preference (based on OLD pred_probs)
            pref_val = 0.0
            if 0 <= item < pred_probs_last.size(1):
                pref_val = float(pred_probs_last[i, item].item())
                reward[i] += self.step_weights["preference"] * pred_probs_last[i, item]

            # repeat penalty (based on chosen path so far; after append is ok because we check[:-1])
            is_repeat = 1 if item in self.paths[i][:-1] else 0
            if is_repeat:
                reward[i] -= self.step_weights["diversity"]

            # record for evaluation (still record OLD preference, correct)
            if self._pref_hist is not None:
                self._pref_hist[i].append(pref_val)
            if self._repeat_hist is not None:
                self._repeat_hist[i].append(is_repeat)

            # difficulty proxy (based on OLD knowledge state)
            if 0 <= item < knowledge_last.size(1):
                k = knowledge_last[i, item]
                reward[i] += self.step_weights["difficulty"] * (1 - torch.abs(k - 0.5))

        return reward

    # ---------------- Metrics outputs ----------------
    def compute_final_metrics(self):
        """
        Episode-level metrics for appended-path RL setting.
        DOES NOT call Metrics.combined_metrics (avoids eff_valid_count==0 division).
        Returns per-sample tensors:
          effectiveness, adaptivity, diversity, preference, final_quality
        """
        if self.original_tgt is None:
            z = torch.zeros(self.batch_size)
            return {
                "effectiveness": z, "adaptivity": z, "diversity": z, "preference": z, "final_quality": z
            }

        device = self.original_tgt.device
        B = self.batch_size

        # 1) effectiveness: use the same as training final reward (episode gain)
        eff_t = self._effectiveness_episode_gain()  # [B]

        # 2) preference: average predicted prob of chosen items (recorded during steps)
        pref_t = torch.zeros(B, device=device)
        if self._pref_hist is not None:
            for i in range(B):
                if len(self._pref_hist[i]) > 0:
                    pref_t[i] = float(sum(self._pref_hist[i]) / max(1, len(self._pref_hist[i])))

        # 3) diversity: simple proxy based on repeats / unique ratio
        div_t = torch.zeros(B, device=device)
        for i in range(B):
            path = self.paths[i] if self.paths is not None else []
            if len(path) > 0:
                unique_ratio = len(set(path)) / len(path)          # in (0,1]
                div_t[i] = float(unique_ratio)
            else:
                div_t[i] = 0.0

        # 4) adaptivity: (safe placeholder) you can refine later
        # --- adaptivity (original definition from Metrics.calculate_adaptivity) ---
        # original_seqs / original_ans: 用真实历史（不要加 PAD，不要加推荐）
        original_seqs = self.original_tgt.detach().cpu().tolist()  # [B, L]
        original_ans = self.original_ans.detach().cpu().tolist()  # [B, L]

        B = self.batch_size
        L = len(original_seqs[0])
        topnum = self.metrics_topnum

        # 构造对齐的 topk_sequence: [B, L-1, topnum] (用 list 形式即可)
        topk_sequence_aligned = []
        for i in range(B):
            recs = [[] for _ in range(L - 1)]
            # 取 RL 最终路径的前 topnum 个作为“最后一步推荐”
            last_recs = (self.paths[i][:topnum] if (self.paths is not None and len(self.paths[i]) > 0) else [])
            recs[L - 2] = [int(x) for x in last_recs]
            topk_sequence_aligned.append(recs)

        # Metrics.calculate_adaptivity 返回的是一个“全局平均标量”
        adaptivity_scalar = self.metric.calculate_adaptivity(
            original_seqs=original_seqs,
            original_ans=original_ans,
            topk_sequence=topk_sequence_aligned,
            data_name=self.data_name,
            T=self.metrics_T
        )

        # 为了保持 compute_final_metrics 返回 [B] 结构，这里把标量扩展成每个样本同值
        ada_t = torch.full((B,), float(adaptivity_scalar), device=self.original_tgt.device)

        # 5) final_quality: weighted sum (you can tune weights)
        # fq_t = (
        #     self.final_weights.get("effectiveness", 0.4) * eff_t
        #     + self.final_weights.get("diversity", 0.2) * div_t
        #     + self.final_weights.get("preference", 0.1) * pref_t
        #     + self.final_weights.get("adaptivity", 0.3) * ada_t
        # )
        fq_t = (
                0.0 * eff_t
                + self.final_weights.get("diversity", 0.2) * div_t
                + self.final_weights.get("preference", 0.1) * pref_t
                + self.final_weights.get("adaptivity", 0.3) * ada_t
        )

        return {
            "effectiveness": eff_t,
            "adaptivity": ada_t,
            "diversity": div_t,
            "preference": pref_t,
            "final_quality": fq_t,
        }


    @torch.no_grad()
    def _effectiveness_episode_gain(self) -> torch.Tensor:
        """
        Episode-level effectiveness reward for appended recommendations:
        Gain = mean( KT(full_seq)_end - KT(history)_end )
        返回: [B]
        """
        device = self.original_tgt.device
        B = self.batch_size

        # 没有 KT 模型则无法算增益（兜底返回 0）
        if self.kt_model is None:
            return torch.zeros(B, device=device)

        # history
        hist_seq = self.original_tgt  # [B, L]
        hist_ans = self.original_ans  # [B, L]
        L = hist_seq.size(1)

        # append chosen recommendations (paths): list of list[int]
        rec_len = len(self.paths[0]) if self.paths is not None and len(self.paths) > 0 else 0
        if rec_len == 0:
            return torch.zeros(B, device=device)

        rec_tensor = torch.tensor(self.paths, device=device, dtype=hist_seq.dtype)  # [B, rec_len]
        full_seq = torch.cat([hist_seq, rec_tensor], dim=1)  # [B, L+rec_len]

        # === 构造推荐题的伪答案 ===
        # 先跑一次 KT 得到 history 末端知识状态（作为伪答题依据）
        # KTOnlyModel.forward: (seq, ans, graph) -> yt_all [B, len, num_skills]
        yt_hist = self.kt_model(hist_seq, hist_ans, self.graph)  # [B, L, K]
        k_end = yt_hist[:, -1, :]  # [B, K]

        # 用 “>=0.5” 或者你原来 “>=0” 的规则生成伪答案（建议先用 >=0.5，更像概率）
        # 这里对每个推荐题取 k_end[item] 作为答对概率阈值生成答案
        full_ans = torch.zeros((B, L + rec_len), device=device, dtype=hist_ans.dtype)
        full_ans[:, :L] = hist_ans

        for i in range(B):
            for j, item in enumerate(self.paths[i]):
                item = int(item)
                if 0 <= item < k_end.size(1):
                    full_ans[i, L + j] = 1 if (k_end[i, item] >= 0) else 0
                else:
                    full_ans[i, L + j] = 0

        # === 跑 KT 得到 full 末端知识状态 ===
        yt_full = self.kt_model(full_seq, full_ans, self.graph)  # [B, L+rec_len, K]
        # gain = (yt_full[:, -1, :] - yt_hist[:, -1, :]).mean(dim=-1)  # [B]
        # skills involved in the appended recs for each sample
        gain = torch.zeros(B, device=device)
        for i in range(B):
            idx = torch.tensor(list(set(self.paths[i])), device=device, dtype=torch.long)
            idx = idx[(idx >= 0) & (idx < yt_hist.size(2))]
            if idx.numel() == 0:
                continue
            gain[i] = (yt_full[i, -1, idx] - yt_hist[i, -1, idx]).mean()

        return gain

    def compute_final_reward(self) -> torch.Tensor:
        # episode-level effectiveness only (no Metrics.combined_metrics, no division by zero)
        return self._effectiveness_episode_gain()





# ======================================================
# Policy Gradient Trainer (REINFORCE + baseline)
# ======================================================
class PPOTrainer:
    def __init__(self, policy_net: PolicyNetwork, env: LearningPathEnv, lr=3e-4, gamma=0.99, entropy_coef=0.0001):
        self.policy_net = policy_net
        self.env = env
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)

    def collect_trajectory(self, tgt, tgt_timestamp, tgt_idx, ans, deterministic: bool = False):
        state = self.env.reset(tgt, tgt_timestamp, tgt_idx, ans)

        rewards = []
        log_probs = []
        entropies = []

        B = tgt.size(0)
        device = tgt.device

        for _ in range(self.env.recommendation_length):
            cand_probs = state["cand_probs"]                 # [B, topk]
            cand_ids = state["cand_ids"]                     # [B, topk]
            cand_feat = cand_probs.unsqueeze(-1)             # [B, topk, 1]

            logits = self.policy_net(state["knowledge_state"], cand_feat)  # [B, topk]
            dist = Categorical(logits=logits)

            if deterministic:
                action_index = torch.argmax(logits, dim=-1)
            else:
                action_index = dist.sample()

            log_prob = dist.log_prob(action_index)
            entropy = dist.entropy()

            chosen_item = cand_ids.gather(1, action_index.view(-1, 1)).squeeze(1)  # [B]

            # Metrics top-2: chosen + best alternative
            topn = self.env.metrics_topnum
            step_topk = torch.full((B, topn), Constants.PAD, device=device, dtype=cand_ids.dtype)
            step_topk[:, 0] = chosen_item
            for k in range(1, topn):
                alt = cand_ids[:, k] if k < cand_ids.size(1) else cand_ids[:, 0]
                if k < cand_ids.size(1) - 1:
                    alt2 = cand_ids[:, k + 1]
                    alt = torch.where(alt == chosen_item, alt2, alt)
                step_topk[:, k] = alt

            next_state, step_reward, done = self.env.step(chosen_item, step_topk)

            rewards.append(step_reward)
            log_probs.append(log_prob)
            entropies.append(entropy)

            state = next_state

        # episode final reward (Metrics) —— ✅ 只在最后一步加
        final_reward = self.env.compute_final_reward()  # [B]
        rewards[-1] = rewards[-1] + final_reward

        return rewards, log_probs, entropies, final_reward

    def update_policy(self, rewards, log_probs, entropies):
        T = len(rewards)
        if T == 0:
            return 0.0

        rewards_t = torch.stack(rewards, dim=0)     # [T, B]
        logp_t = torch.stack(log_probs, dim=0)      # [T, B]
        ent_t = torch.stack(entropies, dim=0)       # [T, B]

        # discounted returns
        returns = torch.zeros_like(rewards_t)
        running = torch.zeros_like(rewards_t[0])
        for t in reversed(range(T)):
            running = rewards_t[t] + self.gamma * running
            returns[t] = running

        # baseline: per-sample mean across time
        baseline = returns.mean(dim=0, keepdim=True)
        adv = returns - baseline

        # 可选：adv 标准化（避免 adv_std 太小导致训练不动）
        adv_std = adv.detach().std(dim=0, keepdim=True).clamp_min(1e-6)
        adv = adv / adv_std

        pg_loss = -(logp_t * adv.detach()).mean()
        ent_loss = -ent_t.mean()
        loss = pg_loss + self.entropy_coef * ent_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        print(f"pg_loss={pg_loss.item():.6f}  entropy={ent_t.mean().item():.6f}  adv_std={adv.detach().std().item():.6f}")
        return float(loss.item())


# ======================================================
# High-level wrappers (keep your old interface needs)
# ======================================================
class RLPathOptimizer:
    def __init__(
        self,
        pretrained_model,
        num_skills,
        batch_size,
        recommendation_length,
        topk,
        data_name,
        graph=None,
        hypergraph_list=None,
        hidden_dim=128,
        lr=3e-4,
        gamma=0.99,
        entropy_coef=1e-4,
        metrics_topnum=2,
        metrics_T=5,
        device=None,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

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
            candidate_feature_dim=1,   # ✅ cand_probs 只有 1 维
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
def evaluate_policy(env: LearningPathEnv, policy_net: PolicyNetwork, data_loader, device, max_batches=20):
    """
    在给定 data_loader 上，用确定性策略（argmax）生成路径，
    输出推荐路径的平均指标：effectiveness/adaptivity/diversity/preference/final_quality
    """
    policy_net.eval()

    all_eff = []
    all_ada = []
    all_div = []
    all_pref = []
    all_fq = []

    for bidx, batch in enumerate(data_loader):
        if bidx >= max_batches:
            break

        tgt, tgt_timestamp, tgt_idx, ans = batch
        tgt = tgt.to(device)
        tgt_timestamp = tgt_timestamp.to(device)
        tgt_idx = tgt_idx.to(device)
        ans = ans.to(device)

        state = env.reset(tgt, tgt_timestamp, tgt_idx, ans)

        for _ in range(env.recommendation_length):
            cand_probs = state["cand_probs"]                 # [B, topk]
            cand_feat = cand_probs.unsqueeze(-1)             # [B, topk, 1]
            logits = policy_net(state["knowledge_state"], cand_feat)
            action_index = torch.argmax(logits, dim=-1)

            cand_ids = state["cand_ids"]
            chosen_item = cand_ids.gather(1, action_index.view(-1, 1)).squeeze(1)

            # step top-2
            B = tgt.size(0)
            topn = env.metrics_topnum
            step_topk = torch.full((B, topn), Constants.PAD, device=device, dtype=cand_ids.dtype)
            step_topk[:, 0] = chosen_item
            for k in range(1, topn):
                alt = cand_ids[:, k] if k < cand_ids.size(1) else cand_ids[:, 0]
                if k < cand_ids.size(1) - 1:
                    alt2 = cand_ids[:, k + 1]
                    alt = torch.where(alt == chosen_item, alt2, alt)
                step_topk[:, k] = alt

            state, _, _ = env.step(chosen_item, step_topk)

        metrics = env.compute_final_metrics()
        all_eff.append(metrics["effectiveness"].detach().cpu())
        all_ada.append(metrics["adaptivity"].detach().cpu())
        all_div.append(metrics["diversity"].detach().cpu())
        all_pref.append(metrics["preference"].detach().cpu())
        all_fq.append(metrics["final_quality"].detach().cpu())

    if len(all_eff) == 0:
        return {
            "effectiveness": 0.0,
            "adaptivity": 0.0,
            "diversity": 0.0,
            "preference": 0.0,
            "final_quality": 0.0,
        }

    eff = torch.cat(all_eff).mean().item()
    ada = torch.cat(all_ada).mean().item()
    div = torch.cat(all_div).mean().item()
    pref = torch.cat(all_pref).mean().item()
    fq = torch.cat(all_fq).mean().item()

    return {
        "effectiveness": eff,
        "adaptivity": ada,
        "diversity": div,
        "preference": pref,
        "final_quality": fq,
    }
