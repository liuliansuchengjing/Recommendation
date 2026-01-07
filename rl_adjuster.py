# rl_adjuster.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

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
    original_seqs,
    original_ans,
    topk_sequence,
    graph,
    yt_before: torch.Tensor,
    batch_size: int,
    topnum: int,
):
    """
    按照 calculate_muti_obj.py 的思想，把推荐资源插入序列并用 KTOnlyModel 跑出 yt_after。
    这是为了给 Metrics.combined_metrics 提供正确的 yt_after。

    输入：
      - original_seqs: List[List[int]] 长度 batch_size
      - original_ans:  List[List[int]] 长度 batch_size
      - topk_sequence: List[List[List[int]]] 形状 [B][seq_len-1][topnum or less]
      - yt_before: Tensor [B, seq_len-1, num_skills]（来自 base_model 输出的 yt/knowledge state）
    输出：
      - yt_after: Tensor [B, seq_len-1, num_skills]
    """
    device = yt_before.device
    B = batch_size
    seq_len_minus_1 = yt_before.size(1)
    num_skills = yt_before.size(2)

    yt_after = torch.zeros((B, seq_len_minus_1, num_skills), device=device, dtype=yt_before.dtype)

    for b in range(B):
        original_seq = list(original_seqs[b])
        original_answer = list(original_ans[b])

        for t in range(seq_len_minus_1):
            recs = topk_sequence[b][t] if (b < len(topk_sequence) and t < len(topk_sequence[b])) else []
            # 只保留有效推荐（非 PAD）
            recommended = [int(r) for r in recs[:topnum] if int(r) != int(Constants.PAD)]

            insert_pos = t + 1
            new_seq = original_seq[:insert_pos] + recommended + original_seq[insert_pos:]

            # 推荐项的“伪答题结果”：用 yt_before 在阈值 0.5 上二值化
            if len(recommended) > 0:
                pred_probs = yt_before[b, t, torch.tensor(recommended, device=device)]
                pred_answers = (pred_probs >= 0.5).long().detach().cpu().tolist()
            else:
                pred_answers = []

            new_answer = original_answer[:insert_pos] + pred_answers + original_answer[insert_pos:]

            # padding 到 max_len（与原始长度一致，否则 KT 输出长度会变）
            max_len = len(original_seq)
            if len(new_seq) < max_len:
                new_seq = new_seq + [Constants.PAD] * (max_len - len(new_seq))
                new_answer = new_answer + [0] * (max_len - len(new_answer))
            else:
                new_seq = new_seq[:max_len]
                new_answer = new_answer[:max_len]

            new_seq_tensor = torch.tensor([new_seq], device=device, dtype=torch.long)
            new_ans_tensor = torch.tensor([new_answer], device=device, dtype=torch.long)

            # KTOnlyModel.forward: (input_seq, answers, graph) -> yt_all  [B, seq_len, num_skills]
            yt_all = kt_model(new_seq_tensor, new_ans_tensor, graph)
            # 对齐取 t 位置（对应 yt_before 的 t）
            yt_after[b, t, :] = yt_all[0, t, :]

    return yt_after


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
        self.final_weights = {
            "effectiveness": 0.4,
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

        self._forward_base(tgt, tgt_timestamp, tgt_idx, ans)
        return self._make_state_from_last()

    def step(self, chosen_item_ids: torch.Tensor, step_topk_items: torch.Tensor):
        """
        chosen_item_ids: LongTensor [B] real item ids.
        step_topk_items: LongTensor [B, metrics_topnum] for Metrics (Top-N at this step).
        """
        B = chosen_item_ids.size(0)
        device = chosen_item_ids.device

        # 1) update chosen paths
        for i in range(B):
            self.paths[i].append(int(chosen_item_ids[i].item()))

        # 2) record step Top-N recs for Metrics
        for i in range(B):
            recs = step_topk_items[i].tolist()
            self.topk_recs[i].append([int(x) for x in recs])

        # 3) build extended inputs
        ext_len = len(self.paths[0])
        path_tensor = torch.tensor(self.paths, device=device, dtype=self.original_tgt.dtype)  # [B, ext_len]
        ext_tgt = torch.cat([self.original_tgt, path_tensor], dim=1)

        ext_timestamp = _pad_and_concat_like(self.original_tgt_timestamp, ext_len, pad_value=0)
        ext_idx = _pad_and_concat_like(self.original_tgt_idx, ext_len, pad_value=0)

        pad_ans = torch.zeros(B, ext_len, device=device, dtype=self.original_ans.dtype)
        ext_ans = torch.cat([self.original_ans, pad_ans], dim=1)

        # 4) forward base model for new state
        self._forward_base(ext_tgt, ext_timestamp, ext_idx, ext_ans)

        # 5) step reward (shaping)
        step_reward = self._compute_step_reward(chosen_item_ids)

        # 6) done
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
            if 0 <= item < pred_probs_last.size(1):
                reward[i] += self.step_weights["preference"] * pred_probs_last[i, item]

            # repeat penalty
            if item in self.paths[i][:-1]:
                reward[i] -= self.step_weights["diversity"]

            # difficulty proxy
            if 0 <= item < knowledge_last.size(1):
                k = knowledge_last[i, item]
                reward[i] += self.step_weights["difficulty"] * (1 - torch.abs(k - 0.5))

        return reward

    # ---------------- Metrics outputs ----------------
    def compute_final_metrics(self):
        """
        Return per-sample metrics tensors:
          effectiveness, adaptivity, diversity, preference, final_quality
        """
        if self.original_tgt is None:
            z = torch.zeros(self.batch_size)
            return {
                "effectiveness": z, "adaptivity": z, "diversity": z, "preference": z, "final_quality": z
            }

        if self.kt_model is None:
            # 没有 KTOnlyModel 时，无法构造 yt_after，避免输出胡乱的 effectiveness
            z = torch.zeros(self.batch_size, device=self.original_tgt.device)
            return {
                "effectiveness": z, "adaptivity": z, "diversity": z, "preference": z, "final_quality": z
            }

        device = self.original_tgt.device
        B = self.batch_size

        rec_len = self.recommendation_length
        hist_len = self.original_tgt.size(1)
        seq_len_full = hist_len + rec_len

        eff_t = torch.zeros(B, device=device)
        ada_t = torch.zeros(B, device=device)
        div_t = torch.zeros(B, device=device)
        pref_t = torch.zeros(B, device=device)
        fq_t = torch.zeros(B, device=device)

        for i in range(B):
            orig_seq_i = _safe_to_cpu_list_1d(self.original_tgt[i])
            orig_ans_i = _safe_to_cpu_list_1d(self.original_ans[i])

            rec_chosen_i = self.paths[i]
            rec_topk_i = self.topk_recs[i]

            full_seq_i = orig_seq_i + rec_chosen_i
            full_ans_i = orig_ans_i + [0] * len(rec_chosen_i)

            # build topk_sequence for Metrics: [1, seq_len-1, topnum]
            seq_steps = seq_len_full - 1
            hist_offset = hist_len - 1

            topk_seq_steps = []
            for t in range(seq_steps):
                rec_t = t - hist_offset
                if 0 <= rec_t < len(rec_topk_i):
                    recs = rec_topk_i[rec_t][:self.metrics_topnum]
                    if len(recs) < self.metrics_topnum:
                        recs = recs + [Constants.PAD] * (self.metrics_topnum - len(recs))
                    topk_seq_steps.append(recs)
                else:
                    topk_seq_steps.append([])

            topk_sequence = [topk_seq_steps]  # batch_size=1 list

            seq_tensor = torch.tensor([full_seq_i], device=device, dtype=self.original_tgt.dtype)
            ans_tensor = torch.tensor([full_ans_i], device=device, dtype=self.original_ans.dtype)

            ts_i = self.original_tgt_timestamp[i:i+1] if self.original_tgt_timestamp is not None else None
            idx_i = self.original_tgt_idx[i:i+1] if self.original_tgt_idx is not None else None
            ts_i = _pad_and_concat_like(ts_i, len(rec_chosen_i), pad_value=0)
            idx_i = _pad_and_concat_like(idx_i, len(rec_chosen_i), pad_value=0)

            pred_probs_1, yt_before, hidden = self._forward_base(seq_tensor, ts_i, idx_i, ans_tensor)

            # ✅ 正确构造 yt_after：用 KTOnlyModel + simulate_learning
            yt_after = simulate_learning_for_metrics(
                kt_model=self.kt_model,
                original_seqs=[full_seq_i],
                original_ans=[full_ans_i],
                topk_sequence=topk_sequence,
                graph=self.graph,
                yt_before=yt_before,
                batch_size=1,
                topnum=self.metrics_topnum,
            )

            pred_probs_flat = pred_probs_1.reshape(-1, pred_probs_1.size(-1)).detach().cpu().numpy()

            m = self.metric.combined_metrics(
                yt_before=yt_before,
                yt_after=yt_after,
                topk_sequence=topk_sequence,
                original_seqs=[full_seq_i],
                hidden=hidden,
                data_name=self.data_name,
                batch_size=1,
                seq_len=seq_len_full,
                pred_probs=pred_probs_flat,
                topnum=self.metrics_topnum,
                T=self.metrics_T,
            )

            eff = float(m.get("effectiveness", 0.0))
            ada = float(m.get("adaptivity", 0.0))
            div = float(m.get("diversity", 0.0))
            pref = float(m.get("preference", 0.0))

            eff_t[i] = eff
            ada_t[i] = ada
            div_t[i] = div
            pref_t[i] = pref

            fq_t[i] = (
                self.final_weights["effectiveness"] * eff
                + self.final_weights["adaptivity"] * ada
                + self.final_weights["diversity"] * div
                + self.final_weights["preference"] * pref
            )

        return {
            "effectiveness": eff_t,
            "adaptivity": ada_t,
            "diversity": div_t,
            "preference": pref_t,
            "final_quality": fq_t,
        }

    def compute_final_reward(self) -> torch.Tensor:
        return self.compute_final_metrics()["final_quality"]


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
