# rl_adjuster.py
"""
A minimal RL adjuster (policy gradient) compatible with the "new-interface" train_rl.py:
- PolicyNetwork
- LearningPathEnv
- PPOTrainer

Key fixes:
1) Action is an index within per-step candidate list (topk), then mapped to a real item id.
2) Final reward uses Metrics.combined_metrics trajectory-level logic. To provide per-sample learning
   signal, compute Metrics per-sample (batch_size=1) without modifying Metrics.py.
3) Safe padding for timestamp/idx: only pad if they are 2D+ sequence tensors.
4) update_policy uses REINFORCE + baseline (avoid z-score that can shrink gradients to ~0).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import Constants
from Metrics import Metrics


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


def _build_topk_sequence(history_len: int, rec_paths: list, seq_len_full: int, topnum: int):
    """
    Build python list topk_sequence of shape [seq_len_full-1, <=topnum] for ONE sample.
    - history steps -> []
    - rec steps -> [item]
    Metrics.combined_metrics will internally pad with PAD to topnum.
    """
    seq_steps = seq_len_full - 1
    topk_seq = []
    hist_offset = history_len - 1
    for t in range(seq_steps):
        rec_t = t - hist_offset
        if 0 <= rec_t < len(rec_paths):
            topk_seq.append([int(rec_paths[rec_t])])
        else:
            topk_seq.append([])
    return topk_seq


# ======================================================
# Learning Path Environment
# ======================================================
class LearningPathEnv:
    """
    Vectorized (batch) environment:
    - reset(...) takes a batch of student histories.
    - step(chosen_item_ids) appends items to each student's path and re-runs base_model to get new state.
    - state includes: knowledge_state, cand_ids, cand_probs (used by policy).
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
        metrics_topnum: int = 1,
        metrics_T: int = 5,
    ):
        self.batch_size = int(batch_size)
        self.base_model = base_model
        self.recommendation_length = int(recommendation_length)
        self.data_name = data_name
        self.graph = graph
        self.hypergraph_list = hypergraph_list

        # candidate set size for policy actions
        self.policy_topk = int(policy_topk)

        # Metrics params (topnum should match how many items you "recommend per step"
        # RL here recommends 1 item per step, so topnum=1 is correct.)
        self.metrics_topnum = int(metrics_topnum)
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

        self._clear_episode_cache()

    def _clear_episode_cache(self):
        self.original_tgt = None
        self.original_tgt_timestamp = None
        self.original_tgt_idx = None
        self.original_ans = None

        self.paths = None
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
            # HGAT.MSHGAT forward returns:
            # pred_flat, pred_res, kt_mask, yt, hidden, status_emb
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
        self.current_step = 0

        self._forward_base(tgt, tgt_timestamp, tgt_idx, ans)
        return self._make_state_from_last()

    def step(self, chosen_item_ids: torch.Tensor):
        """
        chosen_item_ids: LongTensor [B] real item ids.
        """
        B = chosen_item_ids.size(0)
        device = chosen_item_ids.device

        # 1) update paths
        for i in range(B):
            self.paths[i].append(int(chosen_item_ids[i].item()))

        # 2) build extended inputs
        ext_len = len(self.paths[0])
        path_tensor = torch.tensor(self.paths, device=device, dtype=self.original_tgt.dtype)  # [B, ext_len]
        ext_tgt = torch.cat([self.original_tgt, path_tensor], dim=1)

        ext_timestamp = _pad_and_concat_like(self.original_tgt_timestamp, ext_len, pad_value=0)
        ext_idx = _pad_and_concat_like(self.original_tgt_idx, ext_len, pad_value=0)

        pad_ans = torch.zeros(B, ext_len, device=device, dtype=self.original_ans.dtype)
        ext_ans = torch.cat([self.original_ans, pad_ans], dim=1)

        # 3) forward base model for new state
        self._forward_base(ext_tgt, ext_timestamp, ext_idx, ext_ans)

        # 4) step reward (shaping)
        step_reward = self._compute_step_reward(chosen_item_ids)

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

            if 0 <= item < pred_probs_last.size(1):
                reward[i] += self.step_weights["preference"] * pred_probs_last[i, item]

            if item in self.paths[i][:-1]:
                reward[i] -= self.step_weights["diversity"]

            if 0 <= item < knowledge_last.size(1):
                k = knowledge_last[i, item]
                reward[i] += self.step_weights["difficulty"] * (1 - torch.abs(k - 0.5))

        return reward

    def compute_final_reward(self) -> torch.Tensor:
        """
        Final reward aligned with Metrics.combined_metrics.

        IMPORTANT: combined_metrics aggregates over a batch. To provide per-sample learning signal,
        we compute metrics PER SAMPLE (batch_size=1). This is slower but correct and avoids modifying Metrics.py.
        """
        if self.original_tgt is None:
            return torch.zeros(self.batch_size)

        device = self.original_tgt.device
        B = self.batch_size

        rec_len = self.recommendation_length
        hist_len = self.original_tgt.size(1)
        seq_len_full = hist_len + rec_len
        steps_full = seq_len_full - 1

        final_reward = torch.zeros(B, device=device)

        for i in range(B):
            orig_seq_i = self.original_tgt[i].detach().cpu().tolist()
            orig_ans_i = self.original_ans[i].detach().cpu().tolist()
            rec_i = self.paths[i]

            full_seq_i = orig_seq_i + rec_i
            full_ans_i = orig_ans_i + [0] * len(rec_i)

            topk_seq_i = _build_topk_sequence(
                history_len=hist_len,
                rec_paths=rec_i,
                seq_len_full=seq_len_full,
                topnum=self.metrics_topnum,
            )
            topk_sequence = [topk_seq_i]  # batch_size=1

            seq_tensor = torch.tensor([full_seq_i], device=device, dtype=self.original_tgt.dtype)
            ans_tensor = torch.tensor([full_ans_i], device=device, dtype=self.original_ans.dtype)

            ts_i = self.original_tgt_timestamp[i:i+1] if self.original_tgt_timestamp is not None else None
            idx_i = self.original_tgt_idx[i:i+1] if self.original_tgt_idx is not None else None
            ts_i = _pad_and_concat_like(ts_i, len(rec_i), pad_value=0)
            idx_i = _pad_and_concat_like(idx_i, len(rec_i), pad_value=0)

            pred_probs_1, yt_1, hidden = self._forward_base(seq_tensor, ts_i, idx_i, ans_tensor)
            # pred_probs_1: [1, steps_full, num_skills]
            # yt_1:        [1, steps_full, num_skills]

            yt_before = yt_1
            yt_after = torch.zeros_like(yt_before)
            yt_after[:, :-1, :] = yt_before[:, 1:, :]
            yt_after[:, -1, :] = yt_before[:, -1, :]

            pred_probs_flat = pred_probs_1.reshape(-1, pred_probs_1.size(-1)).detach().cpu().numpy()

            metrics = self.metric.combined_metrics(
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

            r = (
                self.final_weights["effectiveness"] * float(metrics.get("effectiveness", 0.0))
                + self.final_weights["adaptivity"] * float(metrics.get("adaptivity", 0.0))
                + self.final_weights["diversity"] * float(metrics.get("diversity", 0.0))
                + self.final_weights["preference"] * float(metrics.get("preference", 0.0))
            )
            final_reward[i] = r

        return final_reward


# ======================================================
# Policy Gradient Trainer (REINFORCE + baseline)
# ======================================================
class PPOTrainer:
    def __init__(self, policy_net: PolicyNetwork, env: LearningPathEnv, lr=3e-4, gamma=0.99, entropy_coef=0.001):
        self.policy_net = policy_net
        self.env = env
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)

    def collect_trajectory(self, tgt, tgt_timestamp, tgt_idx, ans):
        state = self.env.reset(tgt, tgt_timestamp, tgt_idx, ans)

        states_k = []
        actions_idx = []
        rewards = []
        log_probs = []
        entropies = []

        B = tgt.size(0)
        device = tgt.device

        for _ in range(self.env.recommendation_length):
            # Candidate features: use candidate probs as a 1D feature
            cand_probs = state["cand_probs"]                     # [B, topk]
            cand_feat = cand_probs.unsqueeze(-1)                 # [B, topk, 1]

            logits = self.policy_net(state["knowledge_state"], cand_feat)  # [B, topk]
            dist = Categorical(logits=logits)

            action_index = dist.sample()                         # [B] in [0, topk)
            log_prob = dist.log_prob(action_index)               # [B]
            entropy = dist.entropy()                             # [B]

            # Map index -> real item id
            cand_ids = state["cand_ids"]                         # [B, topk]
            chosen_item = cand_ids.gather(1, action_index.view(-1, 1)).squeeze(1)  # [B]

            next_state, step_reward, done = self.env.step(chosen_item)

            states_k.append(state["knowledge_state"])
            actions_idx.append(action_index)
            rewards.append(step_reward)
            log_probs.append(log_prob)
            entropies.append(entropy)

            state = next_state

        # Episode-level final reward (Metrics), broadcast to each step
        final_reward = self.env.compute_final_reward()           # [B]
        for t in range(len(rewards)):
            rewards[t] = rewards[t] + final_reward

        return states_k, actions_idx, rewards, log_probs, entropies

    def update_policy(self, states_k, actions_idx, rewards, log_probs, entropies):
        T = len(rewards)
        if T == 0:
            return 0.0

        rewards_t = torch.stack(rewards, dim=0)     # [T, B]
        logp_t = torch.stack(log_probs, dim=0)      # [T, B]
        ent_t = torch.stack(entropies, dim=0)       # [T, B]

        # Discounted returns
        returns = torch.zeros_like(rewards_t)
        running = torch.zeros_like(rewards_t[0])
        for t in reversed(range(T)):
            running = rewards_t[t] + self.gamma * running
            returns[t] = running

        # Baseline: subtract per-sample mean across time
        baseline = returns.mean(dim=0, keepdim=True)   # [1, B]
        advantages = returns - baseline                # [T, B]

        pg_loss = -(logp_t * advantages.detach()).mean()
        ent_loss = -ent_t.mean()
        loss = pg_loss + self.entropy_coef * ent_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        return float(loss.item())
