# rl_adjuster.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

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
        scores = self.fc2(h).squeeze(-1)
        return scores


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
    ):
        self.batch_size = batch_size
        self.base_model = base_model
        self.recommendation_length = recommendation_length
        self.data_name = data_name

        self.graph = graph
        self.hypergraph_list = hypergraph_list

        # step reward weights
        self.step_weights = {
            "preference": 0.1,
            "diversity": 0.05,
            "difficulty": 0.05,
        }

        # final reward weights (Metrics)
        self.final_weights = {
            "effectiveness": 0.4,
            "adaptivity": 0.3,
            "diversity": 0.2,
            "preference": 0.1,
        }

        self.metric = Metrics()

    # ---------------- reset ----------------
    def reset(self, tgt, tgt_timestamp, tgt_idx, ans):
        """
        tgt: [B, T]
        """
        self.original_tgt = tgt
        self.original_tgt_timestamp = tgt_timestamp
        self.original_tgt_idx = tgt_idx
        self.original_ans = ans

        self.paths = [[] for _ in range(self.batch_size)]
        self.all_knowledge_states = [[] for _ in range(self.batch_size)]
        self.all_predictions = [[] for _ in range(self.batch_size)]

        self.current_step = 0

        # initial forward
        with torch.no_grad():
            pred, _, _, knowledge_state, hidden, *_ = self.base_model(
                tgt,
                tgt_timestamp,
                tgt_idx,
                ans,
                self.graph,
                self.hypergraph_list,
            )

        state = {
            "knowledge_state": knowledge_state[:, -1],
            "pred": pred[:, -1],
        }
        return state

    # ---------------- step ----------------
    def step(self, action):
        """
        action: LongTensor [B]
        """
        B = action.size(0)
        device = action.device

        # 1. update path
        for i in range(B):
            self.paths[i].append(action[i].item())

        # 2. build extended sequence
        path_tensor = torch.tensor(self.paths, device=device)
        ext_tgt = torch.cat([self.original_tgt, path_tensor], dim=1)

        pad_ans = torch.zeros(B, path_tensor.size(1), device=device)
        ext_ans = torch.cat([self.original_ans, pad_ans], dim=1)

        # 3. forward model
        with torch.no_grad():
            pred, _, _, knowledge_state, hidden, *_ = self.base_model(
                ext_tgt,
                self.original_tgt_timestamp,
                self.original_tgt_idx,
                ext_ans,
                self.graph,
                self.hypergraph_list,
            )

        # 4. cache trajectory info
        for i in range(B):
            self.all_knowledge_states[i].append(
                knowledge_state[i, -1].detach()
            )
            self.all_predictions[i].append(
                torch.softmax(pred[i, -1], dim=-1).detach()
            )

        # 5. step reward
        step_reward = self._compute_step_reward(
            action,
            knowledge_state[:, -1],
            pred[:, -1],
        )

        # 6. done
        self.current_step += 1
        done = torch.zeros(B, dtype=torch.bool, device=device)
        if self.current_step >= self.recommendation_length:
            done[:] = True

        next_state = {
            "knowledge_state": knowledge_state[:, -1],
            "pred": pred[:, -1],
        }
        return next_state, step_reward, done

    # ---------------- step reward ----------------
    def _compute_step_reward(self, action, knowledge_state, pred_logits):
        B = action.size(0)
        reward = torch.zeros(B, device=action.device)

        probs = torch.softmax(pred_logits, dim=-1)

        for i in range(B):
            a = action[i].item()

            # preference
            reward[i] += self.step_weights["preference"] * probs[i, a]

            # diversity (repeat penalty)
            if a in self.paths[i][:-1]:
                reward[i] -= self.step_weights["diversity"]

            # difficulty proxy (knowledge ~ 0.5)
            k = knowledge_state[i, a]
            reward[i] += self.step_weights["difficulty"] * (1 - torch.abs(k - 0.5))

        return reward

    # ---------------- final reward ----------------
    def compute_final_reward(self):
        """
        trajectory-level Metrics reward
        """
        B = self.batch_size
        final_reward = torch.zeros(B, device=self.original_tgt.device)

        for i in range(B):
            if len(self.paths[i]) == 0:
                continue

            metrics = self.metric.evaluate_trajectory({
                "original_sequence": self.original_tgt[i].tolist(),
                "original_answers": self.original_ans[i].tolist(),
                "recommended_path": self.paths[i],
                "knowledge_before": self.all_knowledge_states[i][:-1],
                "knowledge_after": self.all_knowledge_states[i][1:],
                "predicted_answers": [0] * len(self.paths[i]),
                "model_predictions": self.all_predictions[i],
                "hidden_embeddings": [],
            })

            final_reward[i] = (
                self.final_weights["effectiveness"] * metrics["effectiveness"]
                + self.final_weights["adaptivity"] * metrics["adaptivity"]
                + self.final_weights["diversity"] * metrics["diversity"]
                + self.final_weights["preference"] * metrics["preference"]
            )

        return final_reward


# ======================================================
# PPO Trainer
# ======================================================
class PPOTrainer:
    def __init__(self, policy_net, env, lr=3e-4, gamma=0.99):
        self.policy_net = policy_net
        self.env = env
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)

    def collect_trajectory(self, tgt, tgt_timestamp, tgt_idx, ans):
        state = self.env.reset(tgt, tgt_timestamp, tgt_idx, ans)

        states, actions, rewards, log_probs = [], [], [], []

        for _ in range(self.env.recommendation_length):
            scores = self.policy_net(
                state["knowledge_state"],
                torch.zeros(
                    tgt.size(0),
                    self.policy_net.topk,
                    5,
                    device=tgt.device,
                ),
            )
            dist = Categorical(logits=scores)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_state, step_reward, done = self.env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(step_reward)
            log_probs.append(log_prob)

            state = next_state

        # add final reward
        final_reward = self.env.compute_final_reward()
        for t in range(len(rewards)):
            rewards[t] = rewards[t] + final_reward

        return states, actions, rewards, log_probs
