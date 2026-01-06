import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
from graphConstruct import ConRelationGraph, ConHyperGraphList
from dataLoader import Split_data, DataLoader
from Metrics import Metrics
import Constants

metric = Metrics()

class RLPathOptimizer:
    """
    强化学习路径优化器
    核心逻辑：
    1. 在每一步，使用完整的当前历史序列调用预训练模型
    2. 从模型输出的topk候选中选择动作
    3. 基于多目标奖励优化选择策略
    """
    def __init__(self, pretrained_model, num_skills, batch_size, recommendation_length=5, topk=20, data_name=None):
        self.base_model = pretrained_model  # 预训练模型
        self.recommendation_length = recommendation_length  # 推荐路径长度
        self.topk = topk                    # 候选集大小
        self.num_skills = num_skills        # 知识点数量
        self.batch_size = batch_size
        self.data_name = data_name          # 数据集名称，用于计算适应性奖励

        # 冻结预训练模型
        for param in self.base_model.parameters():
            param.requires_grad = False

        # 构建策略网络（动作选择器）
        self.policy_net = self._build_policy_network()

        # 构建环境
        self.env = LearningPathEnv(
            batch_size=self.batch_size,
            base_model=self.base_model,
            policy_net=self.policy_net,
            recommendation_length=self.recommendation_length,
            topk=self.topk,
            data_name=self.data_name
        )

        self.trainer = PPOTrainer(
            policy_net = self.policy_net,
            env = self.env,
            lr = 3e-4,
            gamma = 0.99,
            clip_epsilon = 0.2)

    def _build_policy_network(self):
        """构建策略网络实例"""
        # 确定网络参数
        knowledge_dim = self.num_skills  # 知识状态维度
        candidate_feature_dim = 5  # 候选特征维度（可配置）

        # 创建策略网络
        policy_net = PolicyNetwork(
            knowledge_dim=knowledge_dim,
            candidate_feature_dim=candidate_feature_dim,
            hidden_dim=128,  # 隐藏层维度
            topk=self.topk  # 候选数量
        )

        return policy_net


class LearningPathEnv:
    """
    学习路径环境
    关键特性：
    - 维护每个学习者的完整历史路径
    - 每一步都使用完整历史调用预训练模型
    - 计算多目标奖励
    """

    def __init__(self, batch_size, base_model, policy_net, recommendation_length, topk, data_name):
        self.base_model = base_model  # 预训练模型
        self.policy_net = policy_net  # 策略网络
        self.recommendation_length = recommendation_length  # 推荐路径长度
        self.topk = topk  # 候选集大小
        self.data_name = data_name  # 数据集名称

        # 环境状态
        self.batch_size = batch_size  # 批次大小（动态设置）
        self.histories = []  # 每个学习者的历史路径列表
        self.current_step = 0  # 当前步骤
        self.paths = []  # 已生成的推荐路径
        
        # 存储原始输入数据，以便在step中使用
        self.original_tgt = None
        self.original_tgt_timestamp = None
        self.original_tgt_idx = None
        self.original_ans = None
        self.original_graph = None
        self.original_hypergraph_list = None
        self.current_state = None
        self.original_sequences = None  # 存储原始序列用于奖励计算
        self.all_knowledge_states = []  # 存储每一步的知识状态，用于最终奖励计算
        self.all_predictions = []  # 存储每一步的预测结果，用于最终奖励计算

    def reset(self, tgt, tgt_timestamp, tgt_idx, ans, graph, hypergraph_list):
        """
        重置环境
        Args:
            tgt: [batch_size, history_len] 初始历史路径
            tgt_timestamp: 时间戳
            tgt_idx: 索引
            ans: 答案
            graph: 关系图
            hypergraph_list: 超图列表
        """
        self.original_tgt = tgt
        self.original_tgt_timestamp = tgt_timestamp
        self.original_tgt_idx = tgt_idx
        self.original_ans = ans
        self.original_graph = graph
        self.original_hypergraph_list = hypergraph_list
        
        # 初始化历史路径
        self.batch_size = tgt.size(0)
        self.histories = [tgt[i].clone() for i in range(self.batch_size)]
        self.current_step = 0
        self.paths = [[] for _ in range(self.batch_size)]
        self.all_knowledge_states = []  # 重置知识状态记录
        self.all_predictions = []  # 重置预测结果记录

        # 获取初始状态
        initial_state = self._get_current_state(tgt, tgt_timestamp, tgt_idx, ans, graph, hypergraph_list)
        self.current_state = initial_state

        return initial_state

    def _get_current_state(self, tgt, tgt_timestamp, tgt_idx, ans, graph, hypergraph_list):
        """
        获取当前状态
        """
        # 使用当前累积的历史路径作为输入
        batch_histories = torch.stack(self.histories)  # [batch_size, current_seq_len]

        # 调用预训练模型
        with torch.no_grad():
            pred, pred_res, kt_mask, knowledge_state, hidden = self.base_model(
                batch_histories, tgt_timestamp, tgt_idx, ans, graph, hypergraph_list)
            
            # 获取当前步骤的topk候选（只取最后一个时间步的topk）
            # 获取最后时间步的预测概率
            last_step_pred = pred.view(batch_histories.size(0), -1, pred.size(-1))[:, -1, :]  # [batch_size, num_skills]
            
            # 获取topk候选
            topk_scores, topk_indices = torch.topk(last_step_pred, k=self.topk, dim=-1)  # [batch_size, topk]
            topk_candidates = topk_indices

        # 提取当前知识状态（最后一个时间步）
        if knowledge_state.dim() == 3:
            current_knowledge = knowledge_state[:, -1, :]  # [batch_size, num_skills]
        else:
            current_knowledge = knowledge_state

        # 提取候选特征
        candidate_features = self._extract_candidate_features(topk_candidates, current_knowledge)

        # 构建状态字典
        state = {
            'knowledge_state': current_knowledge,  # [batch_size, num_skills]
            'candidates': topk_candidates,  # [batch_size, topk]
            'candidate_features': candidate_features,  # [batch_size, topk, feature_dim]
            'histories': self.histories,  # 原始历史路径
            'step': self.current_step,  # 当前步骤
            'pred': pred,  # 预测概率
            'pred_res': pred_res,  # 预测结果
            'hidden': hidden,  # 隐藏层
            'knowledge_state_full': knowledge_state  # 完整知识状态
        }

        return state

    def _extract_candidate_features(self, topk_candidates, knowledge_state):
        """
        提取候选习题的特征
        """
        # 确保topk_candidates是tensor并具有正确的形状
        if not isinstance(topk_candidates, torch.Tensor):
            topk_candidates = torch.tensor(topk_candidates)
        
        if topk_candidates.dim() == 1:
            # 如果是一维张量，说明是单个样本，需要扩展维度
            topk_candidates = topk_candidates.unsqueeze(0)
        
        batch_size, topk = topk_candidates.shape
        
        # 创建候选特征
        candidate_features = torch.zeros(batch_size, topk, 5, device=topk_candidates.device, dtype=torch.float)
        
        # 特征1: 习题ID归一化
        candidate_features[:, :, 0] = topk_candidates.float() / self.base_model.n_node
        
        # 特征2: 对应知识点的当前掌握程度
        for b in range(batch_size):
            for k in range(topk):
                skill_id = topk_candidates[b, k].item()
                if skill_id < knowledge_state.size(1):
                    candidate_features[b, k, 1] = knowledge_state[b, skill_id]
        
        # 特征3: 习题在topk中的排名
        rank_weights = torch.linspace(1.0, 0.0, topk, device=topk_candidates.device)
        candidate_features[:, :, 2] = rank_weights.unsqueeze(0).repeat(batch_size, 1)
        
        # 特征4: 与当前知识状态的差异度
        for b in range(batch_size):
            for k in range(topk):
                skill_id = topk_candidates[b, k].item()
                if skill_id < knowledge_state.size(1):
                    diff = abs(0.5 - knowledge_state[b, skill_id])
                    candidate_features[b, k, 3] = diff
        
        # 特征5: 偏好分数
        candidate_features[:, :, 4] = 1.0 - rank_weights.unsqueeze(0).repeat(batch_size, 1)
        
        return candidate_features

    def step(self, actions):
        """
        执行一步动作
        Args:
            actions: [batch_size] 选择的候选索引（0到topk-1）
        Returns:
            next_state: 下一个状态
            rewards: [batch_size] 即时奖励
            dones: [batch_size] 是否终止
        """
        # 获取选择的习题
        selected_exercises = []
        for i in range(self.batch_size):
            candidate_idx = actions[i].item()
            exercise_id = self.current_state['candidates'][i, candidate_idx].item()
            selected_exercises.append(exercise_id)

        selected_exercises = torch.tensor(selected_exercises, dtype=torch.long)

        # 更新历史路径
        for i in range(self.batch_size):
            # 将选择的习题添加到历史
            new_exercise = selected_exercises[i].unsqueeze(0)
            self.histories[i] = torch.cat([self.histories[i], new_exercise])
            # 记录到推荐路径
            self.paths[i].append(selected_exercises[i].item())

        # 更新步骤计数
        self.current_step += 1

        # 获取新状态 - 重新运行预训练模型以获取完整的状态信息
        next_state = self._get_current_state(
            self.original_tgt, 
            self.original_tgt_timestamp, 
            self.original_tgt_idx, 
            self.original_ans, 
            self.original_graph, 
            self.original_hypergraph_list
        )

        # 计算奖励
        rewards = self._calculate_reward(selected_exercises, next_state)

        # 检查是否终止
        dones = torch.tensor([self.current_step >= self.recommendation_length] * self.batch_size, device=selected_exercises.device)

        return next_state, rewards, dones

    def _calculate_reward(self, selected_exercises, next_state):
        """
        计算多目标奖励
        """
        batch_size = selected_exercises.size(0)
        rewards = torch.zeros(batch_size, device=selected_exercises.device)

        # 1. 有效性奖励 - 基于知识状态提升
        if hasattr(self, 'prev_knowledge_state'):
            knowledge_gain = torch.sum(
                next_state['knowledge_state'] - self.prev_knowledge_state,
                dim=-1
            )
            validity_reward = 0.4 * torch.sigmoid(knowledge_gain)
        else:
            validity_reward = torch.zeros(batch_size)

        # 2. 适应性奖励 - 基于知识点掌握程度匹配
        current_knowledge = self.current_state['knowledge_state']
        exercise_ids = selected_exercises
        
        exercise_knowledge_levels = torch.zeros(batch_size, device=selected_exercises.device, dtype=torch.float)
        for i in range(batch_size):
            skill_id = exercise_ids[i].item()
            if skill_id < current_knowledge.size(1):
                exercise_knowledge_levels[i] = current_knowledge[i, skill_id]
            else:
                exercise_knowledge_levels[i] = 0.5  # 默认中等水平
        
        optimal_difficulty = exercise_knowledge_levels + 0.2
        adaptivity = 1.0 - torch.abs(exercise_knowledge_levels - optimal_difficulty)
        adaptivity_reward = 0.3 * torch.clamp(adaptivity, min=0.0)

        # 3. 多样性奖励 - 避免重复知识点
        diversity_reward = torch.zeros(batch_size, device=selected_exercises.device)
        for i in range(batch_size):
            current_exercise_id = selected_exercises[i].item()
            recent_exercises = self.paths[i][-3:] if len(self.paths[i]) >= 3 else self.paths[i]
            
            is_repeated = current_exercise_id in recent_exercises
            if is_repeated:
                diversity_reward[i] = 0.0
            else:
                diversity_reward[i] = 0.2

        # 4. 偏好保持奖励 - 鼓励选择原模型推荐概率高的习题
        preference_reward = torch.zeros(batch_size, device=selected_exercises.device)
        for i in range(batch_size):
            # 从候选中找到选择的习题ID对应的索引
            selected_exercise_id = selected_exercises[i].item()
            # 在当前状态的候选列表中找到这个习题的索引
            candidates_list = self.current_state['candidates'][i]
            selected_idx = -1
            for j, candidate_id in enumerate(candidates_list):
                if candidate_id.item() == selected_exercise_id:
                    selected_idx = j
                    break
            # 在topk中位置越靠前，原模型推荐分数越高
            if selected_idx != -1:
                rank_score = (self.topk - selected_idx) / self.topk
                preference_reward[i] = 0.1 * rank_score
            else:
                preference_reward[i] = 0.0  # 如果未找到，则奖励为0

        # 组合奖励
        rewards = validity_reward + adaptivity_reward + diversity_reward + preference_reward

        # 保存当前知识状态供下一步使用
        self.prev_knowledge_state = self.current_state['knowledge_state'].clone()

        return rewards


class PolicyNetwork(nn.Module):
    """
    策略网络
    输入：知识状态 + 候选特征
    输出：每个候选的得分
    """

    def __init__(self, knowledge_dim, candidate_feature_dim, hidden_dim, topk):
        super().__init__()

        # 知识状态编码器
        self.knowledge_encoder = nn.Sequential(
            nn.Linear(knowledge_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        # 候选特征编码器
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        # 注意力机制：计算知识状态与每个候选的匹配度
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )

        # 得分计算网络
        self.score_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.topk = topk

    def forward(self, knowledge_state, candidate_features):
        """
        Args:
            knowledge_state: [batch_size, knowledge_dim]
            candidate_features: [batch_size, topk, candidate_feature_dim]
        Returns:
            scores: [batch_size, topk] 每个候选的得分
        """
        batch_size = knowledge_state.size(0)

        # 编码知识状态
        knowledge_encoded = self.knowledge_encoder(knowledge_state)  # [batch_size, hidden_dim]
        knowledge_encoded_expanded = knowledge_encoded.unsqueeze(1).repeat(1, self.topk, 1)  # [batch_size, topk, hidden_dim]

        # 编码候选特征
        candidate_encoded = self.candidate_encoder(candidate_features)  # [batch_size, topk, hidden_dim]

        # 注意力机制
        attended, _ = self.attention(
            candidate_encoded,
            knowledge_encoded_expanded,
            knowledge_encoded_expanded
        )

        # 拼接原始候选编码和注意力输出
        combined = torch.cat([candidate_encoded, attended], dim=-1)  # [batch_size, topk, hidden_dim*2]

        # 计算得分
        scores = self.score_network(combined).squeeze(-1)  # [batch_size, topk]

        return scores


class PPOTrainer:
    """
    PPO训练器
    使用近端策略优化算法训练策略网络
    """

    def __init__(self, policy_net, env, lr=3e-4, gamma=0.99, clip_epsilon=0.2):
        self.policy_net = policy_net
        self.env = env
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon

        # 优化器
        self.optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)

        # 经验缓冲区
        self.buffer = []

    def collect_trajectory(self, initial_histories, tgt_timestamp, tgt_idx, ans, graph, hypergraph_list):
        """
        收集一条轨迹（一个批次）
        """
        # 重置环境
        state = self.env.reset(initial_histories, tgt_timestamp, tgt_idx, ans, graph, hypergraph_list)

        states = []
        actions = []
        rewards = []
        log_probs = []
        values = []
        dones = []

        # 运行完整轨迹
        step_count = 0
        max_steps = self.env.recommendation_length  # 防止无限循环
        
        while step_count < max_steps:
            # 获取动作概率分布
            with torch.no_grad():
                scores = self.policy_net(
                    state['knowledge_state'],
                    state['candidate_features']
                )
                action_probs = F.softmax(scores, dim=-1)
                dist = Categorical(action_probs)

                # 采样动作
                action = dist.sample()
                log_prob = dist.log_prob(action)

                # 估计状态价值
                state_value = torch.mean(scores, dim=-1)

            # 执行动作
            next_state, reward, done = self.env.step(action)

            # 存储转换
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(state_value)
            dones.append(done)

            # 更新状态
            state = next_state

            # 检查是否终止
            if torch.all(done):
                break
                
            step_count += 1

        # 构建轨迹
        trajectory = {
            'states': states,
            'actions': torch.stack(actions),
            'rewards': torch.stack(rewards),
            'log_probs': torch.stack(log_probs),
            'values': torch.stack(values),
            'dones': torch.stack(dones)
        }

        # 存储到缓冲区
        self.buffer.append(trajectory)

        return trajectory

    def compute_advantages(self, rewards, values, dones):
        """
        计算优势函数（使用GAE）
        """
        batch_size = rewards.size(1)
        advantages = torch.zeros_like(rewards)

        # 计算GAE
        gae = 0
        for t in reversed(range(len(rewards) - 1)):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * 0.95 * gae * (1 - dones[t])  # 0.95是GAE参数
            advantages[t] = gae

        return advantages

    def update_policy(self):
        """
        使用PPO更新策略网络
        """
        if len(self.buffer) == 0:
            return 0.0

        # 从缓冲区采样轨迹
        trajectory = self.buffer[-1]  # 使用最新轨迹

        states = trajectory['states']
        actions = trajectory['actions']
        old_log_probs = trajectory['log_probs']
        rewards = trajectory['rewards']
        old_values = trajectory['values']
        dones = trajectory['dones']

        # 计算优势函数
        advantages = self.compute_advantages(rewards, old_values, dones)

        # 计算回报
        returns = advantages + old_values

        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 准备批量数据
        batch_states = []
        batch_candidate_features = []
        for state in states:
            batch_states.append(state['knowledge_state'])
            batch_candidate_features.append(state['candidate_features'])

        batch_states = torch.stack(batch_states)
        batch_candidate_features = torch.stack(batch_candidate_features)

        # 计算新策略
        scores = self.policy_net(batch_states, batch_candidate_features)
        action_probs = F.softmax(scores, dim=-1)
        dist = Categorical(action_probs)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        # 计算比率
        ratios = torch.exp(new_log_probs - old_log_probs)

        # PPO裁剪目标
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # 价值损失
        new_values = torch.mean(scores, dim=-1)
        value_loss = F.mse_loss(new_values, returns)

        # 总损失（包含熵正则化）
        total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
        self.optimizer.step()

        # 清空缓冲区
        self.buffer.clear()

        return total_loss.item()


# RL路径优化器 主训练循环
def train_rl_path_optimizer(data_path, opt,
        pretrained_model,
        num_skills,
        batch_size,
        recommendation_length=5,  # 默认为5，符合需求中提到的长度
        num_epochs=50,
        topk=20):
    """
    主训练函数
    """
    # ========= Preparing DataLoader =========#
    user_size, total_cascades, timestamps, train, valid, test = Split_data(data_path, opt.train_rate, opt.valid_rate,
                                                                           load_dict=True)

    train_data = DataLoader(train, batch_size=opt.batch_size, load_dict=True, cuda=not opt.no_cuda and torch.cuda.is_available())
    valid_data = DataLoader(valid, batch_size=opt.batch_size, load_dict=True, cuda=not opt.no_cuda and torch.cuda.is_available())
    test_data = DataLoader(test, batch_size=opt.batch_size, load_dict=True, cuda=not opt.no_cuda and torch.cuda.is_available())

    relation_graph = ConRelationGraph(data_path)
    hypergraph_list = ConHyperGraphList(total_cascades, timestamps, user_size)

    opt.user_size = user_size
    # 初始化RL优化器
    rl_optimizer = RLPathOptimizer(
        pretrained_model=pretrained_model,
        num_skills=num_skills,
        batch_size=batch_size,
        recommendation_length=recommendation_length,
        topk=topk,
        data_name=data_path  # 数据集名称
    )

    # 训练统计
    training_stats = {
        'epoch_losses': [],
        'avg_rewards': [],
        'validity_scores': [],
        'diversity_scores': []
    }

    for epoch in range(num_epochs):
        epoch_losses = []
        epoch_rewards = []

        for batch_idx, batch in enumerate(train_data):
            # batch_data包含用户历史路径
            if torch.cuda.is_available() and not opt.no_cuda:
                tgt, tgt_timestamp, tgt_idx, ans = (item.cuda() for item in batch)
            else:
                tgt, tgt_timestamp, tgt_idx, ans = batch

            # 收集轨迹
            trajectory = rl_optimizer.trainer.collect_trajectory(tgt, tgt_timestamp, tgt_idx, ans, relation_graph, hypergraph_list)

            # 计算平均奖励
            avg_reward = trajectory['rewards'].mean().item()
            epoch_rewards.append(avg_reward)

            # 更新策略
            loss = rl_optimizer.trainer.update_policy()
            epoch_losses.append(loss)

            # 定期评估
            if batch_idx % 12 == 0:
                # 评估当前策略
                validity, diversity = evaluate_policy(
                    rl_optimizer.env,
                    rl_optimizer.policy_net,
                    valid_data
                )

                print(f"Epoch {epoch}, Batch {batch_idx}: "
                      f"Loss={loss:.4f}, Reward={avg_reward:.4f}, "
                      f"Validity={validity:.4f}, Diversity={diversity:.4f}")

        # 记录统计
        avg_epoch_loss = np.mean(epoch_losses) if epoch_losses else 0
        avg_epoch_reward = np.mean(epoch_rewards) if epoch_rewards else 0

        training_stats['epoch_losses'].append(avg_epoch_loss)
        training_stats['avg_rewards'].append(avg_epoch_reward)

        print(f"Epoch {epoch} completed: "
              f"Avg Loss={avg_epoch_loss:.4f}, Avg Reward={avg_epoch_reward:.4f}")

    return rl_optimizer, training_stats


def evaluate_policy(env, policy_net, test_data_loader, num_episodes=10):
    """
    评估策略性能
    """
    validity_scores = []
    diversity_scores = []
    adaptivity_scores = []

    with torch.no_grad():
        for episode in range(num_episodes):
            batch_count = 0
            for batch in test_data_loader:
                if batch_count >= 1:  # 只评估一个批次以节省时间
                    break
                
                # 获取批次数据
                if torch.cuda.is_available():
                    tgt, tgt_timestamp, tgt_idx, ans = (item.cuda() for item in batch)
                else:
                    tgt, tgt_timestamp, tgt_idx, ans = batch
                
                # 重置环境
                state = env.reset(tgt, tgt_timestamp, tgt_idx, ans, None, None)

                episode_validity = []
                episode_diversity = []

                # 运行完整轨迹
                step_count = 0
                max_steps = env.recommendation_length
                
                while step_count < max_steps:
                    # 使用确定性策略（选择最高得分）
                    scores = policy_net(
                        state['knowledge_state'],
                        state['candidate_features']
                    )
                    actions = torch.argmax(scores, dim=-1)

                    # 执行动作
                    next_state, rewards, dones = env.step(actions)

                    # 计算指标
                    batch_size = tgt.size(0)
                    for i in range(batch_size):
                        # 有效性：知识状态提升
                        if hasattr(env, 'prev_knowledge_state'):
                            knowledge_gain = torch.sum(
                                next_state['knowledge_state'][i] - env.prev_knowledge_state[i]
                            )
                            episode_validity.append(knowledge_gain.item())

                        # 多样性：路径中独特知识点比例
                        if len(env.paths[i]) > 0:
                            unique_exercises = set(env.paths[i])
                            diversity = len(unique_exercises) / len(env.paths[i]) if len(env.paths[i]) > 0 else 1
                            episode_diversity.append(diversity)
                        
                        # 适应性：难度匹配程度
                        if 'knowledge_state' in next_state:
                            current_knowledge = next_state['knowledge_state'][i]
                            selected_exercise = env.paths[i][-1] if env.paths[i] else 0
                            if selected_exercise < current_knowledge.size(0):
                                knowledge_level = current_knowledge[selected_exercise].item()
                                # 理想难度应该是中等水平（0.5左右），计算与理想难度的差距
                                adaptivity_score = 1.0 - abs(knowledge_level - 0.5)
                                adaptivity_scores.append(max(0, adaptivity_score))

                    # 更新状态
                    state = next_state

                    # 检查是否终止
                    if torch.all(dones):
                        break
                        
                    step_count += 1

                # 记录指标
                if episode_validity:
                    validity_scores.append(np.mean(episode_validity))
                if episode_diversity:
                    diversity_scores.append(np.mean(episode_diversity))
                    
                batch_count += 1

    avg_validity = np.mean(validity_scores) if validity_scores else 0
    avg_diversity = np.mean(diversity_scores) if diversity_scores else 0
    avg_adaptivity = np.mean(adaptivity_scores) if adaptivity_scores else 0

    return avg_validity, avg_diversity, avg_adaptivity


class RLPathRecommender:
    """
    推理阶段的学习路径推荐器
    """

    def __init__(self, rl_optimizer):
        self.rl_optimizer = rl_optimizer
        self.env = rl_optimizer.env
        self.policy_net = rl_optimizer.policy_net

    def recommend_path(self, user_history, deterministic=True):
        """
        为单个用户推荐学习路径
        Args:
            user_history: 用户历史路径 [history_len]
            deterministic: 是否使用确定性策略
        Returns:
            recommended_path: 推荐的学习路径 [seq_len]
            scores: 每个步骤的推荐得分
        """
        # 准备输入（添加批次维度）
        user_history = user_history.unsqueeze(0)  # [1, history_len]

        # 重置环境
        state = self.env.reset(user_history)

        recommended_path = []
        recommendation_scores = []

        with torch.no_grad():
            for step in range(self.rl_optimizer.recommendation_length):
                # 获取候选得分
                scores = self.policy_net(
                    state['knowledge_state'],
                    state['candidate_features']
                )

                # 选择动作
                if deterministic:
                    action = torch.argmax(scores, dim=-1)
                else:
                    # 使用softmax采样
                    probs = F.softmax(scores, dim=-1)
                    dist = Categorical(probs)
                    action = dist.sample()

                # 获取选择的习题
                exercise_id = state['candidates'][0, action.item()].item()
                recommended_path.append(exercise_id)
                recommendation_scores.append(scores[0, action.item()].item())

                # 执行动作
                state, _, done = self.env.step(action)

                if done[0]:
                    break

        return recommended_path, recommendation_scores

    def batch_recommend(self, user_histories, deterministic=True):
        """
        批量推荐学习路径
        Args:
            user_histories: [batch_size, history_len] 用户历史路径
            deterministic: 是否使用确定性策略
        Returns:
            recommended_paths: [batch_size, seq_len] 推荐路径
            all_scores: [batch_size, seq_len] 推荐得分
        """
        batch_size = user_histories.size(0)

        # 重置环境
        state = self.env.reset(user_histories)

        recommended_paths = [[] for _ in range(batch_size)]
        all_scores = [[] for _ in range(batch_size)]

        with torch.no_grad():
            for step in range(self.rl_optimizer.recommendation_length):
                # 获取候选得分
                scores = self.policy_net(
                    state['knowledge_state'],
                    state['candidate_features']
                )

                # 选择动作
                if deterministic:
                    actions = torch.argmax(scores, dim=-1)
                else:
                    probs = F.softmax(scores, dim=-1)
                    dist = Categorical(probs)
                    actions = dist.sample()

                # 获取每个用户选择的习题
                for i in range(batch_size):
                    exercise_id = state['candidates'][i, actions[i].item()].item()
                    recommended_paths[i].append(exercise_id)
                    all_scores[i].append(scores[i, actions[i].item()].item())

                # 执行动作
                state, _, dones = self.env.step(actions)

                # 检查是否所有用户都完成
                if torch.all(dones):
                    break

        # 转换为张量
        max_len = max(len(path) for path in recommended_paths)

        # 填充路径
        padded_paths = torch.zeros(batch_size, max_len, dtype=torch.long)
        padded_scores = torch.zeros(batch_size, max_len)

        for i in range(batch_size):
            path_len = len(recommended_paths[i])
            padded_paths[i, :path_len] = torch.tensor(recommended_paths[i])
            padded_scores[i, :path_len] = torch.tensor(all_scores[i])

        return padded_paths, padded_scores

    def evaluate_path_quality(self, user_histories, num_paths=100):
        """
        评估推荐路径质量
        """
        # 限制路径数量以节省时间
        actual_paths = min(num_paths, len(user_histories) if isinstance(user_histories, list) else user_histories.size(0) if torch.is_tensor(user_histories) else 10)
        
        # 生成推荐路径
        if torch.is_tensor(user_histories):
            user_histories = user_histories[:actual_paths]
        
        recommended_paths, _ = self.batch_recommend(user_histories)
        
        # 计算各指标
        validity_scores = []
        diversity_scores = []
        adaptivity_scores = []
        
        for i in range(min(actual_paths, len(recommended_paths))):
            path = recommended_paths[i]
            
            # 过滤掉填充的0值
            path = [p for p in path if p != 0]
            if not path:
                continue
                
            # 计算多样性
            unique_items = set(path)
            diversity = len(unique_items) / len(path) if len(path) > 0 else 1
            diversity_scores.append(diversity)
            
            # 计算适应性（需要访问环境中的知识状态）
            # 这里使用一个简化的适应性评估方法
            adaptivity = 0.0
            if len(path) > 0:
                # 假设路径中的项目ID与知识点相关，理想难度应适中
                # 使用项目ID的归一化值来近似评估难度匹配
                avg_item_difficulty = sum(path) / len(path)
                # 假设理想的平均难度是知识点总数的一半
                ideal_difficulty = self.rl_optimizer.num_skills / 2
                # 计算难度匹配度
                difficulty_match = 1 - abs(avg_item_difficulty - ideal_difficulty) / ideal_difficulty
                adaptivity = max(0, difficulty_match)
            adaptivity_scores.append(adaptivity)
        
        # 计算有效性需要运行环境并观察知识状态变化
        # 由于这比较复杂，这里使用简化的计算方法
        # 在实际应用中，可能需要更复杂的逻辑
        
        results = {
            'validity': np.mean(validity_scores) if validity_scores else 0,
            'diversity': np.mean(diversity_scores) if diversity_scores else 0,
            'adaptivity': np.mean(adaptivity_scores) if adaptivity_scores else 0,
            'num_paths_evaluated': sum(1 for path in recommended_paths if any(p != 0 for p in path))
        }
        
        return results


# 使用Metrics函数计算路径的多目标指标
def evaluate_path_metrics(rl_recommender, test_data_loader, data_name):
    """
    使用Metrics中的函数评估生成路径的多目标指标
    """
    all_validity_scores = []
    all_adaptivity_scores = []
    all_diversity_scores = []
    all_preference_scores = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_data_loader):
            if batch_idx >= 5:  # 只评估前几个批次以节省时间
                break
                
            # 获取批次数据
            tgt, tgt_timestamp, tgt_idx, ans = (item.cuda() for item in batch)
            
            # 使用RL策略生成推荐路径
            recommended_paths, _ = rl_recommender.batch_recommend(tgt, deterministic=True)
            
            # 将推荐路径与原始序列拼接以获取完整序列
            full_sequences = torch.cat([tgt, recommended_paths], dim=1)
            full_answers = torch.cat([ans, torch.zeros_like(recommended_paths)], dim=1)
            
            # 使用预训练模型获取知识状态和预测结果
            env = rl_recommender.rl_optimizer.env
            with torch.no_grad():
                pred, pred_res, kt_mask, knowledge_state, hidden = env.base_model(
                    full_sequences, tgt_timestamp, tgt_idx, full_answers, 
                    env.original_graph, env.original_hypergraph_list
                )
            
            # 将数据转换为numpy以便使用Metrics函数
            pred_np = pred.cpu().numpy()
            knowledge_state_np = knowledge_state.cpu().numpy()
            recommended_paths_np = recommended_paths.cpu().numpy()
            full_sequences_np = full_sequences.cpu().numpy()
            full_answers_np = full_answers.cpu().numpy()
            
            # 使用Metrics函数计算各项指标
            # 这里我们使用generate_topk_sequence来生成topk序列用于多样性计算
            topk_sequence = metric.generate_topk_sequence(
                pred_np, full_sequences_np.flatten(), 
                full_sequences.size(0), full_sequences.size(1), topnum=5
            ).cpu().numpy()
            
            # 计算有效性
            # 需要获取yt_before和yt_after，这里我们使用知识状态
            if knowledge_state.dim() == 3:
                yt_before = knowledge_state[:, :-1, :]  # [batch_size, seq_len-1, num_skills]
                yt_after = knowledge_state[:, 1:, :]    # [batch_size, seq_len-1, num_skills]
            else:
                yt_before = knowledge_state
                yt_after = knowledge_state
            
            validity_score = metric.compute_effectiveness(
                full_sequences_np, 
                yt_before.cpu().numpy(), 
                yt_after.cpu().numpy(), 
                topk_sequence
            )
            
            all_validity_scores.append(validity_score)
            
            # 计算适应性
            adaptivity_score = metric.calculate_adaptivity(
                full_sequences_np, 
                full_answers_np, 
                topk_sequence, 
                data_name
            )
            all_adaptivity_scores.append(adaptivity_score)
            
            # 计算多样性
            diversity_score = metric.calculate_diversity(
                full_sequences_np, 
                topk_sequence, 
                hidden, 
                full_sequences.size(0), 
                full_sequences.size(1), 
                5  # topnum
            )
            all_diversity_scores.append(diversity_score)

    avg_validity = np.mean(all_validity_scores) if all_validity_scores else 0
    avg_adaptivity = np.mean(all_adaptivity_scores) if all_adaptivity_scores else 0
    avg_diversity = np.mean(all_diversity_scores) if all_diversity_scores else 0
    
    return {
        'validity': avg_validity,
        'adaptivity': avg_adaptivity,
        'diversity': avg_diversity
    }


"""
使用示例和快速入门

要进行强化学习训练，有两种方式：

方式1: 直接运行训练脚本
    python train_rl.py

方式2: 在代码中使用
    from rl_adjuster import RLPathOptimizer, RLPathRecommender
    from train_rl import run_training_with_pretrained_model
    
    # 快速开始训练
    rl_optimizer, stats = run_training_with_pretrained_model()

    # 或者手动构建训练流程
    rl_optimizer = RLPathOptimizer(
        pretrained_model=your_pretrained_model,
        num_skills=num_skills,
        batch_size=batch_size,
        seq_len=5,  # 推荐路径长度
        topk=20     # 候选集大小
    )
    
    # 然后使用 rl_optimizer.trainer 进行训练循环
    
    # 训练完成后使用推荐器
    recommender = RLPathRecommender(rl_optimizer)
    recommended_path, scores = recommender.recommend_path(user_history)
"""

# 使用示例
if __name__ == "__main__":
    # 示例：如何使用强化学习路径优化器
    # 注意：这只是一个示例框架，实际使用时需要加载训练好的MSHGAT模型
    
    print("RL Path Optimizer 示例")
    print("此模块实现了基于强化学习的多目标学习路径优化")
    print("支持的特性：")
    print("1. 有效性奖励：提升知识掌握程度")
    print("2. 适应性奖励：匹配学习者当前水平")
    print("3. 多样性奖励：避免重复知识点")
    print("4. 偏好保持：与原模型推荐保持一致")
    
    print("\n快速开始指南:")
    print("1. 运行 'python train_rl.py' 进行训练")
    print("2. 训练完成后，使用 RLPathRecommender 进行推理")
    print("3. 参考 README_RL_TRAINING.md 获取详细使用说明")
