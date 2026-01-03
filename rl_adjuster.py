# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# from collections import deque, defaultdict
# import random
# from typing import List, Tuple, Dict
#
#
# class LearningPathRLAdjuster:
#     """
#     离线强化学习路径调整器
#     集成到你的现有模型中，为生成的TopK路径提供动态调整能力
#     """
#
#     def __init__(self,
#                  state_dim: int = 300,  # 状态维度，根据你的特征调整
#                  hidden_dim: int = 128,
#                  learning_rate: float = 1e-3,
#                  device: str = "cuda" if torch.cuda.is_available() else "cpu"):
#         """
#         初始化调整器
#
#         参数:
#             state_dim: 状态向量的维度（知识状态 + 资源特征 + 历史上下文）
#             hidden_dim: 神经网络隐藏层维度
#             learning_rate: 学习率
#             device: 计算设备
#         """
#         self.device = torch.device(device)
#         self.state_dim = state_dim
#
#         # Q网络：评估状态-动作价值
#         self.q_network = self._build_q_network(state_dim, hidden_dim).to(self.device)
#         self.target_q_network = self._build_q_network(state_dim, hidden_dim).to(self.device)
#         self.target_q_network.load_state_dict(self.q_network.state_dict())
#
#         self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
#         self.criterion = nn.MSELoss()
#
#         # 经验回放缓冲区
#         self.replay_buffer = deque(maxlen=10000)
#
#         # 训练参数
#         self.gamma = 0.99  # 折扣因子
#         self.tau = 0.005  # 目标网络软更新参数
#         self.batch_size = 64
#
#         # 资源候选池缓存
#         self.resource_pool = []
#         self.resource_embeddings = {}
#
#     def _build_q_network(self, state_dim: int, hidden_dim: int) -> nn.Module:
#         """构建Q网络架构"""
#         return nn.Sequential(
#             nn.Linear(state_dim + 1, hidden_dim),  # +1 for action encoding
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 1)  # 输出Q值
#         )
#
#     def prepare_state(self,
#                       knowledge_state: np.ndarray,
#                       planned_resource: np.ndarray,
#                       recent_history: List[np.ndarray]) -> np.ndarray:
#         """
#         准备强化学习状态向量
#
#         参数:
#             knowledge_state: 知识追踪模型输出的状态向量 [d_k]
#             planned_resource: 原模型计划的资源特征向量 [d_r]
#             recent_history: 最近学习的资源特征列表，每个[d_r]
#
#         返回:
#             状态向量 [state_dim]
#         """
#         # 扁平化历史资源（取最近3个）
#         recent_count = min(3, len(recent_history))
#         if recent_count > 0:
#             history_features = np.concatenate(recent_history[-recent_count:])
#             # 如果历史不足，用零填充
#             if len(history_features) < 3 * planned_resource.shape[0]:
#                 padding = np.zeros(3 * planned_resource.shape[0] - len(history_features))
#                 history_features = np.concatenate([history_features, padding])
#         else:
#             history_features = np.zeros(3 * planned_resource.shape[0])
#
#         # 拼接所有特征
#         state_vector = np.concatenate([
#             knowledge_state.flatten(),
#             planned_resource.flatten(),
#             history_features
#         ])
#
#         # 确保维度一致
#         if len(state_vector) > self.state_dim:
#             state_vector = state_vector[:self.state_dim]
#         elif len(state_vector) < self.state_dim:
#             padding = np.zeros(self.state_dim - len(state_vector))
#             state_vector = np.concatenate([state_vector, padding])
#
#         return state_vector
#
#     def add_experience(self,
#                        state: np.ndarray,
#                        action: int,  # 0:保持原推荐, 1:替换
#                        reward: float,
#                        next_state: np.ndarray,
#                        done: bool):
#         """
#         添加经验到回放缓冲区
#
#         参数:
#             state: 当前状态
#             action: 采取的动作
#             reward: 获得的奖励
#             next_state: 下一状态
#             done: 是否结束
#         """
#         self.replay_buffer.append({
#             'state': state,
#             'action': action,
#             'reward': reward,
#             'next_state': next_state,
#             'done': done
#         })
#
#     def calculate_reward(self,
#                          knowledge_gain: float,
#                          diversity_score: float,
#                          difficulty_match: float,
#                          weights: Tuple[float, float, float] = (0.5, 0.3, 0.2)) -> float:
#         """
#         计算多目标复合奖励
#
#         参数:
#             knowledge_gain: 知识掌握度提升
#             diversity_score: 多样性得分 (0-1)
#             difficulty_match: 难度匹配度 (0-1)
#             weights: 各目标权重 (掌握度, 多样性, 难度)
#
#         返回:
#             综合奖励值
#         """
#         w_mastery, w_diversity, w_difficulty = weights
#         reward = (w_mastery * knowledge_gain +
#                   w_diversity * diversity_score +
#                   w_difficulty * difficulty_match)
#         return reward
#
#     def train_step(self) -> float:
#         """
#         执行一次训练步骤
#
#         返回:
#             损失值
#         """
#         if len(self.replay_buffer) < self.batch_size:
#             return 0.0
#
#         # 从缓冲区采样
#         batch = random.sample(self.replay_buffer, self.batch_size)
#
#         # 准备批量数据
#         states = np.array([exp['state'] for exp in batch])
#         actions = np.array([exp['action'] for exp in batch])
#         rewards = np.array([exp['reward'] for exp in batch])
#         next_states = np.array([exp['next_state'] for exp in batch])
#         dones = np.array([exp['done'] for exp in batch])
#
#         # 转换为张量
#         states_tensor = torch.FloatTensor(states).to(self.device)
#         actions_tensor = torch.FloatTensor(actions).unsqueeze(1).to(self.device)
#         rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
#         next_states_tensor = torch.FloatTensor(next_states).to(self.device)
#         dones_tensor = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
#
#         # 计算当前Q值
#         state_action_pairs = torch.cat([states_tensor, actions_tensor], dim=1)
#         current_q_values = self.q_network(state_action_pairs)
#
#         # 计算目标Q值
#         with torch.no_grad():
#             # 为每个next_state计算两个动作的Q值
#             action0 = torch.zeros(self.batch_size, 1).to(self.device)
#             action1 = torch.ones(self.batch_size, 1).to(self.device)
#
#             next_state_action0 = torch.cat([next_states_tensor, action0], dim=1)
#             next_state_action1 = torch.cat([next_states_tensor, action1], dim=1)
#
#             q0 = self.target_q_network(next_state_action0)
#             q1 = self.target_q_network(next_state_action1)
#             next_q_values = torch.max(torch.cat([q0, q1], dim=1), dim=1, keepdim=True)[0]
#
#             target_q_values = rewards_tensor + (1 - dones_tensor) * self.gamma * next_q_values
#
#         # 计算损失并更新
#         loss = self.criterion(current_q_values, target_q_values)
#
#         self.optimizer.zero_grad()
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)  # 梯度裁剪
#         self.optimizer.step()
#
#         # 软更新目标网络
#         for target_param, param in zip(self.target_q_network.parameters(),
#                                        self.q_network.parameters()):
#             target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
#
#         return loss.item()
#
#     def decide_action(self,
#                       state: np.ndarray,
#                       threshold: float = 0.1) -> Tuple[int, float, float]:
#         """
#         决策是否调整当前推荐
#
#         参数:
#             state: 当前状态
#             threshold: 调整阈值，Q值差异超过此阈值才调整
#
#         返回:
#             (action, q_keep, q_replace)
#             action: 0=保持原推荐, 1=替换
#             q_keep: 保持动作的Q值
#             q_replace: 替换动作的Q值
#         """
#         # 准备两个动作的输入
#         state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
#
#         action0 = torch.zeros(1, 1).to(self.device)
#         action1 = torch.ones(1, 1).to(self.device)
#
#         state_action0 = torch.cat([state_tensor, action0], dim=1)
#         state_action1 = torch.cat([state_tensor, action1], dim=1)
#
#         # 计算Q值
#         with torch.no_grad():
#             q_keep = self.q_network(state_action0).item()
#             q_replace = self.q_network(state_action1).item()
#
#         # 决策逻辑
#         if q_replace > q_keep + threshold:
#             action = 1  # 替换
#         else:
#             action = 0  # 保持
#
#         return action, q_keep, q_replace
#
#     def find_replacement_resource(self,
#                                   current_resource: np.ndarray,
#                                   candidate_resources: List[np.ndarray],
#                                   knowledge_state: np.ndarray,
#                                   recent_history: List[np.ndarray]) -> Tuple[int, np.ndarray]:
#         """
#         寻找最佳替换资源
#
#         参数:
#             current_resource: 当前计划资源特征
#             candidate_resources: 候选资源特征列表
#             knowledge_state: 当前知识状态
#             recent_history: 近期历史资源
#
#         返回:
#             (best_index, best_resource)
#         """
#         if not candidate_resources:
#             return -1, current_resource
#
#         best_index = -1
#         best_q_value = -float('inf')
#         best_resource = current_resource
#
#         for i, candidate in enumerate(candidate_resources):
#             # 准备状态
#             candidate_state = self.prepare_state(knowledge_state, candidate, recent_history)
#
#             # 计算替换动作的Q值
#             _, _, q_replace = self.decide_action(candidate_state, threshold=0)
#
#             if q_replace > best_q_value:
#                 best_q_value = q_replace
#                 best_index = i
#                 best_resource = candidate
#
#         return best_index, best_resource
#
#     def adjust_learning_path(self,
#                              original_path: List[np.ndarray],
#                              knowledge_states: List[np.ndarray],
#                              recent_histories: List[List[np.ndarray]],
#                              candidate_pools: List[List[np.ndarray]]) -> Dict:
#         """
#         调整整个学习路径
#
#         参数:
#             original_path: 原始路径资源特征列表
#             knowledge_states: 每个决策点的知识状态
#             recent_histories: 每个决策点的近期历史
#             candidate_pools: 每个决策点的候选资源池
#
#         返回:
#             调整结果字典
#         """
#         adjusted_path = []
#         adjustments = []
#         decision_log = []
#
#         for t in range(len(original_path)):
#             # 准备当前状态
#             current_state = self.prepare_state(
#                 knowledge_states[t],
#                 original_path[t],
#                 recent_histories[t]
#             )
#
#             # 决策
#             action, q_keep, q_replace = self.decide_action(current_state)
#
#             if action == 0:  # 保持
#                 adjusted_resource = original_path[t]
#                 adjustment_type = "keep"
#                 replacement_idx = -1
#             else:  # 替换
#                 # 寻找最佳替换
#                 replacement_idx, adjusted_resource = self.find_replacement_resource(
#                     original_path[t],
#                     candidate_pools[t],
#                     knowledge_states[t],
#                     recent_histories[t]
#                 )
#                 adjustment_type = "replace"
#
#             # 记录决策
#             decision_info = {
#                 'step': t,
#                 'action': action,
#                 'q_keep': q_keep,
#                 'q_replace': q_replace,
#                 'adjustment_type': adjustment_type,
#                 'replacement_index': replacement_idx,
#                 'original_resource': original_path[t],
#                 'adjusted_resource': adjusted_resource
#             }
#
#             adjusted_path.append(adjusted_resource)
#             adjustments.append(adjustment_type)
#             decision_log.append(decision_info)
#
#         return {
#             'adjusted_path': adjusted_path,
#             'adjustments': adjustments,
#             'decision_log': decision_log,
#             'adjustment_count': sum(1 for a in adjustments if a == "replace")
#         }
#
#     def save_model(self, path: str):
#         """保存模型"""
#         torch.save({
#             'q_network_state_dict': self.q_network.state_dict(),
#             'target_q_network_state_dict': self.target_q_network.state_dict(),
#             'optimizer_state_dict': self.optimizer.state_dict(),
#             'state_dim': self.state_dim
#         }, path)
#
#     def load_model(self, path: str):
#         """加载模型"""
#         checkpoint = torch.load(path, map_location=self.device)
#         self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
#         self.target_q_network.load_state_dict(checkpoint['target_q_network_state_dict'])
#         self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         self.state_dim = checkpoint['state_dim']
#
#
# # ==================== 与现有模型集成的示例 ====================
#
# class YourExistingModelWithRL:
#     """
#     你的现有模型与RL调整器的集成示例
#     """
#
#     def __init__(self):
#         # 你的现有组件
#         self.knowledge_tracker = ...  # 你的知识追踪模型
#         self.path_generator = ...  # 你的TopK路径生成模型
#
#         # 新增的RL调整器
#         self.rl_adjuster = LearningPathRLAdjuster(
#             state_dim=300,  # 根据你的特征维度调整
#             hidden_dim=128,
#             learning_rate=1e-3
#         )
#
#         # 资源管理器（需要你根据实际情况实现）
#         self.resource_manager = ...
#
#     def extract_knowledge_state(self, user_id: str, history: List) -> np.ndarray:
#         """
#         从你的知识追踪模型提取状态向量
#         这是你需要实现的适配器方法
#         """
#         # 调用你的知识追踪模型
#         # 假设返回形状为 [d_k] 的向量
#         return self.knowledge_tracker.predict_state(user_id, history)
#
#     def extract_resource_features(self, resource_id: str) -> np.ndarray:
#         """
#         提取资源特征向量
#         这是你需要实现的适配器方法
#         """
#         # 从你的资源特征库中获取
#         # 假设返回形状为 [d_r] 的向量
#         return self.resource_manager.get_embedding(resource_id)
#
#     def prepare_training_data_from_history(self, user_history: Dict):
#         """
#         从历史数据准备训练样本
#         这是你需要实现的适配器方法
#         """
#         training_samples = []
#
#         for user_id, sessions in user_history.items():
#             for i in range(1, len(sessions)):  # 遍历每个决策点
#                 # 获取t时刻状态
#                 history_up_to_t = sessions[:i]
#                 knowledge_state = self.extract_knowledge_state(user_id, history_up_to_t)
#
#                 # 原模型在t时刻的计划推荐
#                 planned_resource_id = sessions[i]['planned_resource']
#                 planned_resource = self.extract_resource_features(planned_resource_id)
#
#                 # 用户实际学习的资源
#                 actual_resource_id = sessions[i]['actual_resource']
#                 actual_resource = self.extract_resource_features(actual_resource_id)
#
#                 # 近期历史（t-1及之前的实际学习资源）
#                 recent_history = []
#                 for j in range(max(0, i - 3), i):  # 取最近3个
#                     hist_resource_id = sessions[j]['actual_resource']
#                     hist_resource = self.extract_resource_features(hist_resource_id)
#                     recent_history.append(hist_resource)
#
#                 # 准备状态
#                 state = self.rl_adjuster.prepare_state(
#                     knowledge_state, planned_resource, recent_history
#                 )
#
#                 # 定义动作：实际资源是否与计划相同
#                 action = 0 if actual_resource_id == planned_resource_id else 1
#
#                 # 计算奖励（需要你实现奖励计算逻辑）
#                 # 这里简化为基于后续学习效果
#                 future_gain = self._calculate_future_gain(user_id, sessions[i:])
#                 reward = future_gain  # 或使用更复杂的多目标奖励
#
#                 # 准备下一状态（基于实际学习后的状态）
#                 next_knowledge_state = self.extract_knowledge_state(
#                     user_id, sessions[:i + 1]
#                 )
#                 next_state = self.rl_adjuster.prepare_state(
#                     next_knowledge_state,
#                     sessions[i + 1]['planned_resource'] if i + 1 < len(sessions) else planned_resource,
#                     recent_history + [actual_resource]
#                 )
#
#                 # 是否结束
#                 done = (i == len(sessions) - 1)
#
#                 training_samples.append({
#                     'state': state,
#                     'action': action,
#                     'reward': reward,
#                     'next_state': next_state,
#                     'done': done
#                 })
#
#         return training_samples
#
#     def train_rl_adjuster(self, historical_data: List[Dict], epochs: int = 10):
#         """
#         训练RL调整器
#
#         参数:
#             historical_data: 历史用户学习数据
#             epochs: 训练轮数
#         """
#         print("开始训练RL调整器...")
#
#         # 准备训练数据
#         print("准备训练数据...")
#         training_samples = self.prepare_training_data_from_history(historical_data)
#
#         # 添加到经验回放缓冲区
#         print(f"添加 {len(training_samples)} 个训练样本到缓冲区")
#         for sample in training_samples:
#             self.rl_adjuster.add_experience(**sample)
#
#         # 训练循环
#         print("开始训练循环...")
#         for epoch in range(epochs):
#             epoch_loss = 0
#             num_batches = max(1, len(training_samples) // self.rl_adjuster.batch_size)
#
#             for _ in range(num_batches):
#                 loss = self.rl_adjuster.train_step()
#                 epoch_loss += loss
#
#             avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
#             print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, "
#                   f"Buffer size: {len(self.rl_adjuster.replay_buffer)}")
#
#         print("RL调整器训练完成！")
#
#     def generate_adjusted_path(self,
#                                user_id: str,
#                                learning_goal: str,
#                                path_length: int = 10) -> Dict:
#         """
#         生成调整后的学习路径
#
#         参数:
#             user_id: 用户ID
#             learning_goal: 学习目标
#             path_length: 路径长度
#
#         返回:
#             包含原始路径和调整后路径的字典
#         """
#         # 1. 获取用户当前知识状态和历史
#         user_history = self._get_user_history(user_id)
#         current_knowledge_state = self.extract_knowledge_state(user_id, user_history)
#
#         # 2. 用原模型生成TopK路径
#         print("生成原始学习路径...")
#         original_path_ids = self.path_generator.generate_path(
#             user_id, learning_goal, path_length
#         )
#
#         # 转换为特征向量
#         original_path_features = [
#             self.extract_resource_features(res_id) for res_id in original_path_ids
#         ]
#
#         # 3. 为路径上的每个点准备决策信息
#         knowledge_states = []
#         recent_histories = []
#         candidate_pools = []
#
#         # 模拟用户逐步学习的过程
#         simulated_history = list(user_history)  # 复制历史
#
#         for i in range(path_length):
#             # 当前知识状态（基于模拟历史）
#             current_state = self.extract_knowledge_state(user_id, simulated_history)
#             knowledge_states.append(current_state)
#
#             # 近期历史（最近学习的资源）
#             recent = []
#             for j in range(max(0, len(simulated_history) - 3), len(simulated_history)):
#                 hist_res_id = simulated_history[j]['resource_id']
#                 recent.append(self.extract_resource_features(hist_res_id))
#             recent_histories.append(recent)
#
#             # 候选资源池（这里简化为相似资源，你需要根据实际情况实现）
#             current_resource_id = original_path_ids[i]
#             candidates = self._get_similar_resources(current_resource_id, top_n=20)
#             candidate_features = [self.extract_resource_features(res_id) for res_id in candidates]
#             candidate_pools.append(candidate_features)
#
#             # 模拟学习这一步（用于下一步的状态计算）
#             # 注意：这是简化的模拟，实际中RL决策会影响实际学习
#             simulated_history.append({
#                 'resource_id': original_path_ids[i],
#                 'interaction': 'simulated'
#             })
#
#         # 4. 使用RL调整器调整路径
#         print("使用RL调整器优化路径...")
#         adjustment_result = self.rl_adjuster.adjust_learning_path(
#             original_path_features,
#             knowledge_states,
#             recent_histories,
#             candidate_pools
#         )
#
#         # 5. 返回结果
#         result = {
#             'user_id': user_id,
#             'learning_goal': learning_goal,
#             'original_path': original_path_ids,
#             'adjusted_path': [
#                 self._find_resource_id(feature) for feature in adjustment_result['adjusted_path']
#             ],
#             'adjustments': adjustment_result['adjustments'],
#             'adjustment_count': adjustment_result['adjustment_count'],
#             'decision_log': adjustment_result['decision_log']
#         }
#
#         print(f"路径调整完成，共进行了 {adjustment_result['adjustment_count']} 处调整")
#         return result
#
#     def _calculate_future_gain(self, user_id: str, future_sessions: List) -> float:
#         """
#         计算未来学习收益（示例实现）
#         你需要根据实际情况实现
#         """
#         # 简化为未来会话中掌握的知识点数量
#         return len(future_sessions) * 0.1  # 示例值
#
#     def _get_user_history(self, user_id: str) -> List:
#         """获取用户历史（示例）"""
#         # 你需要连接你的数据库
#         return []  # 返回用户学习历史记录
#
#     def _get_similar_resources(self, resource_id: str, top_n: int = 20) -> List[str]:
#         """获取相似资源（示例）"""
#         # 你需要实现资源相似度计算
#         return [resource_id] * top_n  # 示例
#
#     def _find_resource_id(self, feature_vector: np.ndarray) -> str:
#         """根据特征向量查找资源ID（示例）"""
#         # 你需要实现特征到资源的反向查找
#         return "resource_001"
#
#
# # ==================== 快速使用示例 ====================
#
# def quick_demo():
#     """
#     快速演示如何使用这个模块
#     """
#     print("=== 学习路径RL调整器演示 ===")
#
#     # 1. 初始化集成模型
#     print("1. 初始化模型...")
#     integrated_model = YourExistingModelWithRL()
#
#     # 2. 加载历史数据进行训练
#     print("2. 加载历史数据...")
#     # 这里需要你提供实际的历史数据
#     historical_data = [
#         {
#             "user_001": [
#                 {"step": 0, "planned_resource": "res_A", "actual_resource": "res_A", "score": 0.8},
#                 {"step": 1, "planned_resource": "res_B", "actual_resource": "res_C", "score": 0.9},
#                 # ... 更多历史记录
#             ]
#         }
#     ]
#
#     # 3. 训练RL调整器
#     print("3. 训练RL调整器（简化示例）...")
#     integrated_model.train_rl_adjuster(historical_data, epochs=5)
#
#     # 4. 保存训练好的模型
#     print("4. 保存模型...")
#     integrated_model.rl_adjuster.save_model("rl_adjuster_checkpoint.pth")
#
#     # 5. 生成调整后的路径
#     print("5. 为新用户生成调整路径...")
#     result = integrated_model.generate_adjusted_path(
#         user_id="new_user_123",
#         learning_goal="掌握机器学习基础",
#         path_length=8
#     )
#
#     # 6. 展示结果
#     print("\n=== 生成结果 ===")
#     print(f"用户: {result['user_id']}")
#     print(f"学习目标: {result['learning_goal']}")
#     print(f"原始路径: {result['original_path']}")
#     print(f"调整后路径: {result['adjusted_path']}")
#     print(f"调整决策: {result['adjustments']}")
#     print(f"总计调整: {result['adjustment_count']} 处")
#
#     # 显示调整细节
#     print("\n=== 调整细节 ===")
#     for log in result['decision_log'][:3]:  # 显示前3个决策
#         print(f"步骤 {log['step']}: {log['adjustment_type']} "
#               f"(Q保持={log['q_keep']:.3f}, Q替换={log['q_replace']:.3f})")
#
#
# if __name__ == "__main__":
#     # 运行演示
#     quick_demo()