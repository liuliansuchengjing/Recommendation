# # integrated_system.py
# import torch
# import torch.nn as nn
# from HGAT import MSHGAT  # 导入你原有的模型
# from rl_adjuster import LearningPathRLAdjuster  # 导入RL调整器
#
#
# class PersonalizedLearningSystem(nn.Module):
#     """
#     个性化学习系统（集成原有模型 + RL调整器）
#     这个类作为整个系统的统一入口
#     """
#
#     def __init__(self, original_model_path, state_dim=300):
#         super().__init__()
#
#         # 1. 加载并冻结原有模型
#         self.original_model = MSHGAT()
#         self.original_model.load_state_dict(torch.load(original_model_path))
#         self._freeze_original_model()  # 关键：冻结参数
#
#         # 2. 初始化RL调整器
#         self.rl_adjuster = LearningPathRLAdjuster(
#             state_dim=state_dim,
#             hidden_dim=128,
#             learning_rate=1e-3
#         )
#
#         # 3. 其他必要的组件（特征提取器等）
#         self.feature_extractor = ...  # 如果你有单独的特征提取器
#
#     def _freeze_original_model(self):
#         """冻结原模型的所有参数"""
#         self.original_model.eval()  # 设置为评估模式
#         for param in self.original_model.parameters():
#             param.requires_grad = False  # 关键：不计算梯度
#
#         print("原模型参数已冻结（不可训练）")
#         print(f"可训练参数数量（仅RL调整器）: {self.count_trainable_parameters()}")
#
#     def count_trainable_parameters(self):
#         """统计可训练参数数量（应只包含RL调整器）"""
#         return sum(p.numel() for p in self.parameters() if p.requires_grad)
#
#     def train_rl_adjuster(self, train_data_path, val_data_path, epochs=10):
#         """
#         第二阶段：训练RL调整器（原模型参数保持冻结）
#
#         参数:
#             train_data_path: 训练数据路径（用于RL训练）
#             val_data_path: 验证数据路径
#             epochs: 训练轮数
#         """
#         print("=" * 50)
#         print("开始第二阶段：训练RL调整器")
#         print("原模型参数保持冻结")
#         print("=" * 50)
#
#         # 1. 准备RL训练数据
#         print("准备RL训练数据...")
#         train_experiences = self._prepare_rl_training_data(train_data_path)
#         val_experiences = self._prepare_rl_training_data(val_data_path)
#
#         # 2. 添加到经验回放缓冲区
#         print(f"添加 {len(train_experiences)} 条训练经验...")
#         for exp in train_experiences:
#             self.rl_adjuster.add_experience(*exp)
#
#         # 3. 训练循环
#         print(f"开始训练，共 {epochs} 个周期...")
#         for epoch in range(epochs):
#             # 训练步骤
#             train_loss = self._train_one_epoch()
#
#             # 验证步骤
#             val_metrics = self._validate_rl_adjuster(val_experiences)
#
#             print(f"Epoch {epoch + 1}/{epochs}: "
#                   f"Train Loss = {train_loss:.4f}, "
#                   f"Val Reward = {val_metrics['avg_reward']:.4f}")
#
#             # 可选的：保存检查点
#             if (epoch + 1) % 5 == 0:
#                 self.rl_adjuster.save_model(f"rl_adjuster_epoch_{epoch + 1}.pth")
#
#         print("RL调整器训练完成！")
#
#     def _prepare_rl_training_data(self, data_path):
#         """
#         准备RL训练数据：使用冻结的原模型处理数据
#
#         这是你需要实现的核心方法之一！
#         流程：
#         1. 加载数据
#         2. 使用原模型（冻结）为每个用户生成初始路径
#         3. 构建 (s_t, a_t, r_t, s_{t+1}) 经验元组
#         """
#         experiences = []
#
#         # 伪代码，你需要根据实际数据格式实现
#         data = load_your_data(data_path)
#
#         with torch.no_grad():  # 关键：确保原模型不计算梯度
#             for user_data in data:
#                 # 使用冻结的原模型生成初始路径
#                 initial_path = self.original_model.generate_path(user_data)
#
#                 # 为路径上的每个点构建经验
#                 for t in range(len(initial_path)):
#                     # 构建状态s_t（需要你实现状态构建逻辑）
#                     s_t = self._build_state(user_data, t, initial_path)
#
#                     # 定义动作a_t（根据历史判断：保持=0，调整=1）
#                     a_t = self._define_action(user_data, t, initial_path)
#
#                     # 计算奖励r_t（基于后续真实学习效果）
#                     r_t = self._calculate_reward(user_data, t)
#
#                     # 构建下一状态s_{t+1}
#                     s_t1 = self._build_state(user_data, t + 1, initial_path)
#
#                     experiences.append((s_t, a_t, r_t, s_t1))
#
#         return experiences
#
#     def _train_one_epoch(self):
#         """训练一个周期"""
#         total_loss = 0
#         num_batches = max(1, len(self.rl_adjuster.replay_buffer) // self.rl_adjuster.batch_size)
#
#         for _ in range(num_batches):
#             loss = self.rl_adjuster.train_step()
#             total_loss += loss
#
#         return total_loss / num_batches if num_batches > 0 else 0
#
#     def _validate_rl_adjuster(self, val_experiences):
#         """在验证集上评估RL调整器"""
#         # 实现验证逻辑，例如计算平均奖励等
#         rewards = [exp[2] for exp in val_experiences]  # exp[2]是奖励
#         return {'avg_reward': np.mean(rewards)}
#
#     def forward(self, user_ids, learning_goals):
#         """
#         前向传播：生成调整后的学习路径
#
#         参数:
#             user_ids: 用户ID列表（批次）
#             learning_goals: 学习目标列表（批次）
#
#         返回:
#             调整后的路径（批次）
#         """
#         batch_size = len(user_ids)
#         adjusted_paths = []
#
#         with torch.no_grad():  # 推理时也不需要梯度
#             # 1. 原模型生成初始路径（批次）
#             original_paths = self.original_model.batch_generate(user_ids, learning_goals)
#
#             # 2. 对每个用户的路径进行RL调整
#             for i in range(batch_size):
#                 # 获取用户历史状态
#                 user_history = self._get_user_history(user_ids[i])
#                 current_knowledge_state = self._get_knowledge_state(user_ids[i])
#
#                 # 准备调整所需的输入
#                 adjustment_input = self._prepare_adjustment_input(
#                     original_paths[i],
#                     current_knowledge_state,
#                     user_history
#                 )
#
#                 # 调用RL调整器进行调整
#                 adjusted_path = self.rl_adjuster.adjust_learning_path(**adjustment_input)
#                 adjusted_paths.append(adjusted_path['adjusted_path'])
#
#         return {
#             'original_paths': original_paths,
#             'adjusted_paths': adjusted_paths
#         }
#
#     # 以下方法需要你根据实际情况实现
#     def _build_state(self, user_data, step, initial_path):
#         """构建RL状态向量"""
#         # 需要实现：结合知识状态、当前资源、历史等
#         pass
#
#     def _define_action(self, user_data, step, initial_path):
#         """定义动作标签（基于历史数据）"""
#         # 需要实现：判断历史中用户是"保持"还是"调整"了推荐
#         pass
#
#     def _calculate_reward(self, user_data, step):
#         """计算奖励值"""
#         # 需要实现：基于后续学习效果
#         pass
#
#     def _get_user_history(self, user_id):
#         """获取用户历史"""
#         pass
#
#     def _get_knowledge_state(self, user_id):
#         """获取当前知识状态"""
#         pass
#
#     def _prepare_adjustment_input(self, original_path, knowledge_state, user_history):
#         """为RL调整器准备输入"""
#         pass
#
#
# # ==================== 使用示例 ====================
#
# def main():
#     # 1. 初始化集成系统
#     print("初始化集成学习系统...")
#     system = PersonalizedLearningSystem(
#         original_model_path="your_trained_original_model.pth",
#         state_dim=300  # 根据实际情况调整
#     )
#
#     # 2. 检查参数冻结情况
#     print(f"\n系统总参数数: {sum(p.numel() for p in system.parameters())}")
#     print(f"可训练参数数: {system.count_trainable_parameters()}")
#
#     # 3. 训练RL调整器
#     system.train_rl_adjuster(
#         train_data_path="data/train_set.pkl",
#         val_data_path="data/val_set.pkl",
#         epochs=10
#     )
#
#     # 4. 保存完整系统（可选）
#     torch.save(system.state_dict(), "full_integrated_system.pth")
#
#     # 5. 使用系统生成路径
#     test_user_ids = ["user_001", "user_002"]
#     test_goals = ["掌握微积分", "学习Python编程"]
#
#     results = system(test_user_ids, test_goals)
#     print(f"\n生成了 {len(results['adjusted_paths'])} 条调整后的路径")
#
#
# if __name__ == "__main__":
#     main()