import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
from graphConstruct import ConRelationGraph, ConHyperGraphList
from dataLoader import Split_data, DataLoader
from Metrics import Metrics
import Constants
import pickle
from dataLoader import Options

metric = Metrics()


# 轨迹数据管理器类
class TrajectoryDataManager:
    """
    管理RL轨迹中的完整数据，用于最终指标计算
    """

    def __init__(self, batch_size):
        self.batch_size = batch_size
        # 初始化存储结构
        self.reset()

    def reset(self):
        """重置所有存储"""
        # 原始历史数据
        self.original_sequences = [[] for _ in range(self.batch_size)]
        self.original_answers = [[] for _ in range(self.batch_size)]

        # 推荐数据
        self.recommended_paths = [[] for _ in range(self.batch_size)]
        self.predicted_answers = [[] for _ in range(self.batch_size)]

        # 知识状态数据
        self.knowledge_states_before = [[] for _ in range(self.batch_size)]
        self.knowledge_states_after = [[] for _ in range(self.batch_size)]

        # 模型预测数据
        self.model_predictions = [[] for _ in range(self.batch_size)]

        # 资源嵌入（用于多样性计算）
        self.hidden_embeddings = None

        # 当前步骤
        self.current_step = 0

    def add_original_data(self, batch_idx, sequence, answers):
        """添加原始历史数据"""
        self.original_sequences[batch_idx] = sequence
        self.original_answers[batch_idx] = answers

    def add_recommendation(self, batch_idx, exercise_id,
                           knowledge_before, knowledge_after,
                           predicted_answer, model_prediction):
        """添加一个推荐步骤的数据"""
        self.recommended_paths[batch_idx].append(exercise_id)
        self.knowledge_states_before[batch_idx].append(knowledge_before)
        self.knowledge_states_after[batch_idx].append(knowledge_after)
        self.predicted_answers[batch_idx].append(predicted_answer)
        self.model_predictions[batch_idx].append(model_prediction)

    def set_hidden_embeddings(self, hidden_embeddings):
        """设置资源嵌入向量"""
        self.hidden_embeddings = hidden_embeddings

    def get_trajectory_data(self, batch_idx):
        """获取指定batch的完整轨迹数据"""
        return {
            'original_sequence': self.original_sequences[batch_idx],
            'original_answers': self.original_answers[batch_idx],
            'recommended_path': self.recommended_paths[batch_idx],
            'knowledge_before': self.knowledge_states_before[batch_idx],
            'knowledge_after': self.knowledge_states_after[batch_idx],
            'predicted_answers': self.predicted_answers[batch_idx],
            'model_predictions': self.model_predictions[batch_idx],
            'hidden_embeddings': self.hidden_embeddings
        }


# 轨迹指标评估器类
class TrajectoryMetricsEvaluator:
    """
    适配Metrics逻辑到单步推荐轨迹的评估器
    """

    def __init__(self, difficulty_data, idx2u, reward_weights=None):
        self.difficulty_data = difficulty_data
        self.idx2u = idx2u
        self.reward_weights = reward_weights or {
            'validity': 0.4,
            'adaptivity': 0.3,
            'diversity': 0.2,
            'preference': 0.1
        }
        # 缓存难度映射
        self.difficulty_cache = {}

    def _get_difficulty(self, exercise_idx):
        """获取习题难度（带缓存）"""
        if exercise_idx in self.difficulty_cache:
            return self.difficulty_cache[exercise_idx]

        difficulty = 1.0  # 默认难度
        if exercise_idx in self.idx2u:
            original_id = self.idx2u[exercise_idx]
            if original_id in self.difficulty_data:
                difficulty = self.difficulty_data[original_id] / 5.0  # 归一化到0-1

        self.difficulty_cache[exercise_idx] = difficulty
        return difficulty

    def evaluate_trajectory(self, trajectory_data):
        """
        评估完整轨迹的多目标指标
        适配自Metrics.py中的逻辑
        """
        # 解包数据
        original_seq = trajectory_data['original_sequence']
        original_ans = trajectory_data['original_answers']
        recommended_path = trajectory_data['recommended_path']
        knowledge_before = trajectory_data['knowledge_before']
        knowledge_after = trajectory_data['knowledge_after']
        predicted_answers = trajectory_data['predicted_answers']
        model_predictions = trajectory_data['model_predictions']
        hidden_embeddings = trajectory_data['hidden_embeddings']

        # 构建完整序列和答案
        full_sequence = original_seq + recommended_path
        full_answers = original_ans + predicted_answers

        # 计算各项指标
        effectiveness = self._compute_effectiveness(
            recommended_path, knowledge_before, knowledge_after)

        adaptivity = self._compute_adaptivity(
            full_sequence, full_answers, len(original_seq))

        diversity = self._compute_diversity(
            recommended_path, hidden_embeddings)

        preference = self._compute_preference(
            recommended_path, model_predictions)

        # 加权求和得到综合质量分数
        final_quality = (
                self.reward_weights['validity'] * effectiveness +
                self.reward_weights['adaptivity'] * adaptivity +
                self.reward_weights['diversity'] * diversity +
                self.reward_weights['preference'] * preference
        )

        return {
            'effectiveness': effectiveness,
            'adaptivity': adaptivity,
            'diversity': diversity,
            'preference': preference,
            'final_quality': final_quality
        }

    def _compute_effectiveness(self, recommended_path, knowledge_before, knowledge_after):
        """
        计算有效性（适配自Metrics.compute_effectiveness）
        针对单步推荐路径
        """
        total_gain = 0.0
        valid_count = 0

        for i, rec in enumerate(recommended_path):
            # 确保有前后知识状态
            if i < len(knowledge_before) and i < len(knowledge_after):
                before_state = knowledge_before[i]  # 推荐前的知识状态
                after_state = knowledge_after[i]  # 推荐后的知识状态

                # 确保是张量
                if isinstance(before_state, torch.Tensor):
                    before_state = before_state.cpu().numpy()
                if isinstance(after_state, torch.Tensor):
                    after_state = after_state.cpu().numpy()

                # 计算该资源的知识增益
                if rec < len(before_state) and rec < len(after_state):
                    pb = before_state[rec]
                    pa = after_state[rec]

                    # 使用Metrics中的增益公式
                    if pb < 0.9 and pa > 0:
                        gain = (pa - pb) / (1.0 - pb + 1e-8)
                        total_gain += gain
                        valid_count += 1

        return total_gain / valid_count if valid_count > 0 else 0.0

    def _compute_adaptivity(self, full_sequence, full_answers, original_len):
        """
        计算适应性（适配自Metrics.calculate_adaptivity）
        基于完整序列和历史答题记录
        """
        total_adaptivity = 0.0
        valid_count = 0

        # 对推荐部分的每个时间步计算适应性
        for t in range(original_len, len(full_sequence)):
            # 计算当前位置之前的答题历史（用于计算能力值）
            current_pos = t

            # 获取最近10个历史记录
            T = 10
            start_idx = max(0, current_pos - T)

            history_diffs = []
            history_results = []

            for j in range(start_idx, current_pos):
                exercise_id = full_sequence[j]
                if exercise_id > 1:  # 有效资源
                    difficulty = self._get_difficulty(exercise_id)
                    history_diffs.append(difficulty)
                    history_results.append(full_answers[j])

            # 计算能力值
            if history_diffs and sum(history_results) > 0:
                numerator = sum(d * r for d, r in zip(history_diffs, history_results))
                denominator = sum(history_results) + 1e-5
                ability = numerator / denominator
            else:
                ability = 0.5  # 默认能力值

            # 计算推荐资源的适应性
            rec_difficulty = self._get_difficulty(full_sequence[t])
            adaptivity = 1.0 - abs(ability - rec_difficulty)
            total_adaptivity += max(0, adaptivity)
            valid_count += 1

        return total_adaptivity / valid_count if valid_count > 0 else 0.0

    def _compute_diversity(self, recommended_path, hidden_embeddings):
        """
        计算多样性（适配自Metrics.calculate_diversity）
        基于资源嵌入向量的余弦相似度
        """
        if not recommended_path or len(recommended_path) < 2:
            return 0.0

        if hidden_embeddings is None:
            return 0.5  # 默认多样性分数

        # 获取每个推荐资源的嵌入
        rec_embeddings = []
        for rec in recommended_path:
            if rec < hidden_embeddings.size(0):
                emb = hidden_embeddings[rec]
                if isinstance(emb, torch.Tensor):
                    emb = emb.detach().cpu()
                rec_embeddings.append(emb)

        if len(rec_embeddings) < 2:
            return 0.0

        # 计算所有资源对之间的相似度
        total_similarity = 0.0
        pair_count = 0

        for i in range(len(rec_embeddings)):
            for j in range(i + 1, len(rec_embeddings)):
                # 计算余弦相似度
                emb_i = rec_embeddings[i].unsqueeze(0) if rec_embeddings[i].dim() == 1 else rec_embeddings[i]
                emb_j = rec_embeddings[j].unsqueeze(0) if rec_embeddings[j].dim() == 1 else rec_embeddings[j]

                # 归一化
                emb_i_norm = emb_i / (torch.norm(emb_i) + 1e-8)
                emb_j_norm = emb_j / (torch.norm(emb_j) + 1e-8)

                sim = torch.dot(emb_i_norm.flatten(), emb_j_norm.flatten()).item()
                total_similarity += (1 - sim)  # 多样性 = 1 - 相似度
                pair_count += 1

        return total_similarity / pair_count if pair_count > 0 else 0.0

    def _compute_preference(self, recommended_path, model_predictions):
        """
        计算偏好性（基于模型预测概率）
        """
        if not recommended_path:
            return 0.0

        total_preference = 0.0
        valid_count = 0

        for i, rec in enumerate(recommended_path):
            if i < len(model_predictions):
                # 获取模型对该资源的预测概率
                if rec < len(model_predictions[i]):
                    prob = model_predictions[i][rec]
                    total_preference += prob
                    valid_count += 1

        return total_preference / valid_count if valid_count > 0 else 0.0


class LearningPathEnv:
    """
    学习路径环境（合并增强功能）
    关键特性：
    - 维护每个学习者的完整历史路径
    - 每一步都使用完整历史调用预训练模型
    - 计算多目标奖励
    - 支持轨迹数据管理和最终质量奖励计算
    """

    def __init__(self, batch_size, base_model, policy_net, recommendation_length, topk, data_name,
                 graph=None, hypergraph_list=None):
        self.base_model = base_model  # 预训练模型
        self.policy_net = policy_net  # 策略网络
        self.recommendation_length = recommendation_length  # 推荐路径长度
        self.topk = topk  # 候选集大小
        self.data_name = data_name  # 数据集名称

        # 存储默认图数据
        self.default_graph = graph
        self.default_hypergraph_list = hypergraph_list

        # 环境状态
        self.batch_size = batch_size  # 批次大小（动态设置）
        self.histories = []  # 每个学习者的历史路径列表
        self.current_step = 0  # 当前步骤
        self.paths = []  # 已生成的推荐路径

        # 存储原始输入数据，以便在step中使用
        self.original_tgt = None  # [batch_size, history_len] - 目标序列
        self.original_tgt_timestamp = None  # [batch_size, history_len] - 时间戳序列
        self.original_tgt_idx = None  # [batch_size, history_len] - 索引序列
        self.original_ans = None  # [batch_size, history_len] - 答案序列
        self.original_graph = None
        self.original_hypergraph_list = None
        self.current_state = None
        self.original_sequences = None  # 存储原始序列用于奖励计算
        self.all_knowledge_states = []  # 存储每一步的知识状态，用于最终奖励计算
        self.all_predictions = []  # 存储每一步的预测结果，用于最终奖励计算

        # 存储扩展的答案序列
        self.extended_ans = None  # [batch_size, history_len + current_step] - 扩展的答案序列

        # 新增：奖励计算相关
        self.reward_weights = {
            'validity': 0.4,
            'adaptivity': 0.3,
            'diversity': 0.2,
            'preference': 0.1
        }
        self.prev_knowledge_state = None  # 上一步的知识状态

        # 用于存储最终质量奖励
        self.final_quality_scores = []
        self.trajectory_rewards = []  # 存储轨迹的即时奖励

        # 轨迹数据管理器和评估器
        self.trajectory_manager = TrajectoryDataManager(batch_size)
        self.metrics_evaluator = None  # 将在reset时初始化

        # 奖励混合参数
        self.reward_decay_factor = 0.8  # 时间衰减因子
        self.final_reward_weight = 2.0  # 最终奖励权重

        # 加载真实难度数据（用于适应性计算）
        self._load_difficulty_data(data_name)

    def _load_difficulty_data(self, data_name):
        """加载真实难度数据"""
        options = Options(data_name)
        try:
            with open(options.idx2u_dict, 'rb') as handle:
                self.idx2u = pickle.load(handle)

            # 加载难度数据
            self.difficulty_data = {}
            with open(options.difficult_file, 'r') as f:
                next(f)  # 跳过标题行
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        try:
                            challenge_id = int(parts[0].strip())
                            difficulty = int(parts[1].strip())
                            self.difficulty_data[challenge_id] = difficulty
                        except ValueError:
                            continue
            print(f"已加载难度数据，共 {len(self.difficulty_data)} 个习题的难度")
        except Exception as e:
            print(f"加载难度数据失败: {e}")
            self.difficulty_data = {}
            self.idx2u = {}

    def _load_metrics_evaluator(self):
        """加载指标评估器所需的难度数据"""
        # 加载难度数据和映射（重用原有逻辑）
        options = Options(self.data_name)
        try:
            with open(options.idx2u_dict, 'rb') as handle:
                idx2u = pickle.load(handle)

            difficulty_data = {}
            with open(options.difficult_file, 'r') as f:
                next(f)  # 跳过标题行
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        try:
                            challenge_id = int(parts[0].strip())
                            difficulty = int(parts[1].strip())
                            difficulty_data[challenge_id] = difficulty
                        except ValueError:
                            continue

            # 创建评估器
            self.metrics_evaluator = TrajectoryMetricsEvaluator(
                difficulty_data=difficulty_data,
                idx2u=idx2u,
                reward_weights=self.reward_weights
            )

            print(f"指标评估器初始化完成，加载了 {len(difficulty_data)} 个习题的难度数据")

        except Exception as e:
            print(f"加载难度数据失败，使用默认评估器: {e}")
            self.metrics_evaluator = TrajectoryMetricsEvaluator(
                difficulty_data={},
                idx2u={},
                reward_weights=self.reward_weights
            )

    def reset(self, tgt, tgt_timestamp, tgt_idx, ans, graph=None, hypergraph_list=None):
        """
        重置环境
        Args:
            tgt: [batch_size, history_len] 初始历史路径
            tgt_timestamp: [batch_size, history_len] 时间戳
            tgt_idx: [batch_size, history_len] 索引
            ans: [batch_size, history_len] 答案
            graph: 关系图
            hypergraph_list: 超图列表
        """
        self.original_tgt = tgt  # [batch_size, history_len]
        self.original_ans = ans  # [batch_size, history_len]

        # 使用传入的图数据或默认图数据
        if graph is not None:
            self.original_graph = graph
        elif self.default_graph is not None:
            self.original_graph = self.default_graph
        else:
            raise ValueError("Graph must be provided either in init or reset")

        if hypergraph_list is not None:
            self.original_hypergraph_list = hypergraph_list
        elif self.default_hypergraph_list is not None:
            self.original_hypergraph_list = self.default_hypergraph_list
        else:
            raise ValueError("Hypergraph list must be provided either in init or reset")

        # 初始化历史路径
        self.batch_size = tgt.size(0)  # [batch_size, ...]
        self.histories = [tgt[i].clone() for i in range(self.batch_size)]  # List of [history_len] for each batch
        self.current_step = 0
        self.paths = [[] for _ in range(self.batch_size)]  # List of lists for each batch 独立维护batch每个学习者的推荐路径
        self.all_knowledge_states = []  # 重置每一步知识状态记录，step推荐后更新 List[torch.Tensor] [batch_size, num_skills]
        self.all_predictions = []  # 重置每一步预测结果记录，step推荐后更新 List[torch.Tensor] [batch_size, total_seq_len, num_skills]

        # 初始化扩展答案序列，复制原始答案序列
        self.extended_ans = ans.clone()  # [batch_size, history_len] - 扩展的答案序列，随着步骤的推荐，扩展答案序列的长度会增加

        # 获取初始状态
        initial_state = self._get_current_state(tgt, tgt_timestamp, tgt_idx, ans)
        self.current_state = initial_state

        # 新增：初始化 prev_knowledge_state
        self.prev_knowledge_state = initial_state['knowledge_state'].clone()

        # 重置轨迹管理器
        self.trajectory_manager.reset()

        # 存储原始数据
        for i in range(self.batch_size):
            self.trajectory_manager.add_original_data(
                i,
                tgt[i].cpu().tolist(),
                ans[i].cpu().tolist()
            )

        # 初始化评估器（加载难度数据）
        if self.metrics_evaluator is None:
            self._load_metrics_evaluator()

        # 获取并存储资源嵌入
        with torch.no_grad():
            hidden = self.base_model.gnn(self.original_graph)
            self.trajectory_manager.set_hidden_embeddings(hidden)

        return initial_state

    def _get_current_state(self, tgt, tgt_timestamp, tgt_idx, ans):
        """
        获取当前状态
        """
        # 使用当前累积的历史路径作为输入
        batch_histories = torch.stack(self.histories)  # [batch_size, current_seq_len] - 当前累积的历史路径

        # 确保ans与当前历史长度一致
        max_hist_len = batch_histories.size(1)  # 当前历史的最大长度 - scalar

        # 调整其他输入以匹配当前历史长度，确保张量是二维的
        if ans.dim() == 2:
            current_ans = ans[:, :max_hist_len]  # [batch_size, current_seq_len]
        else:
            # 如果是一维张量，扩展为二维
            current_ans = ans.unsqueeze(0)[:, :max_hist_len]  # [1, current_seq_len]

        # 调用预训练模型
        with torch.no_grad():
            # pred: [batch_size, total_seq_len, num_skills] - 预测推荐概率
            # pred_res: [batch_size, total_seq_len] - 预测结果
            # kt_mask: [batch_size, total_seq_len] - KT掩码
            # knowledge_state: [batch_size, total_seq_len, num_skills] - 知识状态
            # hidden: [num_skills, hidden_dim] - 知识初始嵌入
            # status_emb: [batch_size, total_seq_len, hidden_dim] - 状态嵌入
            pred, pred_res, kt_mask, knowledge_state, hidden, status_emb = self.base_model(
                batch_histories, tgt_timestamp, tgt_idx, current_ans,
                self.original_graph, self.original_hypergraph_list)

            # 获取当前步骤的topk候选（只取最后一个时间步的topk）
            # 获取最后时间步的预测概率
            last_step_pred = pred.view(batch_histories.size(0), -1, pred.size(-1))[:, -1, :]  # [batch_size, num_skills]

            # 获取topk候选
            topk_scores, topk_indices = torch.topk(last_step_pred, k=self.topk, dim=-1)  # [batch_size, topk]
            topk_candidates = topk_indices  # [batch_size, topk]

        # 提取当前知识状态（最后一个时间步）
        if knowledge_state.dim() == 3:
            current_knowledge = knowledge_state[:, -1, :]  # [batch_size, num_skills]
        else:
            current_knowledge = knowledge_state  # [num_skills,] or [batch_size, num_skills]

        # 提取候选特征
        candidate_features = self._extract_candidate_features(topk_candidates,
                                                              current_knowledge)  # [batch_size, topk, feature_dim]

        # 构建状态字典
        state = {
            'knowledge_state': current_knowledge,  # [batch_size, num_skills] - 当前知识状态
            'candidates': topk_candidates,  # [batch_size, topk] - TopK候选
            'candidate_features': candidate_features,  # [batch_size, topk, feature_dim] - 候选特征
            'histories': self.histories,  # List of [history_len] for each batch - 原始历史路径
            'step': self.current_step,  # scalar - 当前步骤
            'pred': pred,  # [batch_size, total_seq_len, num_skills] - 预测概率
            'pred_res': pred_res,  # [batch_size, total_seq_len] - 下一题答题预测结果
            'hidden': hidden,  # [skills_num, hidden_dim] - 隐藏层
            'knowledge_state_full': knowledge_state  # [batch_size, total_seq_len, num_skills] - 完整知识状态
        }

        return state

    def _extract_candidate_features(self, topk_candidates, knowledge_state):
        """
        提取候选习题的特征
        Args:
            topk_candidates: [batch_size, topk] - TopK候选习题ID
            knowledge_state: [batch_size, num_skills] - 当前知识状态
        Returns:
            candidate_features: [batch_size, topk, feature_dim] - 候选特征
        """
        # 确保topk_candidates是tensor
        if not isinstance(topk_candidates, torch.Tensor):
            topk_candidates = torch.tensor(topk_candidates)

        # 验证topk_candidates形状
        if topk_candidates.dim() != 2:
            raise ValueError(
                f"topk_candidates should be 2D tensor [batch_size, topk], got shape {topk_candidates.shape}")

        batch_size, topk = topk_candidates.shape  # scalar, scalar

        # 确保所有张量都在同一设备上
        device = knowledge_state.device  # 使用knowledge_state的设备作为目标设备
        topk_candidates = topk_candidates.to(device)

        # 创建候选特征: [batch_size, topk, 5]
        candidate_features = torch.zeros(batch_size, topk, 5, device=device, dtype=torch.float)

        # 特征1: 习题ID归一化 - [batch_size, topk]
        candidate_features[:, :, 0] = topk_candidates.float() / self.base_model.n_node

        # 特征2: 对应知识点的当前掌握程度 - [batch_size, topk]
        for b in range(batch_size):
            for k in range(topk):
                skill_id = topk_candidates[b, k].item()
                if skill_id < knowledge_state.size(1):  # num_skills
                    candidate_features[b, k, 1] = knowledge_state[b, skill_id]

        # 特征3: 习题在topk中的排名 - [batch_size, topk]
        rank_weights = torch.linspace(1.0, 0.0, topk, device=device)
        candidate_features[:, :, 2] = rank_weights.unsqueeze(0).repeat(batch_size, 1)

        # 特征4: 与当前知识状态的差异度 - [batch_size, topk]
        for b in range(batch_size):
            for k in range(topk):
                skill_id = topk_candidates[b, k].item()
                if skill_id < knowledge_state.size(1):
                    diff = abs(0.5 - knowledge_state[b, skill_id])
                    candidate_features[b, k, 3] = diff

        # 特征5: 偏好分数 - [batch_size, topk]
        candidate_features[:, :, 4] = 1.0 - rank_weights.unsqueeze(0).repeat(batch_size, 1)

        return candidate_features

    def step(self, actions):
        """
        执行一步动作，记录轨迹数据
        Args:
            actions: [batch_size] 选择的候选索引（0到topk-1）
        Returns:
            next_state: 下一个状态
            rewards: [batch_size] 即时奖励
            dones: [batch_size] 是否终止
        """
        # 1. 执行动作 获取选择的习题 - [batch_size]
        selected_exercises = []
        for i in range(self.batch_size):
            candidate_idx = actions[i].item()
            exercise_id = self.current_state['candidates'][i, candidate_idx].item()
            selected_exercises.append(exercise_id)

        selected_exercises = torch.tensor(selected_exercises, dtype=torch.long)  # [batch_size]

        # 2. 更新历史路径
        for i in range(self.batch_size):
            # 将选择的习题添加到历史
            new_exercise = selected_exercises[i].unsqueeze(0)  # [1] - 单个习题
            # 确保new_exercise与histories[i]在同一设备上
            new_exercise = new_exercise.to(self.histories[i].device)
            self.histories[i] = torch.cat([self.histories[i], new_exercise])  # [current_history_len + 1]
            # 记录到推荐路径
            self.paths[i].append(selected_exercises[i].item())

        # 3. 扩展答案序列，为新添加的习题添加预测答案
        # 这里使用一个简单的启发式方法：根据当前知识状态预测答案
        # 获取当前知识状态对新习题的预测概率
        current_knowledge = self.current_state['knowledge_state']  # [batch_size, num_skills]

        # 为每个选择的习题获取预测正确率（这里使用当前知识状态作为预测）
        predicted_correct = []
        for i in range(self.batch_size):
            selected_exercise_id = selected_exercises[i].item()
            if selected_exercise_id < current_knowledge.size(1):
                # 使用当前知识状态作为预测正确率
                pred_prob = current_knowledge[i, selected_exercise_id].item()
                # 转换为0或1（正确或错误）
                predicted_answer = 1 if pred_prob > 0.5 else 0
            else:
                # 如果习题ID超出知识点范围，假设答错
                predicted_answer = 0
            predicted_correct.append(predicted_answer)

        # 创建新的答案张量 - [batch_size, 1]
        new_answers = torch.tensor(predicted_correct, dtype=self.extended_ans.dtype,
                                   device=self.extended_ans.device).unsqueeze(1)
        # 扩展答案序列 - [batch_size, current_seq_len + 1]
        self.extended_ans = torch.cat([self.extended_ans, new_answers], dim=1)

        # 4. 记录轨迹数据
        self._record_trajectory_data(actions, selected_exercises)

        # 5. 更新步骤计数
        self.current_step += 1

        # 6. 获取新状态 - 重新运行预训练模型以获取完整的状态信息
        # 使用当前累积的历史来获取状态，并使用扩展后的答案
        next_state = self._get_current_state(
            torch.stack(self.histories),  # [batch_size, current_seq_len + 1] - 当前累积的历史
            self.original_tgt_timestamp,  # [batch_size, original_seq_len] - 原始时间戳
            self.original_tgt_idx,  # [batch_size, original_seq_len] - 原始索引
            self.extended_ans  # [batch_size, current_seq_len + 1] - 扩展后的答案
        )

        # 7. 计算即时奖励 - [batch_size]
        immediate_rewards = self._calculate_immediate_reward(selected_exercises, next_state)

        # 8. 检查是否轨迹结束 - [batch_size]
        dones = torch.tensor([self.current_step >= self.recommendation_length] * self.batch_size,
                             device=selected_exercises.device)

        # 9. 如果是轨迹结束，计算最终质量奖励并混合
        if torch.any(dones):
            # 计算最终质量奖励
            final_quality_bonus = self._calculate_final_quality_bonus()

            # 混合奖励：即时奖励 + 最终质量奖励（反向分配）
            mixed_rewards = self._mix_rewards(immediate_rewards, final_quality_bonus, dones)

            # 存储最终质量分数用于分析
            self.final_quality_scores = final_quality_bonus
        else:
            mixed_rewards = immediate_rewards

        # 10. 保存当前知识状态供下一步使用
        self.prev_knowledge_state = self.current_state['knowledge_state'].clone()

        return next_state, mixed_rewards, dones

    def _record_trajectory_data(self, actions, selected_exercises):
        """记录当前步骤的轨迹数据"""
        # 获取当前知识状态
        knowledge_before = self.current_state['knowledge_state']

        # 预测答案（基于推荐前的知识状态）
        predicted_answers = self._predict_answers_batch(
            knowledge_before, selected_exercises)

        # 获取模型预测概率
        model_predictions = self._get_model_predictions_batch(
            self.current_state, selected_exercises)

        # 临时保存，下一步会更新知识状态
        # 注意：knowledge_after 将在下一步的 _get_current_state 中获取

        # 记录数据
        for i in range(self.batch_size):
            # 注意：knowledge_after 需要在下一步才能获取
            # 这里先记录知识状态前和预测数据，知识状态后将在下一步补全
            self.trajectory_manager.add_recommendation(
                i,
                selected_exercises[i].item(),
                knowledge_before[i].cpu().detach(),
                None,  # 将在下一步补全
                predicted_answers[i],
                model_predictions[i]
            )

    def _predict_answers_batch(self, knowledge_state, selected_exercises):
        """批量预测答案（基于当前知识状态）"""
        predicted_answers = []
        for i, rec in enumerate(selected_exercises):
            if rec < knowledge_state.size(1):
                prob = knowledge_state[i, rec].item()
                # 使用0.5作为阈值
                answer = 1 if prob > 0.5 else 0
            else:
                answer = 0  # 默认答错
            predicted_answers.append(answer)
        return predicted_answers

    def _get_model_predictions_batch(self, state, selected_exercises):
        """获取模型对推荐资源的预测概率"""
        batch_size = len(selected_exercises)
        model_predictions = []

        if 'pred' in state:
            pred_probs = state['pred']  # [batch, seq_len, num_skills]

            for i in range(batch_size):
                # 获取最后一个时间步的预测概率
                last_step_pred = pred_probs[i, -1, :] if pred_probs.dim() == 3 else pred_probs[i]

                # 转换为概率分布
                if isinstance(last_step_pred, torch.Tensor):
                    probs = torch.softmax(last_step_pred, dim=0).cpu().tolist()
                else:
                    probs = last_step_pred.tolist()

                model_predictions.append(probs)
        else:
            # 如果没有预测数据，使用均匀分布
            num_skills = self.base_model.n_node
            uniform_probs = [1.0 / num_skills] * num_skills
            model_predictions = [uniform_probs] * batch_size

        return model_predictions

    def _calculate_immediate_reward(self, selected_exercises, next_state):
        """
        修正的即时奖励计算，与Metrics保持一致
        返回: [batch_size] 即时奖励
        """
        batch_size = selected_exercises.size(0)
        device = selected_exercises.device

        # 初始化各分项奖励
        validity_reward = torch.zeros(batch_size, device=device)
        adaptivity_reward = torch.zeros(batch_size, device=device)
        diversity_reward = torch.zeros(batch_size, device=device)
        preference_reward = torch.zeros(batch_size, device=device)

        # 1. 有效性奖励（基于知识状态变化）
        # 检查 prev_knowledge_state 是否存在且不为 None
        if hasattr(self, 'prev_knowledge_state') and self.prev_knowledge_state is not None:
            prev_knowledge = self.prev_knowledge_state  # [batch, num_skills]
            next_knowledge = next_state['knowledge_state']  # [batch, num_skills]

            # 确保 prev_knowledge 是张量且有正确维度
            if isinstance(prev_knowledge, torch.Tensor) and prev_knowledge.dim() == 2:
                for i in range(batch_size):
                    skill_id = selected_exercises[i].item()
                    if skill_id < prev_knowledge.size(1):
                        pb = prev_knowledge[i, skill_id]
                        pa = next_knowledge[i, skill_id]

                        # 使用Metrics中的有效性计算公式
                        if pb < 0.9 and pa > 0:
                            gain = (pa - pb) / (1.0 - pb + 1e-8)
                            validity_reward[i] = self.reward_weights['validity'] * gain

        # 2. 适应性奖励（使用真实难度数据）
        adaptivity_reward = self._calculate_adaptivity_reward(
            selected_exercises,
            self.current_state['knowledge_state']
        )

        # 3. 多样性奖励（基于近期路径历史）
        diversity_reward = self._calculate_diversity_reward(selected_exercises)

        # 4. 偏好保持奖励（基于原模型预测概率）
        preference_reward = self._calculate_preference_reward(selected_exercises)

        # 组合即时奖励
        immediate_reward = (validity_reward + adaptivity_reward +
                            diversity_reward + preference_reward)

        # 存储用于最终质量评估
        if not hasattr(self, 'immediate_rewards_history'):
            self.immediate_rewards_history = []
        self.immediate_rewards_history.append(immediate_reward.detach().cpu())

        return immediate_reward

    def _calculate_adaptivity_reward(self, selected_exercises, knowledge_state):
        """
        使用真实难度数据计算适应性奖励
        """
        batch_size = selected_exercises.size(0)
        adaptivity_reward = torch.zeros(batch_size, device=selected_exercises.device)

        # 如果没有加载难度数据，返回0
        if not hasattr(self, 'difficulty_data') or not self.difficulty_data:
            return adaptivity_reward

        for i in range(batch_size):
            skill_id = selected_exercises[i].item()

            # 获取习题真实难度
            if skill_id in self.idx2u and self.idx2u[skill_id] in self.difficulty_data:
                original_id = self.idx2u[skill_id]
                difficulty = self.difficulty_data[original_id] / 5.0  # 归一化到0-1

                # 获取当前知识点掌握水平
                if skill_id < knowledge_state.size(1):
                    knowledge_level = knowledge_state[i, skill_id].item()

                    # 理想难度：略高于当前水平
                    optimal_difficulty = min(1.0, knowledge_level + 0.2)

                    # 计算适应性：1 - |实际难度 - 理想难度|
                    adaptivity = 1.0 - abs(difficulty - optimal_difficulty)
                    adaptivity_reward[i] = self.reward_weights['adaptivity'] * max(0, adaptivity)

        return adaptivity_reward

    def _calculate_diversity_reward(self, selected_exercises):
        """
        计算多样性奖励（避免重复推荐）
        """
        batch_size = selected_exercises.size(0)
        diversity_reward = torch.zeros(batch_size, device=selected_exercises.device)

        for i in range(batch_size):
            current_exercise = selected_exercises[i].item()

            # 检查当前路径中的重复情况
            if len(self.paths[i]) > 0:
                # 获取最近推荐的习题（排除当前）
                recent_exercises = self.paths[i][-5:-1] if len(self.paths[i]) >= 5 else self.paths[i][:-1]

                if current_exercise in recent_exercises:
                    # 重复推荐，惩罚
                    repeat_count = recent_exercises.count(current_exercise)
                    penalty = 0.1 * repeat_count  # 重复次数越多，惩罚越大
                    diversity_reward[i] = -penalty
                else:
                    # 新推荐，奖励
                    diversity_reward[i] = self.reward_weights['diversity'] * 0.5
            else:
                # 第一个推荐，给予基础奖励
                diversity_reward[i] = self.reward_weights['diversity'] * 0.3

        return diversity_reward

    def _calculate_preference_reward(self, selected_exercises):
        """
        计算偏好保持奖励（基于原模型预测概率）
        """
        batch_size = selected_exercises.size(0)
        preference_reward = torch.zeros(batch_size, device=selected_exercises.device)

        for i in range(batch_size):
            skill_id = selected_exercises[i].item()

            # 从当前状态获取原模型预测概率
            if 'pred' in self.current_state:
                pred_probs = self.current_state['pred']  # [batch, seq_len, num_skills]

                # 获取最后一个时间步的预测概率
                last_step_pred = pred_probs[i, -1, :] if pred_probs.dim() == 3 else pred_probs[i]

                if skill_id < last_step_pred.size(0):
                    preference_score = last_step_pred[skill_id].item()
                    preference_reward[i] = self.reward_weights['preference'] * preference_score

        return preference_reward

    def _calculate_final_quality_bonus(self):
        """计算最终质量奖励"""
        if self.metrics_evaluator is None:
            return torch.zeros(self.batch_size, device=self.original_tgt.device)

        final_qualities = []

        for i in range(self.batch_size):
            # 获取轨迹数据
            trajectory_data = self.trajectory_manager.get_trajectory_data(i)

            # 补全知识状态后
            # 注意：这里需要从all_knowledge_states中获取完整的知识状态
            # 简化处理：使用当前状态作为最终知识状态
            if len(self.all_knowledge_states) > 0:
                knowledge_after = self.all_knowledge_states[-1][i].cpu().detach()
                trajectory_data['knowledge_after'] = [knowledge_after] * len(trajectory_data['recommended_path'])

            # 评估轨迹
            metrics = self.metrics_evaluator.evaluate_trajectory(trajectory_data)

            # 使用最终质量分数
            final_qualities.append(metrics['final_quality'])

        return torch.tensor(final_qualities, device=self.original_tgt.device)

    def _mix_rewards(self, immediate_rewards, final_quality_bonus, dones):
        """
        混合即时奖励和最终质量奖励
        采用时间衰减的方式分配最终奖励
        """
        batch_size = immediate_rewards.size(0)
        mixed_rewards = immediate_rewards.clone()

        for i in range(batch_size):
            if dones[i].item():
                # 获取轨迹长度
                traj_len = len(self.trajectory_manager.recommended_paths[i])

                if traj_len > 0:
                    # 计算时间衰减权重
                    weights = torch.zeros(traj_len)
                    for t in range(traj_len):
                        # 越接近结束的步骤权重越高
                        weight = self.reward_decay_factor ** (traj_len - t - 1)
                        weights[t] = weight

                    # 归一化权重
                    weights = weights / weights.sum()

                    # 分配最终奖励（加强最终奖励的影响）
                    bonus_per_step = final_quality_bonus[i] * self.final_reward_weight * weights

                    # 由于immediate_rewards是当前步骤的奖励，我们需要将最终奖励
                    # 分配到历史步骤。这里我们简化处理，将总奖励加到当前步骤
                    total_bonus = bonus_per_step.sum()
                    mixed_rewards[i] += total_bonus

                    # 记录分配详情（用于调试）
                    if not hasattr(self, 'reward_allocation_history'):
                        self.reward_allocation_history = []
                    self.reward_allocation_history.append({
                        'batch_idx': i,
                        'traj_len': traj_len,
                        'final_quality': final_quality_bonus[i].item(),
                        'total_bonus': total_bonus.item(),
                        'weights': weights.tolist()
                    })

        return mixed_rewards

    def _calculate_final_quality_bonus_simple(self):
        """
        计算最终路径质量奖励（简化版）
        使用Metrics模块评估整条推荐路径
        """
        batch_size = self.batch_size

        # 1. 获取完整序列
        full_sequences = []
        original_seqs = self.original_tgt.cpu().numpy() if self.original_tgt is not None else []

        for i in range(batch_size):
            # 原始历史 + 推荐路径
            original_len = len(original_seqs[i]) if i < len(original_seqs) else 0
            full_seq = list(original_seqs[i]) + self.paths[i] if original_len > 0 else self.paths[i]
            full_sequences.append(full_seq)

        # 2. 使用预训练模型获取知识状态（用于有效性计算）
        # 这里需要重新运行模型获取完整的知识状态序列
        with torch.no_grad():
            batch_histories = torch.stack(self.histories)

            # 运行模型获取预测和知识状态
            pred, pred_res, kt_mask, knowledge_state, hidden, status_emb = self.base_model(
                batch_histories,
                self.original_tgt_timestamp,
                self.original_tgt_idx,
                self.extended_ans,
                self.original_graph,
                self.original_hypergraph_list
            )
            self.all_knowledge_states.append(knowledge_state)

        # 3. 转换为numpy用于Metrics计算
        pred_np = pred.cpu().numpy()
        knowledge_state_np = knowledge_state.cpu().numpy()

        # 4. 计算各项指标（简化版，实际应调用Metrics中的方法）
        quality_scores = torch.zeros(batch_size, device=batch_histories.device)

        for i in range(batch_size):
            # 有效性：知识状态提升
            if knowledge_state_np.shape[1] > 1:
                # 计算平均知识增益
                initial_knowledge = knowledge_state_np[i, len(original_seqs[i]) - 1, :]
                final_knowledge = knowledge_state_np[i, -1, :]
                knowledge_gain = np.mean(np.maximum(final_knowledge - initial_knowledge, 0))
                validity_score = knowledge_gain / (1.0 - initial_knowledge.mean() + 1e-8)
            else:
                validity_score = 0.0

            # 多样性：路径中独特知识点比例
            unique_exercises = len(set(self.paths[i]))
            total_exercises = len(self.paths[i])
            diversity_score = unique_exercises / total_exercises if total_exercises > 0 else 0.0

            # 适应性：平均难度匹配度（简化计算）
            adaptivity_score = 0.0
            if hasattr(self, 'difficulty_data') and self.difficulty_data:
                for exercise in self.paths[i]:
                    if exercise in self.idx2u and self.idx2u[exercise] in self.difficulty_data:
                        # 这里需要更精确的适应性计算，简化处理
                        adaptivity_score += 0.5  # 简化：假设中等适应性
                adaptivity_score = adaptivity_score / total_exercises if total_exercises > 0 else 0.0

            # 组合最终质量分数
            final_quality = (
                    self.reward_weights['validity'] * validity_score +
                    self.reward_weights['diversity'] * diversity_score +
                    self.reward_weights['adaptivity'] * adaptivity_score
            )

            quality_scores[i] = final_quality

        return quality_scores


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
            knowledge_state: [batch_size, knowledge_dim] - 知识状态
            candidate_features: [batch_size, topk, candidate_feature_dim] - 候选特征
        Returns:
            scores: [batch_size, topk] - 每个候选的得分
        """
        # 验证输入张量维度
        assert knowledge_state.dim() == 2, f"knowledge_state should be 2D, got {knowledge_state.dim()}D: {knowledge_state.shape}"
        assert candidate_features.dim() == 3, f"candidate_features should be 3D, got {candidate_features.dim()}D: {candidate_features.shape}"

        batch_size = knowledge_state.size(0)  # scalar

        # 编码知识状态 - [batch_size, hidden_dim]
        knowledge_encoded = self.knowledge_encoder(knowledge_state)

        # 验证编码后的张量维度
        assert knowledge_encoded.dim() == 2, f"knowledge_encoded should be 2D, got {knowledge_encoded.dim()}D: {knowledge_encoded.shape}"

        # 确保知识状态编码后的维度是正确的hidden_dim
        # 扩展知识编码以匹配候选数量 - [batch_size, topk, hidden_dim]
        knowledge_encoded_expanded = knowledge_encoded.unsqueeze(1).expand(-1, candidate_features.size(1), -1)

        # 验证扩展后的张量维度
        assert knowledge_encoded_expanded.dim() == 3, f"knowledge_encoded_expanded should be 3D, got {knowledge_encoded_expanded.dim()}D: {knowledge_encoded_expanded.shape}"

        # 编码候选特征 - [batch_size, topk, hidden_dim]
        candidate_encoded = self.candidate_encoder(candidate_features)

        # 确保张量维度正确，用于注意力机制
        # 验证所有张量的最后一个维度（嵌入维度）是否一致
        expected_dim = self.attention.embed_dim  # 应该是128

        # 确保张量维度正确
        if candidate_encoded.size(-1) != expected_dim:
            # 如果最后一个维度不匹配，进行线性变换或切片
            if candidate_encoded.size(-1) > expected_dim:
                # 如果维度太大，切片或投影到正确维度
                candidate_encoded = candidate_encoded[..., :expected_dim]
            else:
                # 如果维度太小，需要扩展，但这种情况不太可能
                raise ValueError(
                    f"Candidate encoded dimension {candidate_encoded.size(-1)} is less than expected {expected_dim}")

        if knowledge_encoded_expanded.size(-1) != expected_dim:
            if knowledge_encoded_expanded.size(-1) > expected_dim:
                knowledge_encoded_expanded = knowledge_encoded_expanded[..., :expected_dim]
            else:
                raise ValueError(
                    f"Knowledge encoded expanded dimension {knowledge_encoded_expanded.size(-1)} is less than expected {expected_dim}")

        # 注意力机制
        # query: candidate_encoded [batch_size, topk, hidden_dim]
        # key: knowledge_encoded_expanded [batch_size, topk, hidden_dim]
        # value: knowledge_encoded_expanded [batch_size, topk, hidden_dim]
        attended, _ = self.attention(
            candidate_encoded,  # query - [batch_size, topk, hidden_dim]
            knowledge_encoded_expanded,  # key - [batch_size, topk, hidden_dim]
            knowledge_encoded_expanded  # value - [batch_size, topk, hidden_dim]
        )

        # 拼接原始候选编码和注意力输出 - [batch_size, topk, hidden_dim*2]
        combined = torch.cat([candidate_encoded, attended], dim=-1)

        # 计算得分 - [batch_size, topk, 1] -> [batch_size, topk]
        scores = self.score_network(combined).squeeze(-1)

        return scores


class PPOTrainer:
    """
    PPO训练器（合并增强功能）
    使用近端策略优化算法训练策略网络
    支持轨迹分析和奖励混合
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
        收集一条轨迹（一个批次），支持最终质量奖励
        Args:
            initial_histories: [batch_size, history_len] - 初始历史路径
            tgt_timestamp: [batch_size, history_len] - 时间戳
            tgt_idx: [batch_size, history_len] - 索引
            ans: [batch_size, history_len] - 答案
            graph: 关系图
            hypergraph_list: 超图列表
        Returns:
            trajectory: dict - 轨迹数据
        """
        # 重置环境
        state = self.env.reset(initial_histories, tgt_timestamp, tgt_idx, ans, graph, hypergraph_list)

        states = []  # List of state dicts - 每步的状态
        actions = []  # List of [batch_size] - 每步的动作
        rewards = []  # List of [batch_size] - 每步的奖励 混合奖励
        log_probs = []  # List of [batch_size] - 每步的动作对数概率
        values = []  # List of [batch_size] - 每步的状态价值
        dones = []  # List of [batch_size] - 每步的终止标志

        # 运行完整轨迹
        step_count = 0
        max_steps = self.env.recommendation_length  # 防止无限循环

        while step_count < max_steps:
            # 获取动作概率分布
            with torch.no_grad():
                # scores: [batch_size, topk] - 候选得分
                scores = self.policy_net(
                    state['knowledge_state'],  # [batch_size, num_skills]
                    state['candidate_features']  # [batch_size, topk, feature_dim]
                )
                # 动作概率 - [batch_size, topk]
                action_probs = F.softmax(scores, dim=-1)
                # 分布对象
                dist = Categorical(action_probs)

                # 采样动作 - [batch_size]
                action = dist.sample()
                # 动作对数概率 - [batch_size]
                log_prob = dist.log_prob(action)

                # 估计状态价值 - [batch_size]
                state_value = torch.mean(scores, dim=-1)

            # 执行动作
            next_state, reward, done = self.env.step(action)

            # 存储转换
            states.append(state)
            actions.append(action)
            rewards.append(reward)  # 混合奖励
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
            'actions': torch.stack(actions) if actions else torch.tensor([]),
            'rewards': torch.stack(rewards) if rewards else torch.tensor([]),
            'log_probs': torch.stack(log_probs) if log_probs else torch.tensor([]),
            'values': torch.stack(values) if values else torch.tensor([]),
            'dones': torch.stack(dones) if dones else torch.tensor([]),
            'final_quality': getattr(self.env, 'final_quality_scores', [])  # 新增：最终质量分数
        }

        # 分析轨迹质量
        self._analyze_trajectory_quality(trajectory)

        # 存储到缓冲区
        self.buffer.append(trajectory)

        return trajectory

    def _analyze_trajectory_quality(self, trajectory):
        """分析轨迹质量"""
        if trajectory['final_quality'] is not None and len(trajectory['final_quality']) > 0:
            avg_final_quality = trajectory['final_quality'].mean().item()
            avg_immediate_reward = trajectory['rewards'].mean().item() if trajectory['rewards'].numel() > 0 else 0

            print(f"轨迹分析: 最终质量={avg_final_quality:.4f}, "
                  f"平均即时奖励={avg_immediate_reward:.4f}, "
                  f"总奖励={avg_final_quality + avg_immediate_reward:.4f}")

            # 记录奖励分配详情
            if hasattr(self.env, 'reward_allocation_history'):
                for allocation in self.env.reward_allocation_history:
                    print(f"  批次{allocation['batch_idx']}: "
                          f"轨迹长度={allocation['traj_len']}, "
                          f"最终质量={allocation['final_quality']:.4f}, "
                          f"总奖励加成={allocation['total_bonus']:.4f}")

    def compute_advantages(self, rewards, values, dones):
        """
        计算优势函数（使用GAE）
        Args:
            rewards: [num_steps, batch_size] - 每步的奖励
            values: [num_steps, batch_size] - 每步的状态价值
            dones: [num_steps, batch_size] - 每步的终止标志
        Returns:
            advantages: [num_steps, batch_size] - 优势函数值
        """
        num_steps, batch_size = rewards.size(0), rewards.size(1)
        # 确保advantages张量与rewards在同一设备上
        advantages = torch.zeros_like(rewards)

        # 计算GAE
        gae = torch.zeros(batch_size, device=rewards.device)  # 每个batch的GAE
        for t in reversed(range(num_steps)):
            # 确保所有张量在同一设备上
            reward_t = rewards[t].to(rewards.device)
            value_t = values[t].to(rewards.device)
            # 对于最后一步，使用0作为下一个状态价值
            if t < num_steps - 1:
                value_t_next = values[t + 1].to(rewards.device)
            else:
                value_t_next = torch.zeros(batch_size, device=rewards.device)
            # 将布尔类型的dones转换为浮点类型以进行数学运算
            done_mask = dones[t].float().to(rewards.device)  # 使用经验教训中的处理方式
            delta = reward_t + self.gamma * value_t_next * (1 - done_mask) - value_t
            gae = delta + self.gamma * 0.95 * gae * (1 - done_mask)  # 0.95是GAE参数
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

        # 确保所有张量都在同一设备上
        device = self.policy_net.parameters().__next__().device
        rewards = rewards.to(device)
        old_values = old_values.to(device)
        dones = dones.to(device)
        actions = actions.to(device)
        old_log_probs = old_log_probs.to(device)

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
            # 确保每个state的knowledge_state是2维的 [batch_size, knowledge_dim]
            knowledge_state = state['knowledge_state'].to(device)
            candidate_features = state['candidate_features'].to(device)

            # 验证张量维度
            assert knowledge_state.dim() == 2, f"knowledge_state should be 2D, got {knowledge_state.dim()}D: {knowledge_state.shape}"
            assert candidate_features.dim() == 3, f"candidate_features should be 3D, got {candidate_features.dim()}D: {candidate_features.shape}"

            batch_states.append(knowledge_state)
            batch_candidate_features.append(candidate_features)

        # 堆叠张量
        # batch_states: [num_steps, batch_size, knowledge_dim]
        batch_states = torch.stack(batch_states)
        # batch_candidate_features: [num_steps, batch_size, topk, feature_dim]
        batch_candidate_features = torch.stack(batch_candidate_features)

        # 获取维度信息
        num_steps, batch_size = batch_states.size(0), batch_states.size(1)
        topk, feature_dim = batch_candidate_features.size(2), batch_candidate_features.size(3)
        knowledge_dim = batch_states.size(2)

        # 重塑张量以适应策略网络的输入要求
        # 将 [num_steps, batch_size, ...] 重塑为 [num_steps * batch_size, ...]
        # batch_states: [num_steps * batch_size, knowledge_dim]
        batch_states = batch_states.view(-1, knowledge_dim)
        # batch_candidate_features: [num_steps * batch_size, topk, feature_dim]
        batch_candidate_features = batch_candidate_features.view(num_steps * batch_size, topk, feature_dim)

        # 验证重塑后的张量维度
        assert batch_states.dim() == 2, f"Reshaped batch_states should be 2D, got {batch_states.dim()}D: {batch_states.shape}"
        assert batch_candidate_features.dim() == 3, f"Reshaped batch_candidate_features should be 3D, got {batch_candidate_features.dim()}D: {batch_candidate_features.shape}"

        # 计算新策略
        scores = self.policy_net(batch_states, batch_candidate_features)
        # 重塑scores回到 [num_steps, batch_size, topk]
        scores = scores.view(num_steps, batch_size, -1)

        # 计算动作概率
        action_probs = F.softmax(scores, dim=-1)  # [num_steps, batch_size, topk]

        # 计算新对数概率
        # 使用gather操作来获取对应动作的对数概率
        new_log_probs = torch.log(action_probs.gather(dim=2, index=actions.unsqueeze(-1))).squeeze(
            -1)  # [num_steps, batch_size]

        # 计算熵
        entropy_list = []
        for t in range(num_steps):
            dist = Categorical(action_probs[t])
            entropy_list.append(dist.entropy())
        entropy = torch.stack(entropy_list).mean()  # 平均熵值

        # 确保new_log_probs和old_log_probs形状一致
        # new_log_probs: [num_steps, batch_size]
        # old_log_probs: [num_steps, batch_size]
        assert new_log_probs.shape == old_log_probs.shape, f"new_log_probs {new_log_probs.shape} and old_log_probs {old_log_probs.shape} should have same shape"

        # 计算比率
        ratios = torch.exp(new_log_probs - old_log_probs)

        # PPO裁剪目标
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # 价值损失
        # new_values: [num_steps, batch_size]
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

    def analyze_reward_components(self, trajectory):
        """
        分析奖励构成，用于监控训练
        """
        if 'final_quality' in trajectory and trajectory['final_quality'] and len(trajectory['final_quality']) > 0:
            print(f"最终质量奖励: {trajectory['final_quality'][-1].mean().item():.4f}")

        # 计算奖励的各个部分（如果环境记录了）
        if hasattr(self.env, 'reward_components_history'):
            components = self.env.reward_components_history[-1]
            print(f"奖励构成 - 有效性: {components['validity'].mean():.4f}, "
                  f"适应性: {components['adaptivity'].mean():.4f}, "
                  f"多样性: {components['diversity'].mean():.4f}, "
                  f"偏好: {components['preference'].mean():.4f}")


class RLPathOptimizer:
    """
    强化学习路径优化器
    核心逻辑：
    1. 在每一步，使用完整的当前历史序列调用预训练模型
    2. 从模型输出的topk候选中选择动作
    3. 基于多目标奖励优化选择策略
    """

    def __init__(self, pretrained_model, num_skills, batch_size, recommendation_length=5,
                 topk=20, data_name=None, graph=None, hypergraph_list=None):
        self.base_model = pretrained_model  # 预训练模型
        self.recommendation_length = recommendation_length  # 推荐路径长度
        self.topk = topk  # 候选集大小
        self.num_skills = num_skills  # 知识点数量
        self.batch_size = batch_size
        self.data_name = data_name  # 数据集名称，用于计算适应性奖励

        # 冻结预训练模型
        for param in self.base_model.parameters():
            param.requires_grad = False

        # 构建策略网络（动作选择器）
        self.policy_net = self._build_policy_network()

        # 将策略网络移动到与预训练模型相同的设备
        if next(self.base_model.parameters()).is_cuda:
            device = next(self.base_model.parameters()).device
            self.policy_net = self.policy_net.to(device)
            print(f"策略网络已移动到设备: {device}")
        else:
            print("预训练模型在CPU上，策略网络保持在CPU")

        # 构建环境，传入图数据
        self.env = LearningPathEnv(
            batch_size=self.batch_size,
            base_model=self.base_model,
            policy_net=self.policy_net,
            recommendation_length=self.recommendation_length,
            topk=self.topk,
            data_name=self.data_name,
            graph=graph,  # 传入图数据
            hypergraph_list=hypergraph_list  # 传入超图数据
        )

        self.trainer = PPOTrainer(
            policy_net=self.policy_net,
            env=self.env,
            lr=3e-4,
            gamma=0.99,
            clip_epsilon=0.2)

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


def evaluate_policy(env, policy_net, test_data_loader, relation_graph, hypergraph_list, num_episodes=10):
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
                state = env.reset(tgt, tgt_timestamp, tgt_idx, ans, relation_graph, hypergraph_list)

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
                            # 确保两个张量在同一设备上
                            next_knowledge = next_state['knowledge_state'][i].to(env.prev_knowledge_state.device)
                            prev_knowledge = env.prev_knowledge_state[i]
                            knowledge_gain = torch.sum(
                                next_knowledge - prev_knowledge
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
                yt_after = knowledge_state[:, 1:, :]  # [batch_size, seq_len-1, num_skills]
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
