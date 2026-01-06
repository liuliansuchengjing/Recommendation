"""
测试强化学习路径优化器
"""

import torch
import torch.nn as nn
import numpy as np
from rl_adjuster import RLPathOptimizer, train_rl_path_optimizer, evaluate_path_metrics, RLPathRecommender
from HGAT import MSHGAT
from Constants import Constants
from argparse import Namespace


def test_rl_optimizer():
    """测试RL优化器的基本功能"""
    print("开始测试RL路径优化器...")
    
    # 创建模拟的预训练模型（简化版本）
    class MockPretrainedModel(nn.Module):
        def __init__(self, num_skills):
            super().__init__()
            self.num_skills = num_skills
            self.n_node = num_skills  # 添加n_node属性
            self.embedding = nn.Embedding(100, 64)  # 简单的嵌入层
            self.linear = nn.Linear(64, num_skills)
            
        def forward(self, input, input_timestamp, input_idx, ans, graph, hypergraph_list):
            batch_size, seq_len = input.shape
            # 模拟输出
            pred = torch.randn(batch_size * (seq_len-1), self.num_skills)
            pred_res = torch.randn(batch_size, seq_len-1, self.num_skills)
            kt_mask = torch.ones(batch_size, seq_len-1)
            yt = torch.sigmoid(torch.randn(batch_size, seq_len, self.num_skills))
            hidden = torch.randn(self.num_skills, 64)
            return pred, pred_res, kt_mask, yt, hidden
    
    # 模拟参数
    num_skills = 50
    batch_size = 4
    recommendation_length = 5  # 重命名为recommendation_length
    topk = 10
    
    # 创建模拟的预训练模型
    mock_model = MockPretrainedModel(num_skills)
    
    # 测试RL路径优化器初始化
    print("初始化RL路径优化器...")
    rl_optimizer = RLPathOptimizer(
        pretrained_model=mock_model,
        num_skills=num_skills,
        batch_size=batch_size,
        recommendation_length=recommendation_length,  # 使用新参数名
        topk=topk
    )
    
    print("RL路径优化器初始化成功!")
    print(f"策略网络参数数量: {sum(p.numel() for p in rl_optimizer.policy_net.parameters())}")
    
    # 测试环境重置
    print("测试环境重置功能...")
    batch_histories = torch.randint(1, num_skills, (batch_size, 10))  # 使用历史长度
    batch_timestamps = torch.randint(0, 100, (batch_size, 10))
    batch_indices = torch.randint(0, 10, (batch_size, 10))
    batch_answers = torch.randint(0, 2, (batch_size, 10))
    
    # 模拟图结构
    mock_graph = torch.randn(num_skills, num_skills)  # 模拟图
    mock_hypergraph = [torch.randn(num_skills, num_skills) for _ in range(5)]  # 模拟超图
    
    state = rl_optimizer.env.reset(
        batch_histories, 
        batch_timestamps, 
        batch_indices, 
        batch_answers, 
        mock_graph, 
        mock_hypergraph
    )
    
    print(f"环境重置成功，状态形状:")
    print(f"  知识状态: {state['knowledge_state'].shape}")
    print(f"  候选习题: {state['candidates'].shape}")
    print(f"  候选特征: {state['candidate_features'].shape}")
    
    # 测试策略网络前向传播
    print("测试策略网络前向传播...")
    with torch.no_grad():
        scores = rl_optimizer.policy_net(
            state['knowledge_state'],
            state['candidate_features']
        )
    
    print(f"策略网络输出形状: {scores.shape}")
    print(f"每个候选的得分: {scores[0]}")
    
    # 测试动作采样
    print("测试动作采样...")
    action_probs = torch.softmax(scores, dim=-1)
    print(f"动作概率分布: {action_probs[0]}")
    
    # 测试环境步进
    print("测试环境步进...")
    actions = torch.multinomial(action_probs, 1).squeeze(-1)  # 采样动作
    print(f"选择的动作: {actions}")
    
    # 执行一步
    next_state, rewards, dones = rl_optimizer.env.step(actions)
    print(f"奖励: {rewards}")
    print(f"是否结束: {dones}")
    
    print("所有测试通过！")


def test_with_real_model():
    """使用真实模型结构进行测试（如果可用）"""
    print("\n尝试使用真实MSHGAT模型进行测试...")
    
    # 创建模拟参数
    opt = Namespace()
    opt.d_word_vec = 64
    opt.user_size = 100
    opt.initialFeatureSize = 32
    opt.train_rate = 0.7
    opt.valid_rate = 0.1
    opt.batch_size = 4
    
    try:
        # 创建MSHGAT模型实例
        model = MSHGAT(opt, dropout=0.3)
        
        # 测试模型基本功能
        batch_size = 2
        seq_len = 10
        num_skills = opt.user_size
        
        input_seq = torch.randint(0, num_skills, (batch_size, seq_len))
        input_timestamp = torch.randint(0, 100, (batch_size, seq_len))
        input_idx = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
        ans = torch.randint(0, 2, (batch_size, seq_len))
        graph = torch.randn(num_skills, num_skills)
        hypergraph_list = [torch.randn(num_skills, num_skills) for _ in range(3)]
        
        with torch.no_grad():
            pred, pred_res, kt_mask, yt, hidden = model(input_seq, input_timestamp, input_idx, ans, graph, hypergraph_list)
        
        print(f"MSHGAT模型输出形状:")
        print(f"  pred: {pred.shape}")
        print(f"  yt (知识状态): {yt.shape}")
        
        # 测试RL优化器与真实模型集成
        rl_optimizer = RLPathOptimizer(
            pretrained_model=model,
            num_skills=num_skills,
            batch_size=batch_size,
            recommendation_length=5,  # 使用新参数名
            topk=10
        )
        
        print("MSHGAT模型与RL优化器集成成功!")
        
    except Exception as e:
        print(f"使用真实模型测试失败: {e}")
        print("这可能是由于缺少必要的依赖或配置文件")


def test_env_step():
    """测试环境的完整步进功能"""
    print("\n测试环境的完整步进功能...")
    
    # 创建模拟模型
    class MockPretrainedModel(nn.Module):
        def __init__(self, num_skills):
            super().__init__()
            self.num_skills = num_skills
            self.n_node = num_skills
            self.embedding = nn.Embedding(100, 64)
            self.linear = nn.Linear(64, num_skills)
            
        def forward(self, input, input_timestamp, input_idx, ans, graph, hypergraph_list):
            batch_size, seq_len = input.shape
            pred = torch.randn(batch_size * (seq_len-1), self.num_skills)
            pred_res = torch.randn(batch_size, seq_len-1, self.num_skills)
            kt_mask = torch.ones(batch_size, seq_len-1)
            yt = torch.sigmoid(torch.randn(batch_size, seq_len, self.num_skills))
            hidden = torch.randn(self.num_skills, 64)
            return pred, pred_res, kt_mask, yt, hidden
    
    num_skills = 20
    batch_size = 2
    recommendation_length = 3  # 重命名为recommendation_length
    topk = 5
    
    mock_model = MockPretrainedModel(num_skills)
    
    rl_optimizer = RLPathOptimizer(
        pretrained_model=mock_model,
        num_skills=num_skills,
        batch_size=batch_size,
        recommendation_length=recommendation_length,  # 使用新参数名
        topk=topk
    )
    
    # 准备测试数据
    batch_histories = torch.randint(1, num_skills, (batch_size, 5))  # 使用历史长度
    batch_timestamps = torch.randint(0, 100, (batch_size, 5))
    batch_indices = torch.randint(0, 5, (batch_size, 5))
    batch_answers = torch.randint(0, 2, (batch_size, 5))
    
    mock_graph = torch.randn(num_skills, num_skills)
    mock_hypergraph = [torch.randn(num_skills, num_skills) for _ in range(3)]
    
    # 重置环境
    state = rl_optimizer.env.reset(
        batch_histories,
        batch_timestamps,
        batch_indices,
        batch_answers,
        mock_graph,
        mock_hypergraph
    )
    
    print(f"初始状态知识状态形状: {state['knowledge_state'].shape}")
    print(f"初始状态候选形状: {state['candidates'].shape}")
    
    # 执行多个步骤
    for step in range(2):
        print(f"\n执行步骤 {step+1}")
        
        # 获取策略网络的输出
        with torch.no_grad():
            scores = rl_optimizer.policy_net(
                state['knowledge_state'],
                state['candidate_features']
            )
        
        # 采样动作
        action_probs = torch.softmax(scores, dim=-1)
        actions = torch.multinomial(action_probs, 1).squeeze(-1)
        
        print(f"选择的动作: {actions}")
        
        # 执行步进
        next_state, rewards, dones = rl_optimizer.env.step(actions)
        
        print(f"奖励: {rewards}")
        print(f"是否结束: {dones}")
        print(f"新知识状态形状: {next_state['knowledge_state'].shape}")
        
        state = next_state  # 更新状态
        
        if torch.all(dones):
            print("所有环境已结束")
            break
    
    print("环境步进测试完成!")


def test_path_evaluation():
    """测试路径评估功能"""
    print("\n测试路径评估功能...")
    
    # 创建模拟模型
    class MockPretrainedModel(nn.Module):
        def __init__(self, num_skills):
            super().__init__()
            self.num_skills = num_skills
            self.n_node = num_skills
            self.embedding = nn.Embedding(100, 64)
            self.linear = nn.Linear(64, num_skills)
            
        def forward(self, input, input_timestamp, input_idx, ans, graph, hypergraph_list):
            batch_size, seq_len = input.shape
            pred = torch.randn(batch_size * (seq_len-1), self.num_skills)
            pred_res = torch.randn(batch_size, seq_len-1, self.num_skills)
            kt_mask = torch.ones(batch_size, seq_len-1)
            yt = torch.sigmoid(torch.randn(batch_size, seq_len, self.num_skills))
            hidden = torch.randn(self.num_skills, 64)
            return pred, pred_res, kt_mask, yt, hidden
    
    num_skills = 20
    batch_size = 2
    recommendation_length = 3  # 重命名为recommendation_length
    topk = 5
    
    mock_model = MockPretrainedModel(num_skills)
    
    # 初始化RL优化器
    rl_optimizer = RLPathOptimizer(
        pretrained_model=mock_model,
        num_skills=num_skills,
        batch_size=batch_size,
        recommendation_length=recommendation_length,  # 使用新参数名
        topk=topk
    )
    
    # 创建推荐器
    recommender = RLPathRecommender(rl_optimizer)
    
    # 模拟用户历史
    user_histories = torch.randint(1, num_skills, (batch_size, 5))
    
    # 推荐路径
    recommended_paths, scores = recommender.batch_recommend(user_histories, deterministic=True)
    
    print(f"推荐路径形状: {recommended_paths.shape}")
    print(f"推荐路径: {recommended_paths}")
    print(f"推荐得分形状: {scores.shape}")
    
    print("路径评估测试完成!")


if __name__ == "__main__":
    test_rl_optimizer()
    test_with_real_model()
    test_env_step()
    test_path_evaluation()
    print("\n所有测试完成!")