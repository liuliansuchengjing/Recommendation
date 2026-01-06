"""
强化学习路径优化模型训练脚本
可以直接运行此脚本进行强化学习训练
"""

import torch
import argparse
import os
from HGAT import MSHGAT
from rl_adjuster import RLPathOptimizer, RLPathRecommender, evaluate_policy
from dataLoader import Split_data, DataLoader
from graphConstruct import ConRelationGraph, ConHyperGraphList
import numpy as np


def train_rl_model(data_path, opt, pretrained_model, num_skills, batch_size, recommendation_length=5, num_epochs=50, topk=20):
    """
    完整的强化学习训练函数
    """
    print("开始训练强化学习模型...")
    print(f"训练参数: epochs={num_epochs}, batch_size={batch_size}, recommendation_length={recommendation_length}, topk={topk}")
    
    # 初始化RL优化器
    rl_optimizer = RLPathOptimizer(
        pretrained_model=pretrained_model,
        num_skills=num_skills,
        batch_size=batch_size,
        recommendation_length=recommendation_length,
        topk=topk,
        data_name=data_path
    )
    
    # 获取数据
    user_size, total_cascades, timestamps, train, valid, test = Split_data(
        data_path, opt.train_rate, opt.valid_rate, load_dict=True
    )
    
    # 创建数据加载器
    train_data = DataLoader(train, batch_size=batch_size, load_dict=True, cuda=not opt.no_cuda and torch.cuda.is_available())
    valid_data = DataLoader(valid, batch_size=batch_size, load_dict=True, cuda=not opt.no_cuda and torch.cuda.is_available())
    
    # 创建图结构
    relation_graph = ConRelationGraph(data_path)
    hypergraph_list = ConHyperGraphList(total_cascades, timestamps, user_size)
    
    # 训练统计
    training_stats = {
        'epoch_losses': [],
        'avg_rewards': [],
        'validity_scores': [],
        'diversity_scores': []
    }
    
    # 训练循环
    for epoch in range(num_epochs):
        epoch_losses = []
        epoch_rewards = []

        for batch_idx, batch in enumerate(train_data):
            if batch_idx >= 20:  # 限制每个epoch的批次数量以加快训练
                break
                
            # 准备数据
            if not opt.no_cuda and torch.cuda.is_available():
                tgt, tgt_timestamp, tgt_idx, ans = (item.cuda() for item in batch)
            else:
                tgt, tgt_timestamp, tgt_idx, ans = batch

            # 收集轨迹
            trajectory = rl_optimizer.trainer.collect_trajectory(
                tgt, tgt_timestamp, tgt_idx, ans, relation_graph, hypergraph_list
            )

            # 计算平均奖励
            avg_reward = trajectory['rewards'].mean().item()
            epoch_rewards.append(avg_reward)

            # 更新策略
            loss = rl_optimizer.trainer.update_policy()
            epoch_losses.append(loss)

            # 每几个批次打印一次信息
            if batch_idx % 5 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}: Loss={loss:.4f}, Reward={avg_reward:.4f}")

        # 记录统计
        avg_epoch_loss = np.mean(epoch_losses) if epoch_losses else 0
        avg_epoch_reward = np.mean(epoch_rewards) if epoch_rewards else 0

        training_stats['epoch_losses'].append(avg_epoch_loss)
        training_stats['avg_rewards'].append(avg_epoch_reward)

        # 每隔几个epoch进行评估
        if epoch % 5 == 0:
            validity, diversity, adaptivity = evaluate_policy(
                rl_optimizer.env,
                rl_optimizer.policy_net,
                valid_data,
                num_episodes=5
            )
            training_stats['validity_scores'].append(validity)
            training_stats['diversity_scores'].append(diversity)
            training_stats['adaptivity_scores'].append(adaptivity)
            print(f"Epoch {epoch}: Avg Loss={avg_epoch_loss:.4f}, Avg Reward={avg_reward:.4f}, "
                  f"Validity={validity:.4f}, Diversity={diversity:.4f}, Adaptivity={adaptivity:.4f}")
        else:
            print(f"Epoch {epoch} completed: Avg Loss={avg_epoch_loss:.4f}, Avg Reward={avg_reward:.4f}")

    print("强化学习训练完成!")
    
    # 保存训练好的模型
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "rl_policy_net.pth")
    torch.save(rl_optimizer.policy_net.state_dict(), model_path)
    print(f"模型已保存到 {model_path}")
    
    # 训练完成后在验证集上进行最终评估
    print("\n开始最终验证评估...")
    final_validity, final_diversity, final_adaptivity = evaluate_policy(
        rl_optimizer.env,
        rl_optimizer.policy_net,
        valid_data,
        num_episodes=10  # 使用更多episode获取更稳定的评估结果
    )
    
    print(f"最终验证结果:")
    print(f"  有效性 (Validity): {final_validity:.4f}")
    print(f"  多样性 (Diversity): {final_diversity:.4f}")
    print(f"  适应性 (Adaptivity): {final_adaptivity:.4f}")
    
    # 保存验证结果
    training_stats['final_validity'] = final_validity
    training_stats['final_diversity'] = final_diversity
    training_stats['final_adaptivity'] = final_adaptivity
    
    print(f"\n训练统计:")
    print(f"  最终平均奖励: {training_stats['avg_rewards'][-1] if training_stats['avg_rewards'] else 0:.4f}")
    print(f"  最终有效性: {final_validity:.4f}")
    print(f"  最终多样性: {final_diversity:.4f}")
    print(f"  最终适应性: {final_adaptivity:.4f}")
    
    return rl_optimizer, training_stats


def run_training_with_pretrained_model(data_path="MOO", model_path=None):
    """
    使用预训练模型进行强化学习训练的完整流程
    """
    # 设置参数
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_name', default=data_path)
    parser.add_argument('-batch_size', type=int, default=16)  # 减小批次大小以适应内存
    parser.add_argument('-d_model', type=int, default=64)
    parser.add_argument('-initialFeatureSize', type=int, default=64)  # 保持与原始模型一致
    parser.add_argument('-train_rate', type=float, default=0.8)  # 与run.py保持一致
    parser.add_argument('-valid_rate', type=float, default=0.1)
    parser.add_argument('-n_warmup_steps', type=int, default=1000)
    parser.add_argument('-dropout', type=float, default=0.3)
    parser.add_argument('-save_path', default="./checkpoint/DiffusionPrediction.pt")  # 添加模型保存路径
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-pos_emb', type=bool, default=True)
    opt = parser.parse_args([])
    opt.d_word_vec = opt.d_model
    
    # 获取数据信息以设置opt.user_size
    user_size, _, _, _, _, _ = Split_data(opt.data_name, opt.train_rate, opt.valid_rate, load_dict=True)
    opt.user_size = user_size
    
    # 加载预训练的MSHGAT模型或创建模拟模型
    print("尝试加载预训练模型...")
    try:
        # 初始化MSHGAT模型
        mshgat_model = MSHGAT(opt, dropout=opt.dropout)
        
        # 如果提供了预训练模型路径，则加载权重
        if model_path and os.path.exists(model_path):
            if not opt.no_cuda and torch.cuda.is_available():
                mshgat_model.load_state_dict(torch.load(model_path))
                mshgat_model = mshgat_model.cuda()
                print(f"已从 {model_path} 加载预训练权重并移动到GPU")
            else:
                mshgat_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                print(f"已从 {model_path} 加载预训练权重")
        elif os.path.exists(opt.save_path):  # 检查默认保存路径
            if not opt.no_cuda and torch.cuda.is_available():
                mshgat_model.load_state_dict(torch.load(opt.save_path))
                mshgat_model = mshgat_model.cuda()
                print(f"已从默认路径 {opt.save_path} 加载预训练权重并移动到GPU")
            else:
                mshgat_model.load_state_dict(torch.load(opt.save_path, map_location=torch.device('cpu')))
                print(f"已从默认路径 {opt.save_path} 加载预训练权重")
        else:
            print(f"未找到预训练模型，将在 {opt.save_path} 查找")
            if not opt.no_cuda and torch.cuda.is_available():
                mshgat_model = mshgat_model.cuda()
            
        # 将模型设置为评估模式
        mshgat_model.eval()
        
        print("预训练模型加载成功!")
        print(f"模型参数数量: {sum(p.numel() for p in mshgat_model.parameters())}")
    except Exception as e:
        print(f"加载预训练模型失败: {e}")
    
    num_skills = user_size
    
    print(f"数据加载完成，知识点数量: {num_skills}")
    
    # 运行训练
    print("开始强化学习训练...")
    rl_optimizer, stats = train_rl_model(
        data_path=opt.data_name,
        opt=opt,
        pretrained_model=mshgat_model,
        num_skills=num_skills,
        batch_size=opt.batch_size,
        recommendation_length=5,  # 推荐路径长度，明确参数名
        num_epochs=20,  # 训练轮数
        topk=10         # 候选集大小
    )
    
    print("训练完成！")
    print(f"最终平均奖励: {stats['avg_rewards'][-1] if stats['avg_rewards'] else 0:.4f}")
    
    # 打印最终验证结果
    if 'final_validity' in stats and 'final_diversity' in stats and 'final_adaptivity' in stats:
        print(f"\n强化学习优化效果评估:")
        print(f"  有效性 (Validity): {stats['final_validity']:.4f}")
        print(f"  多样性 (Diversity): {stats['final_diversity']:.4f}")
        print(f"  适应性 (Adaptivity): {stats['final_adaptivity']:.4f}")
    
    return rl_optimizer, stats


if __name__ == "__main__":
    print("="*60)
    print("强化学习路径优化模型训练")
    print("此脚本将使用预训练模型进行强化学习训练")
    print("优化目标：有效性、适应性、多样性、偏好保持")
    print("="*60)
    
    # 运行训练
    try:
        rl_optimizer, stats = run_training_with_pretrained_model()
        print("\n训练成功完成！")
        
        if 'final_validity' in stats and 'final_diversity' in stats and 'final_adaptivity' in stats:
            print(f"\n最终强化学习优化效果:")
            print(f"  有效性 (Validity): {stats['final_validity']:.4f}")
            print(f"  多样性 (Diversity): {stats['final_diversity']:.4f}")
            print(f"  适应性 (Adaptivity): {stats['final_adaptivity']:.4f}")
            print(f"  平均奖励: {stats['avg_rewards'][-1] if stats['avg_rewards'] else 0:.4f}")
        
        # print("您可以使用训练好的模型进行推理：")
        # print("  rl_recommender = RLPathRecommender(rl_optimizer)")
        # print("  recommended_path, scores = rl_recommender.recommend_path(user_history)")
        #
        # # 提供使用示例
        # print("\n使用示例:")
        # print("# 加载训练好的推荐器")
        # print("recommender = RLPathRecommender(rl_optimizer)")
        # print("")
        # print("# 为单个用户推荐路径")
        # print("user_history = torch.randint(1, num_skills, (10,))  # 示例历史")
        # print("path, scores = recommender.recommend_path(user_history)")
        # print("print('推荐路径:', path)")
        # print("")
        # print("# 批量推荐")
        # print("user_histories = torch.randint(1, num_skills, (batch_size, seq_len))")
        # print("paths, all_scores = recommender.batch_recommend(user_histories)")
        # print("print('批量推荐路径形状:', paths.shape)")
        # print("")
        # print("# 评估推荐路径质量")
        # print("quality_results = recommender.evaluate_path_quality(user_histories)")
        # print("print('路径质量评估:', quality_results)")
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()