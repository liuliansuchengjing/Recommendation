"""
强化学习路径优化模型训练脚本
可以直接运行此脚本进行强化学习训练
"""

import torch
import argparse
import os
from HGAT import MSHGAT
from rl_adjuster import RLPathOptimizer, evaluate_policy
from dataLoader import Split_data, DataLoader
from graphConstruct import ConRelationGraph, ConHyperGraphList
import numpy as np
import pickle
import matplotlib.pyplot as plt

def train_rl_model(data_path, opt, pretrained_model, num_skills, batch_size,
                   recommendation_length=5, num_epochs=50, topk=20,
                   graph=None, hypergraph_list=None):
    """
    完整的强化学习训练函数
    """
    print("开始训练强化学习模型...")
    print(f"训练参数: epochs={num_epochs}, batch_size={batch_size}, recommendation_length={recommendation_length}, topk={topk}")

    # 初始化RL优化器，传入图数据
    rl_optimizer = RLPathOptimizer(
        pretrained_model=pretrained_model,
        num_skills=num_skills,
        batch_size=batch_size,
        recommendation_length=recommendation_length,
        topk=topk,
        data_name=data_path,
        graph=graph,  # 传入图数据
        hypergraph_list=hypergraph_list  # 传入超图数据
    )

    # 获取数据
    user_size, total_cascades, timestamps, train, valid, test = Split_data(
        data_path, opt.train_rate, opt.valid_rate, load_dict=True
    )

    # 创建数据加载器
    train_data = DataLoader(train, batch_size=batch_size, load_dict=True, cuda=not opt.no_cuda and torch.cuda.is_available())
    valid_data = DataLoader(valid, batch_size=batch_size, load_dict=True, cuda=not opt.no_cuda and torch.cuda.is_available())

    # 训练统计
    training_stats = {
        'epoch_losses': [],
        'avg_rewards': [],
        'validity_scores': [],
        'diversity_scores': [],
        'adaptivity_scores': [],
        'final_qualities': []
    }

    # 训练循环
    for epoch in range(num_epochs):
        epoch_losses = []
        epoch_rewards = []
        epoch_final_qualities = []

        for batch_idx, batch in enumerate(train_data):
            if batch_idx >= 20:  # 限制每个epoch的批次数量以加快训练
                break

            # 准备数据
            if not opt.no_cuda and torch.cuda.is_available():
                # tgt: [batch_size, seq_len] - 目标序列
                # tgt_timestamp: [batch_size, seq_len] - 时间戳序列
                # tgt_idx: [batch_size, seq_len] - 索引序列
                # ans: [batch_size, seq_len] - 答案序列
                tgt, tgt_timestamp, tgt_idx, ans = (item.cuda() for item in batch)
            else:
                tgt, tgt_timestamp, tgt_idx, ans = batch

            # 收集轨迹
            trajectory = rl_optimizer.trainer.collect_trajectory(
                tgt, tgt_timestamp, tgt_idx, ans, graph, hypergraph_list
            )

            # 计算平均奖励
            if trajectory['rewards'].numel() > 0:
                avg_reward = trajectory['rewards'].mean().item()  # scalar
                epoch_rewards.append(avg_reward)

            # 记录最终质量
            if 'final_quality' in trajectory and trajectory['final_quality'] is not None and len(trajectory['final_quality']) > 0:
                avg_final_quality = trajectory['final_quality'].mean().item()
                epoch_final_qualities.append(avg_final_quality)

            # 更新策略
            loss = rl_optimizer.trainer.update_policy()
            epoch_losses.append(loss)

            # 分析奖励构成
            rl_optimizer.trainer.analyze_reward_components(trajectory)

            # 每几个批次打印一次信息
            if batch_idx % 5 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}: Loss={loss:.4f}, "
                      f"Reward={avg_reward if 'avg_reward' in locals() else 0:.4f}, "
                      f"Final Quality={avg_final_quality if 'avg_final_quality' in locals() else 0:.4f}")

        # 记录统计
        avg_epoch_loss = np.mean(epoch_losses) if epoch_losses else 0  # scalar
        avg_epoch_reward = np.mean(epoch_rewards) if epoch_rewards else 0  # scalar
        avg_epoch_final_quality = np.mean(epoch_final_qualities) if epoch_final_qualities else 0  # scalar

        training_stats['epoch_losses'].append(avg_epoch_loss)
        training_stats['avg_rewards'].append(avg_epoch_reward)
        training_stats['final_qualities'].append(avg_epoch_final_quality)

        # 每隔几个epoch进行评估
        if epoch % 5 == 0:
            validity, diversity, adaptivity = evaluate_policy(
                rl_optimizer.env,
                rl_optimizer.policy_net,
                valid_data,
                relation_graph=graph,
                hypergraph_list=hypergraph_list,
                num_episodes=5
            )
            training_stats['validity_scores'].append(validity)
            training_stats['diversity_scores'].append(diversity)
            training_stats['adaptivity_scores'].append(adaptivity)
            print(f"Epoch {epoch}: Avg Loss={avg_epoch_loss:.4f}, Avg Reward={avg_epoch_reward:.4f}, "
                  f"Final Quality={avg_epoch_final_quality:.4f}, "
                  f"Validity={validity:.4f}, Diversity={diversity:.4f}, Adaptivity={adaptivity:.4f}")
        else:
            print(f"Epoch {epoch} completed: Avg Loss={avg_epoch_loss:.4f}, Avg Reward={avg_epoch_reward:.4f}, "
                  f"Final Quality={avg_epoch_final_quality:.4f}")

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
        graph,
        hypergraph_list,
        num_episodes=10  # 使用更多episode获取更稳定的评估结果
    )

    print(f"最终验证结果:")
    print(f"  有效性 (Validity): {final_validity:.4f}")
    print(f"  多样性 (Diversity): {final_diversity:.4f}")
    print(f"  适应性 (Adaptivity): {final_adaptivity:.4f}")

    # 保存验证结果
    training_stats['final_validity'] = final_validity  # scalar
    training_stats['final_diversity'] = final_diversity  # scalar
    training_stats['final_adaptivity'] = final_adaptivity  # scalar

    print(f"\n训练统计:")
    print(f"  最终平均奖励: {training_stats['avg_rewards'][-1] if training_stats['avg_rewards'] else 0:.4f}")
    print(f"  最终平均质量: {training_stats['final_qualities'][-1] if training_stats['final_qualities'] else 0:.4f}")
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
    user_size, total_cascades, timestamps, train, valid, test = Split_data(opt.data_name, opt.train_rate, opt.valid_rate, load_dict=True)
    opt.user_size = user_size

    # 创建图结构
    relation_graph = ConRelationGraph(opt.data_name)
    hypergraph_list = ConHyperGraphList(total_cascades, timestamps, user_size)

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
        raise

    num_skills = user_size  # scalar

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
        topk=10,        # 候选集大小
        graph=relation_graph,  # 传入图数据
        hypergraph_list=hypergraph_list  # 传入超图数据
    )

    print("训练完成！")
    print(f"最终平均奖励: {stats['avg_rewards'][-1] if stats['avg_rewards'] else 0:.4f}")
    print(f"最终平均质量: {stats['final_qualities'][-1] if stats['final_qualities'] else 0:.4f}")

    # 打印最终验证结果
    if 'final_validity' in stats and 'final_diversity' in stats and 'final_adaptivity' in stats:
        print(f"\n强化学习优化效果评估:")
        print(f"  有效性 (Validity): {stats['final_validity']:.4f}")
        print(f"  多样性 (Diversity): {stats['final_diversity']:.4f}")
        print(f"  适应性 (Adaptivity): {stats['final_adaptivity']:.4f}")

    # 绘制训练曲线
    if len(stats['epoch_losses']) > 1:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # 损失曲线
        axes[0, 0].plot(stats['epoch_losses'])
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')

        # 奖励曲线
        axes[0, 1].plot(stats['avg_rewards'])
        axes[0, 1].set_title('Average Reward')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Reward')

        # 最终质量曲线
        axes[1, 0].plot(stats['final_qualities'])
        axes[1, 0].set_title('Final Quality')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Quality')

        # 总指标曲线
        total_scores = [r + q for r, q in zip(stats['avg_rewards'], stats['final_qualities'])]
        axes[1, 1].plot(total_scores)
        axes[1, 1].set_title('Total Score (Reward + Quality)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')

        plt.tight_layout()
        save_dir = "checkpoints"
        os.makedirs(save_dir, exist_ok=True)
        plot_path = os.path.join(save_dir, "training_curves.png")
        plt.savefig(plot_path)
        print(f"训练曲线已保存到 {plot_path}")

        # 保存训练统计
        stats_path = os.path.join(save_dir, "training_stats.pkl")
        with open(stats_path, 'wb') as f:
            pickle.dump(stats, f)
        print(f"训练统计已保存到 {stats_path}")

    return rl_optimizer, stats


def evaluate_enhanced_rl_model(rl_optimizer, test_data_path):
    """
    评估增强RL模型的性能
    """
    # 加载测试数据
    _, _, _, _, _, test = Split_data(test_data_path, 0.8, 0.1, load_dict=True)
    test_data = DataLoader(test, batch_size=16, load_dict=True, cuda=True)

    all_metrics = []

    for batch in test_data:
        tgt, tgt_timestamp, tgt_idx, ans = (item.cuda() for item in batch)

        # 使用RL模型生成推荐
        state = rl_optimizer.env.reset(tgt, tgt_timestamp, tgt_idx, ans,
                                       rl_optimizer.env.original_graph,
                                       rl_optimizer.env.original_hypergraph_list)

        # 运行轨迹
        for _ in range(rl_optimizer.env.recommendation_length):
            with torch.no_grad():
                scores = rl_optimizer.policy_net(
                    state['knowledge_state'],
                    state['candidate_features']
                )
                actions = torch.argmax(scores, dim=-1)  # 确定性策略
                state, _, _ = rl_optimizer.env.step(actions)

        # 获取轨迹指标
        batch_metrics = []
        for i in range(tgt.size(0)):
            trajectory_data = rl_optimizer.env.trajectory_manager.get_trajectory_data(i)
            metrics = rl_optimizer.env.metrics_evaluator.evaluate_trajectory(trajectory_data)
            batch_metrics.append(metrics)

        all_metrics.extend(batch_metrics)

    # 计算平均指标
    avg_metrics = {
        'effectiveness': np.mean([m['effectiveness'] for m in all_metrics]),
        'adaptivity': np.mean([m['adaptivity'] for m in all_metrics]),
        'diversity': np.mean([m['diversity'] for m in all_metrics]),
        'preference': np.mean([m['preference'] for m in all_metrics]),
        'final_quality': np.mean([m['final_quality'] for m in all_metrics])
    }

    print("\n增强RL模型评估结果:")
    for metric, value in avg_metrics.items():
        print(f"  {metric}: {value:.4f}")

    return avg_metrics


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
            print(f"  平均质量: {stats['final_qualities'][-1] if stats['final_qualities'] else 0:.4f}")

    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()