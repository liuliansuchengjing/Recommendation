"""
train_rl.py
强化学习路径优化训练脚本（兼容整理后的 rl_adjuster.py）
训练后会在验证集上输出推荐路径各指标：
effectiveness / adaptivity / diversity / preference / final_quality
"""

import torch
import argparse
import os
import numpy as np
import pickle

from HGAT import MSHGAT
from dataLoader import Split_data, DataLoader
from graphConstruct import ConRelationGraph, ConHyperGraphList

from rl_adjuster import RLPathOptimizer, evaluate_policy


def train_rl_model(
    data_path,
    opt,
    pretrained_model,
    num_skills,
    batch_size,
    recommendation_length=5,
    num_epochs=10,
    topk=10,
    graph=None,
    hypergraph_list=None,
    max_train_batches=20,
):
    print("开始强化学习训练（RLPathOptimizer）")

    device = torch.device("cuda" if torch.cuda.is_available() and not opt.no_cuda else "cpu")

    rl = RLPathOptimizer(
        pretrained_model=pretrained_model,
        num_skills=num_skills,
        batch_size=batch_size,
        recommendation_length=recommendation_length,
        topk=topk,
        data_name=data_path,
        graph=graph,
        hypergraph_list=hypergraph_list,
        device=device,
    )

    # Data
    user_size, total_cascades, timestamps, train, valid, _ = Split_data(
        data_path, opt.train_rate, opt.valid_rate, load_dict=True
    )

    train_loader = DataLoader(
        train, batch_size=batch_size, load_dict=True, cuda=(device.type == "cuda")
    )
    valid_loader = DataLoader(
        valid, batch_size=batch_size, load_dict=True, cuda=(device.type == "cuda")
    )

    stats = {
        "epoch_loss": [],
        "avg_step_reward": [],
        "avg_final_reward": [],
        "valid_metrics": [],
    }

    for epoch in range(num_epochs):
        epoch_losses = []
        epoch_step_rewards = []
        epoch_final_rewards = []

        rl.policy_net.train()

        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= max_train_batches:
                break

            tgt, tgt_timestamp, tgt_idx, ans = batch
            tgt = tgt.to(device)
            tgt_timestamp = tgt_timestamp.to(device)
            tgt_idx = tgt_idx.to(device)
            ans = ans.to(device)

            rewards, log_probs, entropies, final_reward = rl.trainer.collect_trajectory(
                tgt, tgt_timestamp, tgt_idx, ans, deterministic=False
            )

            loss = rl.trainer.update_policy(rewards, log_probs, entropies)
            epoch_losses.append(loss)

            step_reward = torch.stack(rewards).mean().item()
            fr = final_reward.mean().item()

            epoch_step_rewards.append(step_reward)
            epoch_final_rewards.append(fr)

            if batch_idx % 5 == 0:
                print(f"[Epoch {epoch} | Batch {batch_idx}] Loss={loss:.4f}, Reward={step_reward:.4f}, FinalQuality={fr:.4f}")

        stats["epoch_loss"].append(float(np.mean(epoch_losses) if epoch_losses else 0.0))
        stats["avg_step_reward"].append(float(np.mean(epoch_step_rewards) if epoch_step_rewards else 0.0))
        stats["avg_final_reward"].append(float(np.mean(epoch_final_rewards) if epoch_final_rewards else 0.0))

        print(f"Epoch {epoch} Done | Loss={stats['epoch_loss'][-1]:.4f}, Reward={stats['avg_step_reward'][-1]:.4f}, FinalQuality={stats['avg_final_reward'][-1]:.4f}")

        # 每个 epoch 做一次验证评估并输出指标
        valid_metrics = evaluate_policy(
            env=rl.env,
            policy_net=rl.policy_net,
            data_loader=valid_loader,
            device=device,
            max_batches=20,
        )
        stats["valid_metrics"].append(valid_metrics)

        print("\n[Evaluation Metrics on Recommended Paths]")
        print(f"  effectiveness: {valid_metrics['effectiveness']:.4f}")
        print(f"  adaptivity:    {valid_metrics['adaptivity']:.4f}")
        print(f"  diversity:     {valid_metrics['diversity']:.4f}")
        print(f"  preference:    {valid_metrics['preference']:.4f}")
        print(f"  final_quality: {valid_metrics['final_quality']:.4f}\n")

    return rl.policy_net, rl, stats


def run_training_with_pretrained_model(data_path="MOO", model_path=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_name", default=data_path)
    parser.add_argument("-batch_size", type=int, default=16)
    parser.add_argument("-d_model", type=int, default=64)
    parser.add_argument("-initialFeatureSize", type=int, default=64)
    parser.add_argument("-train_rate", type=float, default=0.8)
    parser.add_argument("-valid_rate", type=float, default=0.1)
    parser.add_argument("-n_warmup_steps", type=int, default=1000)
    parser.add_argument("-dropout", type=float, default=0.3)
    parser.add_argument("-save_path", default="./checkpoint/DiffusionPrediction.pt")
    parser.add_argument("-no_cuda", action="store_true")
    parser.add_argument("-pos_emb", type=bool, default=True)
    opt = parser.parse_args([])

    # ✅ HGAT.MSHGAT 需要 opt.d_word_vec
    opt.d_word_vec = opt.d_model

    # Load data
    user_size, total_cascades, timestamps, train, valid, _ = Split_data(
        opt.data_name, opt.train_rate, opt.valid_rate, load_dict=True
    )
    opt.user_size = user_size

    print(f"training size:{len(train)}\n   valid size:{len(valid)}")
    print(f"user size:{user_size}")

    # Graph
    relation_graph = ConRelationGraph(opt.data_name)
    hypergraph_list = ConHyperGraphList(total_cascades, timestamps, user_size)

    # Base model
    print("尝试加载预训练模型.")
    mshgat_model = MSHGAT(opt, dropout=opt.dropout)

    if model_path and os.path.exists(model_path):
        mshgat_model.load_state_dict(torch.load(model_path, map_location="cpu"))
        print(f"已从 {model_path} 加载预训练权重")
    elif os.path.exists(opt.save_path):
        mshgat_model.load_state_dict(torch.load(opt.save_path, map_location="cpu"))
        print(f"已从默认路径 {opt.save_path} 加载预训练权重")
    else:
        print("未找到预训练权重，将使用随机初始化模型（仅用于调试跑通流程）")

    mshgat_model.eval()
    if torch.cuda.is_available() and not opt.no_cuda:
        mshgat_model = mshgat_model.cuda()
        print("模型已移动到 GPU")

    # Train RL
    policy_net, rl, stats = train_rl_model(
        data_path=opt.data_name,
        opt=opt,
        pretrained_model=mshgat_model,
        num_skills=user_size,
        batch_size=opt.batch_size,
        recommendation_length=5,
        num_epochs=5,
        topk=10,
        graph=relation_graph,
        hypergraph_list=hypergraph_list,
        max_train_batches=20,
    )

    # Save
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    torch.save(policy_net.state_dict(), os.path.join(save_dir, "rl_policy_net.pth"))
    with open(os.path.join(save_dir, "rl_training_stats.pkl"), "wb") as f:
        pickle.dump(stats, f)

    print(f"模型已保存到: {os.path.join(save_dir, 'rl_policy_net.pth')}")
    print(f"训练统计已保存到: {os.path.join(save_dir, 'rl_training_stats.pkl')}")


if __name__ == "__main__":
    run_training_with_pretrained_model()
