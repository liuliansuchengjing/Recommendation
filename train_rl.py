"""
强化学习路径优化训练脚本
【适配新版 rl_adjuster.py（PolicyNetwork + LearningPathEnv + PPOTrainer）】
"""

import torch
import argparse
import os
import numpy as np

from HGAT import MSHGAT
from dataLoader import Split_data, DataLoader
from graphConstruct import ConRelationGraph, ConHyperGraphList

from rl_adjuster import PolicyNetwork, LearningPathEnv, PPOTrainer


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
):
    print("开始强化学习训练（新版 RL 接口）")

    device = torch.device("cuda" if torch.cuda.is_available() and not opt.no_cuda else "cpu")

    policy_net = PolicyNetwork(
        knowledge_dim=num_skills,
        candidate_feature_dim=1,
        hidden_dim=128,
        topk=topk,
    ).to(device)

    env = LearningPathEnv(
        batch_size=batch_size,
        base_model=pretrained_model,
        recommendation_length=recommendation_length,
        data_name=data_path,
        graph=graph,
        hypergraph_list=hypergraph_list,
        policy_topk=topk,  # ✅ 确保 env 候选数与 policy.topk 一致
        metrics_topnum=1,
        metrics_T=5,
    )

    trainer = PPOTrainer(
        policy_net=policy_net,
        env=env,
        lr=3e-4,
        gamma=0.99,
        entropy_coef=0.01,
    )

    user_size, total_cascades, timestamps, train, valid, _ = Split_data(
        data_path, opt.train_rate, opt.valid_rate, load_dict=True
    )

    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        load_dict=True,
        cuda=(device.type == "cuda"),
    )

    stats = {"epoch_loss": [], "avg_step_reward": [], "avg_final_reward": []}

    for epoch in range(num_epochs):
        epoch_losses = []
        epoch_step_rewards = []
        epoch_final_rewards = []

        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 20:
                break

            tgt, tgt_timestamp, tgt_idx, ans = batch
            tgt = tgt.to(device)
            tgt_timestamp = tgt_timestamp.to(device)
            tgt_idx = tgt_idx.to(device)
            ans = ans.to(device)

            states_k, actions, rewards, log_probs, entropies = trainer.collect_trajectory(
                tgt, tgt_timestamp, tgt_idx, ans
            )

            loss = trainer.update_policy(states_k, actions, rewards, log_probs, entropies)
            epoch_losses.append(loss)

            step_reward = torch.stack(rewards).mean().item()
            final_reward = env.compute_final_reward().mean().item()

            epoch_step_rewards.append(step_reward)
            epoch_final_rewards.append(final_reward)

            if batch_idx % 5 == 0:
                print(
                    f"[Epoch {epoch} | Batch {batch_idx}] "
                    f"Loss={loss:.4f}, StepReward={step_reward:.4f}, FinalReward={final_reward:.4f}"
                )

        stats["epoch_loss"].append(float(np.mean(epoch_losses)) if epoch_losses else 0.0)
        stats["avg_step_reward"].append(float(np.mean(epoch_step_rewards)) if epoch_step_rewards else 0.0)
        stats["avg_final_reward"].append(float(np.mean(epoch_final_rewards)) if epoch_final_rewards else 0.0)

        print(
            f"Epoch {epoch} Done | "
            f"Loss={stats['epoch_loss'][-1]:.4f}, "
            f"StepReward={stats['avg_step_reward'][-1]:.4f}, "
            f"FinalReward={stats['avg_final_reward'][-1]:.4f}"
        )

    print("强化学习训练完成")
    return policy_net, stats


def run_training_with_pretrained_model(data_path="MOO", model_path=None):
    # 对齐 run.py 的关键参数（尤其是 d_word_vec）
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

    # ✅ 关键：MSHGAT 需要 d_word_vec
    opt.d_word_vec = opt.d_model

    # load data + set user_size（MSHGAT 也需要 opt.user_size）
    user_size, total_cascades, timestamps, train, valid, _ = Split_data(
        opt.data_name, opt.train_rate, opt.valid_rate, load_dict=True
    )
    opt.user_size = user_size

    graph = ConRelationGraph(opt.data_name)
    hypergraph_list = ConHyperGraphList(total_cascades, timestamps, user_size)

    mshgat = MSHGAT(opt, dropout=opt.dropout)
    if model_path and os.path.exists(model_path):
        mshgat.load_state_dict(torch.load(model_path, map_location="cpu"))
    elif os.path.exists(opt.save_path):
        mshgat.load_state_dict(torch.load(opt.save_path, map_location="cpu"))

    mshgat.eval()
    if torch.cuda.is_available() and not opt.no_cuda:
        mshgat = mshgat.cuda()

    policy_net, stats = train_rl_model(
        data_path=opt.data_name,
        opt=opt,
        pretrained_model=mshgat,
        num_skills=user_size,
        batch_size=opt.batch_size,
        recommendation_length=5,
        num_epochs=5,
        topk=10,
        graph=graph,
        hypergraph_list=hypergraph_list,
    )

    print("训练完成")
    print(f"最终 Step Reward: {stats['avg_step_reward'][-1]:.4f}")
    print(f"最终 Final Reward: {stats['avg_final_reward'][-1]:.4f}")


if __name__ == "__main__":
    run_training_with_pretrained_model()
