"""
强化学习路径优化训练脚本
【适配新版 rl_adjuster.py】
"""

import torch
import argparse
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

from HGAT import MSHGAT
from dataLoader import Split_data, DataLoader
from graphConstruct import ConRelationGraph, ConHyperGraphList

from rl_adjuster import PolicyNetwork, LearningPathEnv, PPOTrainer


# ======================================================
# RL Training
# ======================================================
def train_rl_model(
    data_path,
    opt,
    pretrained_model,
    num_skills,
    batch_size,
    recommendation_length=5,
    num_epochs=30,
    topk=10,
    graph=None,
    hypergraph_list=None,
):
    print("开始强化学习训练（新版 RL 接口）")

    device = torch.device("cuda" if torch.cuda.is_available() and not opt.no_cuda else "cpu")

    # -------- Policy Network --------
    policy_net = PolicyNetwork(
        knowledge_dim=num_skills,
        candidate_feature_dim=5,   # 占位特征（不影响 reward 逻辑）
        hidden_dim=128,
        topk=topk,
    ).to(device)

    # -------- Environment --------
    env = LearningPathEnv(
        batch_size=batch_size,
        base_model=pretrained_model,
        recommendation_length=recommendation_length,
        data_name=data_path,
        graph=graph,
        hypergraph_list=hypergraph_list,
    )

    # -------- PPO Trainer --------
    trainer = PPOTrainer(
        policy_net=policy_net,
        env=env,
        lr=3e-4,
        gamma=0.99,
    )

    # -------- Data --------
    user_size, total_cascades, timestamps, train, valid, _ = Split_data(
        data_path, opt.train_rate, opt.valid_rate, load_dict=True
    )

    train_loader = DataLoader(
        train, batch_size=batch_size, load_dict=True,
        cuda=(device.type == "cuda")
    )

    # -------- Stats --------
    stats = {
        "epoch_loss": [],
        "avg_step_reward": [],
        "avg_final_reward": [],
    }

    # ==================================================
    # Training Loop
    # ==================================================
    for epoch in range(num_epochs):
        epoch_losses = []
        epoch_step_rewards = []
        epoch_final_rewards = []

        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 20:   # 控制 epoch 时间
                break

            tgt, tgt_timestamp, tgt_idx, ans = batch
            tgt = tgt.to(device)
            tgt_timestamp = tgt_timestamp.to(device)
            tgt_idx = tgt_idx.to(device)
            ans = ans.to(device)

            # -------- Collect trajectory --------
            states, actions, rewards, log_probs = trainer.collect_trajectory(
                tgt, tgt_timestamp, tgt_idx, ans
            )

            # -------- PPO Update --------
            loss = trainer.update_policy(states, actions, rewards, log_probs)
            epoch_losses.append(loss)

            # -------- Stats --------
            step_reward = torch.stack(rewards).mean().item()
            final_reward = env.compute_final_reward().mean().item()

            epoch_step_rewards.append(step_reward)
            epoch_final_rewards.append(final_reward)

            if batch_idx % 5 == 0:
                print(
                    f"[Epoch {epoch} | Batch {batch_idx}] "
                    f"Loss={loss:.4f}, "
                    f"StepReward={step_reward:.4f}, "
                    f"FinalReward={final_reward:.4f}"
                )

        # -------- Epoch summary --------
        stats["epoch_loss"].append(np.mean(epoch_losses))
        stats["avg_step_reward"].append(np.mean(epoch_step_rewards))
        stats["avg_final_reward"].append(np.mean(epoch_final_rewards))

        print(
            f"Epoch {epoch} Done | "
            f"Loss={stats['epoch_loss'][-1]:.4f}, "
            f"StepReward={stats['avg_step_reward'][-1]:.4f}, "
            f"FinalReward={stats['avg_final_reward'][-1]:.4f}"
        )

    print("强化学习训练完成")

    return policy_net, stats


# ======================================================
# Entry
# ======================================================
def run_training_with_pretrained_model(data_path="MOO", model_path=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-batch_size", type=int, default=16)
    parser.add_argument("-train_rate", type=float, default=0.8)
    parser.add_argument("-valid_rate", type=float, default=0.1)
    parser.add_argument("-no_cuda", action="store_true")
    opt = parser.parse_args([])

    # -------- Load data --------
    user_size, total_cascades, timestamps, train, valid, _ = Split_data(
        data_path, opt.train_rate, opt.valid_rate, load_dict=True
    )

    # -------- Graph --------
    graph = ConRelationGraph(data_path)
    hypergraph_list = ConHyperGraphList(total_cascades, timestamps, user_size)

    # -------- Base model --------
    mshgat = MSHGAT(opt)
    if model_path and os.path.exists(model_path):
        mshgat.load_state_dict(torch.load(model_path, map_location="cpu"))
    mshgat.eval()

    if torch.cuda.is_available() and not opt.no_cuda:
        mshgat = mshgat.cuda()

    # -------- Train --------
    policy_net, stats = train_rl_model(
        data_path=data_path,
        opt=opt,
        pretrained_model=mshgat,
        num_skills=user_size,
        batch_size=opt.batch_size,
        recommendation_length=5,
        num_epochs=20,
        topk=10,
        graph=graph,
        hypergraph_list=hypergraph_list,
    )

    print("训练完成")
    print(f"最终 Step Reward: {stats['avg_step_reward'][-1]:.4f}")
    print(f"最终 Final Reward: {stats['avg_final_reward'][-1]:.4f}")


if __name__ == "__main__":
    run_training_with_pretrained_model()
