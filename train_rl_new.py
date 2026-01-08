
"""
train_rl.py
强化学习路径优化训练脚本（适配 rl_adjuster_new.py）
"""
import os
import argparse
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt

from HGAT import MSHGAT
from dataLoader import Split_data, DataLoader
from graphConstruct import ConRelationGraph, ConHyperGraphList

from rl_adjuster_new_fixed import RLPathOptimizer, evaluate_policy


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
    device=None,
):
    if device is None:
        device = torch.device("cuda" if (not opt.no_cuda and torch.cuda.is_available()) else "cpu")

    print("开始强化学习训练（RLPathOptimizer）")

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
        metrics_topnum=topk,   # ✅ 与 policy_topk 对齐
    )

    # data
    user_size, total_cascades, timestamps, train, valid, test = Split_data(
        data_path, opt.train_rate, opt.valid_rate, load_dict=True
    )
    train_loader = DataLoader(train, batch_size=batch_size, load_dict=True, cuda=(device.type == "cuda"))
    valid_loader = DataLoader(valid, batch_size=batch_size, load_dict=True, cuda=(device.type == "cuda"))

    stats = {"epoch_loss": [], "avg_step_reward": [], "avg_final_reward": [], "valid_final_quality": []}

    for epoch in range(num_epochs):
        losses = []
        step_rs = []
        final_rs = []

        for bidx, batch in enumerate(train_loader):
            if bidx >= 20:
                break
            tgt, tgt_timestamp, tgt_idx, ans = batch
            tgt = tgt.to(device); tgt_timestamp = tgt_timestamp.to(device); tgt_idx = tgt_idx.to(device); ans = ans.to(device)

            rewards, log_probs, entropies, final_reward = rl.trainer.collect_trajectory(
                tgt, tgt_timestamp, tgt_idx, ans,
                graph=graph, hypergraph_list=hypergraph_list,
                deterministic=False
            )
            loss = rl.trainer.update_policy(rewards, log_probs, entropies)
            losses.append(loss)

            step_rs.append(torch.stack(rewards).mean().item())
            final_rs.append(final_reward.mean().item())

            if bidx % 5 == 0:
                print(f"[Epoch {epoch} | Batch {bidx}] Loss={loss:.4f}, StepReward={step_rs[-1]:.4f}, FinalReward={final_rs[-1]:.4f}")

        stats["epoch_loss"].append(float(np.mean(losses) if losses else 0.0))
        stats["avg_step_reward"].append(float(np.mean(step_rs) if step_rs else 0.0))
        stats["avg_final_reward"].append(float(np.mean(final_rs) if final_rs else 0.0))

        # quick eval
        eff, div, ada, pref, fq = evaluate_policy(rl.env, rl.policy_net, valid_loader, relation_graph=graph, hypergraph_list=hypergraph_list, num_episodes=3)
        stats["valid_final_quality"].append(fq)

        print(f"Epoch {epoch} Done | Loss={stats['epoch_loss'][-1]:.4f}, StepReward={stats['avg_step_reward'][-1]:.4f}, FinalReward={stats['avg_final_reward'][-1]:.4f}")
        print("\n[Evaluation Metrics on Recommended Paths]")
        print(f"  effectiveness: {eff:.4f}")
        print(f"  adaptivity:    {ada:.4f}")
        print(f"  diversity:     {div:.4f}")
        print(f"  preference:    {pref:.4f}")
        print(f"  final_quality: {fq:.4f}\n")

    # save
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(rl.policy_net.state_dict(), "checkpoints/rl_policy_net.pth")
    with open("checkpoints/rl_training_stats.pkl", "wb") as f:
        pickle.dump(stats, f)

    return rl.policy_net, rl, stats


def run_training_with_pretrained_model(data_path="MOO", model_path="./checkpoint/DiffusionPrediction.pt"):
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_name", default=data_path)
    parser.add_argument("-batch_size", type=int, default=16)
    parser.add_argument("-d_model", type=int, default=64)
    parser.add_argument("-initialFeatureSize", type=int, default=64)
    parser.add_argument("-train_rate", type=float, default=0.8)
    parser.add_argument("-valid_rate", type=float, default=0.1)
    parser.add_argument("-dropout", type=float, default=0.3)
    parser.add_argument("-no_cuda", action="store_true")
    opt = parser.parse_args([])

    # ✅ run.py 里需要的字段
    opt.d_word_vec = opt.d_model

    user_size, total_cascades, timestamps, train, valid, test = Split_data(opt.data_name, opt.train_rate, opt.valid_rate, load_dict=True)
    opt.user_size = user_size

    relation_graph = ConRelationGraph(opt.data_name)
    hypergraph_list = ConHyperGraphList(total_cascades, timestamps, user_size)

    print("尝试加载预训练模型.")
    mshgat = MSHGAT(opt, dropout=opt.dropout)
    if os.path.exists(model_path):
        if torch.cuda.is_available() and (not opt.no_cuda):
            mshgat.load_state_dict(torch.load(model_path))
            mshgat = mshgat.cuda()
            print(f"已从 {model_path} 加载预训练权重\n模型已移动到 GPU")
        else:
            mshgat.load_state_dict(torch.load(model_path, map_location="cpu"))
            print(f"已从 {model_path} 加载预训练权重")
    else:
        print(f"未找到预训练权重: {model_path}，将使用随机初始化（不建议）")
        if torch.cuda.is_available() and (not opt.no_cuda):
            mshgat = mshgat.cuda()

    mshgat.eval()

    num_skills = user_size

    policy_net, rl, stats = train_rl_model(
        data_path=opt.data_name,
        opt=opt,
        pretrained_model=mshgat,
        num_skills=num_skills,
        batch_size=opt.batch_size,
        recommendation_length=5,
        num_epochs=5,
        topk=10,
        graph=relation_graph,
        hypergraph_list=hypergraph_list
    )
    return policy_net, rl, stats


if __name__ == "__main__":
    run_training_with_pretrained_model()
