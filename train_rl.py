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
import Constants
from torch.distributions import Categorical

def evaluate_and_print(policy_net, env, data_loader, device, max_batches=10):
    policy_net.eval()
    all_eff, all_ada, all_div, all_pref, all_fq = [], [], [], [], []

    with torch.no_grad():
        for b_idx, batch in enumerate(data_loader):
            if b_idx >= max_batches:
                break
            tgt, tgt_timestamp, tgt_idx, ans = (x.to(device) for x in batch)

            # 用确定性策略跑一条轨迹
            rewards, log_probs, entropies = env_runner(policy_net, env, tgt, tgt_timestamp, tgt_idx, ans, deterministic=True)

            m = env.compute_final_metrics()
            all_eff.append(m["effectiveness"].detach().cpu())
            all_ada.append(m["adaptivity"].detach().cpu())
            all_div.append(m["diversity"].detach().cpu())
            all_pref.append(m["preference"].detach().cpu())
            all_fq.append(m["final_quality"].detach().cpu())

    def _cat_mean(xs):
        x = torch.cat(xs) if xs else torch.tensor([0.0])
        return float(x.mean().item())

    print("\n[Evaluation Metrics on Recommended Paths]")
    print(f"  effectiveness: {_cat_mean(all_eff):.4f}")
    print(f"  adaptivity:    {_cat_mean(all_ada):.4f}")
    print(f"  diversity:     {_cat_mean(all_div):.4f}")
    print(f"  preference:    {_cat_mean(all_pref):.4f}")
    print(f"  final_quality: {_cat_mean(all_fq):.4f}")

    policy_net.train()


def env_runner(policy_net, env, tgt, tgt_timestamp, tgt_idx, ans, deterministic=True):
    # 复用 PPOTrainer.collect_trajectory 的逻辑，但不更新参数
    # 这里直接实例化一个临时 trainer 也行；为了最少改动，我们写个轻量 runner
    state = env.reset(tgt, tgt_timestamp, tgt_idx, ans)
    rewards, log_probs, entropies = [], [], []
    B = tgt.size(0)
    device = tgt.device

    for _ in range(env.recommendation_length):
        cand_probs = state["cand_probs"]
        cand_ids = state["cand_ids"]
        cand_feat = cand_probs.unsqueeze(-1)

        logits = policy_net(state["knowledge_state"], cand_feat)
        dist = Categorical(logits=logits)
        action_index = torch.argmax(logits, dim=-1) if deterministic else dist.sample()

        log_probs.append(dist.log_prob(action_index))
        entropies.append(dist.entropy())

        chosen_item = cand_ids.gather(1, action_index.view(-1,1)).squeeze(1)

        topn = env.metrics_topnum
        step_topk = torch.full((B, topn), Constants.PAD, device=device, dtype=cand_ids.dtype)
        step_topk[:, 0] = chosen_item
        for k in range(1, topn):
            alt = cand_ids[:, k] if k < cand_ids.size(1) else cand_ids[:, 0]
            if k < cand_ids.size(1) - 1:
                alt2 = cand_ids[:, k+1]
                alt = torch.where(alt == chosen_item, alt2, alt)
            step_topk[:, k] = alt

        state, step_reward, done = env.step(chosen_item, step_topk)
        rewards.append(step_reward)

    return rewards, log_probs, entropies


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
        policy_topk=topk,  #  确保 env 候选数与 policy.topk 一致
        metrics_topnum=2,
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

            rewards, log_probs, entropies = trainer.collect_trajectory(tgt, tgt_timestamp, tgt_idx, ans)
            loss = trainer.update_policy(rewards, log_probs, entropies)

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
    evaluate_and_print(policy_net, env, valid, device, max_batches=20)

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

    #  关键：MSHGAT 需要 d_word_vec
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


