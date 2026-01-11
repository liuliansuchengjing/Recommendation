
# -*- coding: utf-8 -*-
"""
train_rl_new_fixed.py

This script trains the RL policy with the reworked rl_adjuster_new_fixed.py:
- Online RL at every valid time step (PAD masked)
- PPO (actor-critic, GAE, clipping)
- Final quality strictly aligned with Eq.(19)(20)(21)
"""

import os
import argparse
import torch

from HGAT import MSHGAT
from dataLoader import Split_data, DataLoader
from graphConstruct import ConRelationGraph, ConHyperGraphList

from rl_adjuster_new import RLPathOptimizer, PPOConfig, evaluate_policy


def run_training_with_pretrained_model():
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_name", type=str, default="assist2009")
    parser.add_argument("-pretrained_path", type=str, default="saved_model.pth")

    parser.add_argument("-batch_size", type=int, default=16)
    parser.add_argument("-d_model", type=int, default=64)
    parser.add_argument("-initialFeatureSize", type=int, default=64)
    parser.add_argument("-train_rate", type=float, default=0.8)
    parser.add_argument("-valid_rate", type=float, default=0.1)
    parser.add_argument("-dropout", type=float, default=0.3)
    parser.add_argument("-no_cuda", action="store_true")

    # RL / PPO
    parser.add_argument("-topk", type=int, default=10)
    parser.add_argument("-cand_k", type=int, default=50)
    parser.add_argument("-history_T", type=int, default=10)
    parser.add_argument("-epochs", type=int, default=5)
    parser.add_argument("-rl_lr", type=float, default=3e-4)
    parser.add_argument("-ppo_epochs", type=int, default=4)
    parser.add_argument("-minibatch_size", type=int, default=256)
    parser.add_argument("-clip_eps", type=float, default=0.2)
    parser.add_argument("-gamma", type=float, default=0.99)
    parser.add_argument("-gae_lambda", type=float, default=0.95)
    parser.add_argument("-ent_coef", type=float, default=0.01)
    parser.add_argument("-vf_coef", type=float, default=0.5)
    parser.add_argument("-terminal_scale", type=float, default=1.0)

    opt = parser.parse_args([])  # keep your original "no CLI" behavior

    # ✅ run.py expects:
    opt.d_word_vec = opt.d_model

    # split data
    user_size, total_cascades, timestamps, train, valid, test = Split_data(
        opt.data_name, opt.train_rate, opt.valid_rate, load_dict=True
    )
    opt.user_size = user_size

    device = torch.device("cuda" if torch.cuda.is_available() and (not opt.no_cuda) else "cpu")
    print("device =", device)

    # dataloaders
    train_loader = DataLoader(train, batch_size=opt.batch_size, cuda=(device.type == "cuda"))
    valid_loader = DataLoader(valid, batch_size=opt.batch_size, cuda=(device.type == "cuda"))

    # graphs
    relation_graph = ConRelationGraph(opt.data_name)
    hypergraph_list = ConHyperGraphList(total_cascades, timestamps, user_size)

    # load base model
    model_path = opt.pretrained_path
    print("Loading base model:", model_path)
    mshgat = MSHGAT(opt, dropout=opt.dropout)
    if os.path.exists(model_path):
        mshgat.load_state_dict(torch.load(model_path, map_location="cpu"))
        print("Loaded pretrained weights.")
    else:
        print(f"[WARN] not found: {model_path}  (training from random init is not recommended)")
    mshgat = mshgat.to(device)
    mshgat.eval()

    num_items = user_size  # your project uses this as node count

    # PPO config
    ppo_cfg = PPOConfig(
        gamma=opt.gamma,
        gae_lambda=opt.gae_lambda,
        clip_eps=opt.clip_eps,
        vf_coef=opt.vf_coef,
        ent_coef=opt.ent_coef,
        ppo_epochs=opt.ppo_epochs,
        minibatch_size=opt.minibatch_size,
    )

    # RL optimizer
    rl = RLPathOptimizer(
        base_model=mshgat,
        num_items=num_items,
        data_name=opt.data_name,
        device=device,
        pad_val=0,
        topk=opt.topk,
        cand_k=opt.cand_k,
        history_window_T=opt.history_T,
        rl_lr=opt.rl_lr,
        ppo_config=ppo_cfg,
        terminal_reward_scale=opt.terminal_scale,
        # step reward weights (you can tune later)
        step_reward_weights={
            "preference": 1.0,
            "adaptivity": 1.0,
            "novelty": 0.2,
        },
        # final weights for Eq.(19)(20)(21)
        final_reward_weights={
            "effectiveness": 1.0,
            "adaptivity": 1.0,
            "diversity": 1.0,
        }
    )

    # training loop
    for epoch in range(1, opt.epochs + 1):
        rl.policy.train()

        epoch_losses = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "total_loss": 0.0}
        epoch_metrics = {"effectiveness": 0.0, "adaptivity": 0.0, "diversity": 0.0, "final_quality": 0.0}
        n_batches = 0

        for batch in train_loader:
            tgt, tgt_timestamp, tgt_idx, ans = batch[0], batch[1], batch[2], batch[3]
            tgt = tgt.to(device)
            tgt_timestamp = tgt_timestamp.to(device)
            tgt_idx = tgt_idx.to(device)
            ans = ans.to(device)

            rollout = rl.collect_trajectory(
                tgt, tgt_timestamp, tgt_idx, ans,
                graph=relation_graph, hypergraph_list=hypergraph_list
            )
            losses = rl.update_policy(rollout)

            for k in epoch_losses:
                epoch_losses[k] += float(losses.get(k, 0.0))

            fm = rollout["final_metrics"]
            for k in epoch_metrics:
                epoch_metrics[k] += float(fm[k].mean().detach().cpu())

            n_batches += 1

        for k in epoch_losses:
            epoch_losses[k] /= max(1, n_batches)
        for k in epoch_metrics:
            epoch_metrics[k] /= max(1, n_batches)

        print(f"\n[Epoch {epoch}] losses={epoch_losses}")
        print(f"[Epoch {epoch}] train_metrics={epoch_metrics}")

        # validation (few batches)
        rl.policy.eval()
        val_metrics = evaluate_policy(
            rl=rl,
            data_loader=valid_loader,
            graph=relation_graph,
            hypergraph_list=hypergraph_list,
            device=device,
            max_batches=20
        )
        print(f"[Epoch {epoch}] valid_metrics={val_metrics}\n")

    return rl


if __name__ == "__main__":
    run_training_with_pretrained_model()
