# -*- coding: utf-8 -*-
"""
train_rl_new.py (fixed v10)

Fully aligned with your repo's run.py + dataLoader.py + HGAT.py.

Key fixes:
- DO NOT call rl.policy.train()/eval() before policy exists.
  RLPathOptimizer is lazy-init; use rl.ensure_initialized(...) once on the first batch.
- Use rl.collect_trajectory(...) -> dict, and rl.update_policy(dict) (PPO update).
"""

import os
import time
import argparse
import numpy as np
import torch

from dataLoader import Split_data, DataLoader
from graphConstruct import ConRelationGraph, ConHyperGraphList
from HGAT import MSHGAT

from rl_adjuster_new import RLPathOptimizer, evaluate_policy


def set_seed(seed: int = 0):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def _shuffle_cas(cas):
    # cas is (tgt, timestamp, idx, ans)
    if cas is None or (not isinstance(cas, (list, tuple))) or len(cas) != 4:
        return cas
    tgt, ts, idx, ans = cas
    n = len(tgt)
    if n <= 1:
        return cas
    perm = np.random.permutation(n)
    return ([tgt[i] for i in perm],
            [ts[i] for i in perm],
            [idx[i] for i in perm],
            [ans[i] for i in perm])


def _load_pretrained(model: torch.nn.Module, path: str, device: torch.device):
    if not path:
        return False
    if not os.path.exists(path):
        print(f"[WARN] not found: {path}  (training from random init is not recommended)")
        return False
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[WARN] missing keys:", missing[:20], ("..." if len(missing) > 20 else ""))
    if unexpected:
        print("[WARN] unexpected keys:", unexpected[:20], ("..." if len(unexpected) > 20 else ""))
    print(f"[OK] loaded pretrained base model from: {path}")
    return True


def build_args():
    p = argparse.ArgumentParser()

    # ===== same as run.py essentials =====
    p.add_argument("-data_name", default="Assist")
    p.add_argument("-batch_size", type=int, default=16)
    p.add_argument("-d_model", type=int, default=64)
    p.add_argument("-initialFeatureSize", type=int, default=64)
    p.add_argument("-train_rate", type=float, default=0.8)
    p.add_argument("-valid_rate", type=float, default=0.1)
    p.add_argument("-dropout", type=float, default=0.3)
    p.add_argument("-pos_emb", type=bool, default=True)

    p.add_argument("--pretrained_path", type=str, default="./checkpoint/DiffusionPrediction_a150.pt")

    # ===== RL/PPO =====
    p.add_argument("--rl_epochs", type=int, default=5)
    p.add_argument("--cand_k", type=int, default=50)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--history_T", type=int, default=10)
    p.add_argument("--rl_lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)

    return p.parse_args()


def train_one_epoch(rl: RLPathOptimizer, train_loader, graph, hypergraph_list, device):
    stats = {"policy_loss": [], "value_loss": [], "entropy": [], "total_loss": [],
             "final_quality": [], "effectiveness": [], "adaptivity": [], "diversity": []}

    for batch in train_loader:
        tgt, tgt_timestamp, tgt_idx, ans = batch
        tgt = tgt.to(device)
        tgt_timestamp = tgt_timestamp.to(device)
        tgt_idx = tgt_idx.to(device)
        ans = ans.to(device)

        # Init policy once
        if rl.policy is None:
            rl.ensure_initialized(tgt, tgt_timestamp, tgt_idx, ans, graph=graph, hypergraph_list=hypergraph_list)

        rl.policy.train()

        rollout = rl.collect_trajectory(tgt, tgt_timestamp, tgt_idx, ans, graph=graph, hypergraph_list=hypergraph_list)
        losses = rl.update_policy(rollout)

        fm = rollout["final_metrics"]
        stats["policy_loss"].append(losses["policy_loss"])
        stats["value_loss"].append(losses["value_loss"])
        stats["entropy"].append(losses["entropy"])
        stats["total_loss"].append(losses["total_loss"])
        stats["final_quality"].append(float(fm["final_quality"].mean().detach().cpu()))
        stats["effectiveness"].append(float(fm["effectiveness"].mean().detach().cpu()))
        stats["adaptivity"].append(float(fm["adaptivity"].mean().detach().cpu()))
        stats["diversity"].append(float(fm["diversity"].mean().detach().cpu()))

    return {k: float(np.mean(v)) if v else 0.0 for k, v in stats.items()}


def main():
    args = build_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    user_size, total_cascades, timestamps, train, valid, test = Split_data(
        args.data_name, args.train_rate, args.valid_rate, load_dict=True
    )

    relation_graph = ConRelationGraph(args.data_name)
    hypergraph_list = ConHyperGraphList(total_cascades, timestamps, user_size)

    args.d_word_vec = args.d_model
    args.user_size = user_size
    base_model = MSHGAT(args, dropout=args.dropout).to(device)
    _load_pretrained(base_model, args.pretrained_path, device)

    rl = RLPathOptimizer(
        base_model=base_model,
        num_items=user_size,
        data_name=args.data_name,
        device=device,
        pad_val=0,
        topk=args.topk,
        cand_k=args.cand_k,
        history_window_T=args.history_T,
        rl_lr=args.rl_lr,
    )

    best_val = -1e18
    for epoch in range(args.rl_epochs):
        print(f"\n[RL Epoch {epoch}]")

        train_shuffled = _shuffle_cas(train)
        train_loader = DataLoader(train_shuffled, batch_size=args.batch_size, load_dict=True, cuda=(device.type == "cuda"))

        t0 = time.time()
        tr = train_one_epoch(rl, train_loader, relation_graph, hypergraph_list, device)
        print(f"  train: {tr}  (elapsed {(time.time() - t0) / 60:.2f} min)")

        valid_loader = DataLoader(valid, batch_size=args.batch_size, load_dict=True, cuda=(device.type == "cuda"), test=True)
        val = evaluate_policy(rl=rl, data_loader=valid_loader, graph=relation_graph, hypergraph_list=hypergraph_list, device=device, compute_all=True)
        print("  valid:", val)

        if val.get("final_quality", -1e18) > best_val:
            best_val = val["final_quality"]
            torch.save({"policy": rl.policy.state_dict()}, "./checkpoint/rl_policy_best.pt")
            print("  [OK] saved best RL policy to ./checkpoint/rl_policy_best.pt")

    test_loader = DataLoader(test, batch_size=args.batch_size, load_dict=True, cuda=(device.type == "cuda"), test=True)
    te = evaluate_policy(rl=rl, data_loader=test_loader, graph=relation_graph, hypergraph_list=hypergraph_list, device=device)
    print("\n[Test]", te)


if __name__ == "__main__":
    main()
