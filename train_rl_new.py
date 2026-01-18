# -*- coding: utf-8 -*-
"""
train_rl_new.py (fixed v10)

Fully aligned with your repo's run.py + dataLoader.py + HGAT.py.

Key fixes:
- DO NOT call rl.policy.train()/eval() before policy exists.
  RLPathOptimizer is lazy-init; use rl.ensure_initialized(...) once on the first batch.
- Use rl.collect_trajectory(...) -> dict, and rl.update_policy(dict) (PPO update).

张量维度说明：
- [B]: Batch size (批处理大小)
- [N]: Number of items/questions (物品/问题总数)
- [L]: Sequence length (序列长度)
- [d]: Embedding dimension (嵌入维度)
- [K]: Top-k/K candidates (候选项目数量)
- [T]: Time steps (时间步数)
"""

import os
import time
import argparse
import numpy as np
import torch

from dataLoader import Split_data, DataLoader
from graphConstruct import ConRelationGraph, ConHyperGraphList
from HGAT import MSHGAT

# from rl_adjuster_new import RLPathOptimizer, evaluate_policy
from rl_adjuster_new_path_metrics import RLPathOptimizer, evaluate_policy
from rl_eval_metrics import evaluate_policy_with_ranking_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    p.add_argument("--cand_k", type=int, default=5)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--history_T", type=int, default=5)
    p.add_argument("--rl_lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)

    return p.parse_args()


def train_one_epoch(rl: RLPathOptimizer, train_loader, graph, hypergraph_list, device):
    """
    训练一个epoch的数据

    Args:
        rl: RLPathOptimizer实例
        train_loader: 训练数据加载器
        graph: 关系图
        hypergraph_list: 超图列表
        device: 计算设备

    Returns:
        stats: 包含各种训练指标的字典
               - policy_loss: 策略损失 [scalar]
               - value_loss: 价值函数损失 [scalar]
               - entropy: 熵值 [scalar]
               - total_loss: 总损失 [scalar]
               - final_quality: 最终质量 [scalar]
               - effectiveness: 有效性 [scalar]
               - adaptivity: 适应性 [scalar]
               - diversity: 多样性 [scalar]
    """
    stats = {"policy_loss": [], "value_loss": [], "entropy": [], "total_loss": [],
             "final_quality": [], "effectiveness": [], "adaptivity": [], "diversity": []}

    for batch in train_loader:
        # batch包含以下张量：
        # tgt: 目标序列 [B, L] - 每个批次的序列
        # tgt_timestamp: 时间戳 [B, L] - 每个序列的时间戳
        # tgt_idx: 索引 [B] - 每个序列的级联ID
        # ans: 答案 [B, L] - 每个交互的正确性标签
        tgt, tgt_timestamp, tgt_idx, ans = batch
        tgt = tgt.to(device)  # [B, L] - 目标序列，批量大小×序列长度
        tgt_timestamp = tgt_timestamp.to(device)  # [B, L] - 时间戳张量
        tgt_idx = tgt_idx.to(device)  # [B] - 级联索引张量
        ans = ans.to(device)  # [B, L] - 答案张量，标记每个交互的正确性

        # 初始化策略一次
        if rl.policy is None:
            # 确保策略初始化，输入张量维度：
            # tgt: [B, L] - 序列数据
            # tgt_timestamp: [B, L] - 时间戳数据
            # tgt_idx: [B] - 级联ID
            # ans: [B, L] - 正确性标签
            rl.ensure_initialized(tgt, tgt_timestamp, tgt_idx, ans, graph=graph, hypergraph_list=hypergraph_list)

        rl.policy.train()

        # 收集轨迹，返回字典包含：
        # - cand_feat: 候选特征 [N*K, F] - N为总时间步，K为候选数，F为特征维数
        # - actions: 动作 [N*K] - 每个时间步选择的动作索引
        # - old_logp: 旧对数概率 [N*K] - 旧策略的概率
        # - old_values: 旧价值 [N*K] - 旧价值估计
        # - advantages: 优势 [N*K] - 优势函数值
        # - returns: 回报 [N*K] - 累积回报
        # - final_metrics: 最终指标字典，包含effectiveness, adaptivity, diversity等 [B]
        rollout = rl.collect_trajectory(tgt, tgt_timestamp, tgt_idx, ans, graph=graph, hypergraph_list=hypergraph_list)

        # 更新策略，返回损失字典：
        # - policy_loss: 策略损失标量
        # - value_loss: 价值损失标量
        # - entropy: 熵损失标量
        # - total_loss: 总损失标量
        losses = rl.update_policy(rollout)

        fm = rollout["final_metrics"]  # 最终评估指标字典
        stats["policy_loss"].append(losses["policy_loss"])  # 策略损失列表
        stats["value_loss"].append(losses["value_loss"])  # 价值损失列表
        stats["entropy"].append(losses["entropy"])  # 熵值列表
        stats["total_loss"].append(losses["total_loss"])  # 总损失列表
        # 最终质量平均值 [scalar]
        stats["final_quality"].append(float(fm["final_quality"].mean().detach().cpu()))
        # 有效性平均值 [scalar]
        stats["effectiveness"].append(float(fm["effectiveness"].mean().detach().cpu()))
        # 适应性平均值 [scalar]
        stats["adaptivity"].append(float(fm["adaptivity"].mean().detach().cpu()))
        # 多样性平均值 [scalar]
        stats["diversity"].append(float(fm["diversity"].mean().detach().cpu()))

    # 对所有统计值取平均
    return {k: float(np.mean(v)) if v else 0.0 for k, v in stats.items()}


def main():
    """
    主训练函数
    """
    args = build_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    # 数据加载返回：
    # user_size: 用户/物品总数 [scalar]
    # total_cascades: 总级联数 [list of lists]
    # timestamps: 时间戳 [list of lists]
    # train, valid, test: 训练/验证/测试集，格式为(tgt, tgt_timestamp, tgt_idx, ans)
    user_size, total_cascades, timestamps, train, valid, test = Split_data(
        args.data_name, args.train_rate, args.valid_rate, load_dict=True
    )

    # 构建关系图和超图
    relation_graph = ConRelationGraph(args.data_name)  # 关系图结构
    hypergraph_list = ConHyperGraphList(total_cascades, timestamps, user_size)  # 超图列表

    args.d_word_vec = args.d_model  # 词向量维度
    args.user_size = user_size  # 用户/物品总数
    # 基础模型MSHGAT，接收参数并构建：
    # - 输入: [B, L] 序列，[B, L] 时间戳，[B] 级联ID，[B, L] 答案
    # - 输出: 预测概率分布和其他状态信息
    base_model = MSHGAT(args, dropout=args.dropout).to(device)
    _load_pretrained(base_model, args.pretrained_path, device)

    # 创建RL路径优化器
    # 参数说明：
    # - base_model: 基础模型
    # - num_items: 物品总数 [scalar] - 用于张量维度
    # - data_name: 数据集名称
    # - device: 计算设备
    # - topk: 每步记录的top-k推荐 [scalar] - 影响最终评估的推荐路径
    # - cand_k: 候选集大小 [scalar] - 每步可选项目的数量
    # - history_window_T: 历史窗口大小 [scalar] - 用于计算适应性指标
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
        target_future_M=3,

        # 固定长度路径 + 方案1：每个时间步做规划，但限制次数
        horizon_H=10,
        min_start=5,
        max_starts_per_seq=5,

        # ===== 终止奖励消融开关（训练用）=====
        # 默认全开；要做 w/o 某项就把它设为 False
        terminal_reward_components={
            "effectiveness": True,
            "adaptivity": False,
            "diversity": False,
        },

        # 训练时：只计算“开着的”终止指标（关掉的就不算，省时间，也符合你的要求）
        train_compute_all_terminal_metrics=False,
    )

    best_val = -1e18  # 最佳验证分数
    for epoch in range(args.rl_epochs):  # RL训练轮次
        print(f"\\n[RL Epoch {epoch}]")

        train_shuffled = _shuffle_cas(train)  # 打乱训练数据
        # 创建训练数据加载器，每个batch输出：
        # - tgt: [B, L] 目标序列
        # - tgt_timestamp: [B, L] 时间戳
        # - tgt_idx: [B] 级联ID
        # - ans: [B, L] 答案标签
        train_loader = DataLoader(train_shuffled, batch_size=args.batch_size, load_dict=True,
                                  cuda=(device.type == "cuda"))

        t0 = time.time()
        # 训练一个epoch，返回平均指标
        tr = train_one_epoch(rl, train_loader, relation_graph, hypergraph_list, device)
        print(f"  train: {tr}  (elapsed {(time.time() - t0) / 60:.2f} min)")

        # 验证集评估
        valid_loader = DataLoader(valid, batch_size=args.batch_size, load_dict=True, cuda=(device.type == "cuda"),
                                  test=True)
        # 评估策略性能，返回：
        # - final_quality: 最终质量 [scalar]
        # - effectiveness: 有效性 [scalar]
        # - adaptivity: 适应性 [scalar]
        # - diversity: 多样性 [scalar]
        val = evaluate_policy(rl=rl, data_loader=valid_loader, graph=relation_graph, hypergraph_list=hypergraph_list,
                              device=device)
        print("  valid:", val)

        # 保存最佳模型
        if val.get("final_quality", -1e18) > best_val:
            best_val = val["final_quality"]
            # 保存策略参数
            torch.save({"policy": rl.policy.state_dict()}, "./checkpoint/A_rl_policy.pt")
            print("  [OK] saved best RL policy to ./checkpoint/A_rl_policy.pt")

    # 测试集评估
    test_loader = DataLoader(test, batch_size=args.batch_size, load_dict=True, cuda=(device.type == "cuda"), test=True)
    te = evaluate_policy(rl=rl, data_loader=test_loader, graph=relation_graph, hypergraph_list=hypergraph_list,
                         device=device)
    print("\\n[Test]", te)


def test_rl_like_training(checkpoint_path: str = "./checkpoint/A_rl_policy.pt", eval_split: str = "test"):
    args = build_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    user_size, total_cascades, timestamps, train_data, valid_data, test_data = Split_data(
        args.data_name, args.train_rate, args.valid_rate, load_dict=True
    )

    relation_graph = ConRelationGraph(args.data_name)
    hypergraph_list = ConHyperGraphList(total_cascades, timestamps, user_size)

    eval_data = valid_data if eval_split.lower() == "valid" else test_data
    eval_loader = DataLoader(eval_data, batch_size=args.batch_size, load_dict=True,
                             cuda=(device.type == "cuda"), test=True)

    args.d_word_vec = args.d_model
    args.user_size = user_size
    base_model = MSHGAT(args, dropout=args.dropout).to(device)
    _load_pretrained(base_model, args.pretrained_path, device)
    base_model.eval()

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
        target_future_M=3,
        horizon_H=10,
        min_start=5,
        max_starts_per_seq=5,  # 训练里是 5
        terminal_reward_components={"effectiveness": True, "adaptivity": False, "diversity": False},
        train_compute_all_terminal_metrics=False,
    )

    # lazy init
    first_batch = next(iter(eval_loader))
    tgt, ts, idx, ans = (x.to(device) for x in first_batch)
    rl.ensure_initialized(tgt, ts, idx, ans, graph=relation_graph, hypergraph_list=hypergraph_list)

    # 1) base top1 baseline
    base_top1 = evaluate_base_top1_terminal_like_training(
        rl, eval_loader, relation_graph, hypergraph_list, device,
        horizon_H=10, min_start=5, max_starts_per_seq=1
    )
    print("[BASE top1 terminal]", base_top1)

    # 2) load RL policy
    ckpt = torch.load(checkpoint_path, map_location=device)
    rl.policy.load_state_dict(ckpt["policy"])
    rl.policy.eval()

    # 3) policy terminal (训练同款)
    policy_metrics = evaluate_policy(
        rl=rl, data_loader=eval_loader,
        graph=relation_graph, hypergraph_list=hypergraph_list, device=device
    )
    print("[POLICY terminal]", policy_metrics)


@torch.no_grad()
def evaluate_base_top1_paths(
    base_model,
    data_loader,
    graph,
    hypergraph_list,
    device,
    k_list=(1, 3, 5, 10),
    m_list=(3, 5, 7, 9),
    pad_id=0,
    skip_id=1,
    min_start=5,
    scheme="windows",   # "prefix" or "windows"
    max_batches=10**9,
):
    """
    Base(无RL)评估：每个时间步只取 top1 作为推荐。
    - next-item 指标：Hits@K, NDCG@K（逐步预测 x[t] -> x[t+1]）
    - path-level 指标：P/R/F1@m（把长度m的预测路径当集合，与真实路径集合算交集）

    scheme:
      - "prefix": 仅用每条序列最前 m 步（类似你run.py里prf_path_level_from_logits那种“前m步”）
      - "windows": 在每条序列上滑窗取多个起点，平均（更像你后来想对齐论文那种）
    """

    base_model.eval()

    # -------- next-item accumulators --------
    hit_sum = {k: 0.0 for k in k_list}
    ndcg_sum = {k: 0.0 for k in k_list}
    denom_next = 0.0  # 有效预测步数（排除PAD）

    # -------- path-level accumulators (micro over windows) --------
    # 用 micro：先累TP/FP/FN，再算P/R/F1
    TP = {m: 0 for m in m_list}
    FP = {m: 0 for m in m_list}
    FN = {m: 0 for m in m_list}
    win_cnt = {m: 0 for m in m_list}

    def _clean_list(x_list):
        return [int(x) for x in x_list if int(x) != pad_id and int(x) != skip_id]

    def _update_prf_micro(true_ids, pred_ids, m):
        true_set = set(_clean_list(true_ids))
        pred_set = set(_clean_list(pred_ids))
        if len(true_set) == 0:
            return  # 跳过空真值窗口
        inter = true_set & pred_set
        TP[m] += len(inter)
        FP[m] += len(pred_set - inter)
        FN[m] += len(true_set - inter)
        win_cnt[m] += 1

    for b_idx, batch in enumerate(data_loader):
        if b_idx >= max_batches:
            break

        tgt, tgt_timestamp, tgt_idx, ans = batch
        tgt = tgt.to(device)
        tgt_timestamp = tgt_timestamp.to(device)
        tgt_idx = tgt_idx.to(device)
        ans = ans.to(device)

        # forward 得到 logits（pred）
        out = base_model(tgt, tgt_timestamp, tgt_idx, ans, graph, hypergraph_list)

        # 兼容不同forward返回
        pred_logits = out[0] if isinstance(out, (tuple, list)) else out

        B, T = tgt.size()
        # pred_logits 一般是 (B*(T-1), V) 或 (B*T, V)；用B推断长度
        V = pred_logits.size(-1)
        t_pred = pred_logits.size(0) // B
        pred_logits = pred_logits.view(B, t_pred, V)

        # gold 对齐到 t_pred 长度：预测步 t 对应真实 tgt[:, t+1]
        gold = tgt[:, 1:1 + t_pred]  # (B, t_pred)

        # ---------- next-item Hits/NDCG ----------
        for k in k_list:
            topk_ids = torch.topk(pred_logits, k=k, dim=-1).indices  # (B, t_pred, k)

            # mask PAD
            valid_mask = gold.ne(pad_id)
            denom_next += valid_mask.sum().item()

            # hits
            hit = (topk_ids == gold.unsqueeze(-1)).any(dim=-1)  # (B, t_pred)
            hit_sum[k] += (hit & valid_mask).sum().item()

            # ndcg: 相关项只有1个，若命中则 1/log2(rank+2)
            if k > 0:
                # 找rank（没命中则rank=-1）
                eq = (topk_ids == gold.unsqueeze(-1))  # (B,t_pred,k)
                # argmax 只能找第一个True；若全False，会返回0，所以要额外mask
                first_pos = eq.float().argmax(dim=-1)  # (B,t_pred)
                is_hit = eq.any(dim=-1)  # (B,t_pred)

                # rank从0开始 -> 分母 log2(rank+2)
                dcg = torch.zeros_like(first_pos, dtype=torch.float, device=device)
                dcg[is_hit] = 1.0 / torch.log2(first_pos[is_hit].float() + 2.0)

                ndcg_sum[k] += (dcg * valid_mask.float()).sum().item()

        # ---------- path-level PRF@m ----------
        # 先取 top1 预测序列 (B, t_pred)
        pred_top1 = pred_logits.argmax(dim=-1)

        for m in m_list:
            if scheme == "prefix":
                # 用每条序列的前m步（预测步0..m-1 对应 gold步0..m-1）
                for bi in range(B):
                    true_ids = gold[bi, :m].tolist()
                    pred_ids = pred_top1[bi, :m].tolist()
                    _update_prf_micro(true_ids, pred_ids, m)

            elif scheme == "windows":
                # 滑窗：窗口起点 s 表示真实未来片段 gold[ s : s+m ]
                # 对应预测片段 pred_top1[ s : s+m ]（因为 pred_top1[t] 预测的是 gold[t]）
                for bi in range(B):
                    # 有效长度（到第一个PAD为止）
                    g = gold[bi].tolist()
                    try:
                        eff_len = g.index(pad_id)
                    except ValueError:
                        eff_len = len(g)

                    # eff_len 是 gold 的有效步数；需要 s+m <= eff_len
                    # min_start 是按“原始tgt位置”理解的，你这里 gold 对应 tgt[:,1:]，所以直接用 min_start-1 做对齐
                    s0 = max(0, min_start - 1)
                    for s in range(s0, eff_len - m + 1):
                        true_ids = g[s:s + m]
                        pred_ids = pred_top1[bi, s:s + m].tolist()
                        # 若窗口内真实长度不足m（含PAD），跳过
                        if pad_id in true_ids:
                            continue
                        _update_prf_micro(true_ids, pred_ids, m)
            else:
                raise ValueError("scheme must be 'prefix' or 'windows'")

    # -------- finalize --------
    results = {}

    denom_next = max(denom_next, 1.0)
    for k in k_list:
        results[f"base_hits@{k}"] = hit_sum[k] / denom_next
        results[f"base_NDCG@{k}"] = ndcg_sum[k] / denom_next

    for m in m_list:
        tp, fp, fn = TP[m], FP[m], FN[m]
        P = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        R = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
        results[f"base_P@{m}"] = P
        results[f"base_R@{m}"] = R
        results[f"base_F1@{m}"] = F1
        results[f"base_windows@{m}"] = win_cnt[m]

    return results


def test_base_top1(eval_split="test", scheme="windows"):
    args = build_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    user_size, total_cascades, timestamps, train_data, valid_data, test_data = Split_data(
        args.data_name, args.train_rate, args.valid_rate, load_dict=True
    )

    relation_graph = ConRelationGraph(args.data_name)
    hypergraph_list = ConHyperGraphList(total_cascades, timestamps, user_size)

    eval_data = valid_data if eval_split.lower() == "valid" else test_data
    eval_loader = DataLoader(
        eval_data,
        batch_size=args.batch_size,
        load_dict=True,
        cuda=(device.type == "cuda"),
        test=True
    )

    args.d_word_vec = args.d_model
    args.user_size = user_size
    base_model = MSHGAT(args, dropout=args.dropout).to(device)
    _load_pretrained(base_model, args.pretrained_path, device)
    base_model.eval()

    res = evaluate_base_top1_paths(
        base_model=base_model,
        data_loader=eval_loader,
        graph=relation_graph,
        hypergraph_list=hypergraph_list,
        device=device,
        k_list=(1, 3, 5, 10),
        m_list=(3, 5, 7, 9),
        min_start=5,
        scheme=scheme,  # "prefix" or "windows"
        pad_id=0,
        skip_id=1,
        max_batches=999999,
    )

    print(f"\n[BASE top1] split={eval_split}, scheme={scheme}")
    print("[Next-item]")
    for k in (1, 3, 5, 10):
        print(f"  hits@{k}={res[f'base_hits@{k}']:.4f}  ndcg@{k}={res[f'base_NDCG@{k}']:.4f}")

    print("[Path-level PRF]")
    for m in (3, 5, 7, 9):
        print(f"  m={m}  P={res[f'base_P@{m}']:.4f}  R={res[f'base_R@{m}']:.4f}  F1={res[f'base_F1@{m}']:.4f}  windows={res[f'base_windows@{m}']}")

import torch

@torch.no_grad()
def evaluate_base_top1_terminal_like_training(
    rl,
    data_loader,
    graph,
    hypergraph_list,
    device,
    horizon_H=10,
    min_start=5,
    max_starts_per_seq=1,
    max_batches=10**9,
):
    """
    不用RL policy：每一步用 base_model 的 top1 作为动作，
    输出与 evaluate_policy 同口径的终止指标（final_quality/effectiveness/adaptivity/diversity）
    """

    env = rl.env
    base_model = rl.base_model
    base_model.eval()

    sums = {"final_quality": 0.0, "effectiveness": 0.0, "adaptivity": 0.0, "diversity": 0.0}
    n = 0

    def _get_candidates(state):
        # 你按实际情况补字段名（常见几种我先列出来）
        if isinstance(state, dict):
            for key in ["cand_items", "candidates", "cand_ids", "candidate_items", "cands"]:
                if key in state and state[key] is not None:
                    x = state[key]
                    if not torch.is_tensor(x):
                        x = torch.tensor(x, device=device)
                    return x.long().to(device)
        # 也可能 env 上缓存了
        for attr in ["cand_items", "candidates", "candidate_items", "cands"]:
            if hasattr(env, attr):
                x = getattr(env, attr)
                if x is None:
                    continue
                if not torch.is_tensor(x):
                    x = torch.tensor(x, device=device)
                return x.long().to(device)
        return None

    def _current_logits():
        """
        从 env 里拿当前序列做一次 forward，得到当前步 logits。
        这里依赖 env 里缓存了 seq/ts/idx/ans（多数实现都有）。
        """
        seq = getattr(env, "seq", None)
        ts  = getattr(env, "ts", None)
        idx = getattr(env, "idx", None)
        ans = getattr(env, "ans", None)
        out = base_model(seq, ts, idx, ans, graph, hypergraph_list)
        pred = out[0] if isinstance(out, (tuple, list)) else out  # [B*(t_pred), V]
        B = seq.size(0)
        V = pred.size(-1)
        t_pred = pred.size(0) // B
        pred = pred.view(B, t_pred, V)
        # 默认用最后一步作为“当前步”
        return pred[:, -1, :]  # [B, V]

    for b_i, batch in enumerate(data_loader):
        if b_i >= max_batches:
            break

        tgt, ts, idx, ans = batch
        tgt = tgt.to(device); ts = ts.to(device); idx = idx.to(device); ans = ans.to(device)

        B = tgt.size(0)

        # windows：对每条序列抽若干 start（先用固定 min_start，完全对齐你训练时的习惯）
        for _ in range(max_starts_per_seq):
            start_t = torch.full((B,), int(min_start), device=device, dtype=torch.long)

            state = env.reset(
                tgt, ts, idx, ans,
                start_t=start_t,
                graph=graph,
                hypergraph_list=hypergraph_list
            )

            # rollout H 步：每一步动作=base top1
            for _step in range(horizon_H):
                cand = _get_candidates(state)
                logits = _current_logits()  # [B,V]

                if cand is None:
                    action = logits.argmax(dim=1).long()  # 全局top1
                else:
                    cand_logits = logits.gather(1, cand)           # [B,K]
                    best_pos = cand_logits.argmax(dim=1)          # [B]
                    action = cand.gather(1, best_pos[:, None]).squeeze(1).long()

                # 大多数 env.step(action_id) 这样就行；若你项目需要别的参数，
                # 报错后把 step 的函数签名贴我，我帮你对齐
                state, reward, done, info = env.step(action)

                # done 可能是 bool，也可能是 (B,) tensor
                if torch.is_tensor(done):
                    if done.all().item():
                        break
                else:
                    if bool(done):
                        break

            # 终止质量：用 env 自带口径算（与训练 evaluate_policy 一致）
            seq_after = getattr(env, "seq_after", getattr(env, "seq", tgt))
            ts_after  = getattr(env, "ts_after",  getattr(env, "ts", ts))
            idx_after = getattr(env, "idx_after", getattr(env, "idx", idx))
            ans_after = getattr(env, "ans_after", getattr(env, "ans", ans))

            metrics = env.compute_final_quality(seq_after, ts_after, idx_after, ans_after)

            for k in sums:
                if k in metrics:
                    sums[k] += float(metrics[k])
            n += 1

    if n == 0:
        return {k: 0.0 for k in sums}
    return {k: v / n for k, v in sums.items()}


def test_rl():
    test_rl_like_training("./checkpoint/A_rl_policy.pt", eval_split="test")
    # args = build_args()
    # set_seed(args.seed)
    #
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("[Device]", device)
    #
    # user_size, total_cascades, timestamps, train_data, valid_data, test_data = Split_data(
    #     args.data_name, args.train_rate, args.valid_rate, load_dict=True
    # )  # 这里返回的 test_data 还是 list 结构，不是 loader :contentReference[oaicite:5]{index=5}
    #
    # relation_graph = ConRelationGraph(args.data_name)
    # hypergraph_list = ConHyperGraphList(total_cascades, timestamps, user_size)
    #
    # #  关键：把 test_data 包装成 DataLoader（与你 RL 训练时一致）
    # test_loader = DataLoader(
    #     test_data,
    #     batch_size=args.batch_size,
    #     load_dict=True,
    #     cuda=(device.type == "cuda"),
    #     test=True
    # )  # DataLoader 每个 batch 返回 (seq, ts, idx, ans) :contentReference[oaicite:6]{index=6}
    #
    # args.d_word_vec = args.d_model
    # args.user_size = user_size
    # base_model = MSHGAT(args, dropout=args.dropout).to(device)
    # _load_pretrained(base_model, args.pretrained_path, device)
    # base_model.eval()
    #
    # rl = RLPathOptimizer(
    #     base_model=base_model,
    #     num_items=user_size,
    #     data_name=args.data_name,
    #     device=device,
    #     pad_val=0,
    #     topk=args.topk,
    #     cand_k=args.cand_k,
    #     history_window_T=args.history_T,
    #     rl_lr=args.rl_lr,
    #     target_future_M=3,
    #
    #     # 固定长度路径 + 方案1：每个时间步做规划，但限制次数
    #     horizon_H=10,
    #     min_start=5,
    #     max_starts_per_seq=1,
    #
    #     # ===== 终止奖励消融开关（训练用）=====
    #     # 默认全开；要做 w/o 某项就把它设为 False
    #     terminal_reward_components={
    #         "effectiveness": True,
    #         "adaptivity": True,
    #         "diversity": True,
    #     },
    #
    #     # 训练时：只计算“开着的”终止指标（关掉的就不算，省时间，也符合你的要求）
    #     train_compute_all_terminal_metrics=False,
    # )
    #
    # # 用一个 batch 触发 lazy init
    # first_batch = next(iter(test_loader))
    # tgt, ts, idx, ans = (x.to(device) for x in first_batch)  # 四元组 :contentReference[oaicite:7]{index=7}
    # rl.ensure_initialized(tgt, ts, idx, ans, graph=relation_graph, hypergraph_list=hypergraph_list)
    #
    # # 加载 policy 权重
    # ckpt = torch.load("./checkpoint/A_rl_policy.pt", map_location=device)
    # rl.policy.load_state_dict(ckpt["policy"])
    # rl.policy.eval()
    #
    # # 评估（两张表）
    # res = evaluate_policy_with_ranking_metrics(
    #     rl=rl,
    #     data_loader=test_loader,
    #     graph=relation_graph,
    #     hypergraph_list=hypergraph_list,
    #     device=device,
    #     max_batches=999999,
    #     k_list=[1, 3, 5, 10],
    #     m_list=[3, 5, 7, 9],
    #     schemeA_skip_short=True
    # )
    #
    # # 打印
    # print("\n[Table 1] Next-item ranking metrics (Base vs Policy)")
    # for k in [1, 3, 5, 10]:
    #     print("K=", k,
    #           res[f"base_hits@{k}"], res[f"policy_hits@{k}"],
    #           res[f"base_NDCG@{k}"], res[f"policy_NDCG@{k}"])
    #
    # print("\n[Table 2] Open-loop path planning metrics (Base rollout vs Policy rollout)")
    # for m in [3, 5, 7, 9]:
    #     print("m=", m,
    #           res[f"base_acc@{m}"], res[f"policy_acc@{m}"],
    #           res[f"base_NDCG@{m}"], res[f"policy_NDCG@{m}"],
    #           "windows", res[f"policy_windows@{m}"])


# def test_rl_from_checkpoint(
#     checkpoint_path: str = "./checkpoint/A_rl_policy.pt",
#     eval_split: str = "test",   # "valid" or "test"
#     max_batches: int = 999999,
#     k_list=(1, 3, 5, 10),
#     m_list=(3, 5, 7, 9),
# ):
#     args = build_args()
#     set_seed(args.seed)
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("[Device]", device)
#
#     user_size, total_cascades, timestamps, train_data, valid_data, test_data = Split_data(
#         args.data_name, args.train_rate, args.valid_rate, load_dict=True
#     )
#
#     relation_graph = ConRelationGraph(args.data_name)
#     hypergraph_list = ConHyperGraphList(total_cascades, timestamps, user_size)
#
#     # 选择评估 split
#     if eval_split.lower() == "valid":
#         eval_data = valid_data
#     else:
#         eval_data = test_data
#
#     eval_loader = DataLoader(
#         eval_data,
#         batch_size=args.batch_size,
#         load_dict=True,
#         cuda=(device.type == "cuda"),
#         test=True
#     )
#
#     # base model
#     args.d_word_vec = args.d_model
#     args.user_size = user_size
#     base_model = MSHGAT(args, dropout=args.dropout).to(device)
#     _load_pretrained(base_model, args.pretrained_path, device)
#     base_model.eval()
#
#     # RL optimizer
#     rl = RLPathOptimizer(
#         base_model=base_model,
#         num_items=user_size,
#         data_name=args.data_name,
#         device=device,
#         pad_val=0,
#         topk=args.topk,
#         cand_k=args.cand_k,
#         history_window_T=args.history_T,
#         rl_lr=args.rl_lr,
#         target_future_M=3,
#
#         horizon_H=10,
#         min_start=5,
#         max_starts_per_seq=1,
#
#         terminal_reward_components={
#             "effectiveness": True,
#             "adaptivity": True,
#             "diversity": True,
#         },
#         train_compute_all_terminal_metrics=False,
#     )
#
#     # 触发 lazy init（你现在就是这么做的）:contentReference[oaicite:2]{index=2}
#     first_batch = next(iter(eval_loader))
#     tgt, ts, idx, ans = (x.to(device) for x in first_batch)
#     rl.ensure_initialized(tgt, ts, idx, ans, graph=relation_graph, hypergraph_list=hypergraph_list)
#
#     # 加载 policy ckpt（你现在固定写死 A_rl_policy.pt）:contentReference[oaicite:3]{index=3}
#     ckpt = torch.load(checkpoint_path, map_location=device)
#     rl.policy.load_state_dict(ckpt["policy"])
#     rl.policy.eval()
#
#     # 评估（两张表）:contentReference[oaicite:4]{index=4}
#     res = evaluate_policy_with_ranking_metrics(
#         rl=rl,
#         data_loader=eval_loader,
#         graph=relation_graph,
#         hypergraph_list=hypergraph_list,
#         device=device,
#         max_batches=max_batches,
#         k_list=list(k_list),
#         m_list=list(m_list),
#         schemeA_skip_short=True
#     )
#
#     print(f"\n[Eval split: {eval_split}] checkpoint={checkpoint_path}")
#
#     print("\n[Table 1] Next-item ranking metrics (Base vs Policy)")
#     for k in k_list:
#         print("K=", k,
#               res[f"base_hits@{k}"], res[f"policy_hits@{k}"],
#               res[f"base_NDCG@{k}"], res[f"policy_NDCG@{k}"])
#
#     print("\n[Table 2] Open-loop path planning metrics (Base rollout vs Policy rollout)")
#     for m in m_list:
#         print("m=", m,
#               res[f"base_acc@{m}"], res[f"policy_acc@{m}"],
#               res[f"base_NDCG@{m}"], res[f"policy_NDCG@{m}"],
#               "windows", res.get(f"policy_windows@{m}", None))
#
#     return res


if __name__ == "__main__":
    # main()
    test_rl()
    # 如果需要使用特定参数测试，可以取消下面的注释
    # test_rl_policy_from_checkpoint(checkpoint_path="./checkpoint/A_rl_policy.pt", data_name="Assist")
