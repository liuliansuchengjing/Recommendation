train_rl_new.py
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
    """设置随机种子以确保结果可重现"""
    torch.backends.cudnn.deterministic = True  # 设置CuDNN为确定性模式
    torch.manual_seed(seed)                     # 设置PyTorch随机种子
    torch.cuda.manual_seed_all(seed)            # 设置CUDA随机种子
    np.random.seed(seed)                        # 设置NumPy随机种子


def _shuffle_cas(cas):
    """
    随机打乱级联数据
    
    Args:
        cas: 级联数据元组 (tgt, timestamp, idx, ans)
        
    Returns:
        打乱后的级联数据
    """
    # cas 是 (tgt, timestamp, idx, ans)
    if cas is None or (not isinstance(cas, (list, tuple))) or len(cas) != 4:
        return cas
    tgt, ts, idx, ans = cas  # tgt: 目标序列, ts: 时间戳, idx: 索引, ans: 答案
    n = len(tgt)  # 获取序列长度
    if n <= 1:   # 如果长度小于等于1则无需打乱
        return cas
    perm = np.random.permutation(n)  # 生成随机排列
    # 按照随机排列重新排序所有元素
    return ([tgt[i] for i in perm],
            [ts[i] for i in perm],
            [idx[i] for i in perm],
            [ans[i] for i in perm])


def _load_pretrained(model: torch.nn.Module, path: str, device: torch.device):
    """
    加载预训练模型
    
    Args:
        model: PyTorch模型
        path: 模型文件路径
        device: 目标设备
        
    Returns:
        是否成功加载
    """
    if not path:  # 如果路径为空则返回False
        return False
    if not os.path.exists(path):  # 如果文件不存在则警告并返回False
        print(f"[WARN] not found: {path}  (training from random init is not recommended)")
        return False
    # 加载模型文件到指定设备
    ckpt = torch.load(path, map_location=device)
    # 检查检查点格式，可能是字典格式包含state_dict或model键
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]  # 获取状态字典
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]       # 获取模型状态
    else:
        state = ckpt                # 直接使用检查点作为状态
    
    # 加载状态到模型，允许部分加载（strict=False）
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:  # 如果有缺失的键则警告
        print("[WARN] missing keys:", missing[:20], ("..." if len(missing) > 20 else ""))
    if unexpected:  # 如果有意外的键则警告
        print("[WARN] unexpected keys:", unexpected[:20], ("..." if len(unexpected) > 20 else ""))
    print(f"[OK] loaded pretrained base model from: {path}")
    return True


def build_args():
    """构建命令行参数解析器"""
    p = argparse.ArgumentParser()  # 创建参数解析器

    # ===== 与run.py保持一致的基础参数 =====
    p.add_argument("-data_name", default="MOO")            # 数据集名称
    p.add_argument("-batch_size", type=int, default=64)    # 批次大小
    p.add_argument("-d_model", type=int, default=64)       # 模型维度
    p.add_argument("-initialFeatureSize", type=int, default=64)  # 初始特征大小
    p.add_argument("-train_rate", type=float, default=0.8) # 训练集比例
    p.add_argument("-valid_rate", type=float, default=0.1) # 验证集比例
    p.add_argument("-dropout", type=float, default=0.3)    # Dropout比率
    p.add_argument("-pos_emb", type=bool, default=True)    # 是否使用位置编码

    p.add_argument("--pretrained_path", type=str, default="./checkpoint/DiffusionPrediction.pt")  # 预训练模型路径

    # ===== RL/PPO相关参数 =====
    p.add_argument("--rl_epochs", type=int, default=5)     # RL训练轮数
    p.add_argument("--cand_k", type=int, default=50)       # 候选项目数量
    p.add_argument("--topk", type=int, default=10)         # Top-K推荐数量
    p.add_argument("--history_T", type=int, default=10)    # 历史窗口大小
    p.add_argument("--rl_lr", type=float, default=3e-4)    # RL学习率
    p.add_argument("--seed", type=int, default=0)          # 随机种子

    return p.parse_args()  # 解析并返回参数


def train_one_epoch(rl: RLPathOptimizer, train_loader, graph, hypergraph_list, device):
    """
    训练单个epoch
    
    Args:
        rl: RL路径优化器
        train_loader: 训练数据加载器
        graph: 关系图
        hypergraph_list: 超图列表
        device: 计算设备
        
    Returns:
        训练统计信息字典
    """
    # 初始化统计信息字典
    stats = {"policy_loss": [], "value_loss": [], "entropy": [], "total_loss": [],
             "final_quality": [], "effectiveness": [], "adaptivity": [], "diversity": []}

    for batch in train_loader:  # 遍历训练批次
        # 解包批次数据
        tgt, tgt_timestamp, tgt_idx, ans = batch  # tgt: tensor of shape [batch_size, seq_len] - 目标序列
                                                # tgt_timestamp: tensor of shape [batch_size, seq_len] - 时间戳序列
                                                # tgt_idx: tensor of shape [batch_size] - 级联索引（每个序列一个索引）
                                                # ans: tensor of shape [batch_size, seq_len] - 答案序列

        tgt = tgt.to(device)                    # 将目标序列移到设备
        tgt_timestamp = tgt_timestamp.to(device)  # 将时间戳移到设备
        tgt_idx = tgt_idx.to(device)            # 将索引移到设备
        ans = ans.to(device)                    # 将答案移到设备

        # 首次初始化策略网络
        if rl.policy is None:  # 如果策略网络尚未初始化
            # 使用第一个批次的数据初始化策略网络
            rl.ensure_initialized(tgt, tgt_timestamp, tgt_idx, ans, graph=graph, hypergraph_list=hypergraph_list)

        rl.policy.train()  # 设置策略网络为训练模式

        # 收集轨迹数据
        rollout = rl.collect_trajectory(tgt, tgt_timestamp, tgt_idx, ans, graph=graph, hypergraph_list=hypergraph_list)
        # rollout是一个字典，包含轨迹相关的数据

        # 更新策略网络
        losses = rl.update_policy(rollout)  # 执行PPO更新步骤

        fm = rollout["final_metrics"]  # 获取最终指标 [batch_size]

        # 记录各种损失和指标
        stats["policy_loss"].append(losses["policy_loss"])      # 策略损失
        stats["value_loss"].append(losses["value_loss"])        # 价值损失
        stats["entropy"].append(losses["entropy"])              # 熵
        stats["total_loss"].append(losses["total_loss"])        # 总损失
        stats["final_quality"].append(float(fm["final_quality"].mean().detach().cpu()))  # 最终质量
        stats["effectiveness"].append(float(fm["effectiveness"].mean().detach().cpu()))  # 效果
        stats["adaptivity"].append(float(fm["adaptivity"].mean().detach().cpu()))        # 适应性
        stats["diversity"].append(float(fm["diversity"].mean().detach().cpu()))          # 多样性

    # 计算并返回平均统计值
    return {k: float(np.mean(v)) if v else 0.0 for k, v in stats.items()}


def main():
    """主函数"""
    args = build_args()  # 构建参数
    set_seed(args.seed)  # 设置随机种子

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择设备
    print("[Device]", device)  # 打印使用的设备

    # 分割数据集
    user_size, total_cascades, timestamps, train, valid, test = Split_data(
        args.data_name, args.train_rate, args.valid_rate, load_dict=True
    )
    # user_size: int - 用户/项目数量
    # total_cascades: list - 总级联数据
    # timestamps: list - 时间戳列表
    # train, valid, test: dict - 训练、验证、测试数据

    # 构建图结构
    relation_graph = ConRelationGraph(args.data_name)  # 关系图
    hypergraph_list = ConHyperGraphList(total_cascades, timestamps, user_size)  # 超图列表

    # 设置额外参数
    args.d_word_vec = args.d_model  # 词向量维度等于模型维度
    args.user_size = user_size      # 用户数量
    # 创建基础模型
    base_model = MSHGAT(args, dropout=args.dropout).to(device)  # MSHGAT模型
    # 加载预训练权重
    _load_pretrained(base_model, args.pretrained_path, device)

    # 创建RL路径优化器
    rl = RLPathOptimizer(
        base_model=base_model,      # 基础模型
        num_items=user_size,        # 项目数量
        data_name=args.data_name,   # 数据集名称
        device=device,              # 设备
        pad_val=0,                  # 填充值
        topk=args.topk,             # Top-K数量
        cand_k=args.cand_k,         # 候选数量
        history_window_T=args.history_T,  # 历史窗口大小
        rl_lr=args.rl_lr,           # RL学习率
    )

    best_val = -1e18  # 初始化最佳验证分数
    for epoch in range(args.rl_epochs):  # 遍历训练轮数
        print(f"\n[RL Epoch {epoch}]")  # 打印当前轮次

        # 打乱训练数据
        train_shuffled = _shuffle_cas(train)  # 打乱训练数据
        # 创建训练数据加载器
        train_loader = DataLoader(train_shuffled, batch_size=args.batch_size, load_dict=True, cuda=(device.type == "cuda"))

        t0 = time.time()  # 记录开始时间
        tr = train_one_epoch(rl, train_loader, relation_graph, hypergraph_list, device)  # 训练一个epoch
        print(f"  train: {tr}  (elapsed {(time.time() - t0) / 60:.2f} min)")  # 打印训练结果和耗时

        # 创建验证数据加载器
        valid_loader = DataLoader(valid, batch_size=args.batch_size, load_dict=True, cuda=(device.type == "cuda"), test=True)
        # 评估验证集性能
        val = evaluate_policy(rl=rl, data_loader=valid_loader, graph=relation_graph, hypergraph_list=hypergraph_list, device=device)
        print("  valid:", val)  # 打印验证结果

        # 保存最佳模型
        if val.get("final_quality", -1e18) > best_val:  # 如果当前验证结果更好
            best_val = val["final_quality"]  # 更新最佳分数
            # 保存策略网络状态
            torch.save({"policy": rl.policy.state_dict()}, "./checkpoint/rl_policy_best.pt")
            print("  [OK] saved best RL policy to ./checkpoint/rl_policy_best.pt")  # 打印保存信息

    # 创建测试数据加载器
    test_loader = DataLoader(test, batch_size=args.batch_size, load_dict=True, cuda=(device.type == "cuda"), test=True)
    # 评估测试集性能
    te = evaluate_policy(rl=rl, data_loader=test_loader, graph=relation_graph, hypergraph_list=hypergraph_list, device=device)
    print("\n[Test]", te)  # 打印测试结果


if __name__ == "__main__":
    main()