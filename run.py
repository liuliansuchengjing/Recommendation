# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 22:42:32 2021

@author: Ling Sun
"""

import argparse
import time
import numpy as np
import Constants
import torch
import torch.nn as nn
from graphConstruct import ConRelationGraph, ConHyperGraphList
from dataLoader import Split_data, DataLoader
from Metrics import Metrics, KTLoss
from HGAT import MSHGAT
from Optim import ScheduledOptim
from calculate_muti_obj import gain_test_model, learning_effect_loss, learning_adaptive_loss
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

torch.backends.cudnn.deterministic = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
torch.cuda.manual_seed(0)

metric = Metrics()

parser = argparse.ArgumentParser()
parser.add_argument('-data_name', default='MOO')
parser.add_argument('-epoch', type=int, default=100)
parser.add_argument('-batch_size', type=int, default=64)
parser.add_argument('-d_model', type=int, default=64)
parser.add_argument('-initialFeatureSize', type=int, default=64)
parser.add_argument('-train_rate', type=float, default=0.8)
parser.add_argument('-valid_rate', type=float, default=0.1)
parser.add_argument('-n_warmup_steps', type=int, default=1000)
parser.add_argument('-dropout', type=float, default=0.3)
parser.add_argument('-log', default=None)
parser.add_argument('-save_rec_path', default="./checkpoint/REC_Prediction_M100.pt")
# parser.add_argument('-save_kt_path', default="./checkpoint/KT_Prediction_M100.pt")  # 注释 KT 相关
parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
parser.add_argument('-no_cuda', action='store_true')
parser.add_argument('-pos_emb', type=bool, default=True)
# --- KT-guided distillation (train-time) ---
# parser.add_argument('--lambda_kt', type=float, default=5000.0)      # 保持你当前 5000 不变
# parser.add_argument('--lambda_distill', type=float, default=0.1)    # 默认关闭，不影响现有结果
parser.add_argument('--distill_k', type=int, default=30)  # topK 候选大小
parser.add_argument('--distill_tau', type=float, default=1.0)  # 温度
parser.add_argument('--distill_eps', type=float, default=1e-12)  # log/softmax 稳定项
# ==================== 新增：超图合成控制参数 ====================
parser.add_argument('-num_stages', type=int, default=3, help='强制划分的子超图(阶段)固定数量')
parser.add_argument('-decay_rate', type=float, default=0.1, help='不同时间窗口的高阶协同衰减权重系数')
# ================================================================
opt = parser.parse_args()
opt.d_word_vec = opt.d_model


def compute_kt_clf_metrics(y_prob, y_true, mask):
    """
    计算二分类任务的 Acc, Precision, Recall, F1
    y_prob: 模型的预测概率 (Tensor)
    y_true: 真实的标签 (Tensor)
    mask: 有效位掩码 (Tensor)
    """
    # 1. 展平并转为 numpy
    y_prob_flat = y_prob.reshape(-1).detach().cpu().numpy()
    y_true_flat = y_true.reshape(-1).detach().cpu().numpy()
    mask_flat = mask.reshape(-1).detach().cpu().bool().numpy()

    # 2. 用 mask 过滤掉 PAD 和 无效部分
    valid_prob = y_prob_flat[mask_flat]
    valid_true = y_true_flat[mask_flat]

    if len(valid_true) == 0:
        return -1, -1, -1, -1

    # 3. 将概率转换为 0 或 1 的硬标签 (阈值 0.5)
    valid_pred = (valid_prob >= 0.5).astype(int)

    # 4. 调用 sklearn 计算指标 (zero_division=0 防止分母为0时报错)
    acc = accuracy_score(valid_true, valid_pred)
    p = precision_score(valid_true, valid_pred, zero_division=0)
    r = recall_score(valid_true, valid_pred, zero_division=0)
    f1 = f1_score(valid_true, valid_pred, zero_division=0)

    return acc, p, r, f1


def compute_rec_clf_metrics(y_pred_logits, y_true, pad_id=0, skip_id=1):
    """
    计算推荐任务 (多分类 Top-1) 的 Acc, Precision, Recall, F1
    y_pred_logits: 推荐模型的预测输出 (N, num_items) numpy array
    y_true: 真实的下一题 ID (N,) numpy array
    """
    # 1. 过滤掉无效的 PAD 和 Skip 标记
    valid_mask = (y_true != pad_id) & (y_true != skip_id)
    valid_true = y_true[valid_mask]

    if len(valid_true) == 0:
        return -1, -1, -1, -1

    # 2. 获取模型预测概率最高的那道题 (Top-1)
    valid_logits = y_pred_logits[valid_mask]
    valid_pred = np.argmax(valid_logits, axis=1)

    # 3. 计算多分类指标 (必须指定 average='weighted' 应对极度不平衡的题库)
    acc = accuracy_score(valid_true, valid_pred)
    p = precision_score(valid_true, valid_pred, average='weighted', zero_division=0)
    r = recall_score(valid_true, valid_pred, average='weighted', zero_division=0)
    f1 = f1_score(valid_true, valid_pred, average='weighted', zero_division=0)

    return acc, p, r, f1


def compute_ranking_metrics(y_pred_logits, y_true, k_list=[1, 5, 10, 20], pad_id=0, skip_id=1):
    """
    计算推荐排序指标: Hit@K, NDCG@K, MRR@K

    Args:
        y_pred_logits: (N, V) 模型预测的 logits
        y_true: (N,) 真实的目标物品 ID
        k_list: 要计算的 K 值列表
        pad_id: PAD 标记 ID
        skip_id: 跳过的标记 ID

    Returns:
        dict: 包含 hit@k, ndcg@k, mrr@k 的字典
    """
    # 过滤无效样本
    valid_mask = (y_true != pad_id) & (y_true != skip_id)
    valid_true = y_true[valid_mask]
    valid_logits = y_pred_logits[valid_mask]

    if len(valid_true) == 0:
        empty_results = {}
        for k in k_list:
            empty_results[f'hit@{k}'] = 0.0
            empty_results[f'ndcg@{k}'] = 0.0
            empty_results[f'mrr@{k}'] = 0.0
        return empty_results

    N = len(valid_true)
    V = valid_logits.shape[1]

    # 获取排序后的索引 (降序)
    sorted_indices = np.argsort(-valid_logits, axis=1)  # (N, V)

    results = {}

    for k in k_list:
        k = min(k, V)  # 确保 k 不超过物品数量
        hits = 0
        ndcgs = 0.0
        mrrs = 0.0

        for i in range(N):
            true_item = valid_true[i]
            top_k_items = sorted_indices[i, :k]

            # Hit@K: 目标物品是否在 Top-K 中
            if true_item in top_k_items:
                hits += 1
                # 找到目标物品在排序中的位置
                rank = np.where(top_k_items == true_item)[0][0] + 1  # rank 从 1 开始
                # NDCG@K: 归一化折损累计增益
                ndcgs += 1.0 / np.log2(rank + 1)
                # MRR@K: 平均倒数排名
                mrrs += 1.0 / rank

        results[f'hit@{k}'] = hits / N
        results[f'ndcg@{k}'] = ndcgs / N
        results[f'mrr@{k}'] = mrrs / N

    return results


def kt_rerank_logits_numpy(logits_np, yt_tensor, base_k=100, beta=1.0, pad_id=0, skip_id=1, eps=1e-12):
    """
    logits_np: (N, V)  numpy, 来自 pred.detach().cpu().numpy()
    yt_tensor: torch.Tensor, (B, T-1, V) 或 (N, V)  —— 你的 model forward 返回的 yt
    base_k: 只在 base_k 个候选上做 KT 重排（推荐 50~200）
    beta: KT 强度，score = logit + beta*log(yt)
    返回: (N, V) numpy，重排后的“可用于 argsort/topk 的分数矩阵”
    """
    # ---- yt reshape -> (N, V) numpy ----
    yt = yt_tensor.detach().cpu()
    if yt.dim() == 3:
        yt = yt.reshape(-1, yt.size(-1))
    yt_np = yt.numpy()  # (N, V)

    N, V = logits_np.shape
    assert yt_np.shape == (N, V), f"yt shape {yt_np.shape} != logits shape {logits_np.shape}"

    # 初始化为很小的分数，确保只在候选集内竞争
    out = np.full_like(logits_np, fill_value=-1e9)

    for n in range(N):
        row = logits_np[n]

        # 取推荐模型的 base_k 候选
        cand = np.argpartition(row, -base_k)[-base_k:]
        # 过滤 PAD / skip
        cand = cand[(cand != pad_id) & (cand != skip_id)]

        # 计算 KT 加权分数
        kt_prob = yt_np[n, cand]
        score = row[cand] + beta * np.log(kt_prob + eps)

        out[n, cand] = score

    return out


# print(opt)

def batch_path_counts_from_logits(pred_logits, tgt, m, pad_id=0, skip_id=1):
    """
    返回该 batch 在 path-level（set）下的 TP/FP/FN 计数（micro 累计用）
    pred_logits: (B*(T-1), V) 或 (B, T-1, V)
    tgt: (B, T)
    """
    B, T = tgt.size()
    gold = tgt[:, 1:]  # (B, T-1)

    if pred_logits.dim() == 2:
        V = pred_logits.size(-1)
        pred_logits = pred_logits.view(B, T - 1, V)

    pred_ids = pred_logits.argmax(dim=-1)  # (B, T-1)

    def _clean(seq):
        return [int(x) for x in seq if int(x) != pad_id and int(x) != skip_id]

    TP = FP = FN = 0
    for b in range(B):
        true_set = set(_clean(gold[b, :m].tolist()))
        pred_set = set(_clean(pred_ids[b, :m].tolist()))

        TP += len(true_set & pred_set)
        FP += len(pred_set - true_set)
        FN += len(true_set - pred_set)

    return TP, FP, FN


def prf_path_level_from_logits(pred_logits, tgt, m, pad_id=0, skip_id=1, average="micro"):
    """
    pred_logits: Tensor, shape (B*(T-1), V)  或 (B, T-1, V)
    tgt:        Tensor, shape (B, T)  (你的 batch 里的 tgt)
    m:          int, 评估的路径长度 (3/5/7/9)
    average:    "micro" 或 "macro"
    return: dict(P, R, F1, TP, FP, FN)
    """
    import torch

    B, T = tgt.size()
    gold = tgt[:, 1:]  # (B, T-1)

    # 1) reshape logits -> (B, T-1, V)
    if pred_logits.dim() == 2:
        V = pred_logits.size(-1)
        pred_logits = pred_logits.view(B, T - 1, V)

    # 2) 取 top-1 生成预测序列 (B, T-1)
    pred_ids = pred_logits.argmax(dim=-1)

    # 3) 对每个样本取前 m 步，并过滤 PAD/skip
    def _clean(seq):
        return [int(x) for x in seq if int(x) != pad_id and int(x) != skip_id]

    if average == "micro":
        TP = FP = FN = 0
        for b in range(B):
            true_path = _clean(gold[b, :m].tolist())
            pred_path = _clean(pred_ids[b, :m].tolist())

            true_set = set(true_path)
            pred_set = set(pred_path)

            TP += len(true_set & pred_set)
            FP += len(pred_set - true_set)
            FN += len(true_set - pred_set)

        P = TP / (TP + FP + 1e-12)
        R = TP / (TP + FN + 1e-12)
        F1 = 0.0 if (P + R) == 0 else 2 * P * R / (P + R)
        return {"P": P, "R": R, "F1": F1, "TP": TP, "FP": FP, "FN": FN}

    elif average == "macro":
        Ps, Rs, F1s = [], [], []
        for b in range(B):
            true_path = _clean(gold[b, :m].tolist())
            pred_path = _clean(pred_ids[b, :m].tolist())

            true_set = set(true_path)
            pred_set = set(pred_path)

            tp = len(true_set & pred_set)
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)

            p = tp / (tp + fp + 1e-12)
            r = tp / (tp + fn + 1e-12)
            f1 = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)

            Ps.append(p);
            Rs.append(r);
            F1s.append(f1)

        return {"P": sum(Ps) / len(Ps), "R": sum(Rs) / len(Rs), "F1": sum(F1s) / len(F1s)}
    else:
        raise ValueError("average must be 'micro' or 'macro'")


def get_performance(crit, pred, gold):
    # loss = crit(pred, gold.contiguous().view(-1))
    # pred = pred.max(1)[1]
    # gold = gold.contiguous().view(-1)
    # n_correct = pred.data.eq(gold.data)
    # n_correct = n_correct.masked_select(gold.ne(Constants.PAD).data).sum().float()
    gold_flat = gold.contiguous().view(-1)

    # 【改动1】：为了让交叉熵损失同时忽略 1，我们临时将 1 替换为 PAD (0)
    # 因为你的 crit(CrossEntropyLoss) 已经设置了 ignore_index=0
    gold_for_loss = gold_flat.clone()
    gold_for_loss[gold_for_loss == 1] = Constants.PAD

    # 这样 loss 就不会去拟合 1 了
    loss = crit(pred, gold_for_loss)

    # 【改动2】：计算准确个数时，掩码不仅过滤 0，还要过滤 1
    pred_id = pred.max(1)[1]
    valid_mask = (gold_flat != Constants.PAD) & (gold_flat != 1)
    n_correct = pred_id.eq(gold_flat).masked_select(valid_mask).sum().float()
    return loss, n_correct


def kt_guided_distill_loss(
        pred_logits,  # (B*(T-1), V) 或 (B, T-1, V)
        yt,  # (B, T-1, V) 或 (B*(T-1), V)
        gold,  # (B, T-1)  真实下一题 id
        k=100,
        tau=1.0,
        pad_id=0,
        eps=1e-12,
):
    """
    只在 topK 候选集上做蒸馏：
      teacher = softmax(log(yt_prob)/tau)
      student = softmax(logits/tau)
      L = KL(teacher || student)
    """
    # ---- reshape logits -> (N, V) ----
    if pred_logits.dim() == 3:
        B, Tm1, V = pred_logits.size()
        pred_logits = pred_logits.reshape(-1, V)
    else:
        V = pred_logits.size(-1)

    # ---- reshape yt -> (N, V) ----
    if yt.dim() == 3:
        yt = yt.reshape(-1, yt.size(-1))
    assert yt.size(-1) == V, f"yt V={yt.size(-1)} != logits V={V}"

    # ---- valid positions (exclude PAD targets) ----
    # valid = gold.ne(pad_id).reshape(-1)  # (N,)
    valid = ((gold != pad_id) & (gold != 1)).reshape(-1)
    if valid.sum().item() == 0:
        return pred_logits.new_tensor(0.0)

    logits_v = pred_logits[valid]  # (Nvalid, V)
    yt_v = yt[valid]  # (Nvalid, V)

    # ---- topK candidate indices from student logits ----
    kk = min(k, V)
    cand_idx = torch.topk(logits_v, k=kk, dim=-1).indices  # (Nvalid, K)

    # ---- gather candidate scores ----
    stu_cand = torch.gather(logits_v, dim=1, index=cand_idx)  # (Nvalid, K)
    tea_prob = torch.gather(yt_v, dim=1, index=cand_idx)  # (Nvalid, K)

    # ---- teacher distribution from KT prob ----
    # 用 log(yt) 再 softmax，更“像分布”，且数值稳定
    tea_logits = torch.log(tea_prob + eps) / tau
    tea_dist = torch.softmax(tea_logits, dim=-1).detach()  # teacher 不回传梯度

    # ---- student log-prob ----
    log_stu = torch.log_softmax(stu_cand / tau, dim=-1)

    # ---- KL(teacher || student) ----
    # kl_div expects input=log-prob, target=prob
    loss = F.kl_div(log_stu, tea_dist, reduction="batchmean")

    # 常见做法：乘 tau^2 保持梯度尺度（可选，建议保留）
    loss = loss * (tau * tau)
    return loss


def train_epoch(model, training_data, graph, hypergraph_list, loss_func, optimizer):  # 移除 kt_loss 参数
    # train

    model.train()

    total_loss = 0.0
    n_total_words = 0.0
    n_total_correct = 0.0
    batch_num = 0.0
    # auc_train = []  # 注释 KT 相关
    # acc_train = []

    for i, batch in enumerate(
            training_data):  # tqdm(training_data, mininterval=2, desc='  - (Training)   ', leave=False):
        # data preparing
        tgt, tgt_timestamp, tgt_idx, ans = (item.cuda() for item in batch)
        batch_size, seq_len = tgt.size()

        np.set_printoptions(threshold=np.inf)
        gold = tgt[:, 1:]

        # n_words = gold.data.ne(Constants.PAD).sum().float()
        valid_mask = (gold != Constants.PAD) & (gold != 1)
        n_words = valid_mask.sum().float()
        n_total_words += n_words
        batch_num += tgt.size(0)

        # training
        optimizer.zero_grad()
        # pred= model(tgt, tgt_timestamp, tgt_idx, ans, graph, hypergraph_list)
        pred = model(tgt, tgt_timestamp, tgt_idx, ans, graph,
                                                  hypergraph_list.cuda())  # ==================================



        # 1. 拿到三个原始损失 (Raw Loss)
        loss_rec_raw, n_correct = get_performance(loss_func, pred, gold)

        # 推荐系统损失
        weight_rec = torch.exp(-model.log_var_rec)
        loss_rec_adaptive = weight_rec * loss_rec_raw + model.log_var_rec


        # 3. 最终的总 Loss
        loss = loss_rec_adaptive  # + loss_kt_adaptive  # 注释 KT 相关，只保留推荐损失

        # 反向传播，这一步会让模型自动去更新那三个 log_var_xxx 参数
        loss.backward()

        # parameter update
        optimizer.step()
        optimizer.update_learning_rate()

        n_total_correct += n_correct
        total_loss += loss.item()
        # if auc != -1 and acc != -1:  # 注释 KT 相关
        #     auc_train.append(auc)
        #     acc_train.append(acc)

    return total_loss / n_total_words, n_total_correct / n_total_words  # , auc_train, acc_train  # 注释 KT 相关


def train_model(MSHGAT, data_path):
    # ========= Preparing DataLoader =========#
    resource_size, total_cascades, timestamps, train, valid, test = Split_data(data_path, opt.train_rate,
                                                                               opt.valid_rate,
                                                                               load_dict=True)

    train_data = DataLoader(train, batch_size=opt.batch_size, load_dict=True, cuda=False)
    valid_data = DataLoader(valid, batch_size=opt.batch_size, load_dict=True, cuda=False)
    test_data = DataLoader(test, batch_size=opt.batch_size, load_dict=True, cuda=False)

    relation_graph = ConRelationGraph(data_path)
    hypergraph_list = ConHyperGraphList(total_cascades, timestamps, resource_size, num_stages=opt.num_stages,
        decay_rate=opt.decay_rate)

    opt.resource_size = resource_size

    # 1. 定义两个不同的最高分记录
    best_rec_hit = 0.0
    # best_kt_auc = 0.0  # 注释 KT 相关

    # 2. 初始化早停机制参数 (新增)
    patience = 5  # 容忍多少个 epoch 没有提升
    patience_counter = 0  # 当前连续没有提升的 epoch 数

    # ========= Preparing Model =========#
    model = MSHGAT(opt, dropout=opt.dropout)
    loss_func = nn.CrossEntropyLoss(size_average=False, ignore_index=Constants.PAD)
    # kt_loss = KTLoss()  # 注释 KT 相关

    params = model.parameters()
    optimizerAdam = torch.optim.Adam(params, betas=(0.9, 0.98), eps=1e-09)
    optimizer = ScheduledOptim(optimizerAdam, opt.d_model, opt.n_warmup_steps)

    if torch.cuda.is_available():
        model = model.cuda()
        loss_func = loss_func.cuda()
        # kt_loss = kt_loss.cuda()  # 注释 KT 相关

    validation_history = 0.0
    best_scores = {}
    # best_kt_metrics = {'auc': 0.0, 'acc': 0.0}  # 注释 KT 相关
    for epoch_i in range(opt.epoch):
        print('\n[ Epoch', epoch_i, ']')

        start = time.time()
        # train_loss, train_accu, train_auc, train_acc = train_epoch(model, train_data, relation_graph, hypergraph_list,
        #                                                            loss_func, kt_loss, optimizer)
        train_loss, train_accu = train_epoch(model, train_data, relation_graph, hypergraph_list,
                                                                   loss_func, optimizer)  # 注释 KT 相关，移除 kt_loss

        # ==================== 新增：获取并计算当前的自适应权重 ====================
        # 因为定义的参数是对数方差 (log_var)，实际的权重是 exp(-log_var)
        # 使用 .item() 将单个元素的 Tensor 转换为普通的 Python 浮点数
        w_rec = torch.exp(-model.log_var_rec).item()
        # w_kt = torch.exp(-model.log_var_kt).item()  # 注释 KT 相关
        # w_distill = torch.exp(-model.log_var_distill).item()
        # =========================================================================

        print('  - (Training)   loss: {loss: 8.5f}, accuracy: {accu:3.3f} %, ' \
              'elapse: {elapse:3.3f} min'.format(
            loss=train_loss, accu=100 * train_accu,
            elapse=(time.time() - start) / 60))
        # print('auc_train: {:.10f}'.format(np.mean(train_auc)),  # 注释 KT 相关
        #       'acc_train: {:.10f}'.format(np.mean(train_acc)))

        # ==================== 新增：打印权重信息 ====================
        # print(f'  - (Weights)    Rec: {w_rec:.4f} | KT: {w_kt:.4f} | Distill: {w_distill:.4f}')
        # print(f'  - (Weights)    Rec: {w_rec:.4f} | KT: {w_kt:.4f} ')  # 注释 KT 相关
        print(f'  - (Weights)    Rec: {w_rec:.4f}')  # 只保留推荐权重
        # ==========================================================

        if epoch_i >= 0:
            start = time.time()
            # scores, auc_valid, acc_valid = test_epoch(model, valid_data, relation_graph, hypergraph_list, kt_loss)  # 注释 KT 相关
            scores = test_epoch(model, valid_data, relation_graph, hypergraph_list)
            print('  - ( Validation )) ')
            for metric in scores.keys():
                print(metric + ' ' + str(scores[metric]))
            # print('auc_valid: {:.10f}'.format(np.mean(auc_valid)),  # 注释 KT 相关
            #       'acc_valid: {:.10f}'.format(np.mean(acc_valid)))
            print("Validation use time: ", (time.time() - start) / 60, "min")

            print('  - (Test) ')
            # scores, auc_test, acc_test = test_epoch(model, test_data, relation_graph, hypergraph_list, kt_loss)  # 注释 KT 相关
            scores = test_epoch(model, test_data, relation_graph, hypergraph_list)
            for metric in scores.keys():
                print(metric + ' ' + str(scores[metric]))
            # print('auc_test: {:.10f}'.format(np.mean(auc_test)),  # 注释 KT 相关
            #       'acc_test: {:.10f}'.format(np.mean(acc_test)))
            # if validation_history <= sum(scores.values()):
            #     print("Best Validation hit@20:{} at Epoch:{}".format(scores["hits@20"], epoch_i))
            #     validation_history = sum(scores.values())
            #     best_scores = scores
            #     print("Save best model!!!")
            #     torch.save(model.state_dict(), opt.save_path)
            # 逻辑 1：保存推荐系统最好的模型
            # 3. 早停逻辑核心判断
            is_improved = False  # 设立一个标志位，记录这一轮是否有任何一个指标变好

            if validation_history <= sum(scores.values()):
                validation_history = sum(scores.values())
                torch.save(model.state_dict(), opt.save_rec_path)
                print("Save Best Recommendation Model!")
                best_scores = scores
                is_improved = True  # 只要推荐变好了，就标记为 True

            # # 逻辑 2：保存知识追踪最好的模型（独立保存！）  # 注释 KT 相关
            # current_kt_auc = np.mean(auc_test)
            # if current_kt_auc > best_kt_auc:
            #     best_kt_auc = current_kt_auc
            #     torch.save(model.state_dict(), opt.save_kt_path)
            #     best_kt_metrics['auc'] = current_kt_auc
            #     best_kt_metrics['acc'] = np.mean(acc_test)
            #     print("Save Best KT Model!")
            #     is_improved = True  # 只要 KT 变好了，也标记为 True

            # 4. 更新耐心计时器
            if is_improved:
                patience_counter = 0  # 只要有任何一个指标提升，清零重新计数
            else:
                patience_counter += 1  # 两个都没提升，计数器 +1
                print(f" Early stopping counter: {patience_counter} out of {patience}")

            # 5. 触发早停
            if patience_counter >= patience:
                print(f"\n Early stopping triggered at epoch {epoch_i}!")
                break  # 直接跳出 for 循环，结束训练

    print(" -(Finished!!) \n Best scores: ")
    for metric in best_scores.keys():
        print(metric + ' ' + str(best_scores[metric]))

    # print("\n Best Knowledge Tracing scores: ")  # 注释 KT 相关
    # for metric in best_kt_metrics.keys():
    #     print(f"    {metric}: {best_kt_metrics[metric]:.4f}")


def generate_ep_greedy_path(model_rec, model_kt, hist_seq, hist_ans, target_set, graph, path_length=10,
                            candidate_size=50):
    """
    单目标 (EP) 贪心路径生成器s
    hist_seq: [1, seq_len] 历史序列
    hist_ans: [1, seq_len] 历史作答记录
    target_set: list 隐式目标题目的 ID
    """
    generated_path = []
    current_seq = hist_seq.clone()
    current_ans = hist_ans.clone()

    with torch.no_grad():
        # 获取 GNN 的图特征
        hidden_kt = model_kt.gnn2(graph)

        # 跑一次 KT 获取初始对所有题目的掌握度
        _, _, yt_init, _, _ = model_kt.ktmodel(hidden_kt, current_seq, current_ans)
        p_current = yt_init[0, -1, :]

        for step in range(path_length):
            # 1. 用推荐模型获取 Top-K 候选池
            pred_logits, _, _, _, _, _ = model_rec(current_seq, current_seq, current_seq, current_ans, graph, None)
            last_step_logits = pred_logits[-1, :]
            topk_candidates = torch.topk(last_step_logits, candidate_size).indices.cpu().numpy()

            best_candidate = -1
            max_ep_gain = -999.0
            best_p_next = None

            # 2. 遍历候选池，进行前瞻模拟
            for cand_id in topk_candidates:
                cand_id = int(cand_id)
                # 过滤掉 PAD(0)、Skip(1) 以及已经做过的题
                if cand_id <= 1 or cand_id in generated_path or cand_id in current_seq[0].cpu().numpy():
                    continue

                # 构造模拟序列: [当前序列, 候选题目]
                sim_seq = torch.cat([current_seq, torch.tensor([[cand_id]], device=current_seq.device)], dim=1)
                sim_ans = torch.cat([current_ans, torch.tensor([[1]], device=current_ans.device)], dim=1)  # 假设作对

                # 送入 KT 模型模拟学习后的状态
                _, _, yt_sim, _, _ = model_kt.ktmodel(hidden_kt, sim_seq, sim_ans)
                p_sim = yt_sim[0, -1, :]  # 模拟学习后的掌握度

                # 3. 计算对 Target Set 的 EP 收益
                ep_gain = 0.0
                for target_id in target_set:
                    gain = p_sim[target_id].item() - p_current[target_id].item()
                    room_for_improvement = 1.0 - p_current[target_id].item()
                    room = max(room, 0.1)
                    # 加上 1e-9 防止除以 0 的极小概率事件
                    ep_gain += gain / (room_for_improvement + 1e-9)

                if ep_gain > max_ep_gain:
                    max_ep_gain = ep_gain
                    best_candidate = cand_id
                    best_p_next = p_sim

            # 4. 确定当前步的选择，更新状态
            if best_candidate != -1:
                generated_path.append(best_candidate)
                current_seq = torch.cat([current_seq, torch.tensor([[best_candidate]], device=current_seq.device)],
                                        dim=1)
                current_ans = torch.cat([current_ans, torch.tensor([[1]], device=current_ans.device)], dim=1)
                p_current = best_p_next
            else:
                break

    return generated_path


def generate_dynamic_ep_path(model_rec, model_kt, hist_seq, hist_ans, target_pred, graph, max_length=10,
                             candidate_size=150):
    """
    动态变长路径生成器：针对预测目标 (target_pred) 寻找最短且 EP 收益最高的路径
    """
    generated_path = []
    generated_ans = []  # <--- ✅ 新增：记录模拟的对错
    current_seq = hist_seq.clone()
    current_ans = hist_ans.clone()

    with torch.no_grad():
        hidden_kt = model_kt.gnn2(graph)
        _, _, yt_init, _, _ = model_kt.ktmodel(hidden_kt, current_seq, current_ans)
        p_current = yt_init[0, -1, :]

        for step in range(max_length):  # 最多允许生成 max_length 道题
            # 1. 动态更新推荐候选池 (随着序列变长，推荐也会变)
            pred_logits, _, _, _, _, _ = model_rec(current_seq, current_seq, current_seq, current_ans, graph, None)
            last_step_logits = pred_logits[-1, :]
            topk_candidates = torch.topk(last_step_logits, candidate_size).indices.cpu().numpy()

            best_candidate = -1
            max_ep_gain = -999.0
            best_p_next = None
            best_ans_val = 1

            # 2. 遍历候选池，计算对预测目标的归一化 EP 收益
            for cand_id in topk_candidates:
                cand_id = int(cand_id)
                # 过滤无效题目和已做题目（注意：这里不禁止推荐 target_pred 里的题了！因为如果是必须做的题，可以直接推）
                if cand_id <= 1 or cand_id in generated_path or cand_id in current_seq[0].cpu().numpy():
                    continue

                #  核心学术逻辑：预测他到底能不能做对这道题！
                sim_ans_val = 1 if p_current[cand_id].item() >= 0.5 else 0

                sim_seq = torch.cat([current_seq, torch.tensor([[cand_id]], device=current_seq.device)], dim=1)
                sim_ans = torch.cat([current_ans, torch.tensor([[1]], device=current_ans.device)], dim=1)

                _, _, yt_sim, _, _ = model_kt.ktmodel(hidden_kt, sim_seq, sim_ans)
                p_sim = yt_sim[0, -1, :]

                # 计算针对 【预测目标 target_pred】 的边际归一化增益
                ep_gain = 0.0
                for t_id in target_pred:
                    gain = p_sim[t_id].item() - p_current[t_id].item()
                    room = 1.0 - p_current[t_id].item()
                    room = max(room, 0.1)
                    ep_gain += gain / (room + 1e-9)

                if ep_gain > max_ep_gain:
                    max_ep_gain = ep_gain
                    best_candidate = cand_id
                    best_p_next = p_sim
                    best_ans_val = sim_ans_val  # <--- ✅ 保存这道题的作答结果

            # 3. 核心机制：边际收益早停 (Early Stopping) 保证路径最短
            # 如果加上这道题，对所有预测目标的总归一化收益提升不到 0.01，说明已经“学透了”，立刻停止生成
            if max_ep_gain < 0.01:
                break

                # 4. 否则，将该题加入路径，推进状态
            if best_candidate != -1:
                generated_path.append(best_candidate)
                generated_ans.append(best_ans_val)  # <--- ✅ 把模拟对错加入记录
                current_seq = torch.cat([current_seq, torch.tensor([[best_candidate]], device=current_seq.device)],
                                        dim=1)
                current_ans = torch.cat([current_ans, torch.tensor([[1]], device=current_ans.device)], dim=1)
                p_current = best_p_next
            else:
                break

    return generated_path, generated_ans


def generate_rec_only_path(model_rec, model_kt, hist_seq, hist_ans, graph, max_length=5):
    """
    纯推荐路径生成器 (Rec-Only Baseline)：
    完全抛弃 EP 收益和 KT 前瞻，单纯依据推荐模型输出的概率最大值 (Top-1) 自回归生成序列。
    """
    generated_path = []
    generated_ans = []
    current_seq = hist_seq.clone()
    current_ans = hist_ans.clone()

    with torch.no_grad():
        hidden_kt = model_kt.gnn2(graph)
        _, _, yt_init, _, _ = model_kt.ktmodel(hidden_kt, current_seq, current_ans)
        p_current = yt_init[0, -1, :]

        for step in range(max_length):  # 推理 max_length 步
            # 1. 纯依靠推荐模型给出下一步的预测
            pred_logits = model_rec(current_seq, current_seq, current_seq, current_ans, graph, None)
            last_step_logits = pred_logits[-1, :]

            # 把所有概率从大到小排序
            sorted_candidates = torch.argsort(last_step_logits, descending=True).cpu().numpy()

            best_candidate = -1
            # 2. 找到概率最大，且没有做过的合法题目
            for cand_id in sorted_candidates:
                cand_id = int(cand_id)
                if cand_id > 1 and cand_id not in generated_path and cand_id not in current_seq[0].cpu().numpy():
                    best_candidate = cand_id
                    break

            if best_candidate == -1:
                break  # 没有有效题目可推，结束

            # 3. 模拟作答状态 (保留 ZPD 机制以保证最终 KT 裁判打分的公平性)
            sim_ans_val = 1 if p_current[best_candidate].item() >= 0.5 else 0

            # 4. 将题目无脑加入路径 (不看它是否带来 EP 收益)
            generated_path.append(best_candidate)
            generated_ans.append(sim_ans_val)

            current_seq = torch.cat([current_seq, torch.tensor([[best_candidate]], device=current_seq.device)], dim=1)
            current_ans = torch.cat([current_ans, torch.tensor([[sim_ans_val]], device=current_ans.device)], dim=1)

            # 更新 p_current (为了下一步的作答模拟)
            _, _, yt_sim, _, _ = model_kt.ktmodel(hidden_kt, current_seq, current_ans)
            p_current = yt_sim[0, -1, :]

    return generated_path, generated_ans

def test_epoch(model, validation_data, graph, hypergraph_list, k_list=[1, 5, 10, 15]):  # 移除 kt_loss 和 kt_evaluator 参数
    # 如果没有传，就用自己；传了，就用最优的 kt 模型当裁判
    # kt_referee = kt_evaluator if kt_evaluator is not None else model  # 注释 KT 相关
    ''' Epoch operation in evaluation phase '''
    model.eval()
    # KT 的指标列表
    # auc_test, acc_test = [], []  # 注释 KT 相关
    # p_test_kt, r_test_kt, f1_test_kt = [], [], []
    # ✅ 新增：推荐任务 (Rec) 的 Top-1 指标列表
    acc_test_rec, p_test_rec, r_test_rec, f1_test_rec = [], [], [], []
    # ✅ 新增：排序指标累计器 (Hit, NDCG, MRR)
    ranking_totals = {k: {'hit': 0.0, 'ndcg': 0.0, 'mrr': 0.0} for k in k_list}
    ranking_count = 0
    # ✅ 新增：用于统计全局 EP 收益的累加器
    # total_ep_real = 0.0  # 注释 KT 相关
    # total_ep_rec = 0.0  # <--- 新增：纯推荐的累加器
    # total_ep_gen = 0.0
    # total_delta_ep = 0.0
    # valid_ep_samples = 0
    scores = {}
    for k in k_list:
        scores['hits@' + str(k)] = 0
        scores['map@' + str(k)] = 0

    n_total_words = 0
    # 全局累计器
    paper_ms = [3, 5, 7, 9]
    paper_totals = {m: {"TP": 0, "FP": 0, "FN": 0} for m in paper_ms}

    with torch.no_grad():
        for i, batch in enumerate(
                validation_data):  # tqdm(validation_data, mininterval=2, desc='  - (Validation) ', leave=False):
            # print("Validation batch ", i)
            # prepare data
            # tgt, tgt_timestamp, tgt_idx = batch
            tgt, tgt_timestamp, tgt_idx, ans = batch
            y_gold = tgt[:, 1:].contiguous().view(-1).detach().cpu().numpy()


            # forward
            # pred = model(tgt, tgt_timestamp, tgt_idx, ans, graph, hypergraph_list)
            pred = model(tgt, tgt_timestamp, tgt_idx, ans, graph,
                                                      hypergraph_list.cuda())  # ==================================


            y_pred = pred.detach().cpu().numpy()
            # =========================================================
            # ✅ 新增：计算推荐任务 (Rec) 的 Top-1 分类指标
            # =========================================================
            rec_acc, rec_p, rec_r, rec_f1 = compute_rec_clf_metrics(y_pred, y_gold)
            if rec_acc != -1:
                acc_test_rec.append(rec_acc)
                p_test_rec.append(rec_p)
                r_test_rec.append(rec_r)
                f1_test_rec.append(rec_f1)


            scores_batch, scores_len = metric.compute_metric(y_pred, y_gold, k_list)

            # =========================================================
            # ✅ 新增：计算排序指标 (Hit@K, NDCG@K, MRR@K)
            # =========================================================
            ranking_batch = compute_ranking_metrics(y_pred, y_gold, k_list, pad_id=Constants.PAD, skip_id=1)
            for k in k_list:
                ranking_totals[k]['hit'] += ranking_batch[f'hit@{k}'] * scores_len
                ranking_totals[k]['ndcg'] += ranking_batch[f'ndcg@{k}'] * scores_len
                ranking_totals[k]['mrr'] += ranking_batch[f'mrr@{k}'] * scores_len
            ranking_count += scores_len


            n_total_words += scores_len
            for k in k_list:
                scores['hits@' + str(k)] += scores_batch['hits@' + str(k)] * scores_len
                scores['map@' + str(k)] += scores_batch['map@' + str(k)] * scores_len

    for k in k_list:
        scores['hits@' + str(k)] = scores['hits@' + str(k)] / n_total_words
        scores['map@' + str(k)] = scores['map@' + str(k)] / n_total_words

    # ✅ 新增：计算并打印排序指标 (Hit@K, NDCG@K, MRR@K)
    print('\n========== 📊 推荐排序指标评估 ==========')
    for k in k_list:
        avg_hit = ranking_totals[k]['hit'] / ranking_count if ranking_count > 0 else 0.0
        avg_ndcg = ranking_totals[k]['ndcg'] / ranking_count if ranking_count > 0 else 0.0
        avg_mrr = ranking_totals[k]['mrr'] / ranking_count if ranking_count > 0 else 0.0
        print(f'  Hit@{k}: {avg_hit:.4f} | NDCG@{k}: {avg_ndcg:.4f} | MRR@{k}: {avg_mrr:.4f}')
    print('==========================================\n')


    print('  [Rec Top-1]   Accuracy: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | F1: {:.4f}'.format(
        np.mean(acc_test_rec), np.mean(p_test_rec), np.mean(r_test_rec), np.mean(f1_test_rec)
    ))

    return scores  # , auc_test, acc_test  # 注释 KT 相关，只返回 scores


def test_model(MSHGAT, data_path):
    # kt_loss = KTLoss()  # 注释 KT 相关
    # 1. 修复：正确接收 Split_data 的 7 个返回值
    resource_size, train_history_cas, train_history_t, train, valid, test = Split_data(
        data_path, opt.train_rate, opt.valid_rate, load_dict=True
    )

    test_data = DataLoader(test, batch_size=opt.batch_size, load_dict=True, cuda=False)

    # 2. 永远只用训练集历史来建图！(防止任何测试集泄露)
    relation_graph = ConRelationGraph(data_path)
    hypergraph_list = ConHyperGraphList(train_history_cas, train_history_t, resource_size, num_stages=opt.num_stages,
        decay_rate=opt.decay_rate)

    opt.resource_size = resource_size


    # 3. 实例化两个模型：一个用于推荐，一个用于知识追踪
    model_rec = MSHGAT(opt, dropout=opt.dropout).cuda()

    # 4. 分别加载它们的最优权重
    model_rec.load_state_dict(torch.load(opt.save_rec_path))
    # model_kt.load_state_dict(torch.load(opt.save_kt_path))  # 注释 KT 相关


    scores = test_epoch(model_rec, test_data, relation_graph, hypergraph_list, k_list=[1, 5, 10, 15])


    print('  - (Test) ')
    for metric in scores.keys():
        print(metric + ' ' + str(scores[metric]))


if __name__ == "__main__":
    model = MSHGAT
    train_model(model, opt.data_name)
    # test_model(model, opt.data_name)
    # 多目标评价指标计算
    # gain_test_model(model, opt.data_name, opt)
