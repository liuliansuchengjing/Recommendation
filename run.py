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
parser.add_argument('-save_kt_path', default="./checkpoint/KT_Prediction_M100.pt")
parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
parser.add_argument('-no_cuda', action='store_true')
parser.add_argument('-pos_emb', type=bool, default=True)
# --- KT-guided distillation (train-time) ---
# parser.add_argument('--lambda_kt', type=float, default=5000.0)      # 保持你当前 5000 不变
# parser.add_argument('--lambda_distill', type=float, default=0.1)    # 默认关闭，不影响现有结果
parser.add_argument('--distill_k', type=int, default=30)           # topK 候选大小
parser.add_argument('--distill_tau', type=float, default=1.0)       # 温度
parser.add_argument('--distill_eps', type=float, default=1e-12)     # log/softmax 稳定项


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
        pred_logits = pred_logits.view(B, T-1, V)

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
    yt,           # (B, T-1, V) 或 (B*(T-1), V)
    gold,         # (B, T-1)  真实下一题 id
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
    yt_v = yt[valid]               # (Nvalid, V)

    # ---- topK candidate indices from student logits ----
    kk = min(k, V)
    cand_idx = torch.topk(logits_v, k=kk, dim=-1).indices  # (Nvalid, K)

    # ---- gather candidate scores ----
    stu_cand = torch.gather(logits_v, dim=1, index=cand_idx)  # (Nvalid, K)
    tea_prob = torch.gather(yt_v, dim=1, index=cand_idx)      # (Nvalid, K)

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

def train_epoch(model, training_data, graph, hypergraph_list, loss_func, kt_loss, optimizer):
    # train

    model.train()

    total_loss = 0.0
    n_total_words = 0.0
    n_total_correct = 0.0
    batch_num = 0.0
    auc_train = []
    acc_train = []

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
        pred, pred_res, kt_mask, yt, _, _ = model(tgt, tgt_timestamp, tgt_idx, ans, graph,
                                                  hypergraph_list)  # ==================================

        # loss
        # loss, n_correct = get_performance(loss_func, pred, gold)

        # y_gold = tgt[:, 1:].contiguous().view(-1).cpu().numpy()  # 维度: [(batch_size * (seq_len - 1))]
        # y_pred = pred.detach().cpu().numpy()  # 维度: [batch_size*seq_len-1, num_skills]
        # scores_batch, topk_sequence, scores_len = metric.gaintest_compute_metric(
        #     y_pred, y_gold, batch_size, seq_len, k_list=[5, 15, 20], topnum=5
        # )
        # loss_eff = learning_effect_loss(model, yt, tgt.tolist(), ans.tolist(), topk_sequence, graph, batch_size, topnum = 1)
        # loss_eff = learning_effect_loss(yt)
        # adaptivity_loss = learning_adaptive_loss(tgt.tolist(), ans.tolist(), topk_sequence, opt.data_name)


        # loss = loss + 5000 * loss_kt
        # loss = loss_rec + opt.lambda_kt * loss_kt + opt.lambda_distill * loss_distill
        # print("loss:", loss)
        # print("loss_kt:", loss_kt)
        # 修改 run.py 中的 train_epoch 函数内部逻辑

        # 1. 拿到三个原始损失 (Raw Loss)
        loss_rec_raw, n_correct = get_performance(loss_func, pred, gold)
        loss_kt_raw, auc, acc = kt_loss(pred_res, ans, kt_mask)
        # loss_distill_raw = kt_guided_distill_loss(
        #     pred_logits=pred,  # (B*(T-1), V)
        #     yt=yt,  # (B, T-1, V)
        #     gold=gold,  # (B, T-1)
        #     k=opt.distill_k,
        #     tau=opt.distill_tau,
        #     pad_id=Constants.PAD,
        #     eps=opt.distill_eps,
        # )

        # 2. 分别计算三个任务的自适应权重，并加权
        # 公式：(1 / e^log_var) * Raw_Loss + log_var

        # 推荐系统损失
        weight_rec = torch.exp(-model.log_var_rec)
        loss_rec_adaptive = weight_rec * loss_rec_raw + model.log_var_rec

        # 知识追踪损失
        weight_kt = torch.exp(-model.log_var_kt)
        loss_kt_adaptive = weight_kt * loss_kt_raw + model.log_var_kt

        # # 蒸馏损失 (或者是你提到的第三个其他损失)
        # weight_distill = torch.exp(-model.log_var_distill)
        # loss_distill_adaptive = weight_distill * loss_distill_raw + model.log_var_distill

        # 3. 最终的总 Loss
        loss = loss_rec_adaptive + loss_kt_adaptive

        # 反向传播，这一步会让模型自动去更新那三个 log_var_xxx 参数
        loss.backward()

        # parameter update
        optimizer.step()
        optimizer.update_learning_rate()

        n_total_correct += n_correct
        total_loss += loss.item()
        if auc != -1 and acc != -1:  # ========================================================================================
            auc_train.append(
                auc)  # ====================================================================================
            acc_train.append(
                acc)  # ==========================================================================================

    return total_loss / n_total_words, n_total_correct / n_total_words, auc_train, acc_train


def train_model(MSHGAT, data_path):
    # ========= Preparing DataLoader =========#
    resource_size, total_cascades, timestamps, train, valid, test = Split_data(data_path, opt.train_rate, opt.valid_rate,
                                                                           load_dict=True)

    train_data = DataLoader(train, batch_size=opt.batch_size, load_dict=True, cuda=False)
    valid_data = DataLoader(valid, batch_size=opt.batch_size, load_dict=True, cuda=False)
    test_data = DataLoader(test, batch_size=opt.batch_size, load_dict=True, cuda=False)

    relation_graph = ConRelationGraph(data_path)
    hypergraph_list = ConHyperGraphList(total_cascades, timestamps, resource_size)

    opt.resource_size = resource_size

    # 1. 定义两个不同的最高分记录
    best_rec_hit = 0.0
    best_kt_auc = 0.0

    # 2. 初始化早停机制参数 (新增)
    patience = 5  # 容忍多少个 epoch 没有提升
    patience_counter = 0  # 当前连续没有提升的 epoch 数

    # ========= Preparing Model =========#
    model = MSHGAT(opt, dropout=opt.dropout)
    loss_func = nn.CrossEntropyLoss(size_average=False, ignore_index=Constants.PAD)
    kt_loss = KTLoss()

    params = model.parameters()
    optimizerAdam = torch.optim.Adam(params, betas=(0.9, 0.98), eps=1e-09)
    optimizer = ScheduledOptim(optimizerAdam, opt.d_model, opt.n_warmup_steps)

    if torch.cuda.is_available():
        model = model.cuda()
        loss_func = loss_func.cuda()
        kt_loss = kt_loss.cuda()

    validation_history = 0.0
    best_scores = {}
    best_kt_metrics = {'auc': 0.0, 'acc': 0.0}  # 新增逻辑：专门记录 KT 的最佳指标
    for epoch_i in range(opt.epoch):
        print('\n[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu, train_auc, train_acc = train_epoch(model, train_data, relation_graph, hypergraph_list,
                                                                   loss_func, kt_loss, optimizer)

        # ==================== 新增：获取并计算当前的自适应权重 ====================
        # 因为定义的参数是对数方差 (log_var)，实际的权重是 exp(-log_var)
        # 使用 .item() 将单个元素的 Tensor 转换为普通的 Python 浮点数
        w_rec = torch.exp(-model.log_var_rec).item()
        w_kt = torch.exp(-model.log_var_kt).item()
        # w_distill = torch.exp(-model.log_var_distill).item()
        # =========================================================================

        print('  - (Training)   loss: {loss: 8.5f}, accuracy: {accu:3.3f} %, ' \
              'elapse: {elapse:3.3f} min'.format(
            loss=train_loss, accu=100 * train_accu,
            elapse=(time.time() - start) / 60))
        print('auc_train: {:.10f}'.format(np.mean(train_auc)),
              'acc_train: {:.10f}'.format(np.mean(train_acc)))

        # ==================== 新增：打印权重信息 ====================
        # print(f'  - (Weights)    Rec: {w_rec:.4f} | KT: {w_kt:.4f} | Distill: {w_distill:.4f}')
        print(f'  - (Weights)    Rec: {w_rec:.4f} | KT: {w_kt:.4f} ')
        # ==========================================================

        if epoch_i >= 0:
            start = time.time()
            scores, auc_valid, acc_valid = test_epoch(model, valid_data, relation_graph, hypergraph_list, kt_loss)
            print('  - ( Validation )) ')
            for metric in scores.keys():
                print(metric + ' ' + str(scores[metric]))
            print('auc_valid: {:.10f}'.format(np.mean(auc_valid)),
                  'acc_valid: {:.10f}'.format(np.mean(acc_valid)))
            print("Validation use time: ", (time.time() - start) / 60, "min")

            print('  - (Test) ')
            scores, auc_test, acc_test = test_epoch(model, test_data, relation_graph, hypergraph_list, kt_loss)
            for metric in scores.keys():
                print(metric + ' ' + str(scores[metric]))
            print('auc_test: {:.10f}'.format(np.mean(auc_test)),
                  'acc_test: {:.10f}'.format(np.mean(acc_test)))
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

            # 逻辑 2：保存知识追踪最好的模型（独立保存！）
            current_kt_auc = np.mean(auc_test)
            if current_kt_auc > best_kt_auc:
                best_kt_auc = current_kt_auc
                torch.save(model.state_dict(), opt.save_kt_path)
                best_kt_metrics['auc'] = current_kt_auc
                best_kt_metrics['acc'] = np.mean(acc_test)
                print("Save Best KT Model!")
                is_improved = True  # 只要 KT 变好了，也标记为 True

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

    print("\n Best Knowledge Tracing scores: ")
    for metric in best_kt_metrics.keys():
        print(f"    {metric}: {best_kt_metrics[metric]:.4f}")


def generate_ep_greedy_path(model_rec, model_kt, hist_seq, hist_ans, target_set, graph, path_length=5,
                            candidate_size=50):
    """
    单目标 (EP) 贪心路径生成器
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
        _, _, yt_init, _,_ = model_kt.ktmodel(hidden_kt, current_seq, current_ans)
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

                _, _, yt_sim, _,_ = model_kt.ktmodel(hidden_kt, sim_seq, sim_ans)
                p_sim = yt_sim[0, -1, :]

                # 计算针对 【预测目标 target_pred】 的边际归一化增益
                ep_gain = 0.0
                for t_id in target_pred:
                    gain = p_sim[t_id].item() - p_current[t_id].item()
                    room = 1.0 - p_current[t_id].item()
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
                current_seq = torch.cat([current_seq, torch.tensor([[best_candidate]], device=current_seq.device)],dim=1)
                current_ans = torch.cat([current_ans, torch.tensor([[1]], device=current_ans.device)], dim=1)
                p_current = best_p_next
            else:
                break

    return generated_path, generated_ans


def test_epoch(model, validation_data, graph, hypergraph_list, kt_loss,kt_evaluator=None, k_list=[1, 5, 10, 20], do_ep_eval=True):
    # 如果没有传，就用自己；传了，就用最优的 kt 模型当裁判
    kt_referee = kt_evaluator if kt_evaluator is not None else model
    ''' Epoch operation in evaluation phase '''
    model.eval()
    # KT 的指标列表
    auc_test, acc_test = [], []
    p_test_kt, r_test_kt, f1_test_kt = [], [], []
    # ✅ 新增：推荐任务 (Rec) 的 Top-1 指标列表
    acc_test_rec, p_test_rec, r_test_rec, f1_test_rec = [], [], [], []
    # ✅ 新增：用于统计全局 EP 收益的累加器
    total_ep_real = 0.0
    total_ep_gen = 0.0
    total_delta_ep = 0.0
    valid_ep_samples = 0
    scores = {}
    for k in k_list:
        scores['hits@' + str(k)] = 0
        scores['map@' + str(k)] = 0

    n_total_words = 0
    #全局累计器
    paper_ms = [3, 5, 7, 9]
    paper_totals = {m: {"TP": 0, "FP": 0, "FN": 0} for m in paper_ms}

    with torch.no_grad():
        for i, batch in enumerate(validation_data):  # tqdm(validation_data, mininterval=2, desc='  - (Validation) ', leave=False):
            # print("Validation batch ", i)
            # prepare data
            # tgt, tgt_timestamp, tgt_idx = batch
            tgt, tgt_timestamp, tgt_idx, ans = batch
            y_gold = tgt[:, 1:].contiguous().view(-1).detach().cpu().numpy()

            # =========================================================
            # ✅ 新增：滑动窗口/多时间步的严谨离线评估 (预测 vs 真实)
            # =========================================================
            # 遍历 Batch 里的前 N 个学生（如果是最终跑数据，改成 range(tgt.size(0))）
            # 这里默认测该 Batch 里的所有学生
            for b in range(tgt.size(0)):
                valid_len = (tgt[b] > 1).sum().item()
                if valid_len > 20 and do_ep_eval:

                    # 💡 核心：设置滑动步长。step_size=5 表示第15题测一次，第20题测一次...
                    # 如果你想每个时间步都测，改成 step_size=1（速度会比较慢）
                    step_size = 5

                    # 从第 15 题开始滑动，直到序列剩下的题目不足 5 道为止
                    for t in range(15, valid_len - 4, step_size):

                        # 1. 动态切分当前历史序列 (长度为 t)
                        hist_seq = tgt[b:b + 1, :t].cuda()
                        hist_ans = ans[b:b + 1, :t].cuda()

                        # 2. 提取【真实学习目标】 (接下来的 5 道题)
                        target_actual_seq = tgt[b:b + 1, t:t + 5]
                        target_actual = [int(x) for x in target_actual_seq[0].cpu().numpy() if x > 1]

                        if len(target_actual) > 0:
                            # 3. 让推荐模型推测【预测学习目标 target_pred】
                            # 注意这里的 tgt_idx 切片也是动态对齐的 tgt_idx[b:b+1]
                            hist_pred, _, _, _, _, _ = model(hist_seq, tgt_timestamp[b:b + 1, :t], tgt_idx[b:b + 1],
                                                             hist_ans, graph, hypergraph_list)
                            last_logits = hist_pred[-1, :]
                            top5_preds = torch.topk(last_logits, 5).indices.cpu().numpy()
                            target_pred = [int(x) for x in top5_preds if x > 1]

                            # 4. 计算做题前对真实目标的初始掌握度
                            hidden_kt_eval = kt_referee.gnn2(graph)
                            # 注意：使用 kt_referee 裁判模型
                            _, _, yt_init_eval, _, _ = kt_referee.ktmodel(hidden_kt_eval, hist_seq, hist_ans)
                            p_init = yt_init_eval[0, -1, :]

                            # 取消学霸过滤，全部评估
                            if True:

                                # --- 5. 评估学生真实的瞎做路径 (Base EP) ---
                                # 把未来真实的题放进去计算
                                real_seq = tgt[b:b + 1, :t + len(target_actual)].cuda()
                                real_ans = ans[b:b + 1, :t + len(target_actual)].cuda()
                                _, _, yt_base, _, _ = kt_referee.ktmodel(hidden_kt_eval, real_seq, real_ans)
                                p_base = yt_base[0, -1, :]

                                ep_base = 0.0
                                for t_id in target_actual:
                                    gain = p_base[t_id].item() - p_init[t_id].item()
                                    ep_base += gain / (1.0 - p_init[t_id].item() + 1e-9)

                                # --- 6. 算法登场：为了 target_pred 生成变长优化路径 ---
                                gen_path, gen_ans = generate_dynamic_ep_path(
                                    model_rec=model,
                                    model_kt=kt_referee,
                                    hist_seq=hist_seq,
                                    hist_ans=hist_ans,
                                    target_pred=target_pred,
                                    graph=graph,
                                    max_length=10,
                                    candidate_size=150
                                )

                                # --- 7. 终极审判：评估生成路径在 target_actual 上的收益 (Opt EP) ---
                                # if len(gen_path) > 0:
                                #     opt_seq = torch.cat([hist_seq, torch.tensor([gen_path]).cuda()], dim=1)
                                #     opt_ans = torch.cat([hist_ans, torch.ones((1, len(gen_path))).cuda()], dim=1)
                                #     _, _, yt_opt, _, _ = kt_referee.ktmodel(hidden_kt_eval, opt_seq, opt_ans)
                                #     p_opt = yt_opt[0, -1, :]
                                #
                                #     ep_opt = 0.0
                                #     for t_id in target_actual:
                                #         gain = p_opt[t_id].item() - p_init[t_id].item()
                                #         ep_opt += gain / (1.0 - p_init[t_id].item() + 1e-9)
                                #
                                #     delta_ep = ep_opt - ep_base
                                #
                                #     # 全局累加
                                #     total_ep_real += ep_base
                                #     total_ep_gen += ep_opt
                                #     total_delta_ep += delta_ep
                                #     valid_ep_samples += 1
                                #
                                #     # ⚠️ 为了防止终端被疯狂刷屏，我们把它改成一行极其精简的输出
                                #     print(
                                #         f"  [对决] 样本 {valid_ep_samples:04d} | 学生 {b:02d} | 步 {t:03d} | Base: {ep_base:+.4f} | Opt: {ep_opt:+.4f} | Delta: {delta_ep:+.4f}")
                                # --- 7. 终极审判：评估生成路径在 target_actual 上的收益 (Opt EP) ---
                                if len(gen_path) > 0:
                                    opt_seq = torch.cat([hist_seq, torch.tensor([gen_path]).cuda()], dim=1)
                                    # opt_ans = torch.cat([hist_ans, torch.ones((1, len(gen_path))).cuda()], dim=1)
                                    opt_ans = torch.cat([hist_ans, torch.tensor([gen_ans]).cuda()], dim=1)
                                    _, _, yt_opt, _, _ = kt_referee.ktmodel(hidden_kt_eval, opt_seq, opt_ans)
                                    p_opt = yt_opt[0, -1, :]

                                    ep_opt = 0.0
                                    for t_id in target_actual:
                                        gain = p_opt[t_id].item() - p_init[t_id].item()
                                        ep_opt += gain / (1.0 - p_init[t_id].item() + 1e-9)
                                else:
                                    # 🚨 算法选择放弃推荐（早停），不造成破坏，但也没有收益
                                    ep_opt = 0.0

                                # 🚨 无论算法是否推荐，都必须参与评测！保证不同模型的评价分母绝对一致！
                                delta_ep = ep_opt - ep_base

                                # 全局累加
                                total_ep_real += ep_base
                                total_ep_gen += ep_opt
                                total_delta_ep += delta_ep
                                valid_ep_samples += 1

                                print(
                                    f"  [对决] 样本 {valid_ep_samples:04d} | 学生 {b:02d} | 步 {t:03d} | Base: {ep_base:+.4f} | Opt: {ep_opt:+.4f} | Delta: {delta_ep:+.4f}")
            # =========================================================
            # # =========================================================
            # # ✅ 新增：严谨的离线评估 (预测目标生成 vs 真实目标评估)
            # # =========================================================
            # valid_len = (tgt[0] > 1).sum().item()
            # if valid_len > 20:
            #     # 1. 切分历史序列
            #     hist_seq = tgt[0:1, :15].cuda()
            #     hist_ans = ans[0:1, :15].cuda()
            #
            #     # 2. 提取【真实学习目标 target_actual】 (也就是未来真实的 5 道题)
            #     target_actual_seq = tgt[0:1, 15:20]
            #     target_actual = [int(x) for x in target_actual_seq[0].cpu().numpy() if x > 1]
            #
            #     if len(target_actual) > 0:
            #         # 3. 让推荐模型推测【预测学习目标 target_pred】
            #         hist_pred, _, _, _, _, _ = model(hist_seq, tgt_timestamp[0:1, :15], tgt_idx[0:1], hist_ans, graph, hypergraph_list)
            #         last_logits = hist_pred[-1, :]
            #         top5_preds = torch.topk(last_logits, 5).indices.cpu().numpy()
            #         target_pred = [int(x) for x in top5_preds if x > 1]
            #
            #         # 4. 计算初始掌握度 (针对真实目标 target_actual 算基准)
            #         hidden_kt_eval = model.gnn(graph)
            #         _, _, yt_init_eval, _,_ = kt_loss.ktmodel(hidden_kt_eval, hist_seq, hist_ans) if hasattr(kt_loss,
            #                                                                                                'ktmodel') else model.ktmodel(
            #             hidden_kt_eval, hist_seq, hist_ans)
            #         p_init = yt_init_eval[0, -1, :]
            #
            #         # 过滤学霸：只看对真实目标提升空间足够大的样本
            #         ep_init_abs = sum([p_init[t_id].item() for t_id in target_actual])
            #         if ep_init_abs < len(target_actual) * 1.0:
            #
            #             # --- 5. 评估学生真实的瞎做路径 (Base EP) ---
            #             # 注意：评估的标准始终是 target_actual
            #             real_seq = tgt[0:1, :20].cuda()
            #             real_ans = ans[0:1, :20].cuda()
            #             _, _, yt_base, _,_ = kt_referee.ktmodel(hidden_kt_eval, real_seq, real_ans)
            #             p_base = yt_base[0, -1, :]
            #
            #             ep_base = 0.0
            #             for t_id in target_actual:
            #                 gain = p_base[t_id].item() - p_init[t_id].item()
            #                 ep_base += gain / (1.0 - p_init[t_id].item() + 1e-9)
            #
            #             # --- 6. 算法登场：为了 target_pred 生成变长优化路径 ---
            #             gen_path = generate_dynamic_ep_path(
            #                 model_rec=model,
            #                 model_kt=model,
            #                 hist_seq=hist_seq,
            #                 hist_ans=hist_ans,
            #                 target_pred=target_pred,  # 🎯 算法只知道预测目标！
            #                 graph=graph,
            #                 max_length=10,  # 允许最多生成10道题
            #                 candidate_size=150
            #             )
            #
            #             # --- 7. 终极审判：评估生成路径在 target_actual 上的收益 (Opt EP) ---
            #             if len(gen_path) > 0:
            #                 opt_seq = torch.cat([hist_seq, torch.tensor([gen_path]).cuda()], dim=1)
            #                 opt_ans = torch.cat([hist_ans, torch.ones((1, len(gen_path))).cuda()], dim=1)
            #                 _, _, yt_opt, _,_ = kt_referee.ktmodel(hidden_kt_eval, opt_seq, opt_ans)
            #                 p_opt = yt_opt[0, -1, :]
            #
            #                 ep_opt = 0.0
            #                 for t_id in target_actual:  # 🎯 打分始终看真实目标！
            #                     gain = p_opt[t_id].item() - p_init[t_id].item()
            #                     ep_opt += gain / (1.0 - p_init[t_id].item() + 1e-9)
            #
            #                 delta_ep = ep_opt - ep_base
            #
            #                 # 全局累加
            #                 total_ep_real += ep_base
            #                 total_ep_gen += ep_opt
            #                 total_delta_ep += delta_ep
            #                 valid_ep_samples += 1
            #
            #                 print(f"\n[终极盲测对决] 算法预测目标: {target_pred} | 真实隐式目标: {target_actual}")
            #                 print(
            #                     f"   => 真实路径 (长度 {len(target_actual)}): {target_actual} | 归一化 EP: {ep_base:.4f}")
            #                 print(f"   => 算法路径 (长度 {len(gen_path)}): {gen_path} | 归一化 EP: {ep_opt:.4f}")
            #                 print(
            #                     f"   => 净收益 (Delta): {delta_ep:+.4f}  <-- {'🚀 盲测有效！' if delta_ep > 0 else '📉 盲测落败'}")
            # =========================================================
            # # =========================================================
            # # ✅ 新增：动态前瞻路径生成与 EP 收益对比验证 (仅在第一维度的 Batch 上采样验证)
            # # 为了节约测试时间，我们只取 Batch 里的第一个学生 (b=0) 并且序列长度足够长的来验证
            # # =========================================================
            # valid_len = (tgt[0] > 1).sum().item()
            # if valid_len > 20:  # 只评估做了 20 题以上的学生
            #     # 1. 拆分历史序列(前15题) 和 隐式目标(第16-20题)
            #     hist_seq = tgt[0:1, :15].cuda()  # [1, 15] 确保在GPU上
            #     hist_ans = ans[0:1, :15].cuda()
            #
            #     target_seq = tgt[0:1, 15:20]
            #     # 提取目标的 ID 列表 (过滤 0 和 1)
            #     target_set = [int(x) for x in target_seq[0].cpu().numpy() if x > 1]
            #
            #     if len(target_set) > 0:
            #         # 提前算一下学生还没做这5道题之前的初始掌握度
            #         hidden_kt_eval = model.gnn(graph)
            #         _, _, yt_init_eval, _, _ = kt_loss.ktmodel(hidden_kt_eval, hist_seq, hist_ans) if hasattr(kt_loss,
            #                                                                                                'ktmodel') else model.ktmodel(
            #             hidden_kt_eval, hist_seq, hist_ans)
            #         p_init = yt_init_eval[0, -1, :]
            #         ep_init = sum([p_init[t_id].item() for t_id in target_set])
            #         # 2. 计算真实轨迹的收益 (Real EP)
            #         real_seq = tgt[0:1, :20]
            #         real_ans = ans[0:1, :20]
            #
            #         hidden_kt_eval = model.gnn(graph)
            #         _, _, yt_real, _, _ = kt_loss.ktmodel(hidden_kt_eval, real_seq, real_ans) if hasattr(kt_loss,
            #                                                                                           'ktmodel') else model.ktmodel(
            #             hidden_kt_eval, real_seq, real_ans)
            #         p_real = yt_real[0, -1, :]  # 真实轨迹做完 20 题后的掌握度
            #         ep_real = 0.0
            #         for t_id in target_set:
            #             gain_real = p_real[t_id].item() - p_init[t_id].item()
            #             room = 1.0 - p_init[t_id].item()
            #             ep_real += gain_real / (room + 1e-9)
            #
            #         # 3. 算法生成：基于前 15 题，生成 5 题的贪心推荐路径
            #         gen_path = generate_ep_greedy_path(
            #             model_rec=model,
            #             model_kt=model,
            #             hist_seq=hist_seq,
            #             hist_ans=hist_ans,
            #             target_set=target_set,
            #             graph=graph,
            #             path_length=len(target_set),
            #             candidate_size=50
            #         )
            #
            #         # 4. 计算生成轨迹的收益 (Generated EP)
            #         if len(gen_path) == len(target_set):
            #             gen_seq = torch.cat([hist_seq, torch.tensor([gen_path], device='cuda')], dim=1)
            #             gen_ans = torch.cat([hist_ans, torch.ones((1, len(gen_path)), device='cuda')], dim=1)
            #
            #             _, _, yt_gen, _, _ = model.ktmodel(hidden_kt_eval, gen_seq, gen_ans)
            #             p_gen = yt_gen[0, -1, :]
            #             ep_gen = 0.0
            #             for t_id in target_set:
            #                 gain_gen = p_gen[t_id].item() - p_init[t_id].item()
            #                 room = 1.0 - p_init[t_id].item()
            #                 ep_gen += gain_gen / (room + 1e-9)
            #
            #             # 5. 打印震撼对比结果！
            #             delta_ep = ep_gen - ep_real
            #             print(f"\n[路径生成对决] Target Set (目标题): {target_set}")
            #             print(
            #                 f"   => 真实路径: {[int(x) for x in real_seq[0, 15:20].cpu().numpy()]} | 真实 EP 得分: {ep_real:.4f}")
            #             print(f"   => 算法路径: {gen_path} | 算法 EP 得分: {ep_gen:.4f}")
            #             print(
            #                 f"   => 净收益 (Delta EP): {delta_ep:+.4f}  <-- {'🚀 算法完胜！' if delta_ep > 0 else '📉 算法落败'}")
            #             # ✅ 新增：将当前样本的得分累加到全局池子里
            #             total_ep_real += ep_real
            #             total_ep_gen += ep_gen
            #             total_delta_ep += delta_ep
            #             valid_ep_samples += 1
            # # =========================================================
            # forward
            # pred = model(tgt, tgt_timestamp, tgt_idx, ans, graph, hypergraph_list)
            pred, pred_res, kt_mask, yt, _, _ = model(tgt, tgt_timestamp, tgt_idx, ans, graph, hypergraph_list)  # ==================================

            # for m in paper_ms:
            #     tp, fp, fn = batch_path_counts_from_logits(pred, tgt, m, pad_id=Constants.PAD, skip_id=1)
            #     paper_totals[m]["TP"] += tp
            #     paper_totals[m]["FP"] += fp
            #     paper_totals[m]["FN"] += fn

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
            # ===== KT rerank (evaluation only) =====
            USE_KT_RERANK = False  # 你也可以换成 argparse 参数
            if USE_KT_RERANK:
                y_pred = kt_rerank_logits_numpy(
                    logits_np=y_pred,
                    yt_tensor=yt,
                    base_k=100,  # 先用 100
                    beta=1.0,  # 先用 1.0
                    pad_id=Constants.PAD,
                    skip_id=1
                )


            loss_kt, auc, acc = kt_loss(pred_res.cpu(), ans.cpu(),
                                        kt_mask.cpu())  # ====================================================================
            if auc != -1 and acc != -1:  # ========================================================================================
                auc_test.append(auc)  # ====================================================================================
                acc_test.append(acc)  # ==========================================================================================

            # =========================================================
            # ✅ 新增：计算 Precision, Recall, F1
            # 这里的 pred_res 是知识追踪预测的概率，ans 是真实答案，kt_mask 是掩码
            batch_acc, batch_p, batch_r, batch_f1 = compute_kt_clf_metrics(pred_res, ans[:, 1:], kt_mask)
            if batch_acc != -1:
                p_test_kt.append(batch_p)
                r_test_kt.append(batch_r)
                f1_test_kt.append(batch_f1)
            # =========================================================
            scores_batch, scores_len = metric.compute_metric(y_pred, y_gold, k_list)

            # # 论文同款：path-level P/R/F1（按不同 m 分别算）
            # for m in [3, 5, 7, 9]:
            #     prf = prf_path_level_from_logits(pred, tgt, m=m, pad_id=Constants.PAD, skip_id=1, average="micro")
            #     print(f"[Paper-style PRF] m={m}  P={prf['P']:.4f} R={prf['R']:.4f} F1={prf['F1']:.4f}")


            n_total_words += scores_len
            for k in k_list:
                scores['hits@' + str(k)] += scores_batch['hits@' + str(k)] * scores_len
                scores['map@' + str(k)] += scores_batch['map@' + str(k)] * scores_len

    for k in k_list:
        scores['hits@' + str(k)] = scores['hits@' + str(k)] / n_total_words
        scores['map@' + str(k)] = scores['map@' + str(k)] / n_total_words

    # ✅ 新增：在终端直接打印出这三个指标的平均值
    print('  [KT Metrics] Precision: {:.4f} | Recall: {:.4f} | F1: {:.4f}'.format(
        np.mean(p_test_kt), np.mean(r_test_kt), np.mean(f1_test_kt)
    ))
    print('  [Rec Top-1]   Accuracy: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | F1: {:.4f}'.format(
        np.mean(acc_test_rec), np.mean(p_test_rec), np.mean(r_test_rec), np.mean(f1_test_rec)
    ))
    # ✅ 新增：计算并打印整个测试集上的最终平均 EP 收益
    if valid_ep_samples > 0:
        avg_ep_real = total_ep_real / valid_ep_samples
        avg_ep_gen = total_ep_gen / valid_ep_samples
        avg_delta_ep = total_delta_ep / valid_ep_samples

        print(f"\n========== 🏆 全局 EP 收益最终评估 ({valid_ep_samples} 个有效测试样本) ==========")
        print(f"  => 平均真实 EP (学生自我摸索): {avg_ep_real:.4f}")
        print(f"  => 平均生成 EP (算法智能推荐): {avg_ep_gen:.4f}")
        print(f"  => 绝对平均净收益 (Average Delta EP): {avg_delta_ep:+.4f}")

        # 计算相对提升百分比
        if avg_ep_real > 0:
            improvement_ratio = (avg_delta_ep / avg_ep_real) * 100
            print(f"  => 相对学习效率提升: +{improvement_ratio:.2f}%")
        print("=========================================================================\n")
    return scores, auc_test, acc_test

def test_model(MSHGAT, data_path):
    kt_loss = KTLoss()
    # 1. 获取数据，务必接收 train_history_cas 和 train_history_t 用于纯净建图
    resource_size, total_cascades, timestamps, train, valid, test = Split_data(data_path, opt.train_rate, opt.valid_rate,
                                                                           load_dict=True)

    test_data = DataLoader(test, batch_size=opt.batch_size, load_dict=True, cuda=False)

    # 2. 永远只用训练集历史来建图！(防止任何测试集泄露)
    relation_graph = ConRelationGraph(data_path)
    hypergraph_list = ConHyperGraphList(total_cascades, timestamps, resource_size)

    opt.resource_size = resource_size

    # model = MSHGAT(opt, dropout=opt.dropout)
    # model.load_state_dict(torch.load(opt.save_path))
    # 3. 实例化两个模型：一个用于推荐，一个用于知识追踪
    model_rec = MSHGAT(opt, dropout=opt.dropout).cuda()
    model_kt = MSHGAT(opt, dropout=opt.dropout).cuda()
    # model.cuda()
    kt_loss = kt_loss.cuda()
    # 4. 分别加载它们的最优权重
    model_rec.load_state_dict(torch.load(opt.save_rec_path))
    model_kt.load_state_dict(torch.load(opt.save_kt_path))

    # 使用 model_rec 跑测试
    scores, _, _ = test_epoch(model_rec, test_data, relation_graph, hypergraph_list, kt_loss,kt_evaluator=model_kt,
                                            k_list=[1, 5, 10, 20])
    # =========================================================
    # 6. 测试知识追踪模型 (KT Model) - 只看 KT 指标
    # =========================================================
    print('\n=======================================')
    print('  Testing Knowledge Tracing Model...')
    print('=======================================')
    # 使用 model_kt 跑测试
    _, auc_test, acc_test = test_epoch(model_kt, test_data, relation_graph, hypergraph_list, kt_loss,
                                       k_list=[5], do_ep_eval=False)  # k_list 随便传个小的省时间，因为我们只取 AUC/ACC
    # 在验证阶段调用
    # 使用带有详细显示的版本
    # scores, auc_test, acc_test = test_epoch(
    #     model, test_data, relation_graph, hypergraph_list, kt_loss,
    #     k_list=[5, 10, 20],
    #     show_examples=True,  # 启用示例显示
    # )

    print('  - (Test) ')
    for metric in scores.keys():
        print(metric + ' ' + str(scores[metric]))
    print('auc_test: {:.10f}'.format(np.mean(auc_test)),
          'acc_test: {:.10f}'.format(np.mean(acc_test)))



if __name__ == "__main__":
    model = MSHGAT
    # train_model(model, opt.data_name)
    test_model(model, opt.data_name)
    # 多目标评价指标计算
    # gain_test_model(model, opt.data_name, opt)
