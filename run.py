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

torch.backends.cudnn.deterministic = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
torch.cuda.manual_seed(0)

metric = Metrics()

parser = argparse.ArgumentParser()
parser.add_argument('-data_name', default='Assist')
parser.add_argument('-epoch', type=int, default=125)
parser.add_argument('-batch_size', type=int, default=64)
parser.add_argument('-d_model', type=int, default=64)
parser.add_argument('-initialFeatureSize', type=int, default=64)
parser.add_argument('-train_rate', type=float, default=0.8)
parser.add_argument('-valid_rate', type=float, default=0.1)
parser.add_argument('-n_warmup_steps', type=int, default=1000)
parser.add_argument('-dropout', type=float, default=0.3)
parser.add_argument('-log', default=None)
parser.add_argument('-save_path', default="./checkpoint/DiffusionPrediction_a125_unrepeat.pt")
parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
parser.add_argument('-no_cuda', action='store_true')
parser.add_argument('-pos_emb', type=bool, default=True)

opt = parser.parse_args()
opt.d_word_vec = opt.d_model

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
    loss = crit(pred, gold.contiguous().view(-1))
    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    n_correct = pred.data.eq(gold.data)
    n_correct = n_correct.masked_select(gold.ne(Constants.PAD).data).sum().float()
    return loss, n_correct


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

        n_words = gold.data.ne(Constants.PAD).sum().float()
        n_total_words += n_words
        batch_num += tgt.size(0)

        # training
        optimizer.zero_grad()
        # pred= model(tgt, tgt_timestamp, tgt_idx, ans, graph, hypergraph_list)
        pred, pred_res, kt_mask, yt, _, _ = model(tgt, tgt_timestamp, tgt_idx, ans, graph,
                                                  hypergraph_list)  # ==================================

        # loss
        loss, n_correct = get_performance(loss_func, pred, gold)

        loss_kt, auc, acc = kt_loss(pred_res, ans,
                                    kt_mask)  # ============================================================================

        y_gold = tgt[:, 1:].contiguous().view(-1).cpu().numpy()  # 维度: [(batch_size * (seq_len - 1))]
        y_pred = pred.detach().cpu().numpy()  # 维度: [batch_size*seq_len-1, num_skills]
        scores_batch, topk_sequence, scores_len = metric.gaintest_compute_metric(
            y_pred, y_gold, batch_size, seq_len, k_list=[5, 15, 20], topnum=5
        )
        # loss_eff = learning_effect_loss(model, yt, tgt.tolist(), ans.tolist(), topk_sequence, graph, batch_size, topnum = 1)
        # loss_eff = learning_effect_loss(yt)
        adaptivity_loss = learning_adaptive_loss(tgt.tolist(), ans.tolist(), topk_sequence, opt.data_name)

        loss = loss + 5000 * loss_kt
        # print("loss:", loss)

        # print("loss_kt:", loss_kt)

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
    user_size, total_cascades, timestamps, train, valid, test = Split_data(data_path, opt.train_rate, opt.valid_rate,
                                                                           load_dict=True)

    train_data = DataLoader(train, batch_size=opt.batch_size, load_dict=True, cuda=False)
    valid_data = DataLoader(valid, batch_size=opt.batch_size, load_dict=True, cuda=False)
    test_data = DataLoader(test, batch_size=opt.batch_size, load_dict=True, cuda=False)

    relation_graph = ConRelationGraph(data_path)
    hypergraph_list = ConHyperGraphList(total_cascades, timestamps, user_size)

    opt.user_size = user_size

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
    for epoch_i in range(opt.epoch):
        print('\n[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu, train_auc, train_acc = train_epoch(model, train_data, relation_graph, hypergraph_list,
                                                                   loss_func, kt_loss, optimizer)

        print('  - (Training)   loss: {loss: 8.5f}, accuracy: {accu:3.3f} %, ' \
              'elapse: {elapse:3.3f} min'.format(
            loss=train_loss, accu=100 * train_accu,
            elapse=(time.time() - start) / 60))
        print('auc_test: {:.10f}'.format(np.mean(train_auc)),
              'acc_test: {:.10f}'.format(np.mean(train_acc)))

        if epoch_i >= 0:
            start = time.time()
            scores, auc_test, acc_test = test_epoch(model, valid_data, relation_graph, hypergraph_list, kt_loss)
            print('  - ( Validation )) ')
            for metric in scores.keys():
                print(metric + ' ' + str(scores[metric]))
            print('auc_test: {:.10f}'.format(np.mean(auc_test)),
                  'acc_test: {:.10f}'.format(np.mean(acc_test)))
            print("Validation use time: ", (time.time() - start) / 60, "min")

            print('  - (Test) ')
            scores, auc_test, acc_test = test_epoch(model, test_data, relation_graph, hypergraph_list, kt_loss)
            for metric in scores.keys():
                print(metric + ' ' + str(scores[metric]))
            print('auc_test: {:.10f}'.format(np.mean(auc_test)),
                  'acc_test: {:.10f}'.format(np.mean(acc_test)))
            if validation_history <= sum(scores.values()):
                print("Best Validation hit@100:{} at Epoch:{}".format(scores["hits@20"], epoch_i))
                validation_history = sum(scores.values())
                best_scores = scores
                print("Save best model!!!")
                torch.save(model.state_dict(), opt.save_path)

    print(" -(Finished!!) \n Best scores: ")
    for metric in best_scores.keys():
        print(metric + ' ' + str(best_scores[metric]))


def test_epoch(model, validation_data, graph, hypergraph_list, kt_loss, k_list=[5, 10, 20]):
    ''' Epoch operation in evaluation phase '''
    model.eval()
    auc_test = []
    acc_test = []
    scores = {}
    for k in k_list:
        scores['hits@' + str(k)] = 0
        scores['map@' + str(k)] = 0

    n_total_words = 0
    #全局累计器
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
            pred, pred_res, kt_mask, yt, _, _ = model(tgt, tgt_timestamp, tgt_idx, ans, graph, hypergraph_list)  # ==================================

            for m in paper_ms:
                tp, fp, fn = batch_path_counts_from_logits(pred, tgt, m, pad_id=Constants.PAD, skip_id=1)
                paper_totals[m]["TP"] += tp
                paper_totals[m]["FP"] += fp
                paper_totals[m]["FN"] += fn

            y_pred = pred.detach().cpu().numpy()
            # ===== KT rerank (evaluation only) =====
            USE_KT_RERANK = True  # 你也可以换成 argparse 参数
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
                auc_test.append(
                    auc)  # ====================================================================================
                acc_test.append(
                    acc)  # ==========================================================================================

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

    paper_scores = {}
    for m in paper_ms:
        TP = paper_totals[m]["TP"]
        FP = paper_totals[m]["FP"]
        FN = paper_totals[m]["FN"]
        P = TP / (TP + FP + 1e-12)
        R = TP / (TP + FN + 1e-12)
        F1 = 0.0 if (P + R) == 0 else 2 * P * R / (P + R)
        paper_scores[m] = (P, R, F1)

    print("==== Paper-style Path-level PRF (FULL TESTSET) ====")
    for m in paper_ms:
        P, R, F1 = paper_scores[m]
        print(f"[FINAL PRF] m={m}  P={P:.4f} R={R:.4f} F1={F1:.4f}")

    return scores, auc_test, acc_test

# def test_epoch(model, validation_data, graph, hypergraph_list, kt_loss, k_list=[5, 10, 20],
#                show_examples=True):
#     ''' Epoch operation in evaluation phase '''
#     model.eval()
#     auc_test = []
#     acc_test = []
#     scores = {}
#     for k in k_list:
#         scores['hits@' + str(k)] = 0
#         scores['map@' + str(k)] = 0
#
#     n_total_words = 0
#
#     with torch.no_grad():
#         for i, batch in enumerate(validation_data):
#             # 准备数据
#             tgt, tgt_timestamp, tgt_idx, ans = batch
#             y_gold = tgt[:, 1:].contiguous().view(-1).detach().cpu().numpy()
#
#             # 前向传播
#             pred, pred_res, kt_mask, _, _, _ = model(tgt, tgt_timestamp, tgt_idx, ans, graph, hypergraph_list)
#
#             # 显示每个样本每个时间步的历史和top5推荐（无限制）
#             if show_examples:
#                 batch_size, seq_len = tgt.size()
#
#                 for b in range(batch_size):  # 遍历所有样本，移除限制
#                     # 获取完整的历史序列
#                     full_history = [int(x) for x in tgt[b].cpu().numpy() if int(x) != Constants.PAD and int(x) != 1]
#
#                     # 遍历每个时间步
#                     for t in range(1, min(seq_len, len(full_history) + 1)):  # 遍历所有时间步，移除限制
#                         # 获取当前时间步之前的history
#                         current_history = full_history[:t - 1] if t > 1 else []
#
#                         # 获取该时间步的预测结果
#                         pred_idx = b * (seq_len - 1) + (t - 1)  # 计算预测tensor中的索引
#
#                         if pred_idx < len(pred):  # 确保索引有效
#                             pred_logits = pred[pred_idx, :]  # 获取该时间步的预测logits
#
#                             # 获取top5推荐
#                             top5_recommendations = torch.topk(pred_logits, k=5).indices.cpu().numpy().tolist()
#                             top1_recommendation = top5_recommendations[0]
#
#                             # 获取真实的下一个项目
#                             if t < seq_len:
#                                 real_next = tgt[b, t].item()
#                                 is_top1_correct = top1_recommendation == real_next
#                                 is_in_top5 = real_next in top5_recommendations
#
#                                 print(f"Batch {i}, Sample {b + 1}, Time step {t}:")
#                                 print(f"  Current history: {current_history}")
#                                 print(f"  Top5 recommendations: {top5_recommendations}")
#                                 print(f"  Top1 recommendation: {top1_recommendation}")
#                                 print(f"  Real next item: {real_next}")
#                                 print(f"  Top1 match: {'✓' if is_top1_correct else '✗'}")
#                                 print(f"  In Top5: {'✓' if is_in_top5 else '✗'}")
#                                 print()
#
#             y_pred = pred.detach().cpu().numpy()
#             loss_kt, auc, acc = kt_loss(pred_res.cpu(), ans.cpu(), kt_mask.cpu())
#
#             if auc != -1 and acc != -1:
#                 auc_test.append(auc)
#                 acc_test.append(acc)
#
#             scores_batch, scores_len = metric.compute_metric(y_pred, y_gold, k_list)
#             n_total_words += scores_len
#
#             for k in k_list:
#                 scores['hits@' + str(k)] += scores_batch['hits@' + str(k)] * scores_len
#                 scores['map@' + str(k)] += scores_batch['map@' + str(k)] * scores_len
#
#     for k in k_list:
#         scores['hits@' + str(k)] = scores['hits@' + str(k)] / n_total_words
#         scores['map@' + str(k)] = scores['map@' + str(k)] / n_total_words
#
#     return scores, auc_test, acc_test



def test_model(MSHGAT, data_path):
    kt_loss = KTLoss()
    user_size, total_cascades, timestamps, train, valid, test = Split_data(data_path, opt.train_rate, opt.valid_rate,
                                                                           load_dict=True)

    test_data = DataLoader(test, batch_size=opt.batch_size, load_dict=True, cuda=False)

    relation_graph = ConRelationGraph(data_path)
    hypergraph_list = ConHyperGraphList(total_cascades, timestamps, user_size)

    opt.user_size = user_size

    model = MSHGAT(opt, dropout=opt.dropout)
    model.load_state_dict(torch.load(opt.save_path))
    model.cuda()
    kt_loss = kt_loss.cuda()
    scores, auc_test, acc_test = test_epoch(model, test_data, relation_graph, hypergraph_list, kt_loss,
                                            k_list=[5, 10, 20, 30, 40, 50])
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
    train_model(model, opt.data_name)
    # test_model(model, opt.data_name)
    # 多目标评价指标计算
    # gain_test_model(model, opt.data_name, opt)

