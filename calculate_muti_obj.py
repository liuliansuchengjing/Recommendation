# 多目标评价指标计算

import numpy as np
import Constants
import torch
from graphConstruct import ConRelationGraph, ConHyperGraphList
from dataLoader import Split_data, DataLoader
from Metrics import Metrics, KTLoss
from HGAT import MSHGAT, KTOnlyModel
from Optim import ScheduledOptim
import torch.nn.functional as F
# 初始化一个Metrics类的实例，用于后续的指标计算
metric = Metrics()


# 新增学习效果损失
# def learning_effect_loss(model, post_test_scores,  original_seqs, original_ans, topk_sequence, graph, batch_size, topnum):
def learning_effect_loss(post_test_scores):
    # 预测学习效果
    # 计算有效性（仅当前时间步）
    # yt_before = post_test_scores
    # # 初始化KTOnlyModel
    # kt_model = KTOnlyModel(model)
    # yt_after = simulate_learning(
    #     kt_model, original_seqs, original_ans, topk_sequence, graph, yt_before, batch_size, topnum
    # )  # 维度: [batch_size, seq_len-1, num_skills]
    # topk_indices = torch.tensor(
    #     [[rec + [Constants.PAD] * (topnum - len(rec)) for rec in sample] for sample in topk_sequence],
    #     dtype=torch.long
    # ).cuda()  # 维度: [batch_size, seq_len-1, topnum]
    # batch_gain = metric.compute_effectiveness_tensor(original_seqs,
    #                                           yt_before,  # [batch_size, seq_len-1, num_skills]
    #                                           yt_after,  # [batch_size, seq_len-1, num_skills]
    #                                           topk_indices  # [batch_size, seq_len-1, topnum]
    #                                           )

    predicted_score_gain = post_test_scores
    # actual_score_gain = torch.ones_like(predicted_score_gain)
    # 创建一个与predicted_score_gain形状相同、所有元素为0.5的张量
    actual_score_gain = torch.full_like(predicted_score_gain, fill_value=0.5)
    # 3. 学习效果损失：鼓励推荐能带来高学习收益的资源
    effect_loss = F.mse_loss(predicted_score_gain, actual_score_gain)

    return effect_loss

def learning_adaptive_loss(original_seqs, original_ans, topk_sequence, data_path):
    batch_adaptivity = metric.calculate_adaptivity_tensor(original_seqs, original_ans, topk_sequence, data_path, T=10)
    # 创建一个与predicted_score_gain形状相同、所有元素为0.5的张量
    # expect_adaptivity = torch.full_like(batch_adaptivity, fill_value=0.5)
    expect_adaptivity = torch.ones_like(batch_adaptivity)
    # 3. 学习效果损失：鼓励推荐能带来高学习收益的资源
    adpativity_loss = F.mse_loss(batch_adaptivity, expect_adaptivity)

    return adpativity_loss

def generate_candidate_seq(y_pred, batch_size, seq_len, topnum, cannum):
    total_samples = batch_size * (seq_len - 1)
    candidate_seq = np.zeros((batch_size, seq_len - 1, cannum), dtype=int)

    for i in range(total_samples):
        p_ = y_pred[i]
        p_sort = p_.argsort()
        topk = p_sort[-topnum:][::-1]

        all_indices = np.arange(len(p_))
        remaining_indices = np.setdiff1d(all_indices, topk)

        selected_indices = np.random.choice(remaining_indices, cannum, replace=False)

        b = i // (seq_len - 1)
        s = i % (seq_len - 1)
        candidate_seq[b, s] = selected_indices

    return candidate_seq

def simulate_learning(kt_model, original_seqs, original_ans, topk_sequence, graph, yt_before, batch_size, K):
    """
    单时间步模拟学习（推荐插入到当前时间步之后，仅保留到t+K）
    输入维度说明：
    original_seqs: list[list[int]] 形状 [batch_size, original_seq_len]
    original_ans: list[list[int]] 形状 [batch_size, original_seq_len]
    topk_sequence: list[list[list[int]] 形状 [batch_size, seq_len-1, K]
    yt_before: Tensor 形状 [batch_size, seq_len-1, num_skills]
    """


    seq_len = len(topk_sequence[0]) # 推荐时间步数（seq_len - 1）
    yt_after_list = []  # 存储每个时间步插入后的知识状态

    for t in range(seq_len):# 遍历每个推荐时间步
        extended_inputs = []# 扩展后的输入序列，形状 [batch_size, insert_pos + K]
        extended_ans = []# 扩展后的答案序列，形状 [batch_size, insert_pos + K]

        for b in range(batch_size):# 遍历每个样本
            original_seq = original_seqs[b]  # 获取当前样本的原始序列, 维度: [original_seq_len]
            original_an = original_ans[b]
            recommended = topk_sequence[b][t]  # 获取当前时间步的推荐资源, 维度: [K]
            insert_pos = t + 1  # 插入位置（当前时间步之后）
            new_seq = original_seq[:insert_pos] + recommended# 构建新的序列，包含原始序列前t+1个元素和推荐的K个元素

            extended_inputs.append(new_seq)# 将新序列添加到扩展输入列表中 [insert_pos + K]

            pred_answers = (yt_before[b, t, recommended] >= 0).float().tolist() # 基于yt_before的当前时间步t生成预测答案
            new_ans = original_an[:insert_pos] + pred_answers # 构建答案序列，前t+1个位置为原始序列答案，推荐位置为预测答案
            extended_ans.append(new_ans)# 将新答案序列添加到扩展答案列表中# [insert_pos + K]

        # 确定扩展后序列的最大长度
        max_len = insert_pos + K
        # 初始化填充后的输入张量，使用PAD填充 [batch_size, max_len]
        padded_inputs = torch.full((batch_size, max_len), Constants.PAD, dtype=torch.long).cuda()  # 维度: [batch_size, max_len]
        # 初始化填充后的答案张量，初始值为0 # [batch_size, max_len]
        padded_ans = torch.zeros((batch_size, max_len), dtype=torch.float).cuda()  # 维度: [batch_size, max_len]

        # 遍历每个样本，将扩展后的序列和答案填充到张量中
        for b in range(batch_size):
            seq = extended_inputs[b]
            ans = extended_ans[b]
            padded_inputs[b, :len(seq)] = torch.LongTensor(seq).cuda()# 填充输入
            padded_ans[b, :len(ans)] = torch.FloatTensor(ans).cuda()# 填充答案

        # 重新预测知识状态，不进行梯度计算
        with torch.no_grad():
            yt_after = kt_model(padded_inputs, padded_ans, graph)  # 维度: [batch_size, max_len, num_skills]
        # 提取插入后的知识状态（时间步t+K，即扩展后的最后一个时间步），并移至CPU
        p_after = yt_after[:, -1, :].detach().cpu()  # 维度: [batch_size, num_skills]
        # 将插入后的知识状态添加到列表中
        yt_after_list.append(p_after)

        # 显式释放GPU内存
        del padded_inputs, padded_ans, yt_after
        torch.cuda.empty_cache()

    # 将所有时间步的插入后的知识状态堆叠成一个张量
    return torch.stack(yt_after_list, dim=1)  # 维度: [batch_size, seq_len-1, num_skills]


def gain_test_epoch(model, kt_model, test_data, graph, hypergraph_list, kt_loss, data_path, k_list=[5, 10, 20], topnum=6, cannum = 14):
    model.eval()# 将模型设置为评估模式
    auc_test, acc_test = [], []
    scores = {'hits@' + str(k): 0.0 for k in k_list}
    scores.update({'map@' + str(k): 0.0 for k in k_list})
    total_valid_samples = 0 # 记录有效的样本总数
    total_gain = 0.0    # 有效性指标的累积值
    total_diff = 0.0
    total_div = 0.0
    total_valid_count = 0     # 有效的批次数量
    # 训练/验证循环外部初始化累加器
    total_metrics = {
        'effectiveness': 0.0,
        'adaptivity': 0.0,
        'diversity': 0.0,
        'preference': 0.0
    }
    n_total_words = 0
    scores_single = {}
    for k in k_list:
        scores_single['hits@' + str(k)] = 0
        scores_single['map@' + str(k)] = 0

    with torch.no_grad():    # 不进行梯度计算
        for i, batch in enumerate(test_data):  # 遍历测试数据的每个批次
            tgt, tgt_timestamp, tgt_idx, ans = batch    # 解包批次数据
            batch_size, seq_len = tgt.size()  # tgt维度: [batch_size, seq_len]
            tgt = tgt.cuda()
            ans = ans.cuda()
            # 前向传播，得到预测结果、预测资源、掩码和原始知识状态
            pred, pred_res, kt_mask, yt_before, hidden = model(tgt, tgt_timestamp.cuda(), tgt_idx.cuda(), ans, graph, hypergraph_list)
            # pred维度: [batch_size*seq_len-1, num_users]
            # pred_res维度: [batch_size, seq_len]
            # kt_mask维度: [batch_size, seq_len]
            # yt_before维度: [batch_size, seq_len-1, num_skills]
            # hidden维度: [seq_len, num_skills]

            loss_kt, auc, acc = kt_loss(pred_res, ans, kt_mask)# 计算当前批次的AUC和ACC
            if auc != -1:    # 如果AUC计算成功，则将其添加到列表中
                auc_test.append(auc)
                acc_test.append(acc)

            # 处理预测结果，生成TopK序列
            y_gold = tgt[:, 1:].contiguous().view(-1).cpu().numpy()  # 维度: [(batch_size * (seq_len - 1))]
            y_pred = pred.detach().cpu().numpy()  # 维度: [batch_size*seq_len-1, num_skills]
            # print("y_pred",y_pred.shape)# 维度: [batch_size, seq_len-1, num_skills]
            # print("y_gold",y_gold.shape) # 维度: [(batch_size * (seq_len - 1))]
            scores_batch, topk_sequence, scores_len = metric.gaintest_compute_metric(
                y_pred, y_gold, batch_size, seq_len, k_list, topnum
            )

            # 累加有效的样本数
            total_valid_samples += scores_len

            # 累加每个指标的值
            for k in k_list:
                scores['hits@' + str(k)] += scores_batch['hits@' + str(k)] * scores_len
                scores['map@' + str(k)] += scores_batch['map@' + str(k)] * scores_len

            # 模拟学习（逐时间步插入）
            original_seqs = tgt.tolist()  # 维度: [batch_size, seq_len]
            original_ans = ans.tolist()  # 维度: [batch_size, seq_len]
            topk_indices = torch.tensor(
                [[rec + [Constants.PAD] * (topnum - len(rec)) for rec in sample] for sample in topk_sequence],
                dtype=torch.long
            ).cuda()  # 维度: [batch_size, seq_len-1, topnum]

            yt_after = simulate_learning(
                kt_model, original_seqs, original_ans, topk_sequence, graph, yt_before, batch_size, topnum
            )  # 维度: [batch_size, seq_len-1, num_skills]

            # 计算有效性（仅当前时间步）
            batch_gain = metric.compute_effectiveness(original_seqs,
                yt_before,  # [batch_size, seq_len-1, num_skills]
                yt_after,  # [batch_size, seq_len-1, num_skills]
                topk_indices  # [batch_size, seq_len-1, topnum]
            )
            # 累加有效性指标
            total_gain += batch_gain
            batch_adaptivity = metric.calculate_adaptivity(original_seqs, original_ans, topk_sequence, data_path)
            total_diff += batch_adaptivity
            batch_diversity = metric.calculate_diversity(original_seqs,topk_indices, hidden, batch_size, seq_len, topnum)
            total_div += batch_diversity
            # 累加有效的批次数量
            total_valid_count += 1
            pred_probs = torch.sigmoid(pred).cpu().numpy() if pred is not None else None
            result = metric.combined_metrics(yt_before, yt_after, topk_sequence, original_seqs, hidden,
                             data_path, batch_size, seq_len, pred_probs, topnum, 5)

            # 累加指标值
            total_metrics['effectiveness'] += result['effectiveness']
            total_metrics['adaptivity'] += result['adaptivity']
            total_metrics['diversity'] += result['diversity']
            total_metrics['preference'] +=result['preference']

            # 单独计算hit和map指标
            scores_batch_single, scores_len_single = metric.compute_metric(y_pred, y_gold, k_list)
            n_total_words += scores_len_single
            for k in k_list:
                scores_single['hits@' + str(k)] += scores_batch_single['hits@' + str(k)] * scores_len_single
                scores_single['map@' + str(k)] += scores_batch_single['map@' + str(k)] * scores_len_single

    for k in k_list:
        scores_single['hits@' + str(k)] = scores_single['hits@' + str(k)] / n_total_words
        scores_single['map@' + str(k)] = scores_single['map@' + str(k)] / n_total_words

    # 计算全局均值
    for k in k_list:
        if total_valid_samples > 0:
            scores['hits@' + str(k)] /= total_valid_samples
            scores['map@' + str(k)] /= total_valid_samples
        else:
            scores['hits@' + str(k)] = 0.0
            scores['map@' + str(k)] = 0.0

    # 计算平均有效性
    E_p = total_gain / total_valid_count if total_valid_count > 0 else 0.0
    Adaptivity = total_diff / total_valid_count if total_valid_count > 0 else 0.0
    Diversity = total_div / total_valid_count if total_valid_count > 0 else 0.0
    # Preference = total_div / total_valid_count if total_valid_count > 0 else 0.0

    # 计算全局平均值
    final_metrics = {
        k: v / total_valid_count
        for k, v in total_metrics.items()
    }

    return scores, auc_test, acc_test, E_p, Adaptivity, Diversity, final_metrics, scores_single


def gain_test_model(model, data_path, opt):
    # 初始化KTLoss类的实例
    kt_loss = KTLoss()
    # 分割数据，得到用户数量、总级联数、时间戳、训练集、验证集和测试集
    user_size, total_cascades, timestamps, train, valid, test = Split_data(data_path, opt.train_rate, opt.valid_rate,
                                                                           load_dict=True)

    # 创建测试数据加载器
    test_data = DataLoader(test, batch_size=opt.batch_size, load_dict=True, cuda=False)

    # 构建关系图
    relation_graph = ConRelationGraph(data_path)
    # 构建超图列表
    hypergraph_list = ConHyperGraphList(total_cascades, timestamps, user_size)

    # 将用户数量添加到选项中
    opt.user_size = user_size

    # 初始化MSHGAT模型
    model = MSHGAT(opt, dropout=opt.dropout)
    # 加载预训练模型参数
    model.load_state_dict(torch.load(opt.save_path))
    # 将模型移到GPU上
    model.cuda()
    # 将KTLoss移到GPU上
    kt_loss = kt_loss.cuda()
    # 初始化KTOnlyModel
    kt_model = KTOnlyModel(model)

    # 运行测试流程
    scores, auc_test, acc_test, E_p, Adaptivity, Diversity, final_metrics, scores_single = gain_test_epoch(
        model, kt_model, test_data, relation_graph, hypergraph_list, kt_loss, data_path
    )

    print('  - (Test--scores_single) ')
    for metric in scores_single.keys():
        print(metric + ' ' + str(scores_single[metric]))

    # 打印结果
    print('\n===== 综合指标 =====')
    for metric in scores.keys():
        print(f'{metric}: {scores[metric]:.4f}')
    print(f'AUC: {np.mean(auc_test):.4f}')
    print(f'ACC: {np.mean(acc_test):.4f}')
    print(f'Effectiveness (E_p): {E_p:.4f}')
    print(f'Adaptivity : {Adaptivity:.4f}')
    print(f'Diversity :{Diversity:.4f}')
    print(f"全局有效性: {final_metrics['effectiveness']:.4f}")
    print(f"全局适应性: {final_metrics['adaptivity']:.4f}")
    print(f"全局多样性: {final_metrics['diversity']:.4f}")
    print(f"全局准确性: {final_metrics['preference']:.4f}")
