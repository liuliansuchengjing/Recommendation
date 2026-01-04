import numpy as np
import pickle
from collections import defaultdict
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score
from dataLoader import Options
import torch.nn.functional as F

class Metrics(object):

    def __init__(self):
        super().__init__()
        self.PAD = 0

    def apk(self, actual, predicted, k=10):
        score = 0.0
        num_hits = 0.0

        for i, p in enumerate(predicted):
            if p in actual and p not in predicted[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)

        # if not actual:
        # 	return 0.0
        return score / min(len(actual), k)

    def compute_metric(self, y_prob, y_true, k_list=[10, 50, 100]):
        '''
            y_true: (#samples, )
            y_pred: (#samples, #users)
        '''
        scores_len = 0
        y_prob = np.array(y_prob)
        y_true = np.array(y_true)

        scores = {'hits@' + str(k): [] for k in k_list}
        scores.update({'map@' + str(k): [] for k in k_list})
        for p_, y_ in zip(y_prob, y_true):
            if y_ != self.PAD:
                scores_len += 1.0
                p_sort = p_.argsort()
                for k in k_list:
                    topk = p_sort[-k:][::-1]
                    scores['hits@' + str(k)].extend([1. if y_ in topk else 0.])
                    scores['map@' + str(k)].extend([self.apk([y_], topk, k)])

        scores = {k: np.mean(v) for k, v in scores.items()}
        return scores, scores_len

    # 多目标评价指标计算
    # --------------------计算击中率和排名质量指标-------------------------------------
    def gaintest_compute_metric(self, y_prob, y_true, batch_size, seq_len, k_list=[5, 15, 20], topnum=5):
        # 初始化所有指标字典，用于存储hits@k和map@k的累积值
        scores = {'hits@' + str(k): 0.0 for k in k_list}
        scores.update({'map@' + str(k): 0.0 for k in k_list})
        # 记录有效的样本数
        valid_samples = 0

        # 预初始化详细结果列表，确保长度与总样本数一致
        total_samples = batch_size * (seq_len - 1)
        detailed_results = [{'topk_resources': [], 'hits': 0.0} for _ in range(total_samples)]

        # 遍历每个样本
        for i in range(total_samples):
            # 获取当前样本的预测概率
            p_ = y_prob[i]  # 维度: [num_users]
            # 获取当前样本的真实标签
            y_ = y_true[i]
            # 如果真实标签为PAD，则跳过当前样本
            if y_ == self.PAD:
                continue  # 保持预初始化的空列表

            # 增加有效的样本数
            valid_samples += 1
            # 对预测概率进行排序
            p_sort = p_.argsort()
            # 获取TopK资源
            topk = p_sort[-topnum:][::-1] if topnum else p_sort[-5:][::-1]  # 维度: [topnum]

            # 直接更新详细结果的对应位置
            detailed_results[i]['topk_resources'] = topk.tolist()
            detailed_results[i]['hits'] = 1.0 if y_ in topk else 0.0

            # 计算hits@k和map@k
            for k in k_list:
                hits = 1.0 if y_ in p_sort[-k:][::-1] else 0.0
                scores['hits@' + str(k)] += hits
                scores['map@' + str(k)] += self.apk([y_], p_sort[-k:][::-1], k)

        # 归一化为均值
        for k in k_list:
            if valid_samples > 0:
                scores['hits@' + str(k)] /= valid_samples
                scores['map@' + str(k)] /= valid_samples
            else:
                scores['hits@' + str(k)] = 0.0
                scores['map@' + str(k)] = 0.0

        # 生成TopK序列
        topk_sequence = []
        for b in range(batch_size):
            seq = []
            for t in range(seq_len - 1):
                idx = b * (seq_len - 1) + t
                if idx < len(detailed_results):
                    seq.append(detailed_results[idx]['topk_resources'][:topnum])
                else:
                    seq.append([])
            topk_sequence.append(seq)  # 维度: [batch_size, seq_len-1, topnum]

        return scores, topk_sequence, valid_samples

    # Metrics.py 的 compute_effectiveness 方法
    # --------------------有效性计算-------------------------------------
    def compute_effectiveness(self, original_seqs, yt_before, yt_after, topk_indices):
        """
        有效性计算（独立处理每个推荐资源）
        输入维度说明：
        yt_before: [B, seq_len-1, num_skills]（原始知识状态）
        yt_after: [B, seq_len-1, num_skills]（每个时间步插入后的最终知识状态）
        topk_indices: [B, seq_len-1, K]
        """
        batch_size, seq_len_minus_1, K = topk_indices.shape
        total_gain = 0.0
        valid_count = 0

        for b in range(batch_size):
            for t in range(seq_len_minus_1):
                if original_seqs[b][t] != self.PAD:
                    recommended = topk_indices[b, t].tolist()
                    valid_rec = [r for r in recommended if 0 <= r < yt_before.shape[2]]
                    if not valid_rec:
                        continue

                    # 原始知识状态（时间步t）
                    pb_values = yt_before[b, t, valid_rec]  # [K_valid]
                    # 插入后的知识状态（时间步t+K）
                    pa_values = yt_after[b, t, valid_rec]  # [K_valid]

                    if valid_rec:
                        for k in range(len(valid_rec)):
                            pb = pb_values[k].item()
                            pa = pa_values[k].item()
                            # if pb < 1.0 - 1e-6 and pa > 0:
                            if pb < 0.9 and pa > 0:
                                gain = (pa - pb) / (1.0 - pb)
                                total_gain += gain
                                valid_count += 1

        return total_gain / valid_count if valid_count > 0 else 0.0

    def compute_effectiveness_tensor(self, original_seqs, yt_before, yt_after, topk_indices):
        """
        有效性计算（独立处理每个推荐资源，返回维度对齐的张量）
        输入维度说明：
        yt_before: [B, seq_len-1, num_skills]（原始知识状态）
        yt_after: [B, seq_len-1, num_skills]（每个时间步插入后的最终知识状态）
        topk_indices: [B, seq_len-1, K]（每个batch-时间步的TopK推荐资源）

        返回值：
        gain_tensor: [B, seq_len-1, K] 每个推荐资源的独立增益值（无效位置填充0）
        avg_gain: 标量（兼容原逻辑的全局平均增益）
        """
        # 1. 统一所有张量到同一设备（核心修复）
        device = yt_before.device
        yt_after = yt_after.to(device)
        topk_indices = topk_indices.to(device)

        # 2. 处理PAD（确保是标量或同设备张量）
        pad_val = self.PAD.item() if isinstance(self.PAD, torch.Tensor) else self.PAD

        batch_size, seq_len_minus_1, K = topk_indices.shape
        num_skills = yt_before.shape[2]

        # 3. 初始化增益张量（指定设备）
        gain_tensor = torch.zeros((batch_size, seq_len_minus_1, K), dtype=torch.float32, device=device)

        total_gain = 0.0
        valid_count = 0

        # 4. 逐batch、逐时间步、逐推荐资源计算
        for b in range(batch_size):
            for t in range(seq_len_minus_1):
                # 跳过PAD位置
                if original_seqs[b][t] == pad_val:
                    continue

                # 当前时间步的TopK推荐资源（已在device上）
                recommended = topk_indices[b, t]  # [K]
                # 过滤有效索引（0 <= r < num_skills）
                valid_mask = (recommended >= 0) & (recommended < num_skills)  # [K]
                valid_rec = recommended[valid_mask]  # [K_valid]
                if len(valid_rec) == 0:
                    continue

                # 索引操作（同设备，无报错）
                pb_values = yt_before[b, t, valid_rec]  # [K_valid]
                pa_values = yt_after[b, t, valid_rec]  # [K_valid]

                # 逐推荐资源填充增益
                for k_idx in range(len(valid_rec)):
                    # 找到该资源在原始TopK中的位置
                    original_k = torch.where(recommended == valid_rec[k_idx])[0].item()

                    pb = pb_values[k_idx].item()
                    pa = pa_values[k_idx].item()

                    if pb < 0.9 and pa > 0:
                        gain = (pa - pb) / (1.0 - pb)
                        gain_tensor[b, t, original_k] = gain
                        total_gain += gain
                        valid_count += 1

        # 计算全局平均增益
        avg_gain = total_gain / valid_count if valid_count > 0 else 0.0

        return gain_tensor

    # --------------------适应性计算-------------------------------------
    def calculate_adaptivity(self, original_seqs, topk_sequence, data_name, T=10, epsilon=1e-5):
        """
        计算适应性表征参数（Adaptivity）

        参数:
            original_seqs: 原始序列列表 [batch_size, seq_len]
            topk_sequence: 推荐序列列表 [batch_size, seq_len-1, K]
            u2idx_path: u2idx.pickle文件路径
            difficulty_path: difficulty.csv文件路径
            T: 历史窗口大小
            epsilon: 平滑项

        返回:
            adaptivity_scores: 每个样本的适应性分数列表
        """
        # 1. 加载u2idx映射和难度数据
        options = Options(data_name)
        with open(options.idx2u_dict, 'rb') as handle:
            idx2u = pickle.load(handle)

        # 加载难度数据 - 修改为使用逗号分隔
        difficulty_data = {}
        with open(options.difficult_file, 'r') as f:
            next(f)  # 跳过标题行
            for line in f:
                line = line.strip()
                if not line:  # 跳过空行
                    continue

                # 使用逗号分割
                parts = line.split(',')
                if len(parts) >= 1:
                    challenge_id = parts[0].strip()
                    diff = parts[1].strip()
                    try:
                        difficulty_data[int(challenge_id)] = int(diff)
                    except (ValueError, IndexError):
                        continue  # 跳过格式错误的行

        # 2. 预处理函数：将习题ID转换为难度
        def get_difficulty(idx):
            """通过索引获取习题难度"""
            challenge_id = int(idx2u[idx])  # 转换为原始ID
            return difficulty_data.get(challenge_id, 1)  # 默认难度为1

        # 3. 计算每个样本的适应性分数
        adaptivity_scores = []
        # 为每个推荐时间步计算适应性
        adaptivity_sum = 0.0
        valid_count = 0

        for b in range(len(original_seqs)):
            seq = original_seqs[b]
            recs = topk_sequence[b]

            # 获取历史答题记录（难度和结果）
            history_diffs = []
            history_results = []

            # 遍历原始序列（去掉最后一个时间步，因为我们要预测它）
            for t in range(len(seq) - 1):
                challenge_idx = seq[t]
                result = 1  # 假设所有历史答题结果都是正确的（根据原始代码逻辑）

                # 获取难度
                if challenge_idx > 1:
                    diff = get_difficulty(challenge_idx)
                    history_diffs.append(diff)
                    history_results.append(result)

            # 遍历推荐序列
            for t in range(len(recs)):
                # 计算当前时间步之前的能力值delta
                if len(history_diffs[:t]) < T // 2:  # 如果历史数据小于窗口一半，使用默认值
                    delta = 1.0
                else:
                    # 使用当前时间步之前的历史数据
                    recent_diffs = history_diffs[max(0, t - T):t]
                    recent_results = history_results[max(0, t - T):t]

                    if len(recent_diffs) > 0:
                        numerator = sum(d * r for d, r in zip(recent_diffs, recent_results))
                        denominator = sum(recent_results) + epsilon
                        delta = numerator / denominator
                    else:
                        delta = 1.0  # 默认能力值

                # 计算当前时间步推荐资源的适应性
                for rec in recs[t]:
                    if rec > 1:
                        # 获取推荐资源的难度
                        rec_diff = get_difficulty(rec)
                        # 计算1-|δ-Dif_i|
                        adaptivity = 1 - abs(delta - rec_diff)
                        adaptivity_sum += adaptivity
                        valid_count += 1

            # # 计算平均适应性
            # if valid_count > 0:
            #     adaptivity_scores.append(adaptivity_sum / valid_count)
            # else:
            #     adaptivity_scores.append(0.0)  # 无有效推荐时得分为0

        return adaptivity_sum / valid_count if valid_count > 0 else 0.0

    # --------------------多样性计算-------------------------------------
    def calculate_diversity(self, original_seqs, topk_sequence, hidden, batch_size, seq_len, topnum):
        """
        计算多样性表征参数（Diversity）

        参数:
            topk_sequence: 推荐序列列表 [batch_size, seq_len-1, topnum]
            hidden: GNN输出的嵌入向量 [num_skills, emb_dim]
            batch_size: 批次大小
            seq_len: 序列长度
            topnum: 每个时间步推荐的资源数量

        返回:
            diversity_scores: 每个样本的多样性分数列表 [batch_size]
        """
        # 将topk_sequence转换为张量（如果还不是张量）
        if not isinstance(topk_sequence, torch.Tensor):
            topk_indices = torch.tensor(
                [[rec + [self.PAD] * (topnum - len(rec)) for rec in sample] for sample in topk_sequence],
                dtype=torch.long
            ).cuda()
        else:
            topk_indices = topk_sequence

        # 初始化结果存储
        diversity_scores = torch.zeros(batch_size).cuda()
        total_diversity = 0.0
        valid_pairs = 0

        for b in range(batch_size):
            # 获取当前样本的所有推荐资源嵌入
            rec_embeddings = hidden[topk_indices[b]]  # [seq_len-1, topnum, emb_dim]

            # 遍历每个时间步的推荐
            for t in range(seq_len - 1):
                if original_seqs[b][t] != self.PAD:
                    # 获取当前时间步的推荐资源嵌入
                    current_embs = rec_embeddings[t]  # [topnum, emb_dim]

                    # 计算所有资源对之间的相似度
                    for i in range(topnum):
                        for j in range(i + 1, topnum):  # 避免重复计算
                            # 跳过无效资源（PAD）
                            if topk_indices[b, t, i] == 0 or topk_indices[b, t, j] == 0:
                                continue

                            # 计算余弦相似度
                            sim = F.cosine_similarity(
                                current_embs[i].unsqueeze(0),
                                current_embs[j].unsqueeze(0)
                            )

                            # 累加多样性贡献
                            total_diversity += (1 - sim.item())
                            valid_pairs += 1

            # 计算平均多样性
            # if valid_pairs > 0:
            #     diversity_scores[b] = total_diversity / valid_pairs
            # else:
            #     diversity_scores[b] = 0.0  # 无有效对时得分为0

        return total_diversity / valid_pairs if valid_pairs > 0 else 0.0

    # --------------------四个指标计算-------------------------------------
    def combined_metrics(self, yt_before, yt_after, topk_sequence, original_seqs, hidden,
                         data_name, batch_size, seq_len, pred_probs, topnum=5, T=10, epsilon=1e-5):
        """
        合并后的目标函数，同时计算有效性、适应性和多样性
        输入维度说明：
        yt_before: [B, seq_len, num_skills]
        yt_after: [B, seq_len-1, num_skills]
        topk_sequence: [B, seq_len-1, K]
        original_seqs: [B, seq_len]
        hidden: [num_skills, emb_dim]
        """
        # 初始化总和存储
        metrics = {
            'total_effectiveness': 0.0,
            'total_adaptivity': 0.0,
            'total_diversity': 0.0,
            'total_preference': 0.0,
            'step_records': defaultdict(list)  # 存储每个时间步的指标
        }
        valid_count = 0
        eff_valid_count = 0
        ada_valid_count = 0
        div_valid_count = 0
        pre_valid_count = 0

        # 预处理：加载难度数据和映射（移出循环）
        options = Options(data_name)
        with open(options.idx2u_dict, 'rb') as handle:
            idx2u = pickle.load(handle)

        # 加载难度数据
        difficulty_data = {}
        with open(options.difficult_file, 'r') as f:
            next(f)
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    try:
                        difficulty_data[int(parts[0])] = int(parts[1])
                    except ValueError:
                        continue

        def get_difficulty(idx):
            """通过索引获取习题难度"""
            challenge_id = int(idx2u[idx])  # 转换为原始ID
            return difficulty_data.get(challenge_id, 1)  # 默认难度为1

        # 转换topk序列为张量
        if not isinstance(topk_sequence, torch.Tensor):
            topk_indices = torch.tensor(
                [[rec + [self.PAD] * (topnum - len(rec)) for rec in sample] for sample in topk_sequence],
                dtype=torch.long
            ).cuda()
        else:
            topk_indices = topk_sequence

        # 遍历每个样本和时间步
        for b in range(batch_size):
            seq = original_seqs[b]
            recs = topk_indices[b]

            # 初始化历史记录
            history_diffs = []
            history_results = []

            # 遍历原始序列（去掉最后一个时间步，因为我们要预测它）
            for t in range(len(seq) - 1):
                challenge_idx = seq[t]
                result = 1  # 假设所有历史答题结果都是正确的（根据原始代码逻辑）

                # 获取难度
                if challenge_idx > 1:
                    diff = get_difficulty(challenge_idx)
                    history_diffs.append(diff)
                    history_results.append(result)

            for t in range(seq_len - 1):
                if original_seqs[b][t] == self.PAD:
                    continue;
                # ========== 有效性计算 ==========
                valid_count += 1
                valid_rec = [r.item() for r in recs[t] if 0 < r < yt_before.shape[2]]
                if valid_rec:

                    pb = yt_before[b, t, valid_rec]  # [K]
                    pa = yt_after[b, t, valid_rec]  # [K]
                    gain = 0.0
                    valid = 0
                    for k in range(len(valid_rec)):
                        if pb[k] < 0.9 and pa[k] > 0:
                            gain += (pa[k] - pb[k]) / (1.0 - pb[k])
                            valid += 1
                    if valid > 0:
                        eff = gain / valid
                        metrics['total_effectiveness'] += eff
                        metrics['step_records'][(b, t)].append(eff)
                        eff_valid_count += 1

                # ========== 适应性计算 ==========
                if t >= 0:  # 需要历史记录
                    recent_diffs = history_diffs[max(0, t - T):t]
                    recent_results = history_results[max(0, t - T):t]
                    if t == 0:
                        delta = 1.0
                    else:
                        delta = sum(d * r for d, r in zip(recent_diffs, recent_results)) / \
                                (sum(recent_results) + epsilon) if recent_results else 1.0

                    adapt_sum = 0.0
                    dif_valid = 0
                    for rec in recs[t]:
                        if rec > 1:
                            try:
                                rec_diff = difficulty_data[int(idx2u[rec.item()])]
                                adapt = 1 - abs(delta - rec_diff)
                                ada_valid_count += 1
                                metrics['total_adaptivity'] += adapt

                            except KeyError:
                                pass
                    # if dif_valid > 0:
                    #   metrics['total_adaptivity'] += (adapt_sum/dif_valid)
                    #   metrics['step_records'][(b, t)].append(adapt)
                    #   ada_valid_count += 1

                # ========== 多样性计算 ==========
                current_embs = hidden[recs[t]]  # [topnum, emb_dim]
                pair_count = 0
                div_sum = 0.0

                for i in range(topnum):
                    for j in range(i + 1, topnum):
                        if recs[t, i] == 0 or recs[t, j] == 0:
                            continue
                        sim = F.cosine_similarity(
                            current_embs[i].unsqueeze(0),
                            current_embs[j].unsqueeze(0))
                        div_sum += (1 - sim.item())
                        pair_count += 1

                if pair_count > 0:
                    div_valid_count += 1
                    metrics['total_diversity'] += (div_sum / pair_count)
                    metrics['step_records'][(b, t)].append(div_sum / pair_count)

                # ========== 喜好计算 ==========
                # 获取当前batch和time_step对应的预测概率
                flat_index = b * (seq_len - 1) + t
                resource_probs = pred_probs[flat_index]  # 形状: [num_resources]

                # 计算推荐资源的平均概率
                prefer_sum = 0.0
                pre_num = 0
                for r in valid_rec:
                    if r < len(resource_probs):
                        metrics['total_preference'] += resource_probs[r]
                        pre_valid_count += 1

                # if pre_num != 0:
                #   metrics['total_adaptivity'] += (prefer_sum/pre_num)
                #   pre_valid_count += 1

                # 计算均值

        # print("total_preference:",metrics['total_preference'])
        # print("pre_valid_count:",pre_valid_count)
        metrics.update({
            'effectiveness': metrics['total_effectiveness'] / eff_valid_count,
            'adaptivity': metrics['total_adaptivity'] / ada_valid_count,
            'diversity': metrics['total_diversity'] / div_valid_count,
            'preference': metrics['total_preference'] / pre_valid_count
        })

        return metrics

    # def compute_effectiveness_single(self, yt_before, yt_after, recommended):
    #     """单时间步有效性计算，输入为numpy数组"""
    #     gain = 0.0
    #     valid = 0
    #     for r in recommended:
    #         if r >= yt_before.shape[0]:  # 过滤无效资源索引
    #             continue
    #         pb = yt_before[r]
    #         pa = yt_after[r]
    #         if pb < 0.9 and pa > 0:
    #             gain += (pa - pb) / (1.0 - pb)
    #             valid += 1
    #     return gain / valid if valid > 0 else 0.0

# Calculate accuracy of prediction result and its corresponding label
# output: tensor, labels: tensor
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels.reshape(-1)).double()
    correct = correct.sum()
    return correct / len(labels)


class KTLoss(nn.Module):
    def __init__(self):
        super(KTLoss, self).__init__()
        self.bce_loss = nn.BCELoss(reduction='none')  # 不自动求平均

    def forward(self, pred_answers, real_answers, kt_mask):
        real_answers = real_answers[:, 1:].float()  # 截取有效部分并转为浮点
        answer_mask = kt_mask.bool()

        # --- 计算 AUC 和 ACC ---
        try:
            y_true = real_answers[answer_mask].cpu().detach().numpy()
            y_pred = pred_answers[answer_mask].cpu().detach().numpy()
            auc = roc_auc_score(y_true, y_pred)
            acc = accuracy_score(y_true, (y_pred >= 0.5).astype(int))  # 直接根据概率阈值计算ACC
        except ValueError:
            auc, acc = -1, -1

        # --- 计算带掩码的 BCE 损失 ---
        loss_per_element = self.bce_loss(pred_answers, real_answers)
        valid_loss = loss_per_element * answer_mask.float()  # 应用掩码
        loss = valid_loss.sum() / answer_mask.float().sum()  # 仅对有效位置求平均

        return loss, auc, acc
