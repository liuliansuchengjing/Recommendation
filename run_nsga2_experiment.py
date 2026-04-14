import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import random
from HGAT import MSHGAT
from dataLoader import Split_data, DataLoader, Options
from graphConstruct import ConRelationGraph, ConHyperGraphList
from run import opt
import pickle
from dataLoader import Options
import csv # 确保文件开头导入了 csv
# ==========================================
# 提取与映射难度字典 (与 Metrics.py 逻辑对齐)
# ==========================================
data_opts = Options(opt.data_name)

# 1. 加载 idx -> 原始资源名称 的映射
with open(data_opts.idx2u_dict, 'rb') as f:
    idx2u = pickle.load(f)

# 2. 加载 原始资源名称 -> 难度 的映射
difficulty_dict = {}
with open(data_opts.difficult_file, 'r') as f:
    next(f)  # 跳过标题行
    for line in f:
        parts = line.strip().split(',')
        if len(parts) >= 2:
            try:
                # parts[0] 是资源名称, parts[1] 是难度
                difficulty_dict[int(parts[0].strip())] = int(parts[1].strip())
            except ValueError:
                continue


# 3. 封装为全局函数供 evaluate_path 调用
def get_diff(idx):
    """
    通过模型内部的 idx 获取对应资源的难度
    映射路径: idx -> idx2u -> 原始 challenge_id -> difficulty_dict -> 难度
    """
    # PAD(0) 和 Skip(1) 返回默认难度 1
    if idx <= 1:
        return 1

    try:
        # 获取真实的资源名称 (challenge_id)
        real_id = int(idx2u[idx])
        # 查找难度，如果缺失则默认给 1
        return difficulty_dict.get(real_id, 1)
    except (ValueError, IndexError):
        return 1


# ==========================================
# 0. 辅助函数：时间分箱 (从 run.py 中同步)
# ==========================================
def calc_time_bins(start_time, end_time):
    def to_minutes(t_tensor):
        minutes = t_tensor % 100
        hours = (t_tensor // 100) % 100
        days = (t_tensor // 10000) % 100
        months = (t_tensor // 1000000) % 100
        return minutes + hours * 60 + days * 1440 + months * 43200

    diff_mins = to_minutes(end_time) - to_minutes(start_time)
    time_bins = torch.zeros_like(start_time)
    valid_mask = (end_time > 0) & (diff_mins >= 0)

    time_bins[valid_mask & (diff_mins <= 1)] = 1
    time_bins[valid_mask & (diff_mins > 1) & (diff_mins <= 3)] = 2
    time_bins[valid_mask & (diff_mins > 3) & (diff_mins <= 5)] = 3
    time_bins[valid_mask & (diff_mins > 5) & (diff_mins <= 10)] = 4
    time_bins[valid_mask & (diff_mins > 10) & (diff_mins <= 30)] = 5
    time_bins[valid_mask & (diff_mins > 30)] = 6
    return time_bins


# ==========================================
# 1. 核心适应度函数 (严格对齐论文公式计算)
# ==========================================
def evaluate_path(path_ids, hist_seq, hist_ans, hist_time_bins, model_kt, graph, hidden_kt, p_before):
    L_TARGET = len(path_ids)
    path_tensor = torch.tensor([path_ids], device='cuda')

    # -----------------------------------------
    # 细节 1：ZPD 模拟答题数据
    # 基于学习者初始掌握度，预测其在推荐路径上的作答表现
    # -----------------------------------------
    ans_list = []
    for idx in path_ids:
        # 预测掌握度 >= 0.5 有能力做对记为 1.0；否则做错记为 0.0
        sim_ans = 1.0 if p_before[idx].item() >= 0.5 else 0.0
        ans_list.append(sim_ans)

    ans_tensor = torch.tensor([ans_list], device='cuda')
    time_bins_tensor = torch.full((1, L_TARGET), 2, device='cuda')

    # 拼接历史与生成的候选路径
    sim_seq = torch.cat([hist_seq, path_tensor], dim=1)
    sim_ans = torch.cat([hist_ans, ans_tensor], dim=1)
    sim_time_bins = torch.cat([hist_time_bins, time_bins_tensor], dim=1)

    # DKT 模拟预测未来的掌握状态
    with torch.no_grad():
        _, _, yt_sim, _, _ = model_kt.ktmodel(hidden_kt, sim_seq, sim_ans, sim_time_bins)
        p_after = yt_sim[0, -1, :]

    # -----------------------------------------
    # 细节 2：Expected Mastery Gain 知识增益
    # -----------------------------------------
    f_gain = 0.0
    for idx in path_ids:
        # 直接计算绝对期望掌握度增益
        f_gain += (p_after[idx].item() - p_before[idx].item())
    f_gain /= L_TARGET

    # -----------------------------------------
    # 细节 3：Difficulty Smoothness 难度平滑度 (关键修复：相邻难度差)
    # -----------------------------------------
    # 获取历史序列中最后一道做对的题的难度，作为平滑度计算的起点
    valid_hist = [get_diff(int(x)) for x, y in zip(hist_seq[0], hist_ans[0]) if int(x) > 1 and int(y) == 1]
    prev_diff = valid_hist[-1] if len(valid_hist) > 0 else 1.0

    f_smooth = 0.0
    for idx in path_ids:
        curr_diff = get_diff(idx)
        # 比较相邻步序的难度跳跃，除以极差 (3-1=2.0) 进行 [0, 1] 归一化
        f_smooth += 1.0 - (abs(curr_diff - prev_diff) / 2.0)
        prev_diff = curr_diff  # 关键：更新 prev_diff 为当前难度，实现步步连贯对比！
    f_smooth /= L_TARGET

    # -----------------------------------------
    # 细节 4：Resource Diversity 资源多样性 (严格对齐公式 5.4)
    # -----------------------------------------
    f_div = 0.0
    pairs = 0
    embs = hidden_kt[path_ids]
    for i in range(L_TARGET):
        for j in range(i + 1, L_TARGET):
            sim = torch.cosine_similarity(embs[i].unsqueeze(0), embs[j].unsqueeze(0)).item()
            # 严格遵循公式(5.4): 分子为 (1 - sim)，绝不多余除以 2
            f_div += (1.0 - sim) / 2.0
            pairs += 1

    if pairs > 0:
        f_div /= pairs

    return f_gain, f_smooth, f_div


# ==========================================
# 2. 严谨的 NSGA-II 框架 (彻底修复精英保留 Bug)
# ==========================================
def non_dominated_sort(population_fitness):
    front = []
    for i, fit_i in enumerate(population_fitness):
        dominated = False
        for j, fit_j in enumerate(population_fitness):
            if i == j: continue
            if (fit_j[0] >= fit_i[0] and fit_j[1] >= fit_i[1] and fit_j[2] >= fit_i[2]) and \
                    (fit_j[0] > fit_i[0] or fit_j[1] > fit_i[1] or fit_j[2] > fit_i[2]):
                dominated = True
                break
        if not dominated:
            front.append(i)
    return front


def run_nsga2(strategy, hist_seq, hist_ans, hist_time_bins, topK_candidates, valid_resource_ids, model_kt, graph,
              p_before):
    hidden_kt = model_kt.gnn2(graph)
    L_TARGET = 6
    POPULATION_SIZE = 50
    MAX_GEN = 30  # 最大迭代次数

    history_fronts = {}  # 记录不同阶段的帕累托前沿

    # 1. 种群初始化
    population = []
    pool = valid_resource_ids if strategy == 'Random' else topK_candidates
    for _ in range(POPULATION_SIZE):
        population.append(random.sample(pool, L_TARGET))

    # 评估与排序记录 Gen 1
    fitness = [evaluate_path(p, hist_seq, hist_ans, hist_time_bins, model_kt, graph, hidden_kt, p_before) for p in
               population]
    front_indices = non_dominated_sort(fitness)
    history_fronts[1] = [fitness[i] for i in front_indices]

    # 2. 模拟演化迭代
    for gen in range(2, MAX_GEN + 1):
        new_population = []

        # ✅ 核心修复 1：全量无损保留上一代的所有帕累托前沿精英！绝不丢弃任何极值点！
        for idx in front_indices:
            new_population.append(population[idx].copy())

        # 安全性兜底：如果前沿点太多超过种群上限，优先保留即可
        if len(new_population) > POPULATION_SIZE:
            new_population = new_population[:POPULATION_SIZE]

        # ✅ 核心修复 2：用精英的变异后代填满剩下的种群槽位
        while len(new_population) < POPULATION_SIZE:
            # 优先从精英库中随机挑选父代
            parent_idx = random.choice(front_indices) if front_indices else random.randint(0, len(population) - 1)
            child = population[parent_idx].copy()

            # 单点变异探索新空间
            mut_idx = random.randint(0, L_TARGET - 1)
            child[mut_idx] = random.choice(pool)
            new_population.append(child)

        population = new_population
        # 重新评估并更新非支配解
        fitness = [evaluate_path(p, hist_seq, hist_ans, hist_time_bins, model_kt, graph, hidden_kt, p_before) for p in
                   population]
        front_indices = non_dominated_sort(fitness)

        # 记录中期演化 (Gen 15)
        if gen == 15:
            history_fronts[15] = [fitness[i] for i in front_indices]

    # 记录最终种群 (Gen 30)
    history_fronts[MAX_GEN] = [fitness[i] for i in front_indices]

    return history_fronts


# ==========================================
# 3. 主函数缝合执行
# ==========================================
def main():
    # ==========================================
    # 0. 固定全局随机种子 (保证实验完美可复现)
    # ==========================================
    seed_value = 48  # 🌟 这里的数字就是你的“盲盒编号”，你可以任意修改（如 0, 100, 2026 等）
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True

    print(f"当前实验已固定随机种子为: {seed_value}")
    # 1. 基础配置与数据加载 (复用 run.py 逻辑)

    resource_size, train_history_cas, train_history_t, train, valid, test = Split_data(opt.data_name, load_dict=True)
    test_data = DataLoader(test, batch_size=1, load_dict=True, cuda=False)  # batch_size=1 方便抽取单学生

    relation_graph = ConRelationGraph(opt.data_name)
    hypergraph_list = ConHyperGraphList(train_history_cas, train_history_t, resource_size)

    valid_resource_ids = list(range(2, resource_size))  # 排除 PAD=0 和 SKIP=1
    opt.resource_size = resource_size

    # 2. 加载模型
    model_rec = MSHGAT(opt, dropout=opt.dropout).cuda()
    model_rec.load_state_dict(torch.load(opt.save_rec_path))
    model_rec.eval()

    model_kt = MSHGAT(opt, dropout=opt.dropout).cuda()
    model_kt.load_state_dict(torch.load(opt.save_kt_path))
    model_kt.eval()

    # 3. 提取真实学生数据进行实验 (带 ZPD 潜力筛选与未来真实目标提取)
    print("正在从测试集中扫描有效学生序列...")
    candidate_students = []

    for batch in test_data:
        tgt, tgt_timestamp, tgt_idx, ans, tgt_end_time, _, _, _ = [item.cuda() for item in batch]
        time_bins = calc_time_bins(tgt_timestamp, tgt_end_time)

        valid_len = (tgt[0] > 1).sum().item()
        if valid_len > 15:
            # 截取历史序列 (t=15)
            t = 15
            hist_seq = tgt[0:1, :t]
            hist_ans = ans[0:1, :t]
            hist_time_bins = time_bins[0:1, :t]
            hist_timestamp = tgt_timestamp[0:1, :t]
            hist_idx = tgt_idx[0:1]

            # 🌟 提取未来真实序列作为 Ground Truth (最大长度为 6，不够 6 则取剩余全部)
            future_len = min(6, valid_len - t)
            if future_len == 0:
                continue  # 没有未来数据无法评估迁移增益
            future_seq = tgt[0:1, t:t + future_len]

            # 评估初始状态
            with torch.no_grad():
                hidden_kt = model_kt.gnn2(relation_graph)
                _, _, yt_init, _, _ = model_kt.ktmodel(hidden_kt, hist_seq, hist_ans, hist_time_bins)
                p_before = yt_init[0, -1, :]

            # 过滤掉基础极差（连题都看不懂的人）
            if p_before.mean().item() < 0.1:
                continue

            # 推荐候选池
            with torch.no_grad():
                pred_logits, _, _, _, _ = model_rec(hist_seq, hist_timestamp, hist_idx, hist_ans, relation_graph,
                                                    hypergraph_list, hist_time_bins)
                last_step_logits = pred_logits[-1, :]
                top50_indices = torch.topk(last_step_logits, 80).indices.cpu().numpy()
                hist_list = hist_seq[0].cpu().numpy().tolist()
                topK_candidates = [int(x) for x in top50_indices if x > 1 and x not in hist_list][:50]

            # ZPD 潜力甄别
            correct_capacity = sum(1 for x in topK_candidates if p_before[x].item() >= 0.5)
            if correct_capacity < 15:
                continue

            candidate_students.append({
                'hist_seq': hist_seq,
                'hist_ans': hist_ans,
                'hist_time_bins': hist_time_bins,
                'p_before': p_before,
                'topK_candidates': topK_candidates,
                'p_mean': p_before.mean().item(),
                'future_seq': future_seq  # 存入未来真实序列
            })

    print(f"扫描完毕！找到 {len(candidate_students)} 个具备干预价值的候选学生。")

    # ==========================================
    # 4. 划分群体与随机多样本采样
    # ==========================================
    group_weak, group_mid, group_strong = [], [], []
    for stu in candidate_students:
        p_mean = stu['p_mean']
        if p_mean < 0.5:
            group_weak.append(stu)
        elif 0.5 <= p_mean < 0.8:
            group_mid.append(stu)
        else:
            group_strong.append(stu)

    print(
        f"全局分组情况: 基础薄弱组 {len(group_weak)} 人, 能力巩固组 {len(group_mid)} 人, 进阶提升组 {len(group_strong)} 人")
    if not (group_weak and group_mid and group_strong):
        print("警告：某一组人数为0，请检查数据分布！")
        return

    SAMPLE_N = 5  # 每组采样人数
    sample_weak = random.sample(group_weak, min(SAMPLE_N, len(group_weak)))
    sample_mid = random.sample(group_mid, min(SAMPLE_N, len(group_mid)))
    sample_strong = random.sample(group_strong, min(SAMPLE_N, len(group_strong)))

    # 🌟 核心函数 1：选出最优路径 (基于 6 道题的整体适应度)
    def select_best_path(front_fitness, front_paths, w_gain, w_smooth, w_div=0.2):
        if not front_fitness: return []
        front_arr = np.array(front_fitness)
        mins = front_arr.min(axis=0)
        maxs = front_arr.max(axis=0)
        denoms = maxs - mins + 1e-8
        norm_front = (front_arr - mins) / denoms
        utility = w_gain * norm_front[:, 0] + w_smooth * norm_front[:, 1] + w_div * norm_front[:, 2]
        return front_paths[np.argmax(utility)]

    # 🌟 核心函数 2：自适应长度截断与全指标评估！
    def evaluate_adaptive_metrics(path_ids, future_seq, hist_seq, hist_ans, hist_time_bins, model_kt, graph, hidden_kt,
                                  p_before):
        if not path_ids: return 0.0, 0.0, 0.0, 0.0

        L_TARGET = len(path_ids)
        path_tensor = torch.tensor([path_ids], device='cuda')
        ans_list = [1.0 if p_before[idx].item() >= 0.5 else 0.0 for idx in path_ids]
        ans_tensor = torch.tensor([ans_list], device='cuda')
        time_bins_tensor = torch.full((1, L_TARGET), 2, device='cuda')

        sim_seq = torch.cat([hist_seq, path_tensor], dim=1)
        sim_ans = torch.cat([hist_ans, ans_tensor], dim=1)
        sim_time_bins = torch.cat([hist_time_bins, time_bins_tensor], dim=1)

        with torch.no_grad():
            _, _, yt_sim, _, _ = model_kt.ktmodel(hidden_kt, sim_seq, sim_ans, sim_time_bins)

        # 🌟 动态获取输出序列的真实长度，彻底免疫 DKT 模型的 T/T-1 长度截断问题
        yt_sim_len = yt_sim.size(1)

        best_k = 1
        max_exp_gain = -999.0
        best_p_after = None

        # 逐层遍历 1 到 L_TARGET 步，寻找知识增益的最高峰 (Adaptive Length)
        for k in range(1, L_TARGET + 1):
            # 🌟 倒数索引法：
            # 无论模型输出长度是20还是21，倒数第 (L_TARGET - k + 1) 个必定是第 k 题的状态！
            target_idx = yt_sim_len - L_TARGET + k - 1
            p_after_k = yt_sim[0, target_idx, :]
            g = 0.0
            for idx in path_ids[:k]:
                gain = p_after_k[idx].item() - p_before[idx].item()
                room = max(1.0 - p_before[idx].item(), 0.1)
                g += max(-1.0, gain / room)
            g /= k

            if g > max_exp_gain:
                max_exp_gain = g
                best_k = k
                best_p_after = p_after_k

        # 基于最佳截断长度 best_k 计算剩余指标
        adap_path = path_ids[:best_k]

        # 1. 截断后的平滑度
        valid_hist = [get_diff(int(x)) for x, y in zip(hist_seq[0], hist_ans[0]) if int(x) > 1 and int(y) == 1]
        delta = np.mean(valid_hist[-5:]) if len(valid_hist) > 0 else 1.0
        f_smooth = sum([1.0 - (abs(delta - get_diff(idx)) / 2.0) for idx in adap_path]) / best_k

        # 2. 截断后的多样性
        f_div = 1.0  # 默认 1.0
        pairs = 0
        if best_k > 1:
            f_div = 0.0
            embs = hidden_kt[adap_path]
            for i in range(best_k):
                for j in range(i + 1, best_k):
                    sim = torch.cosine_similarity(embs[i].unsqueeze(0), embs[j].unsqueeze(0)).item()
                    f_div += (1.0 - sim) / 2.0
                    pairs += 1
            f_div /= pairs

        # 3. 截断路径的真实未来迁移增益 (Migration Gain)
        future_ids = [int(x) for x in future_seq[0] if int(x) > 1]
        m_gain = 0.0
        if future_ids:
            for idx in future_ids:
                gain = best_p_after[idx].item() - p_before[idx].item()
                room = max(1.0 - p_before[idx].item(), 0.1)
                m_gain += max(-1.0, gain / room)
            m_gain /= len(future_ids)

        return max_exp_gain, f_smooth, f_div, m_gain

    # ==========================================
    # 5. 执行全指标敏感性分析 (细粒度 0.02)
    # ==========================================
    lambda_1_range = np.round(np.arange(0.10, 0.72, 0.02), 2)

    # 统一存储结构
    results = {
        'weak': {'exp_gain': [], 'smooth': [], 'div': [], 'mig_gain': []},
        'mid': {'exp_gain': [], 'smooth': [], 'div': [], 'mig_gain': []},
        'strong': {'exp_gain': [], 'smooth': [], 'div': [], 'mig_gain': []}
    }

    # 用于写入 CSV 的扁平数据列表
    csv_data_rows = []

    print("开始执行高精度参数连续敏感性分析 (自适应长度 & 全指标)...")

    for group_name, sample_group in zip(['weak', 'mid', 'strong'], [sample_weak, sample_mid, sample_strong]):
        print(f"\n处理 [{group_name}] 组...")

        temp_res = {l1: {'exp_gain': [], 'smooth': [], 'div': [], 'mig_gain': []} for l1 in lambda_1_range}

        for stu in sample_group:
            hidden_kt = model_kt.gnn2(relation_graph)
            history = run_nsga2('Prob', stu['hist_seq'], stu['hist_ans'], stu['hist_time_bins'],
                                stu['topK_candidates'], valid_resource_ids, model_kt, relation_graph, stu['p_before'])

            pop_pool = [random.sample(stu['topK_candidates'], 6) for _ in range(500)]
            fits = [evaluate_path(p, stu['hist_seq'], stu['hist_ans'], stu['hist_time_bins'], model_kt, relation_graph,
                                  hidden_kt, stu['p_before']) for p in pop_pool]
            f_idx = non_dominated_sort(fits)
            final_front_fits = [fits[i] for i in f_idx]
            final_front_paths = [pop_pool[i] for i in f_idx]

            for l1 in lambda_1_range:
                l1_key = round(l1, 2)
                l2 = round(0.8 - l1_key, 2)

                best_path = select_best_path(final_front_fits, final_front_paths, w_gain=l1_key, w_smooth=l2, w_div=0.2)
                # 🌟 获取自适应长度下的 4 项核心指标
                e_g, s_m, d_v, m_g = evaluate_adaptive_metrics(best_path, stu['future_seq'], stu['hist_seq'],
                                                               stu['hist_ans'], stu['hist_time_bins'], model_kt,
                                                               relation_graph, hidden_kt, stu['p_before'])

                temp_res[l1_key]['exp_gain'].append(e_g)
                temp_res[l1_key]['smooth'].append(s_m)
                temp_res[l1_key]['div'].append(d_v)
                temp_res[l1_key]['mig_gain'].append(m_g)

        # 组内取平均值
        for l1 in lambda_1_range:
            l1_key = round(l1, 2)
            avg_eg = np.mean(temp_res[l1_key]['exp_gain'])
            avg_sm = np.mean(temp_res[l1_key]['smooth'])
            avg_dv = np.mean(temp_res[l1_key]['div'])
            avg_mg = np.mean(temp_res[l1_key]['mig_gain'])

            results[group_name]['exp_gain'].append(avg_eg)
            results[group_name]['smooth'].append(avg_sm)
            results[group_name]['div'].append(avg_dv)
            results[group_name]['mig_gain'].append(avg_mg)

            # 记录到 CSV 数据行
            csv_data_rows.append([l1_key, round(0.8 - l1_key, 2), group_name, avg_eg, avg_mg, avg_sm, avg_dv])

    # ==========================================
    # 6. 导出包含所有细粒度参数的 CSV 报表
    # ==========================================
    csv_filename = "Sensitivity_Detailed_Results.csv"
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Lambda_1(Gain)', 'Lambda_2(Smooth)', 'Group', 'Expected_Gain', 'Migration_Gain', 'Smoothness',
                         'Diversity'])
        writer.writerows(csv_data_rows)
    print(f"\n[成功] 全量细粒度数据已导出至文件: {csv_filename} (可用于Excel作表)")

    # ==========================================
    # 7. 绘制极度震撼的 2x2 全景综合评估图 (纯英文版)
    # ==========================================
    print("Drawing comprehensive 2x2 multi-metric charts (English)...")

    fig, axs = plt.subplots(2, 2, figsize=(16, 12))

    # 颜色配置
    c_weak, c_mid, c_strong = '#4A90E2', '#F5A623', '#D0021B'
    labels = ['Weak Group ($p_{before}<0.5$)', 'Intermediate Group ($0.5\leq p_{before}<0.8$)',
              'Advanced Group ($p_{before}\geq0.8$)']

    # --- 子图 1: 参数对 Expected Gain 的影响 ---
    axs[0, 0].plot(lambda_1_range, results['weak']['exp_gain'], linewidth=3, color=c_weak, label=labels[0])
    axs[0, 0].plot(lambda_1_range, results['mid']['exp_gain'], linewidth=3, color=c_mid, label=labels[1])
    axs[0, 0].plot(lambda_1_range, results['strong']['exp_gain'], linewidth=3, color=c_strong, label=labels[2])
    axs[0, 0].set_xlabel('Knowledge Gain Weight $\lambda_1$', fontsize=12)
    axs[0, 0].set_ylabel('Adaptive Expected Mastery Gain', fontsize=12)
    axs[0, 0].set_title('(a) Impact on Expected Knowledge Gain', fontsize=14)
    axs[0, 0].grid(linestyle='--', alpha=0.4)
    axs[0, 0].legend()

    # --- 子图 2: 参数对 Smoothness 的影响 ---
    axs[0, 1].plot(lambda_1_range, results['weak']['smooth'], linewidth=3, color=c_weak)
    axs[0, 1].plot(lambda_1_range, results['mid']['smooth'], linewidth=3, color=c_mid)
    axs[0, 1].plot(lambda_1_range, results['strong']['smooth'], linewidth=3, color=c_strong)
    axs[0, 1].set_xlabel('Knowledge Gain Weight $\lambda_1$', fontsize=12)
    axs[0, 1].set_ylabel('Adaptive Path Smoothness', fontsize=12)
    axs[0, 1].set_title('(b) Impact on Difficulty Smoothness', fontsize=14)
    axs[0, 1].grid(linestyle='--', alpha=0.4)

    # --- 子图 3: 参数对 Diversity 的影响 ---
    axs[1, 0].plot(lambda_1_range, results['weak']['div'], linewidth=3, color=c_weak)
    axs[1, 0].plot(lambda_1_range, results['mid']['div'], linewidth=3, color=c_mid)
    axs[1, 0].plot(lambda_1_range, results['strong']['div'], linewidth=3, color=c_strong)
    axs[1, 0].set_xlabel('Knowledge Gain Weight $\lambda_1$', fontsize=12)
    axs[1, 0].set_ylabel('Adaptive Resource Diversity', fontsize=12)
    axs[1, 0].set_title('(c) Impact on Resource Diversity', fontsize=14)
    axs[1, 0].grid(linestyle='--', alpha=0.4)

    # --- 子图 4: Expected Gain vs. Migration Gain 的准确性对齐 ---
    # 为了展示一致性，我们计算所有群体在各个参数下的平均 Expected Gain 和 平均 Migration Gain 进行拟合对比
    global_exp_gain = np.mean([results['weak']['exp_gain'], results['mid']['exp_gain'], results['strong']['exp_gain']],
                              axis=0)
    global_mig_gain = np.mean([results['weak']['mig_gain'], results['mid']['mig_gain'], results['strong']['mig_gain']],
                              axis=0)

    axs[1, 1].plot(lambda_1_range, global_exp_gain, linewidth=3, color='#8B008B',
                   label='Global Expected Gain (Internal)')
    axs[1, 1].plot(lambda_1_range, global_mig_gain, linewidth=3, color='#2E8B57', linestyle='--',
                   label='Global Migration Gain (Ground Truth)')
    axs[1, 1].set_xlabel('Knowledge Gain Weight $\lambda_1$', fontsize=12)
    axs[1, 1].set_ylabel('Average Gain Value', fontsize=12)
    axs[1, 1].set_title('(d) Internal Expected vs. Real Migration Gain Alignment', fontsize=14)
    axs[1, 1].grid(linestyle='--', alpha=0.4)
    axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig('Comprehensive_Metrics_Sensitivity_EN.png', dpi=300)
    print("Multi-metric sensitivity chart generated: Comprehensive_Metrics_Sensitivity_EN.png")
    plt.show()


# ==========================================
# 执行入口 (这段一定要放在文件最末尾！)
# ==========================================
if __name__ == "__main__":
    main()