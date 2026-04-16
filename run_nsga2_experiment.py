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
import csv

# ==========================================
# 提取与映射难度字典
# ==========================================
data_opts = Options(opt.data_name)

with open(data_opts.idx2u_dict, 'rb') as f:
    idx2u = pickle.load(f)

difficulty_dict = {}
with open(data_opts.difficult_file, 'r') as f:
    next(f)
    for line in f:
        parts = line.strip().split(',')
        if len(parts) >= 2:
            try:
                difficulty_dict[int(parts[0].strip())] = int(parts[1].strip())
            except ValueError:
                continue


def get_diff(idx):
    if idx <= 1: return 1
    try:
        real_id = int(idx2u[idx])
        return difficulty_dict.get(real_id, 1)
    except (ValueError, IndexError):
        return 1


# ==========================================
# 0. 辅助函数：时间分箱
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
# 1. 核心适应度函数 (用于遗传算法内评估)
# ==========================================
# 🌟 增加 target_ids 参数
def evaluate_path(path_ids, hist_seq, hist_ans, hist_time_bins, model_kt, graph, hidden_kt, p_before, target_ids):
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
        p_after = yt_sim[0, -1, :]

    # 🌟 彻底修复知识增益计算逻辑 (严格对齐论文公式)
    f_gain = 0.0
    for idx in target_ids:
        # 1. 计算绝对增益
        gain = p_after[idx].item() - p_before[idx].item()

        # 2. 计算剩余可提升空间 (Room for improvement)
        # 设定 0.1 的保护阈值，防止高分段 (如 p_before=0.99) 导致除以极小数发生数值爆炸
        room = max(1.0 - p_before[idx].item(), 0.1)

        # 3. 计算相对增益，并施加 -1.0 的下限截断防御认知反噬惩罚过大
        f_gain += max(-1.0, gain / room)

    f_gain /= len(target_ids)

    valid_hist = [get_diff(int(x)) for x, y in zip(hist_seq[0], hist_ans[0]) if int(x) > 1 and int(y) == 1]
    prev_diff = valid_hist[-1] if len(valid_hist) > 0 else 1.0

    f_smooth = 0.0
    for idx in path_ids:
        curr_diff = get_diff(idx)
        f_smooth += 1.0 - (abs(curr_diff - prev_diff) / 2.0)
        prev_diff = curr_diff
    f_smooth /= L_TARGET

    f_div = 0.0
    pairs = 0
    embs = hidden_kt[path_ids]
    for i in range(L_TARGET):
        for j in range(i + 1, L_TARGET):
            sim = torch.cosine_similarity(embs[i].unsqueeze(0), embs[j].unsqueeze(0)).item()
            f_div += (1.0 - sim) / 2.0
            pairs += 1

    if pairs > 0: f_div /= pairs

    return f_gain, f_smooth, f_div


# ==========================================
# 2. NSGA-II 框架
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


def run_nsga2(strategy, hist_seq, hist_ans, hist_time_bins, topK_candidates, valid_resource_ids, model_kt, graph, p_before, target_ids):
    hidden_kt = model_kt.gnn2(graph)
    L_TARGET = 6
    POPULATION_SIZE = 50
    MAX_GEN = 30

    history_fronts = {}
    population = []
    pool = valid_resource_ids if strategy == 'Random' else topK_candidates
    for _ in range(POPULATION_SIZE):
        population.append(random.sample(pool, L_TARGET))

    fitness = [evaluate_path(p, hist_seq, hist_ans, hist_time_bins, model_kt, graph, hidden_kt, p_before, target_ids) for p in population]
    front_indices = non_dominated_sort(fitness)
    history_fronts[1] = [fitness[i] for i in front_indices]

    for gen in range(2, MAX_GEN + 1):
        new_population = []
        for idx in front_indices:
            new_population.append(population[idx].copy())

        if len(new_population) > POPULATION_SIZE:
            new_population = new_population[:POPULATION_SIZE]

        while len(new_population) < POPULATION_SIZE:
            parent_idx = random.choice(front_indices) if front_indices else random.randint(0, len(population) - 1)
            child = population[parent_idx].copy()
            mut_idx = random.randint(0, L_TARGET - 1)
            child[mut_idx] = random.choice(pool)
            new_population.append(child)

        population = new_population
        fitness = [evaluate_path(p, hist_seq, hist_ans, hist_time_bins, model_kt, graph, hidden_kt, p_before) for p in
                   population]
        front_indices = non_dominated_sort(fitness)

        if gen == 15: history_fronts[15] = [fitness[i] for i in front_indices]
    history_fronts[MAX_GEN] = [fitness[i] for i in front_indices]

    return history_fronts


# ==========================================
# 3. 主函数缝合执行
# ==========================================
def main():
    # 🌟 调试神器：只计算基准真实指标开关
    # True  = 只算真实学生的三指标，算完立即结束程序 (秒出结果)
    # False = 跑完基准后，继续跑完整的遗传算法和敏感性分析图
    QUICK_BASELINE_ONLY = True

    # 🌟 核心函数 2：自适应长度截断与全指标评估！
    def evaluate_adaptive_metrics(path_ids, future_seq, hist_seq, hist_ans, hist_time_bins, model_kt, graph, hidden_kt, p_before, target_ids):
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

        yt_sim_len = yt_sim.size(1)

        best_k = 1
        max_exp_gain = -999.0
        best_p_after = None

        for k in range(1, L_TARGET + 1):
            target_idx = yt_sim_len - L_TARGET + k - 1
            p_after_k = yt_sim[0, target_idx, :]
            g = 0.0

            # 🌟 确保这里遍历的是 target_ids，且严格执行空间归一化和下限截断
            for idx in target_ids:
                gain = p_after_k[idx].item() - p_before[idx].item()
                room = max(1.0 - p_before[idx].item(), 0.1)
                g += max(-1.0, gain / room)
            g /= len(target_ids)

            if g > max_exp_gain:
                max_exp_gain = g
                best_k = k
                best_p_after = p_after_k

        adap_path = path_ids[:best_k]

        valid_hist = [get_diff(int(x)) for x, y in zip(hist_seq[0], hist_ans[0]) if int(x) > 1 and int(y) == 1]
        delta = np.mean(valid_hist[-5:]) if len(valid_hist) > 0 else 1.0
        f_smooth = sum([1.0 - (abs(delta - get_diff(idx)) / 2.0) for idx in adap_path]) / best_k

        f_div = 1.0
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

        future_ids = [int(x) for x in future_seq[0] if int(x) > 1]
        m_gain = 0.0
        if future_ids:
            for idx in future_ids:
                gain = best_p_after[idx].item() - p_before[idx].item()
                room = max(1.0 - p_before[idx].item(), 0.1)
                m_gain += max(-1.0, gain / room)
            m_gain /= len(future_ids)

        return max_exp_gain, f_smooth, f_div, m_gain

    # 🌟 复用核心评估函数的基准计算逻辑
    def calculate_ground_truth_all_metrics(model_kt, relation_graph, sample_weak, sample_mid, sample_strong,
                                           L_TARGET=6):
        print("\n" + "=" * 60)
        print("开始计算各组学生【真实原始路径】的综合三指标 (Gain, Smoothness, Diversity)...")

        groups_dict = {
            'Low-level (<0.65)': sample_weak,
            'Mid-level (0.65~0.8)': sample_mid,
            'High-level (>=0.8)': sample_strong
        }

        # 提前算好 hidden_kt 节省算力
        with torch.no_grad():
            hidden_kt = model_kt.gnn2(relation_graph)

        for group_name, students in groups_dict.items():
            g_gains, g_smooths, g_divs = [], [], []

            for stu in students:
                actual_path = stu['future_seq'][0][:L_TARGET].tolist()

                # 把真实的 actual_path 当作系统推荐的路径传进去！完美对齐！
                e_g, s_m, d_v, _ = evaluate_adaptive_metrics(
                    actual_path,
                    stu['future_seq'],
                    stu['hist_seq'],
                    stu['hist_ans'],
                    stu['hist_time_bins'],
                    model_kt,
                    relation_graph,
                    hidden_kt,
                    stu['p_before']
                )

                g_gains.append(e_g)
                g_smooths.append(s_m)
                g_divs.append(d_v)

            print(f"[{group_name}] 样本数: {len(students)} 人")
            print(f"    -> 真实 Target-Oriented Gain : {np.mean(g_gains):.4f}")
            print(f"    -> 真实 Difficulty Smoothness: {np.mean(g_smooths):.4f}")
            print(f"    -> 真实 Resource Diversity   : {np.mean(g_divs):.4f}")
            print("-" * 50)
        print("=" * 60 + "\n")

    # ==========================================
    # 0. 固定全局随机种子
    # ==========================================
    seed_value = 48
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True

    print(f"当前实验已固定随机种子为: {seed_value}")

    resource_size, train_history_cas, train_history_t, train, valid, test = Split_data(opt.data_name, load_dict=True)
    test_data = DataLoader(test, batch_size=1, load_dict=True, cuda=False)

    relation_graph = ConRelationGraph(opt.data_name)
    hypergraph_list = ConHyperGraphList(train_history_cas, train_history_t, resource_size)

    valid_resource_ids = list(range(2, resource_size))
    opt.resource_size = resource_size

    # 加载模型
    model_rec = MSHGAT(opt, dropout=opt.dropout).cuda()
    model_rec.load_state_dict(torch.load(opt.save_rec_path))
    model_rec.eval()

    model_kt = MSHGAT(opt, dropout=opt.dropout).cuda()
    model_kt.load_state_dict(torch.load(opt.save_kt_path))
    model_kt.eval()

    print("正在从测试集中扫描有效学生序列...")
    candidate_students = []

    for batch in test_data:
        tgt, tgt_timestamp, tgt_idx, ans, tgt_end_time, _, _, _ = [item.cuda() for item in batch]
        time_bins = calc_time_bins(tgt_timestamp, tgt_end_time)

        valid_len = (tgt[0] > 1).sum().item()
        if valid_len > 15:
            t = 15
            hist_seq = tgt[0:1, :t]
            hist_ans = ans[0:1, :t]
            hist_time_bins = time_bins[0:1, :t]
            hist_timestamp = tgt_timestamp[0:1, :t]
            hist_idx = tgt_idx[0:1]

            future_len = min(6, valid_len - t)
            if future_len == 0:
                continue
            future_seq = tgt[0:1, t:t + future_len]

            with torch.no_grad():
                hidden_kt = model_kt.gnn2(relation_graph)
                _, _, yt_init, _, _ = model_kt.ktmodel(hidden_kt, hist_seq, hist_ans, hist_time_bins)
                p_before = yt_init[0, -1, :]

            if p_before.mean().item() < 0.1:
                continue

            with torch.no_grad():
                pred_logits, _, _, _, _ = model_rec(hist_seq, hist_timestamp, hist_idx, hist_ans, relation_graph,
                                                    hypergraph_list, hist_time_bins)
                last_step_logits = pred_logits[-1, :]
                top50_indices = torch.topk(last_step_logits, 80).indices.cpu().numpy()
                hist_list = hist_seq[0].cpu().numpy().tolist()
                topK_candidates = [int(x) for x in top50_indices if x > 1 and x not in hist_list][:50]

            correct_capacity = sum(1 for x in topK_candidates if p_before[x].item() >= 0.5)
            if correct_capacity < 15:
                continue

            candidate_students.append({
                'hist_seq': hist_seq,
                'hist_ans': hist_ans,
                'hist_time_bins': hist_time_bins,
                'p_before': p_before,
                'topK_candidates': topK_candidates,
                'target_ids': topK_candidates[:6],  # 🌟 新增：提取前 6 个作为刚性学习目标
                'p_mean': p_before.mean().item(),
                'future_seq': future_seq
            })

    print(f"扫描完毕！找到 {len(candidate_students)} 个具备干预价值的候选学生。")

    # ==========================================
    # 4. 划分群体与随机多样本采样
    # ==========================================
    group_weak, group_mid, group_strong = [], [], []
    for stu in candidate_students:
        p_mean = stu['p_mean']
        if p_mean < 0.65:
            group_weak.append(stu)
        elif 0.65 <= p_mean < 0.8:
            group_mid.append(stu)
        else:
            group_strong.append(stu)

    print(
        f"全局分组情况: 低掌握度组 {len(group_weak)} 人, 中掌握度组 {len(group_mid)} 人, 高掌握度组 {len(group_strong)} 人")
    if not (group_weak and group_mid and group_strong):
        print("警告：某一组人数为0，请检查数据分布！")
        return

    SAMPLE_N = 50
    L_TARGET = 6
    sample_weak = random.sample(group_weak, min(SAMPLE_N, len(group_weak)))
    sample_mid = random.sample(group_mid, min(SAMPLE_N, len(group_mid)))
    sample_strong = random.sample(group_strong, min(SAMPLE_N, len(group_strong)))

    # 🌟 调用基准评估函数
    calculate_ground_truth_all_metrics(model_kt, relation_graph, sample_weak, sample_mid, sample_strong, L_TARGET)

    # 🌟 工程化开关断点：如果只是想看基准数据，在这里直接停止程序
    if QUICK_BASELINE_ONLY:
        print("\n[系统提示] QUICK_BASELINE_ONLY = True, 极速基准评估结束。如需运行完整实验请将其改为 False。")
        return

    # 🌟 核心函数 1：选出最优路径
    def select_best_path(front_fitness, front_paths, w_gain, w_smooth, w_div=0.2):
        if not front_fitness: return []
        front_arr = np.array(front_fitness)
        mins = front_arr.min(axis=0)
        maxs = front_arr.max(axis=0)
        denoms = maxs - mins + 1e-8
        norm_front = (front_arr - mins) / denoms
        utility = w_gain * norm_front[:, 0] + w_smooth * norm_front[:, 1] + w_div * norm_front[:, 2]
        return front_paths[np.argmax(utility)]

    # ==========================================
    # 5. 执行全指标敏感性分析 (细粒度 0.02)
    # ==========================================
    lambda_1_range = np.round(np.arange(0.10, 0.72, 0.02), 2)

    results = {
        'weak': {'exp_gain': [], 'smooth': [], 'div': [], 'mig_gain': []},
        'mid': {'exp_gain': [], 'smooth': [], 'div': [], 'mig_gain': []},
        'strong': {'exp_gain': [], 'smooth': [], 'div': [], 'mig_gain': []}
    }
    csv_data_rows = []

    print("开始执行高精度参数连续敏感性分析 (自适应长度 & 全指标)...")

    for group_name, sample_group in zip(['weak', 'mid', 'strong'], [sample_weak, sample_mid, sample_strong]):
        print(f"\n处理 [{group_name}] 组...")
        temp_res = {l1: {'exp_gain': [], 'smooth': [], 'div': [], 'mig_gain': []} for l1 in lambda_1_range}

        for stu in sample_group:
            hidden_kt = model_kt.gnn2(relation_graph)

            # 🌟 入口 1：喂给 NSGA-II 算法，让进化方向被靶心牵引
            history = run_nsga2('Prob', stu['hist_seq'], stu['hist_ans'], stu['hist_time_bins'],
                                stu['topK_candidates'], valid_resource_ids, model_kt, relation_graph,
                                stu['p_before'], stu['target_ids'])

            pop_pool = [random.sample(stu['topK_candidates'], 6) for _ in range(500)]

            # 🌟 入口 2：喂给初代种群评估函数
            fits = [evaluate_path(p, stu['hist_seq'], stu['hist_ans'], stu['hist_time_bins'], model_kt, relation_graph,
                                  hidden_kt, stu['p_before'], stu['target_ids']) for p in pop_pool]

            f_idx = non_dominated_sort(fits)
            final_front_fits = [fits[i] for i in f_idx]
            final_front_paths = [pop_pool[i] for i in f_idx]

            for l1 in lambda_1_range:
                l1_key = round(l1, 2)
                l2 = round(0.8 - l1_key, 2)

                best_path = select_best_path(final_front_fits, final_front_paths, w_gain=l1_key, w_smooth=l2, w_div=0.2)

                # 🌟 入口 3：喂给最终生成与截断评估函数
                e_g, s_m, d_v, m_g = evaluate_adaptive_metrics(best_path, stu['future_seq'], stu['hist_seq'],
                                                               stu['hist_ans'], stu['hist_time_bins'], model_kt,
                                                               relation_graph, hidden_kt, stu['p_before'],
                                                               stu['target_ids'])

                temp_res[l1_key]['exp_gain'].append(e_g)
                temp_res[l1_key]['smooth'].append(s_m)
                temp_res[l1_key]['div'].append(d_v)
                temp_res[l1_key]['mig_gain'].append(m_g)

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
            csv_data_rows.append([l1_key, round(0.8 - l1_key, 2), group_name, avg_eg, avg_mg, avg_sm, avg_dv])

    csv_filename = "Sensitivity_Detailed_Results.csv"
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Lambda_1(Gain)', 'Lambda_2(Smooth)', 'Group', 'Expected_Gain', 'Migration_Gain', 'Smoothness',
                         'Diversity'])
        writer.writerows(csv_data_rows)
    print(f"\n[成功] 全量细粒度数据已导出至文件: {csv_filename}")

    # ==========================================
    # 7. 绘制 2x2 全景综合评估图 (修复图例为 0.65 与 0.8)
    # ==========================================
    print("Drawing comprehensive 2x2 multi-metric charts...")
    fig1, axs1 = plt.subplots(2, 2, figsize=(16, 12))

    c_weak, c_mid, c_strong = '#4A90E2', '#F5A623', '#D0021B'
    labels = ['Low-level Group ($p_{before}<0.65$)', 'Mid-level Group ($0.65\leq p_{before}<0.8$)',
              'High-level Group ($p_{before}\geq0.8$)']

    axs1[0, 0].plot(lambda_1_range, results['weak']['exp_gain'], linewidth=3, color=c_weak, label=labels[0])
    axs1[0, 0].plot(lambda_1_range, results['mid']['exp_gain'], linewidth=3, color=c_mid, label=labels[1])
    axs1[0, 0].plot(lambda_1_range, results['strong']['exp_gain'], linewidth=3, color=c_strong, label=labels[2])
    axs1[0, 0].set_xlabel('Knowledge Gain Weight $\lambda_1$', fontsize=12)
    axs1[0, 0].set_ylabel('Adaptive Expected Mastery Gain', fontsize=12)
    axs1[0, 0].set_title('(a) Impact on Expected Knowledge Gain', fontsize=14)
    axs1[0, 0].grid(linestyle='--', alpha=0.4)
    axs1[0, 0].legend()

    axs1[0, 1].plot(lambda_1_range, results['weak']['smooth'], linewidth=3, color=c_weak)
    axs1[0, 1].plot(lambda_1_range, results['mid']['smooth'], linewidth=3, color=c_mid)
    axs1[0, 1].plot(lambda_1_range, results['strong']['smooth'], linewidth=3, color=c_strong)
    axs1[0, 1].set_xlabel('Knowledge Gain Weight $\lambda_1$', fontsize=12)
    axs1[0, 1].set_ylabel('Adaptive Path Smoothness', fontsize=12)
    axs1[0, 1].set_title('(b) Impact on Difficulty Smoothness', fontsize=14)
    axs1[0, 1].grid(linestyle='--', alpha=0.4)

    axs1[1, 0].plot(lambda_1_range, results['weak']['div'], linewidth=3, color=c_weak)
    axs1[1, 0].plot(lambda_1_range, results['mid']['div'], linewidth=3, color=c_mid)
    axs1[1, 0].plot(lambda_1_range, results['strong']['div'], linewidth=3, color=c_strong)
    axs1[1, 0].set_xlabel('Knowledge Gain Weight $\lambda_1$', fontsize=12)
    axs1[1, 0].set_ylabel('Adaptive Resource Diversity', fontsize=12)
    axs1[1, 0].set_title('(c) Impact on Resource Diversity', fontsize=14)
    axs1[1, 0].grid(linestyle='--', alpha=0.4)

    global_exp_gain = np.mean([results['weak']['exp_gain'], results['mid']['exp_gain'], results['strong']['exp_gain']],
                              axis=0)
    global_mig_gain = np.mean([results['weak']['mig_gain'], results['mid']['mig_gain'], results['strong']['mig_gain']],
                              axis=0)

    axs1[1, 1].plot(lambda_1_range, global_exp_gain, linewidth=3, color='#8B008B',
                    label='Global Expected Gain (Internal)')
    axs1[1, 1].plot(lambda_1_range, global_mig_gain, linewidth=3, color='#2E8B57', linestyle='--',
                    label='Global Migration Gain (Ground Truth)')
    axs1[1, 1].set_xlabel('Knowledge Gain Weight $\lambda_1$', fontsize=12)
    axs1[1, 1].set_ylabel('Average Gain Value', fontsize=12)
    axs1[1, 1].set_title('(d) Internal Expected vs. Real Migration Gain Alignment', fontsize=14)
    axs1[1, 1].grid(linestyle='--', alpha=0.4)
    axs1[1, 1].legend()

    plt.tight_layout()
    plt.savefig('Comprehensive_Metrics_Sensitivity_EN.png', dpi=300)
    plt.close(fig1)

    # ==========================================
    # 8. 绘制 1x3 各认知阶段对齐图
    # ==========================================
    fig2, axs2 = plt.subplots(1, 3, figsize=(16, 5.5))
    groups_list = ['weak', 'mid', 'strong']
    titles = ['(a) Low-level Group ($p_{before} < 0.65$)', '(b) Mid-level Group ($0.65 \leq p_{before} < 0.8$)',
              '(c) High-level Group ($p_{before} \geq 0.8$)']

    for i, (grp, title) in enumerate(zip(groups_list, titles)):
        axs2[i].plot(lambda_1_range, results[grp]['exp_gain'], linewidth=3.5, color='#8B008B',
                     label='Expected Gain (Internal)')
        axs2[i].plot(lambda_1_range, results[grp]['mig_gain'], linewidth=3.5, color='#2E8B57', linestyle='--',
                     label='Migration Gain (Ground Truth)')
        axs2[i].set_xlabel('Knowledge Gain Weight $\lambda_1$', fontsize=12)
        axs2[i].set_ylabel('Gain Value', fontsize=12)
        axs2[i].set_title(title, fontsize=14, pad=12)
        axs2[i].grid(linestyle='--', alpha=0.4)
        axs2[i].legend(loc='best', fontsize=10)

    plt.tight_layout()
    plt.savefig('Groupwise_Expected_vs_Migration_EN.png', dpi=300)
    plt.close(fig2)
    print("All charts generated and memory cleared successfully.")


if __name__ == "__main__":
    main()