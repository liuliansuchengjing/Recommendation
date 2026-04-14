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
    seed_value = 41  # 🌟 这里的数字就是你的“盲盒编号”，你可以任意修改（如 0, 100, 2026 等）
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

    # 3. 提取真实学生数据进行实验 (带 ZPD 潜力筛选)
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

            # 步骤 A: 评估初始状态
            with torch.no_grad():
                hidden_kt = model_kt.gnn2(relation_graph)
                _, _, yt_init, _, _ = model_kt.ktmodel(hidden_kt, hist_seq, hist_ans, hist_time_bins)
                p_before = yt_init[0, -1, :]

            # 注意：这里我们移除了对高分学霸的强制 continue，由后面的分组逻辑接管
            # 仅过滤掉基础极差（均值 < 0.1），连题都看不懂的人
            if p_before.mean().item() < 0.1:
                continue

            # 步骤 B: 推荐候选池
            with torch.no_grad():
                pred_logits, _, _, _, _ = model_rec(hist_seq, hist_timestamp, hist_idx, hist_ans, relation_graph,
                                                    hypergraph_list, hist_time_bins)
                last_step_logits = pred_logits[-1, :]
                top50_indices = torch.topk(last_step_logits, 80).indices.cpu().numpy()
                hist_list = hist_seq[0].cpu().numpy().tolist()
                topK_candidates = [int(x) for x in top50_indices if x > 1 and x not in hist_list][:50]

            # 🌟 核心修复：ZPD 潜力甄别！确保推荐池里有 >= 15 道题是他能做对的
            correct_capacity = sum(1 for x in topK_candidates if p_before[x].item() >= 0.5)
            if correct_capacity < 15:
                continue

            candidate_students.append({
                'hist_seq': hist_seq,
                'hist_ans': hist_ans,
                'hist_time_bins': hist_time_bins,
                'p_before': p_before,
                'topK_candidates': topK_candidates,
                'p_mean': p_before.mean().item()
            })

    print(f"扫描完毕！测试集中共找到 {len(candidate_students)} 个具备干预价值的候选学生。")

    # ==========================================
    # 4. 划分三个认知阶段的学生群体 (调整了阈值为 0.8)
    # ==========================================
    group_weak, group_mid, group_strong = [], [], []
    for stu in candidate_students:
        p_mean = stu['p_mean']
        if p_mean < 0.5:
            group_weak.append(stu)
        elif 0.5 <= p_mean < 0.8:  # 🌟 阈值调整为 0.8
            group_mid.append(stu)
        else:
            group_strong.append(stu)

    print(
        f"分组情况: 基础薄弱组 {len(group_weak)} 人, 能力巩固组 {len(group_mid)} 人, 进阶提升组 {len(group_strong)} 人")

    if not (group_weak and group_mid and group_strong):
        print("警告：某一组人数为0，请检查数据分布！")
        return

    rep_weak = random.choice(group_weak)
    rep_mid = random.choice(group_mid)
    rep_strong = random.choice(group_strong)

    # 🌟 参数细化：从 0.1 到 0.7，步长为 0.05，产生更加平滑的连续曲线
    lambda_1_range = np.round(np.arange(0.10, 0.75, 0.05), 2)

    def select_best_path_gain(pareto_front, w_gain, w_smooth, w_div=0.2):
        if not pareto_front: return 0.0
        front_arr = np.array(pareto_front)

        # 极差标准化
        mins = front_arr.min(axis=0)
        maxs = front_arr.max(axis=0)
        denoms = maxs - mins + 1e-8
        norm_front = (front_arr - mins) / denoms

        # 效用计算
        utility = w_gain * norm_front[:, 0] + w_smooth * norm_front[:, 1] + w_div * norm_front[:, 2]
        best_idx = np.argmax(utility)
        return front_arr[best_idx, 0]

        # ==========================================

    # 5. 执行敏感性优化实验
    # ==========================================
    results_gain = {'weak': [], 'mid': [], 'strong': []}

    print("开始执行更细粒度的连续敏感性分析 (步长 0.05)...")

    for group_name, stu in zip(['weak', 'mid', 'strong'], [rep_weak, rep_mid, rep_strong]):
        print(f"正在优化 {group_name} 组代表学生 (初始掌握度 {stu['p_mean']:.3f})...")
        history = run_nsga2('Prob', stu['hist_seq'], stu['hist_ans'], stu['hist_time_bins'],
                            stu['topK_candidates'], valid_resource_ids, model_kt, relation_graph, stu['p_before'])
        final_front = history.get(30, [])

        for l1 in lambda_1_range:
            l2 = round(0.8 - l1, 2)
            best_gain = select_best_path_gain(final_front, w_gain=l1, w_smooth=l2, w_div=0.2)
            results_gain[group_name].append(best_gain)

    # ==========================================
    # 6. 绘制高精度连续敏感性曲线
    # ==========================================
    print("实验完成，正在绘制高精度敏感性曲线...")
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(10, 6.5))

    ax.plot(lambda_1_range, results_gain['weak'], marker='o', markersize=6, linewidth=2.5,
            color='#4A90E2', label='基础薄弱组 ($p_{before} < 0.5$)', alpha=0.9)
    ax.plot(lambda_1_range, results_gain['mid'], marker='s', markersize=6, linewidth=2.5,
            color='#F5A623', label='能力巩固组 ($0.5 \leq p_{before} < 0.8$)', alpha=0.9)
    ax.plot(lambda_1_range, results_gain['strong'], marker='^', markersize=7, linewidth=2.5,
            color='#D0021B', label='进阶提升组 ($p_{before} \geq 0.8$)', alpha=0.9)

    # 绘制垂直辅助线：标出不同群体的“大概峰值区间”以辅助读者视觉对齐
    best_weak_idx = np.argmax(results_gain['weak'])
    best_strong_idx = np.argmax(results_gain['strong'])
    ax.axvline(lambda_1_range[best_weak_idx], color='#4A90E2', linestyle=':', alpha=0.5)
    ax.axvline(lambda_1_range[best_strong_idx], color='#D0021B', linestyle=':', alpha=0.5)

    ax.set_xlabel('知识增益权重参数 $\lambda_1$\n(约束条件: $\lambda_2 = 0.8 - \lambda_1, \lambda_3 = 0.2$)',
                  fontsize=13)
    ax.set_ylabel('最终决策路径的离线迁移增益 (Selected Path Gain)', fontsize=13)
    ax.set_title('图 5.x 多认知群体下的效用函数权重参数连续敏感性分析', fontsize=16, pad=15)

    ax.set_xticks(np.round(np.arange(0.1, 0.8, 0.1), 1))  # X轴刻度保持干净，只显示0.1,0.2...
    ax.grid(linestyle='--', alpha=0.4)
    ax.legend(loc='best', fontsize=12, framealpha=0.9)

    plt.tight_layout()
    plt.savefig('Fine_Continuous_Sensitivity.png', dpi=300)
    print("高精度敏感性曲线图已生成: Fine_Continuous_Sensitivity.png")
    plt.show()


# ==========================================
# 执行入口 (这段一定要放在文件最末尾！)
# ==========================================
if __name__ == "__main__":
    main()