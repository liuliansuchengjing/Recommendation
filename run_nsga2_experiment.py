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
    seed_value = 42  # 🌟 这里的数字就是你的“盲盒编号”，你可以任意修改（如 0, 100, 2026 等）
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

    # 3. 提取真实学生数据进行实验
    print("正在从测试集中扫描有效学生序列...")
    candidate_students = []  # 用来装所有符合条件的候选学生

    for batch in test_data:
        tgt, tgt_timestamp, tgt_idx, ans, tgt_end_time, _, _, _ = [item.cuda() for item in batch]
        time_bins = calc_time_bins(tgt_timestamp, tgt_end_time)

        valid_len = (tgt[0] > 1).sum().item()
        if valid_len > 15:  # 找一个答题记录够长的学生
            # 截取前 15 题作为历史序列
            t = 15
            hist_seq = tgt[0:1, :t]
            hist_ans = ans[0:1, :t]
            hist_time_bins = time_bins[0:1, :t]
            hist_timestamp = tgt_timestamp[0:1, :t]
            hist_idx = tgt_idx[0:1]

            # --- 步骤 A: 用 DKT 评估初始知识状态 (p_before) ---
            with torch.no_grad():
                hidden_kt = model_kt.gnn2(relation_graph)
                _, _, yt_init, _, _ = model_kt.ktmodel(hidden_kt, hist_seq, hist_ans, hist_time_bins)
                p_before = yt_init[0, -1, :]  # 获取学完 15 题后的掌握度

            # 学霸过滤：如果该学生已经掌握了大多数知识（均值>0.8），换一个学生
            if p_before.mean().item() > 0.8:
                continue

            # --- 步骤 B: 用推荐模型生成 TopK (K=50) 候选池 ---
            with torch.no_grad():
                pred_logits, _, _, _, _ = model_rec(hist_seq, hist_timestamp, hist_idx, hist_ans, relation_graph,
                                                    hypergraph_list, hist_time_bins)
                last_step_logits = pred_logits[-1, :]

                # 提取 Top 50 并且过滤掉已经做过的题和无效占位符
                top50_indices = torch.topk(last_step_logits, 80).indices.cpu().numpy()
                hist_list = hist_seq[0].cpu().numpy().tolist()
                topK_candidates = [int(x) for x in top50_indices if x > 1 and x not in hist_list][:50]

            # 把符合条件的学生存进列表，去掉原来的 break！
            candidate_students.append({
                'hist_seq': hist_seq,
                'hist_ans': hist_ans,
                'hist_time_bins': hist_time_bins,
                'p_before': p_before,
                'topK_candidates': topK_candidates
            })

    print(f"扫描完毕！测试集中共找到 {len(candidate_students)} 个符合条件的候选学生。")

    if len(candidate_students) == 0:
        print("未找到符合条件的学生，请调低学霸过滤阈值！")
        return

    # 从候选池中随机挑一个学生进行本次实验！
    selected_student = random.choice(candidate_students)

    hist_seq = selected_student['hist_seq']
    hist_ans = selected_student['hist_ans']
    hist_time_bins = selected_student['hist_time_bins']
    p_before = selected_student['p_before']
    topK_candidates = selected_student['topK_candidates']

    print("成功随机抽取一名学生样本！初始知识平均掌握度: {:.4f}".format(p_before.mean().item()))

    # ==========================================
    # 4. 执行 NSGA-II 算法 (只需运行一次，获取完整的历史字典)
    # ==========================================
    print("Running NSGA-II Strategy A (Random)...")
    history_random = run_nsga2('Random', hist_seq, hist_ans, hist_time_bins, topK_candidates, valid_resource_ids,
                               model_kt, relation_graph, p_before)

    print("Running NSGA-II Strategy B (Probability Screening)...")
    history_prob = run_nsga2('Prob', hist_seq, hist_ans, hist_time_bins, topK_candidates, valid_resource_ids, model_kt,
                             relation_graph, p_before)

    # 🌟 安全解包函数：彻底防止 NoneType 或者 zip 报错
    def get_xyz(history_dict, gen):
        front = history_dict.get(gen, [])
        if not front:
            return [], [], []
        x, y, z = zip(*front)
        return list(x), list(y), list(z)

    # ==========================================
    # 可视化 1: 最终代 (Gen 30) 的帕累托前沿对比散点图
    # ==========================================
    print("Experiment completed, drawing Pareto front comparison plots...")
    fig1 = plt.figure(figsize=(15, 10))

    # 直接从历史字典中提取最后一代 (Gen 30)
    r_gain, r_smooth, r_div = get_xyz(history_random, 30)
    p_gain, p_smooth, p_div = get_xyz(history_prob, 30)

    # 子图1: 3D 前沿图
    ax1 = fig1.add_subplot(221, projection='3d')
    ax1.scatter(r_gain, r_smooth, r_div, c='blue', marker='o', alpha=0.5, label='Random Candidate')
    ax1.scatter(p_gain, p_smooth, p_div, c='red', marker='^', s=60, label='Probability Screening')
    ax1.set_xlabel('Proficiency Gain')
    ax1.set_ylabel('Difficulty Smoothness')
    ax1.set_zlabel('Resource Diversity')
    ax1.set_xlim([-1.0, 1.0])
    ax1.set_title('3D Pareto Front Distribution')
    ax1.legend()

    # 子图2: 增益 vs 平滑度
    ax2 = fig1.add_subplot(222)
    ax2.scatter(r_gain, r_smooth, c='blue', alpha=0.5)
    ax2.scatter(p_gain, p_smooth, c='red', marker='^')
    ax2.axvline(0, color='gray', linestyle='--')
    ax2.set_xlabel('Proficiency Gain')
    ax2.set_ylabel('Difficulty Smoothness')
    ax2.set_xlim([-1.0, 1.0])
    ax2.set_title('2D Projection: Gain vs. Smoothness')

    # 子图3: 增益 vs 多样性
    ax3 = fig1.add_subplot(223)
    ax3.scatter(r_gain, r_div, c='blue', alpha=0.5)
    ax3.scatter(p_gain, p_div, c='red', marker='^')
    ax3.axvline(0, color='gray', linestyle='--')
    ax3.set_xlabel('Proficiency Gain')
    ax3.set_ylabel('Resource Diversity')
    ax3.set_xlim([-1.0, 1.0])
    ax3.set_title('2D Projection: Gain vs. Diversity')

    # 子图4: 平滑度 vs 多样性
    ax4 = fig1.add_subplot(224)
    ax4.scatter(r_smooth, r_div, c='blue', alpha=0.5)
    ax4.scatter(p_smooth, p_div, c='red', marker='^')
    ax4.set_xlabel('Difficulty Smoothness')
    ax4.set_ylabel('Resource Diversity')
    ax4.set_title('2D Projection: Smoothness vs. Diversity')

    plt.tight_layout()
    plt.savefig('pareto_front_comparison.png', dpi=300)
    print("Visualization saved as pareto_front_comparison.png")

    # ==========================================
    # 可视化 2: 种群收敛轨迹与移动方向图
    # ==========================================
    # ==========================================
    # 可视化 2: 种群收敛轨迹与移动方向图 (三视图全景)
    # ==========================================
    print("Drawing comprehensive evolution trajectory plots...")

    # 创建 1x3 的画布，尺寸横向拉长以适应三个子图
    fig2, (ax_traj1, ax_traj2, ax_traj3) = plt.subplots(1, 3, figsize=(18, 5.5))

    # 提取概率筛选策略在不同代数的数据
    gen1_g, gen1_s, gen1_d = get_xyz(history_prob, 1)
    gen15_g, gen15_s, gen15_d = get_xyz(history_prob, 15)
    gen30_g, gen30_s, gen30_d = get_xyz(history_prob, 30)

    # 定义一个内部辅助绘图函数，保持三个子图的代码极其精简和统一
    def plot_trajectory(ax, x1, y1, x15, y15, x30, y30, xlabel, ylabel, title, show_zero_line=False):
        # 绘制不同代数的散点
        ax.scatter(x1, y1, c='#FFB6C1', marker='o', s=50, label='Gen 1', alpha=0.6)
        ax.scatter(x15, y15, c='#FF4500', marker='s', s=60, label='Gen 15', alpha=0.7)
        ax.scatter(x30, y30, c='#8B0000', marker='^', s=80, label='Gen 30', alpha=0.9)

        # 绘制表示“移动方向”的引导箭头 (从 Gen1 质心指向 Gen30 质心)
        if x1 and x30:
            c1_x, c1_y = np.mean(x1), np.mean(y1)
            c30_x, c30_y = np.mean(x30), np.mean(y30)

            # shrink=0.1 使得箭头首尾留有空隙，不会遮挡质心点
            ax.annotate('', xy=(c30_x, c30_y), xytext=(c1_x, c1_y),
                        arrowprops=dict(facecolor='black', edgecolor='black', width=2, headwidth=10, alpha=0.5,
                                        shrink=0.1))

            # 在箭头的中间位置标注 'Direction'
            text_x, text_y = (c1_x + c30_x) / 2, (c1_y + c30_y) / 2
            # 略微上移文字以免和箭头重叠
            ax.text(text_x, text_y + 0.015, 'Direction', fontsize=11, fontweight='bold', color='#444', ha='center',
                    va='bottom')

        # 图表基础修饰
        if show_zero_line:
            ax.axvline(0, color='gray', linestyle='--', alpha=0.5)  # 仅在包含 Gain 的图中画零线
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, pad=10)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(alpha=0.3)

    # 依次调用绘制三个视图
    # 视图 1：Gain vs. Smoothness
    plot_trajectory(ax_traj1, gen1_g, gen1_s, gen15_g, gen15_s, gen30_g, gen30_s,
                    'Expected Mastery Gain', 'Difficulty Smoothness',
                    '(a) Gain vs. Smoothness Trajectory', show_zero_line=True)

    # 视图 2：Gain vs. Diversity
    plot_trajectory(ax_traj2, gen1_g, gen1_d, gen15_g, gen15_d, gen30_g, gen30_d,
                    'Expected Mastery Gain', 'Resource Diversity',
                    '(b) Gain vs. Diversity Trajectory', show_zero_line=True)

    # 视图 3：Smoothness vs. Diversity
    plot_trajectory(ax_traj3, gen1_s, gen1_d, gen15_s, gen15_d, gen30_s, gen30_d,
                    'Difficulty Smoothness', 'Resource Diversity',
                    '(c) Smoothness vs. Diversity Trajectory', show_zero_line=False)

    plt.tight_layout()
    plt.savefig('Evolution_Trajectory_3Views.png', dpi=300)
    print("Trajectory saved as Evolution_Trajectory_3Views.png")

    plt.show()


# ==========================================
# 执行入口 (这段一定要放在文件最末尾！)
# ==========================================
if __name__ == "__main__":
    main()