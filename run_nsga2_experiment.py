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
# 1. 核心适应度函数 (接入 DKT 模拟)
# ==========================================
def evaluate_path(path_ids, hist_seq, hist_ans, hist_time_bins, model_kt, graph, hidden_kt, p_before):
    L_TARGET = len(path_ids)
    path_tensor = torch.tensor([path_ids], device='cuda')

    # 模拟作答状态: 假设都作对(1)，时间分箱设为2(1-3分钟)
    ans_tensor = torch.ones((1, L_TARGET), device='cuda')
    time_bins_tensor = torch.full((1, L_TARGET), 2, device='cuda')

    # 拼接历史与生成的候选路径
    sim_seq = torch.cat([hist_seq, path_tensor], dim=1)
    sim_ans = torch.cat([hist_ans, ans_tensor], dim=1)
    sim_time_bins = torch.cat([hist_time_bins, time_bins_tensor], dim=1)

    # DKT 模拟预测未来的掌握状态 (注意 DKT.py forward 需要 4 个参数)
    with torch.no_grad():
        _, _, yt_sim, _, _ = model_kt.ktmodel(hidden_kt, sim_seq, sim_ans, sim_time_bins)
        p_after = yt_sim[0, -1, :]  # 获取路径学完后的最终状态

    # ---- 目标1: 知识增益 (f_gain) ∈ [-1, 1] ----
    f_gain = 0.0
    for idx in path_ids:
        gain = p_after[idx].item() - p_before[idx].item()
        room = max(1.0 - p_before[idx].item(), 0.1)  # 剩余提升空间
        f_gain += max(-1.0, gain / room)
    f_gain /= L_TARGET

    # ---- 目标2: 难度平滑度 (f_smooth) ∈ [0, 1] ----
    valid_hist = [get_diff(int(x)) for x, y in zip(hist_seq[0], hist_ans[0]) if int(x) > 1 and int(y) == 1]
    delta = np.mean(valid_hist[-5:]) if len(valid_hist) > 0 else 1.0

    f_smooth = 0.0
    for idx in path_ids:
        f_smooth += 1.0 - (abs(delta - get_diff(idx)) / 2.0)
    f_smooth /= L_TARGET

    # ---- 目标3: 资源多样性 (f_div) ∈ [0, 1] ----
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
# 2. 简易 NSGA-II 框架
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

    # 初始化种群
    population = []
    for _ in range(POPULATION_SIZE):
        if strategy == 'Random':
            # 全局随机策略：从所有有效资源中盲抽
            population.append(random.sample(valid_resource_ids, L_TARGET))
        elif strategy == 'Prob':
            # 概率筛选策略：仅从推荐模型给出的 TopK (K=50) 候选集中抽取
            population.append(random.sample(topK_candidates, L_TARGET))

    # 评估与非支配排序
    fitness = [evaluate_path(p, hist_seq, hist_ans, hist_time_bins, model_kt, graph, hidden_kt, p_before) for p in
               population]
    front_indices = non_dominated_sort(fitness)
    return [fitness[i] for i in front_indices]


# ==========================================
# 3. 主函数缝合执行
# ==========================================
def main():
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
    print("正在从测试集中提取有效学生序列...")
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
                top50_indices = torch.topk(last_step_logits, 80).indices.cpu().numpy()  # 多取一点用来过滤
                hist_list = hist_seq[0].cpu().numpy().tolist()
                topK_candidates = [int(x) for x in top50_indices if x > 1 and x not in hist_list][:50]

            # 成功提取，跳出循环
            break

    print("成功提取学生样本！初始知识平均掌握度: {:.4f}".format(p_before.mean().item()))

    # 4. 执行 NSGA-II 对比实验
    print("Running NSGA-II Strategy A (Random)...")
    pareto_random = run_nsga2('Random', hist_seq, hist_ans, hist_time_bins, topK_candidates, valid_resource_ids,
                              model_kt, relation_graph, p_before)

    print("Running NSGA-II Strategy B (Probability Screening)...")
    pareto_prob = run_nsga2('Prob', hist_seq, hist_ans, hist_time_bins, topK_candidates, valid_resource_ids, model_kt,
                            relation_graph, p_before)

    # ======= 可视化绘制 =======
    print("实验完成，正在绘制帕累托前沿对比图...")
    # 设置中文字体，防止图表中的中文变方块
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']  # Windows用SimHei，Mac用Arial Unicode MS
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

    fig = plt.figure(figsize=(15, 10))

    # 解析数据 (增加防空判断)
    r_gain, r_smooth, r_div = zip(*pareto_random) if pareto_random else ([], [], [])
    p_gain, p_smooth, p_div = zip(*pareto_prob) if pareto_prob else ([], [], [])

    # 子图1: 3D 前沿图
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.scatter(r_gain, r_smooth, r_div, c='blue', marker='o', alpha=0.5, label='随机候选策略')
    ax1.scatter(p_gain, p_smooth, p_div, c='red', marker='^', s=60, label='概率筛选策略')
    ax1.set_xlabel('期望掌握度增益 (Gain)')
    ax1.set_ylabel('难度平滑度 (Smoothness)')
    ax1.set_zlabel('资源多样性 (Diversity)')
    ax1.set_xlim([-1.0, 1.0])  # 明确标出负增益区间
    ax1.set_title('三维帕累托前沿分布对比')
    ax1.legend()

    # 子图2: 增益 vs 平滑度
    ax2 = fig.add_subplot(222)
    ax2.scatter(r_gain, r_smooth, c='blue', alpha=0.5)
    ax2.scatter(p_gain, p_smooth, c='red', marker='^')
    ax2.axvline(0, color='gray', linestyle='--')  # 零增益基准线
    ax2.set_xlabel('期望掌握度增益 (Gain)')
    ax2.set_ylabel('难度平滑度 (Smoothness)')
    ax2.set_xlim([-1.0, 1.0])
    ax2.set_title('2D 投影: 增益 vs 平滑度')

    # 子图3: 增益 vs 多样性
    ax3 = fig.add_subplot(223)
    ax3.scatter(r_gain, r_div, c='blue', alpha=0.5)
    ax3.scatter(p_gain, p_div, c='red', marker='^')
    ax3.axvline(0, color='gray', linestyle='--')
    ax3.set_xlabel('期望掌握度增益 (Gain)')
    ax3.set_ylabel('资源多样性 (Diversity)')
    ax3.set_xlim([-1.0, 1.0])
    ax3.set_title('2D 投影: 增益 vs 多样性')

    # 子图4: 平滑度 vs 多样性
    ax4 = fig.add_subplot(224)
    ax4.scatter(r_smooth, r_div, c='blue', alpha=0.5)
    ax4.scatter(p_smooth, p_div, c='red', marker='^')
    ax4.set_xlabel('难度平滑度 (Smoothness)')
    ax4.set_ylabel('资源多样性 (Diversity)')
    ax4.set_title('2D 投影: 平滑度 vs 多样性')

    plt.tight_layout()
    plt.savefig('pareto_front_comparison.png', dpi=300)
    print("可视化图像已成功保存为 pareto_front_comparison.png")
    plt.show()


# ==========================================
# 执行入口 (这段一定要放在文件最末尾！)
# ==========================================
if __name__ == "__main__":
    main()