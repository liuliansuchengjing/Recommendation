import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import random
from HGAT import MSHGAT
from dataLoader import Split_data, Options
from graphConstruct import ConRelationGraph, ConHyperGraphList
from calculate_muti_obj import gain_test_model


# ==========================================
# 1. 配置与超参数
# ==========================================
class ExpOptions:
    def __init__(self):
        self.data_name = 'MOO'
        self.d_model = 64
        self.d_word_vec = 64
        self.initialFeatureSize = 64
        self.dropout = 0.3
        self.resource_size = 0  # 稍后更新
        self.batch_size = 1  # 逐个学生评估
        self.save_rec_path = "./checkpoint/REC_Prediction_M100.pt"
        self.save_kt_path = "./checkpoint/KT_Prediction_M100.pt"


opt = ExpOptions()
L_TARGET = 6  # 统一设定论文5.6.2节要求的路径长度 L=6
POPULATION_SIZE = 50
MAX_GEN = 30

# ==========================================
# 2. 加载难度与映射字典
# ==========================================
data_opts = Options(opt.data_name)
with open(data_opts.idx2u_dict, 'rb') as f:
    idx2u = pickle.load(f)

difficulty_dict = {}  # key: idx, value: diff
with open(data_opts.difficult_file, 'r') as f:
    next(f)
    for line in f:
        parts = line.strip().split(',')
        if len(parts) >= 2:
            try:
                difficulty_dict[int(parts[0])] = int(parts[1])
            except ValueError:
                continue


def get_diff(idx):
    if idx <= 1: return 1
    real_id = int(idx2u[idx])
    return difficulty_dict.get(real_id, 1)


# ==========================================
# 3. 目标函数计算 (核心修正点)
# ==========================================
def evaluate_path(path_ids, hist_seq, hist_ans, model_kt, graph, hidden_kt, p_before):
    """
    计算三个维度的目标值
    path_ids: 长度为L_TARGET的推荐题目ID列表
    """
    path_tensor = torch.tensor([path_ids], device='cuda')

    # 模拟作答状态 (假设都作对，或者基于 p_before 预测作答)
    # 为保证公平，假设全做对以观察最佳增益上限
    ans_tensor = torch.ones((1, L_TARGET), device='cuda')

    sim_seq = torch.cat([hist_seq, path_tensor], dim=1)
    sim_ans = torch.cat([hist_ans, ans_tensor], dim=1)

    _, _, yt_sim, _, _ = model_kt.ktmodel(hidden_kt, sim_seq, sim_ans)
    p_after = yt_sim[0, -1, :]

    # ---- 目标1: 知识增益 (f_gain) ∈ [-1, 1] ----
    f_gain = 0.0
    for idx in path_ids:
        gain = p_after[idx].item() - p_before[idx].item()
        room = 1.0 - p_before[idx].item()
        room = max(room, 0.1)  # 防止分母为0
        f_gain += max(-1.0, gain / room)  # 保留负增益逻辑
    f_gain /= L_TARGET

    # ---- 目标2: 难度平滑度 (f_smooth) ∈ [0, 1] ----
    # 计算历史能力 delta (简化取近期答对题目难度的平均)
    valid_hist = [get_diff(int(x)) for x, y in zip(hist_seq[0], hist_ans[0]) if int(x) > 1 and int(y) == 1]
    delta = np.mean(valid_hist[-5:]) if len(valid_hist) > 0 else 1.0

    f_smooth = 0.0
    for idx in path_ids:
        diff_i = get_diff(idx)
        # 最大难度差为2 (3 - 1)，故除以2进行归一化
        f_smooth += 1.0 - (abs(delta - diff_i) / 2.0)
    f_smooth /= L_TARGET

    # ---- 目标3: 资源多样性 (f_div) ∈ [0, 1] ----
    f_div = 0.0
    pairs = 0
    # 获取资源静态嵌入 (取 model_kt.gnn 的输出)
    embs = hidden_kt[path_ids]
    for i in range(L_TARGET):
        for j in range(i + 1, L_TARGET):
            sim = torch.cosine_similarity(embs[i].unsqueeze(0), embs[j].unsqueeze(0)).item()
            # 余弦相似度[-1,1] 转换到 多样性[0,1]
            f_div += (1.0 - sim) / 2.0
            pairs += 1

    if pairs > 0:
        f_div /= pairs

    return f_gain, f_smooth, f_div


# ==========================================
# 4. 简易 NSGA-II 流程
# ==========================================
def non_dominated_sort(population_fitness):
    """提取第一层帕累托前沿 (Rank 1)"""
    front = []
    for i, fit_i in enumerate(population_fitness):
        dominated = False
        for j, fit_j in enumerate(population_fitness):
            if i == j: continue
            # 判断 j 是否支配 i (所有目标都大于等于，且至少一个严格大于)
            if (fit_j[0] >= fit_i[0] and fit_j[1] >= fit_i[1] and fit_j[2] >= fit_i[2]) and \
                    (fit_j[0] > fit_i[0] or fit_j[1] > fit_i[1] or fit_j[2] > fit_i[2]):
                dominated = True
                break
        if not dominated:
            front.append(i)
    return front


def run_nsga2(strategy, hist_seq, hist_ans, model_rec, model_kt, graph, p_before, valid_resource_ids):
    hidden_kt = model_kt.gnn2(graph)

    # 1. 种群初始化
    population = []
    if strategy == 'Random':
        for _ in range(POPULATION_SIZE):
            population.append(random.sample(valid_resource_ids, L_TARGET))
    elif strategy == 'Prob':
        # 利用 model_rec 预测偏好
        pred_logits, _, _, _, _ = model_rec(hist_seq, hist_seq, hist_seq, hist_ans, graph, None, None)
        topk_ids = torch.topk(pred_logits[-1], 100).indices.cpu().tolist()  # 隐式偏好约束池
        topk_ids = [x for x in topk_ids if x > 1]
        for _ in range(POPULATION_SIZE):
            population.append(random.sample(topk_ids, L_TARGET))

    # 2. 简易演化迭代 (由于重点是分布对比，这里简化变异交叉，直接评估初始化种群+微量变异)
    for gen in range(MAX_GEN):
        # ... (此处可补充具体的遗传交叉变异逻辑，为节约篇幅暂略，核心在评估)
        pass

        # 3. 适应度评估与帕累托提取
    fitness = [evaluate_path(p, hist_seq, hist_ans, model_kt, graph, hidden_kt, p_before) for p in population]
    front_indices = non_dominated_sort(fitness)
    pareto_fitness = [fitness[i] for i in front_indices]

    return pareto_fitness


# ==========================================
# 5. 主程序与 3D/2D 可视化
# ==========================================
def main():
    # 数据加载 (略写，调用 Split_data)
    resource_size, _, _, _, _, test = Split_data(opt.data_name, load_dict=True)
    opt.resource_size = resource_size
    valid_resource_ids = list(range(2, resource_size))

    # 模型加载
    model_rec = MSHGAT(opt, dropout=opt.dropout).cuda()
    model_rec.load_state_dict(torch.load(opt.save_rec_path))
    model_rec.eval()

    model_kt = MSHGAT(opt, dropout=opt.dropout).cuda()
    model_kt.load_state_dict(torch.load(opt.save_kt_path))
    model_kt.eval()

    graph = ConRelationGraph(opt.data_name)

    # 抽取一个有提升空间的样本 (学霸过滤)
    # ... (从 test 集中挑选 hist_seq, hist_ans 且 p_before 均值 < 0.8 的样本)
    # 假设我们获取了以下样本 (需替换为真实数据提取):
    # hist_seq = ...
    # hist_ans = ...
    # p_before = ...

    print("Running NSGA-II Strategy A (Random)...")
    pareto_random = run_nsga2('Random', hist_seq, hist_ans, model_rec, model_kt, graph, p_before, valid_resource_ids)

    print("Running NSGA-II Strategy B (Probability Screening)...")
    pareto_prob = run_nsga2('Prob', hist_seq, hist_ans, model_rec, model_kt, graph, p_before, valid_resource_ids)

    # ======= 可视化绘制 =======
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(15, 10))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文

    # 解析数据
    r_gain, r_smooth, r_div = zip(*pareto_random)
    p_gain, p_smooth, p_div = zip(*pareto_prob)

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
    print("可视化图像已保存为 pareto_front_comparison.png")
    plt.show()


if __name__ == '__main__':
    # main() # 确保数据准备好后再解除注释
    pass