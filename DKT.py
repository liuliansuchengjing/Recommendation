import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import os
import pickle

class DKT(nn.Module):

    def __init__(self, emb_dim, hidden_dim, num_skills, data_name, dropout=0.2, bias=True):
        super(DKT, self).__init__()
        self.emb_dim = emb_dim  # 嵌入维度
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.correct_embed = nn.Embedding(2, emb_dim)  # 答案结果嵌入（正确、错误）

        # --- 新增：难度嵌入层 (0用于PAD/EOS，1,2,3用于真实难度) ---
        self.diff_embed = nn.Embedding(5, emb_dim)

        # ==========================================================
        # 核心逻辑：离线构建“内部ID -> 难度”的映射张量
        # ==========================================================
        diff_map = torch.zeros(num_skills, dtype=torch.long)
        idx2u_path = f'data/{data_name}/idx2u.pickle'
        diff_path = f'data/{data_name}/difficulty.csv'

        if os.path.exists(idx2u_path) and os.path.exists(diff_path):
            with open(idx2u_path, 'rb') as f:
                idx2u = pickle.load(f)

            # 1. 读取 difficulty.csv 构建字典
            diff_dict = {}
            with open(diff_path, 'r') as f:
                next(f)  # 跳过表头 challenge_id,difficulty
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        diff_dict[str(parts[0])] = int(parts[1])

            # 2. 遍历内部 ID (0 到 num_skills-1)，生成映射表
            for i in range(num_skills):
                if i < len(idx2u):
                    orig_id = str(idx2u[i])
                    # PAD (0) 和 EOS (1) 没有实际难度，设为 0
                    if i == 0 or i == 1:
                        diff_map[i] = 0
                    else:
                        # 如果某个题在CSV没找到，默认给难度 1
                        diff_map[i] = diff_dict.get(orig_id, 1)

        # 将 diff_map 注册为 buffer，它会自动随 model.cuda() 转移到 GPU，但不会更新梯度
        self.register_buffer('difficulty_map', diff_map)
        # ==========================================================

        self.rnn = nn.LSTM(emb_dim * 3, hidden_dim, bias=bias, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_skills, bias=bias)

    def forward(self, dynamic_skill_embeds, questions, correct_seq):
        """
                Parameters:
                    dynamic_skill_embeds: 动态生成的题目嵌入 [num_skills, emb_dim]
                    questions: 题目ID序列 [batch_size, seq_len]
                    correct_seq: 答题结果序列 [batch_size, seq_len]（0错误/1正确）
                Returns:
                    pred: 下一题正确概率预测 [batch_size, seq_len-1]
                """
        batch_size, max_seq_len = questions.shape
        mask = (questions[:, 1:] >= 2).float()

        # --- 步骤1：生成每个时间步的输入特征 ---
        # 根据题目ID获取动态嵌入 [batch_size, seq_len, emb_dim]
        skill_embeds = dynamic_skill_embeds[questions]  # 索引操作

        # 生成答题结果嵌入 [batch_size, seq_len, emb_dim]
        correct_embeds = self.correct_embed(correct_seq.long().to(questions.device))

        # 通过查表，直接获得当前 Batch 里所有题目的难度 [batch_size, seq_len]
        diff_seq = self.difficulty_map[questions]
        # 转化为嵌入向量 [batch_size, seq_len, emb_dim]
        diff_embeds = self.diff_embed(diff_seq)

        # 拼接题目嵌入和答题结果及难度嵌入 [batch_size, seq_len, emb_dim*3]
        # lstm_input = torch.cat([skill_embeds, correct_embeds], dim=-1)
        lstm_input = torch.cat([skill_embeds, correct_embeds, diff_embeds], dim=-1)

        # seq_lens = ((questions != 0) & (questions != 1)).sum(dim=1)

        # # --- 步骤2：处理变长序列 ---
        # packed_input = pack_padded_sequence(
        #     lstm_input, seq_lens.cpu(),
        #     batch_first=True, enforce_sorted=False
        # )

        # --- 步骤3：LSTM时序建模 ---
        output, (hn, cn) = self.rnn(lstm_input)
        # output, _ = pad_packed_sequence(packed_output, batch_first=True)  # [batch, seq_len, hidden_dim]

        # --- 步骤4：预测下一题正确概率 ---
        yt = torch.sigmoid(self.fc(output))  # [batch, seq_len, num_skills]
        yt_all = yt
        yt = yt[:, :-1, :]  # 对齐下一题预测 [batch, seq_len-1, num_skills]

        # --- 步骤5：提取目标题概率 ---
        next_skill_ids = questions[:, 1:]  # 下一题的skill_id [batch, seq_len-1]
        pred = torch.gather(yt, dim=2, index=next_skill_ids.unsqueeze(-1).to('cuda')).squeeze(-1)# [batch, seq_len-1]

        return pred, mask, yt, yt_all
