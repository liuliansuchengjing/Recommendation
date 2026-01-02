import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class DKT(nn.Module):

    def __init__(self, emb_dim, hidden_dim, num_skills, dropout=0.2, bias=True):
        super(DKT, self).__init__()
        self.emb_dim = emb_dim  # 嵌入维度
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.correct_embed = nn.Embedding(2, emb_dim)  # 答案结果嵌入（正确、错误）
        self.rnn = nn.LSTM(emb_dim * 2, hidden_dim, bias=bias, dropout=dropout, batch_first=True)
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
        correct_embeds = self.correct_embed(correct_seq.long().to('cuda'))

        # 拼接题目嵌入和答题结果嵌入 [batch_size, seq_len, emb_dim*2]
        lstm_input = torch.cat([skill_embeds, correct_embeds], dim=-1)

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
        yt = yt[:, :-1, :]  # 对齐下一题预测 [batch, seq_len-1, num_skills]

        # --- 步骤5：提取目标题概率 ---
        next_skill_ids = questions[:, 1:]  # 下一题的skill_id [batch, seq_len-1]
        pred = torch.gather(yt, dim=2, index=next_skill_ids.unsqueeze(-1).to('cuda')).squeeze(-1)

        return pred, mask  # [batch, seq_len-1]
