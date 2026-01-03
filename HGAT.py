import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from layer import HGATLayer, TransformerEncoder
from torch_geometric.nn import GCNConv
import torch.nn.init as init
import Constants
from TransformerBlock import TransformerBlock
from torch.autograd import Variable
from DKT import DKT

class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):  #
        super(HGNN_conv, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(in_ft, out_ft))
        self.weight1 = nn.Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.weight1.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, G):  # x: torch.Tensor, G: torch.Tensor

        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        edge = G.t().matmul(x)
        edge = edge.matmul(self.weight1)
        x = G.matmul(edge)

        return x, edge


class HGNNLayer(nn.Module):
    def __init__(self, emb_dim, dropout=0.15):
        super(HGNNLayer, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(emb_dim, emb_dim)
        self.hgc2 = HGNN_conv(emb_dim, emb_dim)
        self.hgc3 = HGNN_conv(emb_dim, emb_dim)

    def forward(self, x, G):
        x, edge = self.hgc1(x, G)
        x, edge = self.hgc2(x, G)
        x = F.softmax(x, dim=1)
        x, edge = self.hgc3(x, G)
        x = F.dropout(x, self.dropout)
        x = F.tanh(x)
        return x, edge


def get_previous_user_mask(seq, user_size):
    ''' Mask previous activated users.'''
    assert seq.dim() == 2
    prev_shape = (seq.size(0), seq.size(1), seq.size(1))
    seqs = seq.repeat(1, 1, seq.size(1)).view(seq.size(0), seq.size(1), seq.size(1))
    previous_mask = np.tril(np.ones(prev_shape)).astype('float32')
    previous_mask = torch.from_numpy(previous_mask)
    if seq.is_cuda:
        previous_mask = previous_mask.cuda()
    masked_seq = previous_mask * seqs.data.float()

    # force the 0th dimension (PAD) to be masked
    PAD_tmp = torch.zeros(seq.size(0), seq.size(1), 1)
    if seq.is_cuda:
        PAD_tmp = PAD_tmp.cuda()
    masked_seq = torch.cat([masked_seq, PAD_tmp], dim=2)
    ans_tmp = torch.zeros(seq.size(0), seq.size(1), user_size)
    if seq.is_cuda:
        ans_tmp = ans_tmp.cuda()
    masked_seq = ans_tmp.scatter_(2, masked_seq.long(), float(-1000))
    masked_seq = Variable(masked_seq, requires_grad=False)
    # print("masked_seq ",masked_seq.size())
    return masked_seq.cuda()


# Fusion gate
class Fusion(nn.Module):
    def __init__(self, input_size, out=1, dropout=0.2):
        super(Fusion, self).__init__()
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, out)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

    def forward(self, hidden, dy_emb):
        emb = torch.cat([hidden.unsqueeze(dim=0), dy_emb.unsqueeze(dim=0)], dim=0)
        emb_score = F.softmax(self.linear2(torch.tanh(self.linear1(emb))), dim=0)
        emb_score = self.dropout(emb_score)
        out = torch.sum(emb_score * emb, dim=0)
        return out


'''Learn friendship network'''


class GraphNN(nn.Module):
    def __init__(self, ntoken, ninp, dropout=0.5, is_norm=True):
        super(GraphNN, self).__init__()
        self.embedding = nn.Embedding(ntoken, ninp, padding_idx=0)
        # in:inp,out:nip*2
        self.gnn1 = GCNConv(ninp, ninp * 2)
        self.gnn2 = GCNConv(ninp * 2, ninp)
        self.is_norm = is_norm

        self.dropout = nn.Dropout(dropout)
        if self.is_norm:
            self.batch_norm = torch.nn.BatchNorm1d(ninp)
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.embedding.weight)

    def forward(self, graph):
        graph_edge_index = graph.edge_index.cuda()
        graph_x_embeddings = self.gnn1(self.embedding.weight, graph_edge_index)
        graph_x_embeddings = self.dropout(graph_x_embeddings)
        graph_output = self.gnn2(graph_x_embeddings, graph_edge_index)
        if self.is_norm:
            graph_output = self.batch_norm(graph_output)
        # print(graph_output.shape)
        return graph_output.cuda()


'''Learn diffusion network'''


class HGNN_ATT(nn.Module):
    def __init__(self, input_size, n_hid, output_size, dropout=0.3, is_norm=True):
        super(HGNN_ATT, self).__init__()
        self.dropout = dropout
        self.is_norm = is_norm
        if self.is_norm:
            self.batch_norm1 = torch.nn.BatchNorm1d(output_size)
        self.gat1 = HGATLayer(input_size, output_size, dropout=self.dropout, transfer=False, concat=True, edge=True)
        self.hgnn = HGNNLayer(input_size, 0.1)
        self.fus1 = Fusion(output_size)

    def forward(self, x, hypergraph_list):
        root_emb = F.embedding(hypergraph_list[1].cuda(), x)

        hypergraph_list = hypergraph_list[0]

        embedding_list = {}
        for sub_key in hypergraph_list.keys():
            sub_graph = hypergraph_list[sub_key]
            # sub_node_embed, sub_edge_embed = self.gat1(x, sub_graph.cuda(), root_emb)
            sub_node_embed, sub_edge_embed = self.hgnn(x, sub_graph.cuda())
            sub_node_embed = F.dropout(sub_node_embed, self.dropout, training=self.training)

            if self.is_norm:
                sub_node_embed = self.batch_norm1(sub_node_embed)
                sub_edge_embed = self.batch_norm1(sub_edge_embed)

            xl = x
            x = self.fus1(x, sub_node_embed)
            embedding_list[sub_key] = [x.cpu(), sub_edge_embed.cpu(), xl.cpu()]

        return embedding_list


class MLPReadout(nn.Module):
    def __init__(self, in_dim, out_dim, act):
        """
        out_dim: the final prediction dim, usually 1
        act: the final activation, if rating then None, if CTR then sigmoid
        """
        super(MLPReadout, self).__init__()
        self.layer1 = nn.Linear(in_dim, out_dim)
        self.act = nn.ReLU()
        self.out_act = act

    def forward(self, x):
        ret = self.layer1(x)
        return ret


class MSHGAT(nn.Module):
    def __init__(self, num_skills, opt, dropout=0.3):
        super(MSHGAT, self).__init__()
        self.hidden_size = opt.d_word_vec
        self.n_node = opt.user_size
        self.dropout = nn.Dropout(dropout)
        self.initial_feature = opt.initialFeatureSize

        self.hgnn = HGNN_ATT(self.initial_feature, self.hidden_size * 2, self.hidden_size, dropout=dropout)
        self.gnn = GraphNN(self.n_node, self.initial_feature, dropout=dropout)
        self.fus = Fusion(self.hidden_size)
        self.fus1 = Fusion(self.hidden_size)
        self.fus2 = Fusion(self.hidden_size)

        self.embedding = nn.Embedding(self.n_node, self.initial_feature, padding_idx=0)
        self.reset_parameters()
        self.readout = MLPReadout(self.hidden_size, self.n_node, None)
        self.gru1 = nn.GRU(self.hidden_size, self.hidden_size, num_layers=1, batch_first=True)
        self.gru2 = nn.GRU(self.hidden_size, self.hidden_size, num_layers=1, batch_first=True)

        self.n_layers = 1
        self.n_heads = 2
        self.inner_size = 64
        self.hidden_dropout_prob = 0.3
        self.attn_dropout_prob = 0.3
        self.layer_norm_eps = 1e-12
        self.hidden_act = 'gelu'
        self.item_embedding = nn.Embedding(self.n_node + 1, self.hidden_size, padding_idx=0)  # mask token add 1
        self.position_embedding = nn.Embedding(500, self.hidden_size)  # add mask_token at the last
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            multiscale=False
        )

        self.num_skills = num_skills
        self.ktmodel = DKT(self.hidden_size, self.hidden_size, self.num_skills)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def get_attention_mask(self, item_seq):
        """Generate bidirectional attention mask for multi-scale attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # bidirectional mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def pred(self, pred_logits):
        predictions = self.readout(pred_logits)
        return predictions

    def forward(self, input, input_timestamp, input_idx, ans, graph, hypergraph_list):
        # 只使用图神经网络部分，跳过超图处理
        original_input = input
        input = input[:, :-1]  # 保持原始处理方式
        input_timestamp = input_timestamp[:, :-1]  # 保持原始处理方式
        
        # 从original_input中提取对应的ans部分
        # original_ans = ans
        # ans = ans[:, :-1] if ans.size(1) > input.size(1) else ans

        # 仅使用图神经网络获取节点嵌入
        hidden = self.dropout(self.gnn(graph))

        # 使用DKT模型获取知识追踪结果
        pred_res, kt_mask = self.ktmodel(hidden, original_input, ans)

        # 直接使用图神经网络的输出作为序列嵌入
        batch_size, max_len = input.size()

        # 使用图嵌入作为序列处理的输入
        sequence_embeddings = F.embedding(input.cuda(), hidden.cuda())

        # 添加位置编码
        input_embeddings = sequence_embeddings
        position_ids = torch.arange(input.size(1), dtype=torch.long, device=input.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input)
        position_embedding = self.position_embedding(position_ids.cuda())
        input_embeddings = input_embeddings + position_embedding
        input_embeddings = self.LayerNorm(input_embeddings)
        input_embeddings = self.dropout(input_embeddings)

        # 应用注意力掩码
        extended_attention_mask = self.get_attention_mask(input)

        # Transformer处理
        trm_output = self.trm_encoder(input_embeddings, extended_attention_mask, output_all_encoded_layers=False)

        # 预测
        pred = self.pred(trm_output)
        mask = get_previous_user_mask(input.cpu(), self.n_node)

        return (pred + mask).view(-1, pred.size(-1)).cuda(), pred_res, kt_mask
