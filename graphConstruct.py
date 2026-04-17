import numpy as np
import torch
import pickle
import Constants
import os
from torch_geometric.data import Data
from dataLoader import Options
import scipy.sparse as sp
import torch.nn.functional as F
import math
'''Friendship network'''


def ConRelationGraph(data):
    options = Options(data)
    _u2idx = {}

    with open(options.u2idx_dict, 'rb') as handle:
        _u2idx = pickle.load(handle)

    edges_list = []
    if os.path.exists(options.net_data):
        with open(options.net_data, 'r') as handle:
            relation_list = handle.read().strip().split("\n")
            relation_list = [edge.split(',') for edge in relation_list]

            relation_list = [(_u2idx[edge[0]], _u2idx[edge[1]]) for edge in relation_list if
                             edge[0] in _u2idx and edge[1] in _u2idx]
            # relation_list_reverse = [edge[::-1] for edge in relation_list]
            # edges_list += relation_list_reverse
            edges_list += relation_list
    else:
        return []
    edges_list_tensor = torch.LongTensor(edges_list).t()
    edges_weight = torch.ones(edges_list_tensor.size(1)).float()
    data = Data(edge_index=edges_list_tensor, edge_attr=edges_weight)

    return data


'''Diffusion hypergraph'''


# 将参数 window_size 替换为 num_stages (设定期望的子超图固定个数，如 3 或 5)
def ConHyperGraphList(cascades, timestamps, user_size, num_stages=3, decay_rate=0.1):
    """
    通过强制划分为固定个数的子超图，构建加权合成转移矩阵
    """
    num_nodes = user_size
    synthesized_adj = sp.csr_matrix((num_nodes, num_nodes))

    for seq in cascades:
        seq_len = len(seq)

        # 处理序列长度小于设定切分份数的极端情况
        if seq_len < num_stages:
            actual_stages = seq_len
            dynamic_window_size = 1
        else:
            actual_stages = num_stages
            # 动态计算该序列对应的窗口大小
            dynamic_window_size = math.ceil(seq_len / num_stages)

        for w_idx in range(actual_stages):
            start_idx = w_idx * dynamic_window_size
            end_idx = min((w_idx + 1) * dynamic_window_size, seq_len)
            window_nodes = seq[start_idx:end_idx]

            valid_nodes = [n for n in window_nodes if n > 1]

            if len(valid_nodes) >= 2:
                # 权重计算依赖于实际切分出的阶段数
                time_weight = np.exp(-decay_rate * (actual_stages - 1 - w_idx))

                row = valid_nodes
                col = [0] * len(valid_nodes)
                data = [1.0] * len(valid_nodes)
                H_t = sp.coo_matrix((data, (row, col)), shape=(num_nodes, 1))

                De_inv_val = 1.0 / len(valid_nodes)
                adj_t = H_t @ H_t.T * De_inv_val

                synthesized_adj = synthesized_adj + adj_t * time_weight

    Dv = np.array(synthesized_adj.sum(axis=1)).flatten()
    Dv_inv_half = np.power(Dv, -0.5, where=Dv != 0)
    Dv_inv_half[Dv == 0] = 0.0
    Dv_mat_inv_half = sp.diags(Dv_inv_half)

    H_norm_weighted = Dv_mat_inv_half @ synthesized_adj @ Dv_mat_inv_half

    return torch.FloatTensor(H_norm_weighted.todense())


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    mx = torch.Tensor(mx).to_sparse()
    return mx


def get_NodeAttention(x, adjt, root_emb):
    x1 = x[adjt.nonzero().t()[1]]
    # print(x1.shape)

    q1 = torch.cat([root_emb[i].repeat(len(adjt[i].nonzero()), 1) for i in torch.arange(root_emb.shape[0])], dim=0)
    # similarity with the roots
    distance = torch.norm(q1.float() - x1.float(), dim=1).cpu()
    n2e_att = torch.sparse_coo_tensor(adjt.nonzero().t(), distance, adjt.shape).to_dense()  # e*n

    zero_vec = 9e15 * torch.ones_like(n2e_att)
    n2e_att = torch.where(n2e_att > 0, n2e_att, zero_vec)
    n2e_att = F.softmax(-n2e_att, dim=1)  # e*n
    return n2e_att.cuda()


def get_EdgeAttention(adj):
    return adj.cuda()

