import numpy as np
import torch
import pickle
import Constants
import os 
from torch_geometric.data import Data
from dataLoader import Options
import scipy.sparse as sp
import torch.nn.functional as F

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

                relation_list = [(_u2idx[edge[0]], _u2idx[edge[1]]) for edge in relation_list if edge[0] in _u2idx and edge[1] in _u2idx]
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
# def ConHyperGraphList(cascades, timestamps, user_size, step_split=Constants.step_split):
#     '''split the graph to sub graphs, return the list'''
#
#     times, root_list = ConHyperDiffsuionGraph(cascades, timestamps, user_size)
#     zero_vec = torch.zeros_like(times)
#     one_vec = torch.ones_like(times)
#
#     time_sorted = []
#     graph_list = {}
#
#     for time in timestamps:
#         time_sorted += time[:-1]
#     time_sorted = sorted(time_sorted)
#     split_length = len(time_sorted) // step_split
#
#     for x in range(split_length, split_length * step_split , split_length):
#         if x == split_length:
#             sub_graph = torch.where(times > 0, one_vec, zero_vec) - torch.where(times > time_sorted[x], one_vec, zero_vec)
#         else:
#             sub_graph = torch.where(times > time_sorted[x-split_length], one_vec, zero_vec) - torch.where(times > time_sorted[x], one_vec, zero_vec)
#
#         graph_list[time_sorted[x]] = sub_graph
#
#     graphs = [graph_list, root_list]
#
#     return graphs

'''Diffusion hypergraph (重构版：全局协同超图)'''


def ConHyperGraphList(cascades, timestamps, user_size, step_split=None):
    """
    抛弃绝对时间切片，将每一条学生训练序列视为一条超边。
    构建 全局协同超图发生矩阵 (Global Collaborative Incidence Matrix)。
    """
    n_size = user_size
    e_size = len(cascades)

    rows = []
    cols = []

    for i, cas in enumerate(cascades):
        # 过滤掉 PAD(0) 和 Skip/EOS(1) 等无效占位符
        unique_items = list(set([x for x in cas if x > 1]))
        if len(unique_items) == 0:
            continue

        rows.extend(unique_items)
        cols.extend([i] * len(unique_items))

    # 1. 构建超图关联矩阵 H [节点数, 超边数]
    vals = [1.0] * len(rows)
    H = torch.sparse_coo_tensor(torch.LongTensor([rows, cols]), torch.FloatTensor(vals), [n_size, e_size]).to_dense()

    # 2. 计算节点度矩阵 Dv 和 超边度矩阵 De
    D_v = H.sum(dim=1, keepdim=True)  # [n_size, 1]
    D_e = H.sum(dim=0, keepdim=True)  # [1, e_size]

    # 3. 计算度矩阵的 -1/2 次方
    D_v_invsqrt = torch.pow(D_v, -0.5)
    D_v_invsqrt[torch.isinf(D_v_invsqrt)] = 0.

    D_e_invsqrt = torch.pow(D_e, -0.5)
    D_e_invsqrt[torch.isinf(D_e_invsqrt)] = 0.

    # 4. 拉普拉斯度归一化: G_norm = Dv^{-1/2} * H * De^{-1/2}
    # 这一步极其关键，能有效防止热门题目主导图网络梯度
    G_norm = D_v_invsqrt * H * D_e_invsqrt

    return G_norm
    
    
def ConHyperDiffsuionGraph(cascades, timestamps, user_size):
    '''return the adj. and time adj. of hypergraph'''
    e_size = len(cascades)+1
    n_size = user_size
    rows = []
    cols = []
    vals_time = []
    root_list = [0]

        
    for i in range(e_size-1):
        root_list.append(cascades[i][0])
        rows += cascades[i][:-1]
        cols +=[i+1]*(len(cascades[i])-1)
        #vals +=[1.0]*(len(cascades[i])-1)
        vals_time += timestamps[i][:-1]
        
    root_list = torch.tensor(root_list)
    Times = torch.sparse_coo_tensor(torch.Tensor([rows,cols]), torch.Tensor(vals_time), [n_size,e_size])
        
        
    return Times.to_dense(), root_list
 

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
    #print(x1.shape)
    
    q1 = torch.cat([root_emb[i].repeat(len(adjt[i].nonzero()),1) for i in torch.arange(root_emb.shape[0])], dim = 0)
    #similarity with the roots
    distance = torch.norm(q1.float()-x1.float(),dim = 1).cpu()
    n2e_att = torch.sparse_coo_tensor(adjt.nonzero().t(), distance, adjt.shape).to_dense() #e*n
        
    zero_vec = 9e15*torch.ones_like(n2e_att)
    n2e_att = torch.where(n2e_att > 0, n2e_att, zero_vec)
    n2e_att = F.softmax(-n2e_att, dim = 1) #e*n
    return n2e_att.cuda()

def get_EdgeAttention(adj):                
    return adj.cuda()
    
