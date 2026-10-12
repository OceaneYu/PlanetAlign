# 文件作用：批量计算带重启随机游走（RWR），描述节点相对指定锚点的结构位置。
# 数据来源：PlanetAlign/algorithms/parrot.py → PARROT.get_transport_cost() 传入图和一侧训练锚点。
# 结果去向：返回节点×锚点矩阵，由调用者用于匹配代价或训练输入。

import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj


# 核心思路：把图邻接矩阵按行归一化，从各锚点同时传播，反复加入重启项直到收敛或用尽迭代次数。
# 输出结果：每行表示一个节点相对所有 landmarks 的分数；connect_isolated 控制孤立节点的处理。
def get_batch_rwr_scores(graph, landmarks, restart_prob=0.15, max_iters=1000, tol=1e-6, connect_isolated=False, dtype=torch.float32, device='cpu'):
    r"""Compute Random Walk with Restart (RWR) scores for a batch of landmarks in a single graph."""
    batch_landmark_vecs = torch.zeros(graph.num_nodes, len(landmarks)).to(dtype).to(device)
    batch_landmark_vecs[landmarks, torch.arange(len(landmarks))] = 1
    batch_rwr_vecs = torch.ones(graph.num_nodes, len(landmarks)).to(dtype).to(device)

    adj = to_dense_adj(graph.edge_index, max_num_nodes=graph.num_nodes).squeeze().to(dtype)
    if connect_isolated:
        adj[torch.where(~adj.sum(1).bool())] = torch.ones(graph.num_nodes, dtype=dtype)
    trans_mat = F.normalize(adj.to(device), p=1, dim=1).T
    for i in range(max_iters):
        batch_rwr_vecs_old = torch.clone(batch_rwr_vecs)
        batch_rwr_vecs = (1 - restart_prob) * trans_mat @ batch_rwr_vecs + restart_prob * batch_landmark_vecs
        diff = torch.max(torch.abs(batch_rwr_vecs - batch_rwr_vecs_old))
        if diff.item() < tol:
            break

    return batch_rwr_vecs
