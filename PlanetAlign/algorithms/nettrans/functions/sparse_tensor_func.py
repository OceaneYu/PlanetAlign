# 文件作用：对稀疏节点分配权重做归一化，并提供可采样的 Gumbel-softmax 处理。
# 数据来源：PlanetAlign/algorithms/nettrans/model.py → TransPool.forward() 传入候选分配边与分数。
# 结果去向：返回边编号与更新权重，供图压缩矩阵构造使用。

import torch


# 功能：按节点累积边权并做度数归一化。
def degree_normalize_sparse_tensor(edge_index, edge_weight, shape):
    """degree_normalize_sparse_tensor.
    """

    row, col = edge_index
    num_nodes = shape[0]
    deg = torch.zeros(num_nodes, dtype=edge_weight.dtype, device=edge_weight.device)
    deg = deg.scatter_reduce(0, row, edge_weight, reduce="sum", include_self=False)
    deg_inv_sqrt = deg.pow(-1)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    edge_weight = deg_inv_sqrt[row] * edge_weight

    return edge_index, edge_weight


# 功能：在稀疏边上取指数并按分组和归一化，返回概率权重。
def sparse_softmax(edge_index, edge_weight, shape):

    row, col = edge_index
    num_nodes = shape[0]
    edge_weight = torch.exp(edge_weight)
    deg = torch.zeros(num_nodes, dtype=edge_weight.dtype, device=edge_weight.device)
    deg = deg.scatter_reduce(0, row, edge_weight, reduce="sum", include_self=False)
    deg_inv_sqrt = deg.pow(-1)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    edge_weight = deg_inv_sqrt[row] * edge_weight

    return edge_index, edge_weight


def sample_gumbel(shape, device, eps=1e-20):
    U = torch.rand(shape)
    U = U.to(device)
    return -torch.log(-torch.log(U + eps) + eps)


# 功能：向边分数加入 Gumbel 随机噪声，再用 sparse_softmax() 形成采样式分配。
def sparse_gumbel_softmax(src, index, temperature, shape):
    y = src + sample_gumbel(len(src), src.device)
    edge_index, edge_weight = sparse_softmax(index, y / temperature, shape)
    return edge_index, edge_weight
