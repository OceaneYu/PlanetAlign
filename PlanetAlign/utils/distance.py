# 文件作用：把两组节点向量归一化，计算两两相似度或负指数匹配代价。
# 结果去向：PlanetAlign/m2m/base.py → _extract_s() 使用余弦矩阵；PlanetAlign/algorithms/parrot.py → PARROT.get_transport_cost() 使用负指数结果。

import torch
import torch.nn.functional as F


# 功能：归一化两组向量，将两两内积取负指数；默认 p=2 时越相似代价越小。
def get_normalized_neg_exp_dist(emb1, emb2, p=2, device='cpu'):
    emb1 = emb1.to(device)
    emb2 = emb2.to(device)
    normalized_emb1 = F.normalize(emb1, p=p, dim=1)
    normalized_emb2 = F.normalize(emb2, p=p, dim=1)
    return torch.exp(-(normalized_emb1 @ normalized_emb2.T))


# 输出结果：返回两组向量的归一化内积矩阵，默认 p=2 时为余弦相似度，供嵌入算法转成 S。
def pairwise_cosine_similarity(x, y, p=2):
    normalized_x = F.normalize(x, p=p, dim=1)
    normalized_y = F.normalize(y, p=p, dim=1)
    return normalized_x @ normalized_y.T
