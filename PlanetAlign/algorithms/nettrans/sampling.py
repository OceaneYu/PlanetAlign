# 文件作用：为 NetTrans 的锚点对齐和上下文训练生成负例节点。
# 数据来源：PlanetAlign/algorithms/nettrans/main.py → NetTrans.train() 传入训练锚点与采样概率。
# 结果去向：负例编号用于 PlanetAlign/algorithms/nettrans/model.py → NetTransModel 的对齐与邻接损失。

import torch


# 功能：在两侧分别排除已知正对应，按均匀分布采样负例。
def uniform_negative_sampling(anchor1, anchor2, n1, n2, n_negs):
    assert len(anchor1) == len(anchor2), "The number of anchors should be the same."

    anchor1, anchor2 = anchor1.cpu(), anchor2.cpu()
    num_anchors = len(anchor1)
    p1 = torch.ones((num_anchors, n2))
    p2 = torch.ones((num_anchors, n1))
    negs1, negs2 = [], []
    for i in range(num_anchors):
        p = p1[i]
        p[anchor2[i]] = 0
        p = p / torch.sum(p)
        samples = torch.multinomial(p, n_negs, replacement=True)
        negs1.append(samples)

        p = p2[i]
        p[anchor1[i]] = 0
        p = p / torch.sum(p)
        samples = torch.multinomial(p, n_negs, replacement=True)
        negs2.append(samples)

    negs1 = torch.vstack(negs1)
    negs2 = torch.vstack(negs2)

    return negs1, negs2


# 功能：按概率从非正上下文节点中采样，返回每个锚点的负例编号。
def negative_edge_sampling(prob, anchors, n_negs):
    anchors = anchors.cpu()
    prob = prob.cpu()
    prob = prob[anchors]
    negs = []
    for p in prob:
        p1 = p / torch.sum(p)
        neg_idx = torch.multinomial(p1, n_negs, replacement=True)
        negs.append(neg_idx)
    negs = torch.vstack(negs)
    return negs
