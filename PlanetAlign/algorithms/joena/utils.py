# 文件作用：提供 JOENA 使用的稳定 Sinkhorn 迭代，将匹配代价转换为传输权重。
# 数据来源：PlanetAlign/algorithms/joena/model.py → FusedGWLoss.forward() 传入节点代价、两侧结构与参数。
# 结果去向：sinkhorn_stable() 返回 S，交回损失函数计算训练目标。

import torch


# 核心思路：交替更新行列缩放变量，用减去最小值的 soft-min 减少指数运算的数值问题。
# 输出结果：两侧均匀权重约束下的节点传输矩阵。
def sinkhorn_stable(inter_c, intra_c1, intra_c2, threshold_lambda=0, in_iter=5, out_iter=10, gw_weight=20, gamma_p=1e-2,
                    dtype=torch.float32, device='cpu'):
    n1, n2 = inter_c.shape
    # marginal distribution
    a = torch.ones(n1).to(dtype).to(device) / n1
    b = torch.ones(n2).to(dtype).to(device) / n2
    # lagrange multiplier
    f = torch.ones(n1).to(dtype).to(device) / n1
    g = torch.ones(n2).to(dtype).to(device) / n2
    # transport plan
    s = torch.ones((n1, n2)).to(dtype).to(device) / (n1 * n2)

    def soft_min_row(z_in, eps):
        hard_min = torch.min(z_in, dim=1, keepdim=True)[0]
        soft_min = hard_min - eps * torch.log(torch.sum(torch.exp(-(z_in - hard_min) / eps), dim=1, keepdim=True))
        return soft_min.squeeze(-1)

    def soft_min_col(z_in, eps):
        hard_min = torch.min(z_in, dim=0, keepdim=True)[0]
        soft_min = hard_min - eps * torch.log(torch.sum(torch.exp(-(z_in - hard_min) / eps), dim=0, keepdim=True))
        return soft_min.squeeze(0)

    for i in range(out_iter):
        a_hat = torch.sum(s - threshold_lambda, dim=1)
        b_hat = torch.sum(s - threshold_lambda, dim=0)
        temp = (intra_c1 ** 2 @ a_hat.view(-1, 1) @ torch.ones((1, n2)).to(dtype).to(device) +
                torch.ones((n1, 1)).to(dtype).to(device) @ b_hat.view(1, -1) @ intra_c2 ** 2)
        L = temp - 2 * intra_c1 @ (s - threshold_lambda) @ intra_c2.T
        cost = inter_c + gw_weight * L

        Q = cost
        for j in range(in_iter):
            # log-sum-exp stabilization
            f = soft_min_row(Q - g.view(1, -1), gamma_p) + gamma_p * torch.log(a)
            g = soft_min_col(Q - f.view(-1, 1), gamma_p) + gamma_p * torch.log(b)
        s = 0.05 * s + 0.95 * torch.exp((f.view(-1, 1) + g.view(-1, 1).T - Q) / gamma_p)

    return s
