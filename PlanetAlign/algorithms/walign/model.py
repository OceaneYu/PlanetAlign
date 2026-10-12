# 文件作用：定义图编码器、跨图线性变换、分布判别器与特征重构网络。
# 数据来源：PlanetAlign/algorithms/walign/main.py → WAlign.train() 创建模型并传入图特征或训练样本。
# 结果去向：前向计算返回嵌入或损失，交回训练入口评估与反向传播。

import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import GATConv, GCNConv, MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm


# 功能：用两层注意力图卷积生成节点嵌入。
class GATNet(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim=512, heads=1):
        super(GATNet, self).__init__()
        self.conv1 = GATConv(in_dim, hid_dim, heads=heads)
        self.conv2 = GATConv(hid_dim * heads, out_dim)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    # 功能：对特征做 dropout 和图卷积，返回节点表示。
    def forward(self, feature, edge_index):
        x = F.dropout(feature, p=0.5, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


# 功能：用两层 GCN 汇总邻居信息。
class GCNNet(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim=512):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(in_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, out_dim)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    # 输出结果：图卷积后的节点嵌入供跨图变换和损失使用。
    def forward(self, feature, edge_index):
        x = F.dropout(feature, p=0.5, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


# 功能：将多跳无参数传播结果拼接，再线性投影。
class LGCN(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim=512, K=8):
        super(LGCN, self).__init__()
        self.conv1 = CombUnweighted(K=K)
        self.linear = torch.nn.Linear(in_dim * (K + 1), out_dim)

    # 功能：调用 CombUnweighted 传播并将结果映射到目标维度。
    def forward(self, feature, edge_index):
        x = self.conv1(feature, edge_index)
        x = self.linear(x)
        return x


# 功能：归一化图边并收集从自身到 K 跳的传播特征。
class CombUnweighted(MessagePassing):
    def __init__(self, K=1, **kwargs):
        super(CombUnweighted, self).__init__(aggr='add', **kwargs)
        self.K = K

    # 输出结果：拼接各跳特征，交给 LGCN 的线性层。
    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(edge_index, edge_weight, x.size(0), dtype=x.dtype)
        xs = [x]
        for k in range(self.K):
            xs.append(self.propagate(edge_index, x=xs[-1], norm=norm))
        return torch.cat(xs, dim=1)

    def message(self, x_j, norm=None):
        if norm is not None:
            return norm.view(-1, 1) * x_j
        return x_j

    def edge_update(self):
        pass

    def message_and_aggregate(self, edge_index):
        pass

    def __repr__(self):
        return '{}({}, {}, K={})'.format(self.__class__.__name__, self.in_channels, self.out_channels, self.K)


# 功能：按配置将一侧嵌入线性变换到另一侧空间，关闭时直接返回输入。
class TransLayer(torch.nn.Module):
    def __init__(self, out_dim=512, transform=True):
        super(TransLayer, self).__init__()
        self.transform = transform
        if transform:
            self.trans = torch.nn.Parameter(torch.eye(out_dim))

    # 输出结果：原向量或变换后的向量，供训练辅助函数比较两侧分布。
    def forward(self, x):
        if self.transform:
            return x @ self.trans
        return x


# 功能：对节点嵌入输出标量评分，供 Wasserstein 分布对抗训练。
class WDiscriminator(torch.nn.Module):
    def __init__(self, hidden_sizes=(512, 512)):
        super(WDiscriminator, self).__init__()
        self.hidden1 = torch.nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.hidden2 = torch.nn.Linear(hidden_sizes[1], hidden_sizes[1])
        self.output = torch.nn.Linear(hidden_sizes[1], 1)

    # 输出结果：每个嵌入的判别分数，交给 train_wgan_adv_pseudo_self()。
    def forward(self, x):
        x = self.hidden1(x)
        x = F.leaky_relu(x, 0.2, inplace=True)
        x = self.hidden2(x)
        x = F.leaky_relu(x, 0.2, inplace=True)
        return self.output(x)


# 功能：从节点嵌入重构输入属性。
class ReconDNN(torch.nn.Module):
    def __init__(self, out_dim, in_dim, hid_dim=512):
        super(ReconDNN, self).__init__()
        self.hidden = torch.nn.Linear(out_dim, hid_dim)
        self.output = torch.nn.Linear(hid_dim, in_dim)

    # 输出结果：重构属性交给 FeatureReconstructLoss。
    def forward(self, x):
        x = self.hidden(x)
        x = F.relu(x)
        return self.output(x)


# 功能：计算重构属性与原属性的平均向量误差。
class FeatureReconstructLoss(torch.nn.Module):
    def __init__(self):
        super(FeatureReconstructLoss, self).__init__()

    # 输出结果：重构误差标量，供 PlanetAlign/algorithms/walign/train.py → train_feature_recon() 使用。
    @staticmethod
    def forward(embd, x, recon_model):
        recon_x = recon_model(embd)
        return torch.norm(recon_x - x, dim=1, p=2).mean()
