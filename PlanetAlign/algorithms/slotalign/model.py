# 文件作用：实现无可学习参数的图卷积，给 SLOTAlign 生成不同层的结构属性表示。
# 数据来源：PlanetAlign/algorithms/slotalign/main.py → SLOTAlign.get_intra_graph_similarity() 传入特征与图边。
# 结果去向：卷积输出用于构造图内相似度通道。

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, degree


# 功能：加自环并按节点度数归一化，用 MessagePassing 汇总邻居特征。
class ParamFreeGraphConv(MessagePassing):
    def __init__(self):
        super(ParamFreeGraphConv, self).__init__(aggr='add')  # using sum aggregation

    # 功能：准备自环和度数权重，调用 propagate() 触发 message()。
    def forward(self, x, edge_index):
        # Add self-loops to mimic dgl.add_self_loop()
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.shape[0])

        # Compute symmetric normalization (norm='both')
        row, col = edge_index
        deg = degree(row, x.shape[0], dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Propagate messages without applying any weight
        return self.propagate(edge_index, x=x, norm=norm)

    # 输出结果：按边归一化权重缩放邻居向量，交给框架聚合。
    def message(self, x_j, norm=None):
        if norm is not None:
            return norm.view(-1, 1) * x_j
        return x_j

    def edge_update(self):
        pass

    def message_and_aggregate(self, edge_index):
        pass