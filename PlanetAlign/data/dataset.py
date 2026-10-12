# 文件作用：用 Dataset 读取 .pt 中的图和已知节点对应关系，或用 BaseData 接收已有图对象。
# 核心处理：检查数据是否匹配，并把节点对应表分成训练用的 train_data 和测试用的 test_data。
# 结果去向：scripts/run_quotient_compare.py → main() 将加载好的 Dataset 传给 PlanetAlign/m2m/base.py → train_base_S()。
# 阅读建议：先看 Dataset.__init__()/load_dataset()；多对多训练节点对的恢复在 PlanetAlign/m2m/core.py → use_full_anchor_supervision()。

import copy
import os
from pathlib import Path
from typing import Union
import torch
from torch_geometric.data import Data


# 功能：包装调用者已有的图和锚点，提供与 Dataset 相同的校验、划分和复制操作。
class BaseData:
    r"""A dataset object storing multiple graphs and ground-truth anchor links for alignment.

    Parameters
    ----------
    graphs: list[torch_geometric.data.Data]
        List of PyTorch Geometric Data objects representing the graphs.
    anchor_links: torch.Tensor
        Tensor containing the anchor links between the graphs.
    name: str
        Name of the dataset.
    train_ratio: float, optional
        Ratio of anchor links to be used for training. Default is 0.2.
    dtype: torch.dtype, optional
        Data type of the tensors, either torch.float32 or torch.float64. Default is torch.float32.
    seed: int, optional
        Random seed for shuffling the anchor links. Default is 0.
    """
    # 输入来源：调用者直接传入 PyG 图列表和锚点张量，无需读取磁盘。
    # 输出结果：初始化 pyg_graphs、train_data、test_data，供算法的 train() 使用。
    def __init__(self,
                 graphs: list[Data],
                 anchor_links: torch.Tensor,
                 name: str,
                 train_ratio: float = 0.2,
                 dtype: torch.dtype = torch.float32,
                 seed: int = 0):
        self.pyg_graphs = graphs
        self._anchor_links = anchor_links
        self.name = name
        self.train_ratio = train_ratio
        self.seed = seed
        self.dtype = dtype

        self._validate()
        self.train_data, self.test_data = self.train_test_split()

    # 功能：按 seed 打乱锚点行并按 train_ratio 划分，返回训练和测试张量。
    # 核心思路：保存并恢复 PyTorch 随机状态，避免这次划分影响后续随机操作。
    def train_test_split(self):
        num_anchor = self._anchor_links.shape[0]
        rng_state = torch.get_rng_state()
        torch.manual_seed(self.seed)
        perm = torch.randperm(num_anchor)
        torch.set_rng_state(rng_state)
        train_size = int(num_anchor * self.train_ratio)
        return self._anchor_links[perm[:train_size]], self._anchor_links[perm[train_size:]]

    def _check_integrity(self, root): # 检查数据集是否完整
        raise NotImplementedError

    # 功能：检查图对象、锚点维度、节点编号及属性数量是否匹配。
    def _validate(self): # 验证数据集是否符合要求
        assert hasattr(self, 'pyg_graphs') and hasattr(self, '_anchor_links'), 'Dataset has not been loaded yet, wrong place for validation'
        assert type(self.pyg_graphs) in [list, tuple], 'Graphs must be stored in a list or tuple'
        assert len(self.pyg_graphs) > 1, 'At least two graphs are required for alignment'
        assert all([isinstance(g, Data) for g in self.pyg_graphs]), 'Each graph must be a PyG Data object'
        assert isinstance(self._anchor_links, torch.Tensor), 'Anchor links must be stored in a PyTorch tensor'
        assert self._anchor_links.dim() == 2, 'Anchor links must be a 2D tensor'
        assert len(self.pyg_graphs) == self._anchor_links.shape[1], 'Number of PyG graphs and anchor links dimension do not match'
        for i, g in enumerate(self.pyg_graphs):
            assert g.edge_index.max() < g.num_nodes, f'Node index must be less than the number of nodes in graph {i}'
            assert self._anchor_links[:, i].max() < g.num_nodes, f'Anchor link must be less than the number of nodes in graph {i}'
            if g.x is not None:
                assert g.x.shape[0] == g.num_nodes, f'Number of nodes and node attributes must match in graph {i}'
            if g.edge_attr is not None:
                assert g.edge_attr.shape[0] == g.edge_index.shape[1], f'Number of edges and edge attributes must match in graph {i}'

    def __str__(self):
        network_count = len(self.pyg_graphs)  # Number of networks (Assuming stored in a list)
        # Header
        output = (
                f"Dataset: {self.name}\n"
                f"{'=' * 60}\n"
                f"{'Graphs':<20}" + "".join([f"{self.pyg_graphs[i].name:>15}" for i in range(network_count)]) +
                f"\n{'-' * 60}\n"
        )
        # Number of Nodes per graph
        output += (
                f"{'# Nodes':<20}" +
                "".join([f"{self.pyg_graphs[i].num_nodes:>15,}" for i in range(network_count)]) + "\n"
        )
        # Number of Edges per graph
        output += (
                f"{'# Edges':<20}" +
                "".join([f"{self.pyg_graphs[i].num_edges:>15,}" for i in range(network_count)]) + "\n"
        )
        # Node Attributes Dimension per graph
        output += (
                f"{'# Node Attributes':<20}" +
                "".join([
                    f"{self.pyg_graphs[i].num_node_features:>15,}"
                    for i in range(network_count)
                ]) + "\n"
        )
        # Edge Attributes Dimension per graph
        output += (
                f"{'# Edge Attributes':<20}" +
                "".join([
                    f"{self.pyg_graphs[i].num_edge_features:>15,}"
                    for i in range(network_count)
                ]) + "\n"
        )
        # Anchor Links (Train/Test)
        output += (
            f"{'=' * 60}\n"
            f"{'Anchor Links':<20}{'Train':>15}{'Test':>15}\n"
            f"{'-' * 60}\n"
            f"{f'Count (ratio: {self.train_ratio})':<20}{self.train_data.shape[0]:>15,}{self.test_data.shape[0]:>15,}\n"
            f"{'=' * 60}"
        )
        return output

    # 输出结果：返回独立的深拷贝，便于噪声实验修改副本。
    def clone(self):
        return copy.deepcopy(self)


# 功能：统一加载 .pt 图数据；PlanetAlign/datasets/ 的具体数据集类可重写 load_dataset()。
class Dataset:
    r"""A dataset object storing multiple graphs and ground-truth anchor links for alignment.

    Parameters
    ----------
    root: str or pathlib.Path
        Root directory where the dataset is stored.

    name: str
        Name of the dataset.

    train_ratio: float, optional
        Ratio of anchor links to be used for training. Default is 0.2.

    dtype: torch.dtype, optional
        Data type of the tensors, either torch.float32 or torch.float64. Default is torch.float32.

    seed: int, optional
        Random seed for shuffling the anchor links. Default is 0.

    """
    # 流程位置：实验入口创建数据集时执行；依次加载图、校验数据、划分锚点。
    # 输出结果：pyg_graphs 提供图结构与属性，train_data/test_data 提供已知对应关系。
    def __init__(self,
                 root: Union[str, Path],
                 name: str,
                 train_ratio: float = 0.2,
                 dtype: torch.dtype = torch.float32,
                 seed: int = 0):
        assert 0 <= train_ratio < 1, 'Training ratio must be in [0, 1)'
        assert dtype in [torch.float32, torch.float64], 'Data type must be either torch.float32 or torch.float64'

        self.root = Path(root)
        self.name = name
        self.train_ratio = train_ratio
        self.seed = seed
        self.dtype = dtype

        print(f'Loading dataset "{self.name} ({self.__class__.__name__})"...', end=' ')
        self.pyg_graphs, self._anchor_links = self.load_dataset()
        print('Done')
        self._validate()
        self.train_data, self.test_data = self.train_test_split()

    # 输入来源：读取 root/<name>.pt 中的 edges、number_of_nodes、可选属性和 anchor_links。
    # 输出结果：返回 PyG Data 列表和锚点张量，由 __init__() 继续校验与划分。
    def load_dataset(self) -> tuple[list[Data], torch.Tensor]:
        data_dict = torch.load(Path.joinpath(self.root, f'{self.name}.pt'), weights_only=True)
        pyg_graphs = list()
        for gid, gname in enumerate(data_dict['graphs']):
            num_nodes = data_dict['number_of_nodes'][gid]
            x = data_dict['node_attributes'][gid].to(self.dtype) if 'node_attributes' in data_dict else None
            edge_attr = data_dict['edge_attributes'][gid].to(self.dtype) if 'edge_attributes' in data_dict else None
            edge_index = data_dict['edges'][gid]
            pyg_graph = Data(name=gname, num_nodes=num_nodes, x=x, edge_index=edge_index, edge_attr=edge_attr)
            pyg_graphs.append(pyg_graph)
        anchor_links = data_dict['anchor_links']
        return pyg_graphs, anchor_links

    # 功能：按 seed 和 train_ratio 将锚点拆成训练与测试部分。
    # 流程位置：生成的多对多基准会再由 PlanetAlign/m2m/core.py → use_full_anchor_supervision() 恢复完整训练锚点。
    def train_test_split(self):
        num_anchor = self._anchor_links.shape[0]
        rng_state = torch.get_rng_state()
        torch.manual_seed(self.seed)
        perm = torch.randperm(num_anchor)
        torch.set_rng_state(rng_state)
        train_size = int(num_anchor * self.train_ratio)
        return self._anchor_links[perm[:train_size]], self._anchor_links[perm[train_size:]]

    def _check_integrity(self, root):
        raise NotImplementedError

    # 功能：检查加载后的图、锚点和属性形状；错误数据在进入算法前触发断言。
    def _validate(self):
        assert hasattr(self, 'pyg_graphs') and hasattr(self, '_anchor_links'), 'Dataset has not been loaded yet, wrong place for validation'
        assert type(self.pyg_graphs) in [list, tuple], 'Graphs must be stored in a list or tuple'
        assert len(self.pyg_graphs) > 1, 'At least two graphs are required for alignment'
        assert all([isinstance(g, Data) for g in self.pyg_graphs]), 'Each graph must be a PyG Data object'
        assert isinstance(self._anchor_links, torch.Tensor), 'Anchor links must be stored in a PyTorch tensor'
        assert self._anchor_links.dim() == 2, 'Anchor links must be a 2D tensor'
        assert len(self.pyg_graphs) == self._anchor_links.shape[1], 'Number of PyG graphs and anchor links dimension do not match'
        for i, g in enumerate(self.pyg_graphs):
            assert g.edge_index.max() < g.num_nodes, f'Node index must be less than the number of nodes in graph {i}'
            assert self._anchor_links[:, i].max() < g.num_nodes, f'Anchor link must be less than the number of nodes in graph {i}'
            if g.x is not None:
                assert g.x.shape[0] == g.num_nodes, f'Number of nodes and node attributes must match in graph {i}'
            if g.edge_attr is not None:
                assert g.edge_attr.shape[0] == g.edge_index.shape[1], f'Number of edges and edge attributes must match in graph {i}'

    def __str__(self):
        network_count = len(self.pyg_graphs)  # Number of networks (Assuming stored in a list)
        # Header
        output = (
                f"Dataset: {self.name}\n"
                f"{'=' * 60}\n"
                f"{'Graphs':<20}" + "".join([f"{self.pyg_graphs[i].name:>15}" for i in range(network_count)]) +
                f"\n{'-' * 60}\n"
        )
        # Number of Nodes per graph
        output += (
                f"{'# Nodes':<20}" +
                "".join([f"{self.pyg_graphs[i].num_nodes:>15,}" for i in range(network_count)]) + "\n"
        )
        # Number of Edges per graph
        output += (
                f"{'# Edges':<20}" +
                "".join([f"{self.pyg_graphs[i].num_edges:>15,}" for i in range(network_count)]) + "\n"
        )
        # Node Attributes Dimension per graph
        output += (
                f"{'# Node Attributes':<20}" +
                "".join([
                    f"{self.pyg_graphs[i].num_node_features:>15,}"
                    for i in range(network_count)
                ]) + "\n"
        )
        # Edge Attributes Dimension per graph
        output += (
                f"{'# Edge Attributes':<20}" +
                "".join([
                    f"{self.pyg_graphs[i].num_edge_features:>15,}"
                    for i in range(network_count)
                ]) + "\n"
        )
        # Anchor Links (Train/Test)
        output += (
            f"{'=' * 60}\n"
            f"{'Anchor Links':<20}{'Train':>15}{'Test':>15}\n"
            f"{'-' * 60}\n"
            f"{f'Count (ratio: {self.train_ratio})':<20}{self.train_data.shape[0]:>15,}{self.test_data.shape[0]:>15,}\n"
            f"{'=' * 60}"
        )
        return output

    # 输出结果：深拷贝数据集，供 PlanetAlign/utils/noise.py 的噪声函数选择修改副本。
    def clone(self):
        return copy.deepcopy(self)


if __name__ == '__main__':
    for f in os.listdir('../../datasets/pyg'):
        if f.endswith('.pt'):
            dataset_name = f[:-3]
            dataset = Dataset(root='../../datasets/pyg', name=dataset_name, train_ratio=0.2, dtype=torch.float64)
            print(dataset)
