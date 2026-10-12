# 文件作用：保留 MEAformer 算法的接口占位与论文说明，当前 train() 的函数体只有 pass。
# 核心依赖：PlanetAlign/algorithms/__init__.py 导出 MEAformer；导出类不代表训练功能已实现。

import time
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj

from PlanetAlign.data import Dataset
from PlanetAlign.utils import get_anchor_pairs
from PlanetAlign.metrics import hits_ks_scores, mrr_score
from PlanetAlign.algorithms.base_model import BaseModel


# 功能：继承 BaseModel 初始化 dtype 等公共状态，目前没有实现对齐过程。
class MEAformer(BaseModel):
    """
    Embedding-based method MEAformer for pairwise entity alignment. DualMatch is proposed by the paper
    "`MEAformer: Multi-modal Entity Alignment Transformer for Meta Modality Hybrid <https://arxiv.org/pdf/2212.14454>`_"
    in ACM MM 2023.
    """

    def __init__(self,
                 dtype: torch.dtype = torch.float32):
        super(MEAformer, self).__init__(dtype=dtype)

    # 输出结果：当前未执行训练或产生矩阵，直接调用默认返回 None。
    def train(self,
              dataset: Dataset,
              gid1: int,
              gid2: int,
              use_attr: bool = True,
              save_log: bool = True,
              verbose: bool = True):
        pass
