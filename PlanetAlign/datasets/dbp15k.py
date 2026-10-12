# 文件作用：提供内置一对一图数据集入口；DBP15K_FR_EN 读取 dbp15k_fr-en.pt；DBP15K_JA_EN 读取 dbp15k_ja-en.pt；DBP15K_ZH_EN 读取 dbp15k_zh-en.pt。
# 核心依赖：PlanetAlign/datasets/utils.py → download_file_from_google_drive() 可选下载，PlanetAlign/data/dataset.py → Dataset 完成实际加载。
# 结果去向：创建的对象由调用者传给算法 train()；本文件不会自动启动训练。

from typing import Union, Optional
from pathlib import Path
import os
import torch

from PlanetAlign.data import Dataset

from .utils import download_file_from_google_drive


# 功能：提供这个内置基准的创建入口，图统计与含义见保留的类文档。
class DBP15K_FR_EN(Dataset):
    """A pair of French to English version of multi-lingual DBpedia networks. The dataset is proposed by the
    `"Cross-lingual Entity Alignment via Joint Attribute-Preserving Embedding" <https://arxiv.org/abs/1708.05045>`_ paper,
    and the node attributes are given by pre-trained and aligned monolingual word embeddings from the
    `"Cross-lingual Knowledge Graph Alignment via Graph Matching Neural Network" <https://arxiv.org/abs/1905.11605>`_ paper.
    There are 15,000 pairs of aligned entities in DBP15K (French to English).

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - Graph
          - #nodes
          - #edges
          - #node attrs
          - #edge attrs
        * - FR
          - 19,661
          - 105,997
          - 300
          - 0
        * - EN
          - 19,993
          - 115,722
          - 300
          - 0
    """
    # 输入来源：调用者指定 root；download=True 时下载 dbp15k_fr-en.pt，否则要求文件已存在。
    # 流程位置：检查文件存在后调用 PlanetAlign/data/dataset.py → Dataset.__init__() 加载图并划分锚点。
    def __init__(self,
                 root: Union[str, Path],
                 download: Optional[bool] = False,
                 train_ratio: Optional[float] = 0.2,
                 dtype: torch.dtype = torch.float32,
                 seed: Optional[int] = 0):

        if download:
            download_file_from_google_drive(
                remote_file_id='1_uMPync1Y1yXsNPldtqGAG7VRLmtlt2b',
                save_filename='dbp15k_fr-en.pt',
                root=root)

        if not self._check_integrity(root):
            raise RuntimeError('DBP15K_FR-EN dataset not found or corrupted. You can use download=True to download it')

        super(DBP15K_FR_EN, self).__init__(root=root, name='dbp15k_fr-en', train_ratio=train_ratio, dtype=dtype, seed=seed)

    # 功能：只检查 root/dbp15k_fr-en.pt 是否存在；内容校验由父类 Dataset._validate() 执行。
    def _check_integrity(self, root):
        return os.path.exists(os.path.join(root, 'dbp15k_fr-en.pt'))


# 功能：提供这个内置基准的创建入口，图统计与含义见保留的类文档。
class DBP15K_JA_EN(Dataset):
    """A pair of Japanese to English version of multi-lingual DBpedia networks. The dataset is proposed by the
    `"Cross-lingual Entity Alignment via Joint Attribute-Preserving Embedding" <https://arxiv.org/abs/1708.05045>`_ paper,
    and the node attributes are given by pre-trained and aligned monolingual word embeddings from the
    `"Cross-lingual Knowledge Graph Alignment via Graph Matching Neural Network" <https://arxiv.org/abs/1905.11605>`_ paper.
    There are 15,000 pairs of aligned entities in DBP15K (Japanese to English).

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - Graph
          - #nodes
          - #edges
          - #node attrs
          - #edge attrs
        * - JA
          - 19,814
          - 77,214
          - 300
          - 0
        * - EN
          - 19,780
          - 93,484
          - 300
          - 0
    """
    # 输入来源：调用者指定 root；download=True 时下载 dbp15k_ja-en.pt，否则要求文件已存在。
    # 流程位置：检查文件存在后调用 PlanetAlign/data/dataset.py → Dataset.__init__() 加载图并划分锚点。
    def __init__(self,
                 root: Union[str, Path],
                 download: Optional[bool] = False,
                 train_ratio: Optional[float] = 0.2,
                 dtype: torch.dtype = torch.float32,
                 seed: Optional[int] = 0):

        if download:
            download_file_from_google_drive(
                remote_file_id='1Xc4sWilV7v804JZHRQqo2LzAM8O2_lib',
                save_filename='dbp15k_ja-en.pt',
                root=root)

        if not self._check_integrity(root):
            raise RuntimeError('DBP15K_JA-EN dataset not found or corrupted. You can use download=True to download it')

        super(DBP15K_JA_EN, self).__init__(root=root, name='dbp15k_ja-en', train_ratio=train_ratio, dtype=dtype, seed=seed)

    # 功能：只检查 root/dbp15k_ja-en.pt 是否存在；内容校验由父类 Dataset._validate() 执行。
    def _check_integrity(self, root):
        return os.path.exists(os.path.join(root, 'dbp15k_ja-en.pt'))


# 功能：提供这个内置基准的创建入口，图统计与含义见保留的类文档。
class DBP15K_ZH_EN(Dataset):
    """A pair of Chinese to English version of multi-lingual DBpedia networks. The dataset is proposed by the
    `"Cross-lingual Entity Alignment via Joint Attribute-Preserving Embedding" <https://arxiv.org/abs/1708.05045>`_ paper,
    and the node attributes are given by pre-trained and aligned monolingual word embeddings from the
    `"Cross-lingual Knowledge Graph Alignment via Graph Matching Neural Network" <https://arxiv.org/abs/1905.11605>`_ paper.
    There are 15,000 pairs of aligned entities in DBP15K (Chinese to English).

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - Graph
          - #nodes
          - #edges
          - #node attrs
          - #edge attrs
        * - ZH
          - 19,388
          - 70,414
          - 300
          - 0
        * - EN
          - 19,572
          - 95,142
          - 300
          - 0
    """
    # 输入来源：调用者指定 root；download=True 时下载 dbp15k_zh-en.pt，否则要求文件已存在。
    # 流程位置：检查文件存在后调用 PlanetAlign/data/dataset.py → Dataset.__init__() 加载图并划分锚点。
    def __init__(self,
                 root: Union[str, Path],
                 download: Optional[bool] = False,
                 train_ratio: Optional[float] = 0.2,
                 dtype: torch.dtype = torch.float32,
                 seed: Optional[int] = 0):

        if download:
            download_file_from_google_drive(
                remote_file_id='1ccub0If-q6_6MotYVOyFZzSc5SDJ2Izr',
                save_filename='dbp15k_zh-en.pt',
                root=root)

        if not self._check_integrity(root):
            raise RuntimeError('DBP15K_ZH-EN dataset not found or corrupted. You can use download=True to download it')

        super(DBP15K_ZH_EN, self).__init__(root=root, name='dbp15k_zh-en', train_ratio=train_ratio, dtype=dtype, seed=seed)

    # 功能：只检查 root/dbp15k_zh-en.pt 是否存在；内容校验由父类 Dataset._validate() 执行。
    def _check_integrity(self, root):
        return os.path.exists(os.path.join(root, 'dbp15k_zh-en.pt'))
