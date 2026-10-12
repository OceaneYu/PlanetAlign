# 文件作用：提供内置一对一图数据集入口；Arenas 读取 arenas.pt。
# 核心依赖：PlanetAlign/datasets/utils.py → download_file_from_google_drive() 可选下载，PlanetAlign/data/dataset.py → Dataset 完成实际加载。
# 结果去向：创建的对象由调用者传给算法 train()；本文件不会自动启动训练。

from typing import Union, Optional
from pathlib import Path
import os
import torch

from PlanetAlign.data import Dataset
from .utils import download_file_from_google_drive


# 功能：提供这个内置基准的创建入口，图统计与含义见保留的类文档。
class Arenas(Dataset):
    """A pair of networks synthesized from the email communication network Arenas at the University Rovira i Virgili.
    Nodes are users and each edge represents that at least one email was sent. The two networks are noisy permutations
    of each other. The dataset is proposed by the paper: "`KONECT – The Koblenz Network Collection.
    <https://dl.acm.org/doi/pdf/10.1145/2487788.2488173>`_" in WWW 2013. There are in total 1,135 common nodes across two
    networks.

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - Graph
          - #nodes
          - #edges
          - #node attrs
          - #edge attrs
        * - Arenas1
          - 1,135
          - 10,902
          - 50
          - 0
        * - Arenas2
          - 1,135
          - 10,800
          - 50
          - 0
    """
    # 输入来源：调用者指定 root；download=True 时下载 arenas.pt，否则要求文件已存在。
    # 流程位置：检查文件存在后调用 PlanetAlign/data/dataset.py → Dataset.__init__() 加载图并划分锚点。
    def __init__(self,
                 root: Union[str, Path],
                 download: Optional[bool] = False,
                 train_ratio: Optional[float] = 0.2,
                 dtype: torch.dtype = torch.float32,
                 seed: Optional[int] = 0):

        if download:
            download_file_from_google_drive(
                remote_file_id='1Eup0eKuKOv9Pb5E70_0-BOM2igV95NdC',
                save_filename='arenas.pt',
                root=root)

        if not self._check_integrity(root):
            raise RuntimeError('Arenas email dataset not found or corrupted. You can use download=True to download it')

        super(Arenas, self).__init__(root=root, name='arenas', train_ratio=train_ratio, dtype=dtype, seed=seed)

    # 功能：只检查 root/arenas.pt 是否存在；内容校验由父类 Dataset._validate() 执行。
    def _check_integrity(self, root):
        return os.path.exists(os.path.join(root, 'arenas.pt'))
