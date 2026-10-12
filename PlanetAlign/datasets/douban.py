# 文件作用：提供内置一对一图数据集入口；Douban 读取 douban.pt。
# 核心依赖：PlanetAlign/datasets/utils.py → download_file_from_google_drive() 可选下载，PlanetAlign/data/dataset.py → Dataset 完成实际加载。
# 结果去向：创建的对象由调用者传给算法 train()；本文件不会自动启动训练。

from typing import Union, Optional
from pathlib import Path
import os
import torch

from PlanetAlign.data import Dataset
from .utils import download_file_from_google_drive


# 功能：提供这个内置基准的创建入口，图统计与含义见保留的类文档。
class Douban(Dataset):
    """
    A pair of online-offline social networks constructed from Douban. Nodes represent
    users and edges represent user interactions on the website. The location of a suer is treated as the node attribute,
    and the contact/friend relationship are treated as the edge attributes.
    There are 1,118 common user across the two networks.

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - Graph
          - #nodes
          - #edges
          - #node attrs
          - #edge attrs
        * - Douban(online)
          - 3,906
          - 8,164
          - 538
          - 2
        * - Douban(offline)
          - 1,118
          - 1,511
          - 538
          - 2

    """
    # 输入来源：调用者指定 root；download=True 时下载 douban.pt，否则要求文件已存在。
    # 流程位置：检查文件存在后调用 PlanetAlign/data/dataset.py → Dataset.__init__() 加载图并划分锚点。
    def __init__(self,
                 root: Union[str, Path],
                 download: Optional[bool] = False,
                 train_ratio: Optional[float] = 0.2,
                 dtype: torch.dtype = torch.float32,
                 seed: Optional[int] = 0):

        if download:
            download_file_from_google_drive(
                remote_file_id='1PMtPhwEPKUr-PP8LnYoYADh7-x22xbZ4',
                save_filename='douban.pt',
                root=root)

        if not self._check_integrity(root):
            raise RuntimeError('Douban dataset not found or corrupted. You can use download=True to download it')

        super(Douban, self).__init__(root=root, name='douban', train_ratio=train_ratio, dtype=dtype, seed=seed)

    # 功能：只检查 root/douban.pt 是否存在；内容校验由父类 Dataset._validate() 执行。
    def _check_integrity(self, root):
        return os.path.exists(os.path.join(root, 'douban.pt'))
