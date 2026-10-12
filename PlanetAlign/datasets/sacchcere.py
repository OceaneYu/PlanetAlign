# 文件作用：提供内置一对一图数据集入口；SacchCere 读取 SacchCere.pt。
# 核心依赖：PlanetAlign/datasets/utils.py → download_file_from_google_drive() 可选下载，PlanetAlign/data/dataset.py → Dataset 完成实际加载。
# 结果去向：创建的对象由调用者传给算法 train()；本文件不会自动启动训练。

from typing import Union, Optional
from pathlib import Path
import os
import torch

from PlanetAlign.data import Dataset
from .utils import download_file_from_google_drive


# 功能：提供这个内置基准的创建入口，图统计与含义见保留的类文档。
class SacchCere(Dataset):
    """A pair of direct interaction layer and association layer from the SacchCere multiplex GPI network. The
    SacchCere network consider different kinds of protein and genetic interactions for Saccharomyces Cerevisiae in
    `BioGRID <https://thebiogrid.org>`_, a public database that archives and disseminates genetic and protein interaction data from humans and model
    organisms. There are in total 1,337 common nodes across two layers of networks. The original multi-layer networks can be found `here <https://manliodedomenico.com/data.php>`_.

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - Graph
          - #nodes
          - #edges
          - #node attrs
          - #edge attrs
        * - SacchCere-direct
          - 5,042
          - 54,045
          - 0
          - 0
        * - SacchCere-association
          - 1,401
          - 3,918
          - 0
          - 0
    """

    # 输入来源：调用者指定 root；download=True 时下载 SacchCere.pt，否则要求文件已存在。
    # 流程位置：检查文件存在后调用 PlanetAlign/data/dataset.py → Dataset.__init__() 加载图并划分锚点。
    def __init__(self,
                 root: Union[str, Path],
                 download: Optional[bool] = False,
                 train_ratio: Optional[float] = 0.2,
                 dtype: torch.dtype = torch.float32,
                 seed: Optional[int] = 0):

        if download:
            download_file_from_google_drive(
                remote_file_id='1xD6ilqX6ICfi0jSkCTd1Jf17sG3xx_yN',
                save_filename='SacchCere.pt',
                root=root)

        if not self._check_integrity(root):
            raise RuntimeError('SacchCere dataset not found or corrupted. You can use download=True to download it')

        super(SacchCere, self).__init__(root=root, name='SacchCere', train_ratio=train_ratio, dtype=dtype, seed=seed)

    # 功能：只检查 root/SacchCere.pt 是否存在；内容校验由父类 Dataset._validate() 执行。
    def _check_integrity(self, root):
        return os.path.exists(os.path.join(root, 'SacchCere.pt'))
