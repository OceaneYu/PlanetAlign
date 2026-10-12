# 文件作用：提供内置一对一图数据集入口；FlickrLastFM 读取 flickr-lastfm.pt。
# 核心依赖：PlanetAlign/datasets/utils.py → download_file_from_google_drive() 可选下载，PlanetAlign/data/dataset.py → Dataset 完成实际加载。
# 结果去向：创建的对象由调用者传给算法 train()；本文件不会自动启动训练。

from typing import Union, Optional
from pathlib import Path
import os
import torch

from PlanetAlign.data import Dataset
from .utils import download_file_from_google_drive


# 功能：提供这个内置基准的创建入口，图统计与含义见保留的类文档。
class FlickrLastFM(Dataset):
    """
    A pair of social networks from Flickr and LastFM. Nodes in both networks represent users,
    and edges represent friend / following relationships in Flickr and LastFM, respectively. The gender
    of a user is treated as the node attributes (male, female, unknown), and the level of people a user is connected to
    is treated as the edge attributes (e.g., leader with leader). There are 452 common users across two networks.

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - Graph
          - #nodes
          - #edges
          - #node attrs
          - #edge attrs
        * - Flickr
          - 12,974
          - 16,149
          - 3
          - 3
        * - LastFM
          - 15,436
          - 16,319
          - 3
          - 3
    """
    # 输入来源：调用者指定 root；download=True 时下载 flickr-lastfm.pt，否则要求文件已存在。
    # 流程位置：检查文件存在后调用 PlanetAlign/data/dataset.py → Dataset.__init__() 加载图并划分锚点。
    def __init__(self,
                 root: Union[str, Path],
                 download: Optional[bool] = False,
                 train_ratio: Optional[float] = 0.2,
                 dtype: torch.dtype = torch.float32,
                 seed: Optional[int] = 0):

        if download:
            download_file_from_google_drive(
                remote_file_id='1zfZc6So7UEyVZHN-Uas-eAgckqk28gal',
                save_filename='flickr-lastfm.pt',
                root=root)

        if not self._check_integrity(root):
            raise RuntimeError('Flickr-LastFM dataset not found or corrupted. You can use download=True to download it')

        super(FlickrLastFM, self).__init__(root=root, name='flickr-lastfm', train_ratio=train_ratio, dtype=dtype, seed=seed)

    # 功能：只检查 root/flickr-lastfm.pt 是否存在；内容校验由父类 Dataset._validate() 执行。
    def _check_integrity(self, root):
        return os.path.exists(os.path.join(root, 'flickr-lastfm.pt'))
