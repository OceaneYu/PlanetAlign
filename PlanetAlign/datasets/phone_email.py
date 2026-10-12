# 文件作用：提供内置一对一图数据集入口；PhoneEmail 读取 phone-email.pt。
# 核心依赖：PlanetAlign/datasets/utils.py → download_file_from_google_drive() 可选下载，PlanetAlign/data/dataset.py → Dataset 完成实际加载。
# 结果去向：创建的对象由调用者传给算法 train()；本文件不会自动启动训练。

from typing import Union, Optional
from pathlib import Path
import os
import torch

from PlanetAlign.data import Dataset
from .utils import download_file_from_google_drive


# 功能：提供这个内置基准的创建入口，图统计与含义见保留的类文档。
class PhoneEmail(Dataset):
    """
    A pair of communication networks among people via phone or email.
    Nodes represent people and an edge exists between two people if they communicate via
    phone or email at least once. Phone network includes 1,000 nodes and 41,191 edges.
    Email network includes 1,003 nodes and 4,627 edges. Both networks are plain networks.
    There are 1,000 common people across two networks.

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - Graph
          - #nodes
          - #edges
          - #node attrs
          - #edge attrs
        * - Phone
          - 1,000
          - 41,191
          - 0
          - 0
        * - Email
          - 1,003
          - 4,628
          - 0
          - 0
    """
    # 输入来源：调用者指定 root；download=True 时下载 phone-email.pt，否则要求文件已存在。
    # 流程位置：检查文件存在后调用 PlanetAlign/data/dataset.py → Dataset.__init__() 加载图并划分锚点。
    def __init__(self,
                 root: Union[str, Path],
                 download: Optional[bool] = False,
                 train_ratio: Optional[float] = 0.2,
                 dtype: torch.dtype = torch.float32,
                 seed: Optional[int] = 0):

        if download:
            download_file_from_google_drive(
                remote_file_id='1RTovO4olFFiXhXbDeO_X2NSDB8wkq742',
                save_filename='phone-email.pt',
                root=root)

        if not self._check_integrity(root):
            raise RuntimeError('Phone-Email dataset not found or corrupted. You can use download=True to download it to the specified "root" directory.')

        super(PhoneEmail, self).__init__(root=root, name='phone-email', train_ratio=train_ratio, dtype=dtype, seed=seed)

    # 功能：只检查 root/phone-email.pt 是否存在；内容校验由父类 Dataset._validate() 执行。
    def _check_integrity(self, root):
        return os.path.exists(os.path.join(root, 'phone-email.pt'))
