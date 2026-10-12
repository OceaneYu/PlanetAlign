# 文件作用：按权重阈值过滤稀疏分配连接。
# 数据来源：PlanetAlign/algorithms/nettrans/model.py → TransPool.forward() 传入分配边与权重。

import torch


# 输出结果：只保留权重满足阈值的 row/col/value，供压缩图构造使用。
def remove_by_threshold(index, value, threshold):

    mask = (value >= threshold)
    row, col = index
    value = value[mask]
    row, col = row[mask], col[mask]

    return torch.stack([row, col]), value
