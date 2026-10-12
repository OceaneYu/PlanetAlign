# 文件作用：把两侧等长的上下文节点对包装成 PyTorch 小批次数据集。
# 数据来源：PlanetAlign/algorithms/nettrans/main.py 的 NetTrans 训练过程传入采样结果。
# 结果去向：DataLoader 通过 __getitem__() 按索引取两侧样本交给训练循环。

import torch
from torch.utils.data import Dataset


# 功能：保存两侧上下文样本和数量，支持 DataLoader 按行读取。
class ContextDataset(Dataset):
    def __init__(self, context_pairs1, context_pairs2):
        # data loading
        self.context_pairs1 = context_pairs1
        self.context_pairs2 = context_pairs2
        self.n_samples = context_pairs1.shape[0]

    # 输出结果：返回同一索引处的两侧节点对。
    def __getitem__(self, index):
        return self.context_pairs1[index], self.context_pairs2[index]

    def __len__(self):
        return self.n_samples
