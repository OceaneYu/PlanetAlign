# 文件作用：通过随机重复较短的一组样本，让两组训练样本数量一致。
# 数据来源：PlanetAlign/algorithms/nettrans/main.py → NetTrans.train() 和 PlanetAlign/algorithms/nextalign/main.py 的上下文采样流程传入节点对。

import torch


# 输出结果：返回等长的两组样本；只补足较短的一组，不删除较长一组。
def balance_samples(sample_pairs1, sample_pairs2):
    if len(sample_pairs1) < len(sample_pairs2):
        len_diff = len(sample_pairs2) - len(sample_pairs1)
        idx = torch.randint(len(sample_pairs1), (len_diff,))
        imputes = sample_pairs1[idx]
        balanced_sample_pairs1 = torch.vstack([sample_pairs1, imputes])
        balanced_sample_pairs2 = sample_pairs2
    else:
        len_diff = len(sample_pairs1) - len(sample_pairs2)
        idx = torch.randint(len(sample_pairs2), (len_diff,))
        imputes = sample_pairs2[idx]
        balanced_sample_pairs2 = torch.vstack([sample_pairs2, imputes])
        balanced_sample_pairs1 = sample_pairs1
    return balanced_sample_pairs1, balanced_sample_pairs2
