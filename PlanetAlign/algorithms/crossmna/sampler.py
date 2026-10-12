# 文件作用：为给定离散概率建立别名表，快速抽取负例节点。
# 数据来源：PlanetAlign/algorithms/crossmna/main.py → CrossMNA._init_sampler() 传入度数分布。
# 结果去向：AliasSampler.sample() 的节点编号交给 CrossMNA._generate_samples()。

import numpy as np


# 功能：预先保存每个桶的概率与替代桶，减少反复按权重采样的开销。
class AliasSampler:
    def __init__(self, prob):
        self.n = len(prob)
        self.U = np.array(prob) * self.n
        self.K = [i for i in range(len(prob))]
        overfull, underfull = [], []
        for i, U_i in enumerate(self.U.tolist()):
            if U_i > 1:
                overfull.append(i)
            elif U_i < 1:
                underfull.append(i)
        while len(overfull) and len(underfull):
            i, j = overfull.pop(), underfull.pop()
            self.K[j] = i
            self.U[i] = self.U[i] - (1 - self.U[j])
            if self.U[i] > 1:
                overfull.append(i)
            elif self.U[i] < 1:
                underfull.append(i)

    # 输出结果：按初始化概率返回指定数量的节点编号数组。
    def sample(self, num_samples=1):
        x = np.random.rand(num_samples)
        i = np.floor(self.n * x).astype(np.int32)
        y = self.n * x - i
        samples = np.array([i[k] if y[k] < self.U[i[k]] else self.K[i[k]] for k in range(num_samples)])
        return samples