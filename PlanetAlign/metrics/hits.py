# 文件作用：计算真实对应节点进入前 K 名的比例，即 Hits@K。
# 数据来源：PlanetAlign/algorithms/base_model.py → BaseModel.test() 或算法训练日志流程传入 S 与评估节点对。
# 核心依赖：PlanetAlign/metrics/ranking.py → pairwise_ranks() 统一处理排名与同分。

from typing import Union
import torch

from .ranking import DEFAULT_TIE_ATOL, DEFAULT_TIE_RTOL, pairwise_ranks


# 功能：按 mode 选择左到右、右到左、双向均值或双向最大值，返回各 K 对应的命中比例。
def hits_ks_scores(simiarity: torch.Tensor,
                   test_pairs: torch.Tensor,
                   ks: Union[list[int], tuple[int, ...]] = (1, 10, 30, 50),
                   mode: str = 'mean',
                   tie_policy: str = 'optimistic',
                   atol: float = DEFAULT_TIE_ATOL,
                   rtol: float = DEFAULT_TIE_RTOL) -> dict[int, float]:
    r"""Hits@K scores of pairwise alignment results.

    Parameters
    ----------
    simiarity : torch.Tensor
        Similarity matrix of shape (n1, n2) where n1 and n2 are the number of nodes in graph1 and graph2.
    test_pairs : torch.Tensor
        Test pairs of shape (m, 2) where m is the number of test pairs.
    ks : list[int] or tuple[int, ...], optional
        List of k values for Hits@K scores. Default is (1, 10, 30, 50).
    mode : str, optional
        Mode for Hits@K scores. Options are 'mean', 'max', 'ltr' (left-to-right), 'rtl' (right-to-left). Default is 'mean'.
    """

    if mode == 'mean':
        hits_ks = hits_ks_mean_scores(
            simiarity, test_pairs, ks=ks, tie_policy=tie_policy,
            atol=atol, rtol=rtol
        )
    elif mode == 'max':
        hits_ks = hits_ks_max_scores(
            simiarity, test_pairs, ks=ks, tie_policy=tie_policy,
            atol=atol, rtol=rtol
        )
    elif mode == 'ltr':
        hits_ks = hits_ks_ltr_scores(
            simiarity, test_pairs, ks=ks, tie_policy=tie_policy,
            atol=atol, rtol=rtol
        )
    elif mode == 'rtl':
        hits_ks = hits_ks_rtl_scores(
            simiarity, test_pairs, ks=ks, tie_policy=tie_policy,
            atol=atol, rtol=rtol
        )
    else:
        raise ValueError(f"Invalid mode: {mode}")
    return hits_ks


# 功能：按 S 的行查询目标节点，统计真实目标排名不超过 K 的比例。
def hits_ks_ltr_scores(
    similarity, test_pairs, ks=(1, 10, 30, 50),
    tie_policy='optimistic', atol=DEFAULT_TIE_ATOL, rtol=DEFAULT_TIE_RTOL,
):
    r"""Hits@K scores of graph1(left) to graph2(right) alignment."""
    ranks = pairwise_ranks(
        similarity, test_pairs, 'ltr', tie_policy, atol, rtol
    )
    return {k: float((ranks <= k).to(torch.float64).mean()) for k in ks}


# 功能：按 S 的列反向查询源节点，统计真实源节点进入前 K 名的比例。
def hits_ks_rtl_scores(
    similarity, test_pairs, ks=(1, 10, 30, 50),
    tie_policy='optimistic', atol=DEFAULT_TIE_ATOL, rtol=DEFAULT_TIE_RTOL,
):
    r"""Hits@K scores of graph2(right) to graph1(left) alignment."""
    ranks = pairwise_ranks(
        similarity, test_pairs, 'rtl', tie_policy, atol, rtol
    )
    return {k: float((ranks <= k).to(torch.float64).mean()) for k in ks}


# 输出结果：对每个 K 取双向 Hits 的较大值，不是合并两侧候选。
def hits_ks_max_scores(
    similarity, test_pairs, ks=(1, 10, 30, 50),
    tie_policy='optimistic', atol=DEFAULT_TIE_ATOL, rtol=DEFAULT_TIE_RTOL,
):
    r"""Max Hits@K scores of left-to-right and right-to-left alignments."""
    hits_ks = {}

    hits_ks_ltr = hits_ks_ltr_scores(
        similarity, test_pairs, ks, tie_policy, atol, rtol
    )
    hits_ks_rtl = hits_ks_rtl_scores(
        similarity, test_pairs, ks, tie_policy, atol, rtol
    )
    for k in ks:
        hits_ks[k] = max(hits_ks_ltr[k], hits_ks_rtl[k])

    return hits_ks


# 输出结果：对每个 K 平均左到右与右到左的命中比例。
def hits_ks_mean_scores(
    similarity, test_pairs, ks=(1, 10, 30, 50),
    tie_policy='optimistic', atol=DEFAULT_TIE_ATOL, rtol=DEFAULT_TIE_RTOL,
):
    r"""Mean Hits@K scores of left-to-right and right-to-left alignments."""
    hits_ks = {}

    hits_ks_ltr = hits_ks_ltr_scores(
        similarity, test_pairs, ks, tie_policy, atol, rtol
    )
    hits_ks_rtl = hits_ks_rtl_scores(
        similarity, test_pairs, ks, tie_policy, atol, rtol
    )
    for k in ks:
        hits_ks[k] = (hits_ks_ltr[k] + hits_ks_rtl[k]) / 2

    return hits_ks
