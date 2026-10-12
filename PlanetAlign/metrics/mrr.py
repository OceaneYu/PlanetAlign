# 文件作用：计算真实对应节点排名倒数的平均值（MRR），越接近 1 表示排名越靠前。
# 数据来源：PlanetAlign/algorithms/base_model.py → BaseModel.test() 等传入 S 和评估节点对。
# 核心依赖：PlanetAlign/metrics/ranking.py → pairwise_ranks() 提供同分策略一致的排名。

import torch

from .ranking import DEFAULT_TIE_ATOL, DEFAULT_TIE_RTOL, pairwise_ranks


# 功能：按 mode 调用单向或双向汇总函数，返回一个 MRR 分数。
def mrr_score(similarity: torch.Tensor,
              test_pairs: torch.Tensor,
              mode: str = 'mean',
              tie_policy: str = 'optimistic',
              atol: float = DEFAULT_TIE_ATOL,
              rtol: float = DEFAULT_TIE_RTOL) -> float:
    r"""Mean Reciprocal Rank (MRR) score of pairwise alignment results.

    Parameters
    ----------
    similarity : torch.Tensor
        Similarity matrix of shape (n1, n2) where n1 and n2 are the number of nodes in graph1 and graph2.
    test_pairs : torch.Tensor
        Test pairs of shape (m, 2) where m is the number of test pairs.
    mode : str, optional
        Mode for MRR score. Options are 'mean', 'max', 'ltr' (left-to-right), 'rtl' (right-to-left). Default is 'mean'.

    """

    if mode == 'mean':
        mrr = mrr_mean_score(
            similarity, test_pairs, tie_policy, atol, rtol
        )
    elif mode == 'max':
        mrr = mrr_max_score(
            similarity, test_pairs, tie_policy, atol, rtol
        )
    elif mode == 'ltr':
        mrr = mrr_ltr_score(
            similarity, test_pairs, tie_policy, atol, rtol
        )
    elif mode == 'rtl':
        mrr = mrr_rtl_score(
            similarity, test_pairs, tie_policy, atol, rtol
        )
    else:
        raise ValueError(f"Invalid mode: {mode}")
    return mrr


# 功能：按 S 的行查询目标，计算真实目标排名倒数的均值。
def mrr_ltr_score(
    similarity, test_pairs, tie_policy='optimistic',
    atol=DEFAULT_TIE_ATOL, rtol=DEFAULT_TIE_RTOL,
):
    r"""Mean Reciprocal Rank (MRR) score of graph1(left) to graph2(right) alignment."""
    ranks = pairwise_ranks(
        similarity, test_pairs, 'ltr', tie_policy, atol, rtol
    )
    return float((1.0 / ranks).mean())


# 功能：按 S 的列查询源节点，计算反向排名倒数的均值。
def mrr_rtl_score(
    similarity, test_pairs, tie_policy='optimistic',
    atol=DEFAULT_TIE_ATOL, rtol=DEFAULT_TIE_RTOL,
):
    r"""Mean Reciprocal Rank (MRR) score of graph2(right) to graph1(left) alignment."""
    ranks = pairwise_ranks(
        similarity, test_pairs, 'rtl', tie_policy, atol, rtol
    )
    return float((1.0 / ranks).mean())


# 输出结果：取两个方向 MRR 的较大值。
def mrr_max_score(
    similarity, test_pairs, tie_policy='optimistic',
    atol=DEFAULT_TIE_ATOL, rtol=DEFAULT_TIE_RTOL,
):
    r"""Max Mean Reciprocal Rank (MRR) score of left-to-right and right-to-left alignments."""
    mrr_ltr = mrr_ltr_score(similarity, test_pairs, tie_policy, atol, rtol)
    mrr_rtl = mrr_rtl_score(similarity, test_pairs, tie_policy, atol, rtol)
    mrr = max(mrr_ltr, mrr_rtl)

    return mrr


# 输出结果：平均两个方向的 MRR。
def mrr_mean_score(
    similarity, test_pairs, tie_policy='optimistic',
    atol=DEFAULT_TIE_ATOL, rtol=DEFAULT_TIE_RTOL,
):
    r"""Mean Mean Reciprocal Rank (MRR) score of left-to-right and right-to-left alignments."""
    mrr_ltr = mrr_ltr_score(similarity, test_pairs, tie_policy, atol, rtol)
    mrr_rtl = mrr_rtl_score(similarity, test_pairs, tie_policy, atol, rtol)
    mrr = (mrr_ltr + mrr_rtl) / 2

    return mrr
