# 文件作用：统一计算节点对齐排名，明确同分候选采用最好、平均还是最差名次。
# 数据来源：PlanetAlign/metrics/hits.py、PlanetAlign/metrics/mrr.py 的评分函数传入 S 和真实节点对。
# 结果去向：pairwise_ranks() 返回从 1 开始的排名，供 Hits、MRR 和 MeanRank 汇总。

"""Tie-aware ranks for pairwise network alignment metrics."""

import torch


TIE_POLICIES = ("optimistic", "average", "pessimistic")
DEFAULT_TIE_ATOL = 1e-8
DEFAULT_TIE_RTOL = 1e-7


def _validate_tie_arguments(tie_policy, atol, rtol):
    if tie_policy not in TIE_POLICIES:
        raise ValueError(
            f"Invalid tie_policy '{tie_policy}'; expected one of {TIE_POLICIES}"
        )
    if atol < 0 or rtol < 0:
        raise ValueError("Tie tolerances must be non-negative")


# 核心思路：统计严格高于真实节点的候选数，再按 tie_policy 处理近似同分候选。
# 输出结果：每个查询的排名张量；direction 决定查看 S 的行还是列。
def pairwise_ranks(
    similarity: torch.Tensor,
    test_pairs: torch.Tensor,
    direction: str = "ltr",
    tie_policy: str = "optimistic",
    atol: float = DEFAULT_TIE_ATOL,
    rtol: float = DEFAULT_TIE_RTOL,
) -> torch.Tensor:
    """Return tie-aware one-based ranks for explicit ground-truth pairs.

    ``num_equal`` includes the true candidate. Values within ``atol +
    rtol * abs(truth_score)`` are treated as equal and are excluded from the
    strictly-greater count.
    """
    _validate_tie_arguments(tie_policy, atol, rtol)
    if direction not in ("ltr", "rtl"):
        raise ValueError("direction must be 'ltr' or 'rtl'")
    if similarity.ndim != 2 or test_pairs.ndim != 2 or test_pairs.shape[1] != 2:
        raise ValueError("Expected a 2D similarity matrix and [N, 2] pairs")
    pairs = test_pairs.to(device=similarity.device, dtype=torch.long)
    if direction == "ltr":
        scores = similarity[pairs[:, 0]]
        truth_ids = pairs[:, 1]
    else:
        scores = similarity.T[pairs[:, 1]]
        truth_ids = pairs[:, 0]
    truth = scores[torch.arange(len(pairs), device=scores.device), truth_ids]
    equal = torch.isclose(scores, truth[:, None], atol=atol, rtol=rtol)
    strictly_greater = (scores > truth[:, None]) & ~equal
    num_greater = strictly_greater.sum(dim=1).to(torch.float64)
    num_equal = equal.sum(dim=1).to(torch.float64)
    if tie_policy == "optimistic":
        return num_greater + 1.0
    if tie_policy == "average":
        return num_greater + (num_equal + 1.0) / 2.0
    return num_greater + num_equal


# 功能：统计最高分候选是否并列、平均并列数量和真实节点独占第一名的比例，供诊断脚本使用。
def rank_diagnostics(
    similarity: torch.Tensor,
    test_pairs: torch.Tensor,
    direction: str = "ltr",
    atol: float = DEFAULT_TIE_ATOL,
    rtol: float = DEFAULT_TIE_RTOL,
) -> dict[str, float]:
    """Summarize maximum-score ties for explicit query pairs."""
    _validate_tie_arguments("optimistic", atol, rtol)
    pairs = test_pairs.to(device=similarity.device, dtype=torch.long)
    if direction == "ltr":
        scores = similarity[pairs[:, 0]]
        truth_ids = pairs[:, 1]
    elif direction == "rtl":
        scores = similarity.T[pairs[:, 1]]
        truth_ids = pairs[:, 0]
    else:
        raise ValueError("direction must be 'ltr' or 'rtl'")
    maxima = scores.max(dim=1).values
    maximum_mask = torch.isclose(
        scores, maxima[:, None], atol=atol, rtol=rtol
    )
    maximum_counts = maximum_mask.sum(dim=1).to(torch.float64)
    truth = scores[torch.arange(len(pairs), device=scores.device), truth_ids]
    truth_is_maximum = torch.isclose(truth, maxima, atol=atol, rtol=rtol)
    return {
        "maximum_tie_rate": float(
            (maximum_counts > 1).to(torch.float64).mean()
        ),
        "average_maximum_candidates": float(maximum_counts.mean()),
        "unique_top1_ratio": float(
            (truth_is_maximum & (maximum_counts == 1)).to(torch.float64).mean()
        ),
    }


# 功能：按单向、双向均值或双向较好结果汇总排名；平均名次越小越好。
def mean_rank_score(
    similarity: torch.Tensor,
    test_pairs: torch.Tensor,
    mode: str = "mean",
    tie_policy: str = "optimistic",
    atol: float = DEFAULT_TIE_ATOL,
    rtol: float = DEFAULT_TIE_RTOL,
) -> float:
    """Compute mean rank using an explicit tie policy."""
    ltr = float(pairwise_ranks(
        similarity, test_pairs, "ltr", tie_policy, atol, rtol
    ).mean())
    rtl = float(pairwise_ranks(
        similarity, test_pairs, "rtl", tie_policy, atol, rtol
    ).mean())
    if mode == "ltr":
        return ltr
    if mode == "rtl":
        return rtl
    if mode == "mean":
        return (ltr + rtl) / 2.0
    if mode == "max":
        return min(ltr, rtl)
    raise ValueError(f"Invalid mode: {mode}")
