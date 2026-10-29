# 文件作用：统一导出节点排名指标和多对多实体集合指标，供算法与评估入口导入。
# 阅读建议：节点指标看 PlanetAlign/metrics/hits.py → hits_ks_scores() 与 PlanetAlign/metrics/mrr.py → mrr_score()，同分排名在 PlanetAlign/metrics/ranking.py → pairwise_ranks()；实体指标看 PlanetAlign/metrics/many_to_many.py → many_to_many_scores()。

r"""Utility package for computing evaluation metrics for network aligment."""

import copy

from .hits import *
from .mrr import *
from .ranking import mean_rank_score, pairwise_ranks, rank_diagnostics
from .metrics_MSF1 import InvalidPredictionError, MSF1_PROTOCOL, macro_f1_score, macro_set_f1
from .metrics_Micro_SF1 import global_set_f1, micro_f1_score
from .metrics_AUCS import aucs_score, weighted_aucs_score
from .metrics_AUIS import auis_score, weighted_auis_score
from .many_to_many import (
    DEFAULT_MANY_TO_MANY_METRICS,
    many_to_many_scores,
)

__all__ = [
    'hits_ks_scores',
    'mrr_score',
    'mean_rank_score',
    'pairwise_ranks',
    'rank_diagnostics',
    'macro_f1_score',
    'InvalidPredictionError',
    'MSF1_PROTOCOL',
    'micro_f1_score',
    'auis_score',
    'weighted_auis_score',
    'aucs_score',
    'weighted_aucs_score',
    'macro_set_f1',
    'global_set_f1',
    'DEFAULT_MANY_TO_MANY_METRICS',
    'many_to_many_scores',
]

classes = copy.copy(__all__)
