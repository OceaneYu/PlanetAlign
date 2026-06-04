r"""Utility package for computing evaluation metrics for network aligment."""

import copy

from .hits import *
from .mrr import *
from .metrics_ACS import acs_score, concentration_accuracy_score
from .metrics_MSF1 import macro_f1_score, macro_set_f1
from .metrics_Micro_SF1 import global_set_f1, micro_f1_score
from .metrics_m2m_EGS import egs_score, egs_weighted_score, m2m_egs_score
from .metrics_m2m_SGS import m2m_sgs_score, sgs_score, sgs_weighted_score
from .many_to_many import (
    DEFAULT_MANY_TO_MANY_METRICS,
    many_to_many_scores,
    pred_entities_from_similarity,
    similarity_to_pred_entities,
)

__all__ = [
    'hits_ks_scores',
    'mrr_score',
    'acs_score',
    'macro_f1_score',
    'micro_f1_score',
    'sgs_score',
    'sgs_weighted_score',
    'egs_score',
    'egs_weighted_score',
    'concentration_accuracy_score',
    'macro_set_f1',
    'global_set_f1',
    'm2m_sgs_score',
    'm2m_egs_score',
    'DEFAULT_MANY_TO_MANY_METRICS',
    'many_to_many_scores',
    'similarity_to_pred_entities',
    'pred_entities_from_similarity',
]

classes = copy.copy(__all__)
