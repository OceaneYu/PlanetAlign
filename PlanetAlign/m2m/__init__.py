# 文件作用：导出 PlanetAlign/m2m/core.py 的实体格式、基准读写、预测评分和监督调整接口。
# 阅读建议：训练矩阵看 PlanetAlign/m2m/base.py → train_base_S()，预测实体看 PlanetAlign/m2m/quotient.py → quotient_decode()，两者需导入子模块。

"""Many-to-many alignment interfaces, decoders, and research utilities.

The common entity-map and benchmark interfaces are available directly here.
Use the base, blind, quotient, contrastive, and diagnostics submodules for
training, decoding, and experiments.
"""

from .core import (
    DEFAULT_ENTITY_SIDES,
    EntityMap,
    LoadedM2MBenchmark,
    align_prediction_to_ground_truth,
    complete_prediction_entities,
    evaluate_predictions,
    ground_truth_path,
    load_entity_map,
    load_m2m_benchmark,
    normalize_entity_map,
    save_entity_map,
    use_full_anchor_supervision,
)

__all__ = [
    "DEFAULT_ENTITY_SIDES",
    "EntityMap",
    "LoadedM2MBenchmark",
    "align_prediction_to_ground_truth",
    "complete_prediction_entities",
    "evaluate_predictions",
    "ground_truth_path",
    "load_entity_map",
    "load_m2m_benchmark",
    "normalize_entity_map",
    "save_entity_map",
    "use_full_anchor_supervision",
]
