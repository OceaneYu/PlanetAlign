# 文件作用：记录论文声明的 GADL 参数与其公开参考实现的差异，提供对应模型工厂。
# 核心依赖：PlanetAlign/algorithms/gadl/profiles.py → ConfigurableGADL 实现实际可选行为。
# 结果去向：scripts/evaluate_gadl_paper_declared.py 使用 paper_declared_gadl() 构建受控比较模型。

"""Opt-in configuration declared by the GADL paper for Douban Online-Offline.

This module intentionally does not change :class:`GADL` defaults.  The paper
profile follows the equations and Appendix C.4 of arXiv:2509.09597v3, while
the public-reference profile follows the behavior of the released code.  The
two profiles must remain separate because those materials disagree.
"""

from __future__ import annotations

from typing import Any, Dict

from .profiles import ConfigurableGADL


PAPER_DECLARED_PROFILE: Dict[str, Any] = {
    "graph_protocol": "full-graph",
    "online_edge_entries": 16_328,
    "hidden_dim": 256,
    "num_hidden_layers": 6,
    "spectral_rank": 300,
    "lr": 1e-3,
    "weight_decay": 5e-4,
    "descriptor_weight": 1e-3,
    "commutativity_weight": 1e-2,
    "functional_map_weight": 1.0,
    "bijectivity_weight": 0.1,
    "orthogonality_weight": 0.1,
    "residual": False,
    "reconstruction_mode": "unweighted_dense_bce",
    "functional_map_schedule": "constant",
    "activation_mode": "planetalign_prelu",
    "total_epochs": 100,
}

PUBLIC_REFERENCE_DIFFERENCES: Dict[str, Dict[str, Any]] = {
    "graph_protocol": {"paper": "full-graph", "public_reference": "baseline-compatible"},
    "online_edge_entries": {"paper": 16_328, "public_reference": 11_502},
    "hidden_dim": {"paper": 256, "public_reference": 512},
    "num_hidden_layers": {"paper": 6, "public_reference": 3},
    "lr": {"paper": 1e-3, "public_reference": 1e-4},
    "reconstruction_mode": {
        "paper": "unweighted_dense_bce",
        "public_reference": "reference_weighted_bce",
    },
    "functional_map_schedule": {
        "paper": "constant",
        "public_reference": "reference_linear_floor_half",
    },
    "activation_mode": {
        "paper": "planetalign_prelu",
        "public_reference": "reference_identity",
    },
}

_MODEL_KEYS = {
    "hidden_dim",
    "num_hidden_layers",
    "spectral_rank",
    "lr",
    "weight_decay",
    "descriptor_weight",
    "commutativity_weight",
    "functional_map_weight",
    "bijectivity_weight",
    "orthogonality_weight",
    "residual",
    "reconstruction_mode",
    "functional_map_schedule",
    "activation_mode",
}


# 输出结果：将论文声明的配置与调用者覆盖项合并，返回 ConfigurableGADL 对象；不启动训练。
def paper_declared_gadl(**overrides: Any) -> ConfigurableGADL:
    """Construct GADL with the paper-declared model and objective semantics."""

    config = {key: value for key, value in PAPER_DECLARED_PROFILE.items() if key in _MODEL_KEYS}
    config.update(overrides)
    return ConfigurableGADL(**config)
