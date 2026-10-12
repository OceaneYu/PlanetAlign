# 文件作用：为 GADL 复现研究提供可选的重构损失、映射权重计划和编码器行为。
# 核心依赖：PlanetAlign/algorithms/gadl/main.py → GADL；ConfigurableGADL.train() 临时替换目标函数后复用其训练流程。
# 结果去向：scripts/evaluate_gadl_public_reference.py 等调用这些配置做受控比较。
# 阅读建议：先看 ConfigurableGADL.train() 的作用域与配置，默认 GADL 行为见 PlanetAlign/algorithms/gadl/main.py。

"""Opt-in experimental training profiles for GADL.

The base :class:`GADL` implementation and its defaults remain unchanged.  This
module adds explicit reconstruction and functional-map schedule choices for
controlled reference-code attribution experiments.
"""

from __future__ import annotations

from contextlib import ExitStack
from typing import Callable, Dict, Optional
from unittest.mock import patch

import torch
import torch.nn.functional as F

import PlanetAlign.algorithms.gadl.main as gadl_main
from PlanetAlign.algorithms.gadl.main import GADL


RECONSTRUCTION_MODES = (
    "unweighted_dense_bce",
    "reference_weighted_bce",
)
FUNCTIONAL_MAP_SCHEDULES = (
    "constant",
    "reference_linear_floor_half",
)
ACTIVATION_MODES = (
    "planetalign_prelu",
    "reference_identity",
)


PUBLIC_REFERENCE_COMPATIBLE_PROFILE = {
    "hidden_dim": 512,
    "num_hidden_layers": 3,
    "spectral_rank": 300,
    "lr": 1e-4,
    "weight_decay": 5e-4,
    "descriptor_weight": 1e-3,
    # This follows the requested paper-style alpha/beta mapping.  The public
    # source itself applies beta to an internal C^T C term and leaves its
    # Laplacian commutativity term unweighted; experiments record that mismatch.
    "commutativity_weight": 1e-2,
    "functional_map_weight": 1.0,
    "bijectivity_weight": 0.1,
    "orthogonality_weight": 0.1,
    "residual": False,
    "reconstruction_mode": "reference_weighted_bce",
    "functional_map_schedule": "reference_linear_floor_half",
    "activation_mode": "reference_identity",
}


# 功能：按公开参考实现的正负边权重计算 sigmoid 后的二元交叉熵。
def reference_weighted_reconstruction_loss(
    embeddings: torch.Tensor,
    adjacency: torch.Tensor,
) -> torch.Tensor:
    """Reproduce the public reference's weighted sigmoid-BCE formula."""
    target = adjacency.to_dense() if adjacency.is_sparse else adjacency
    target = target.to(device=embeddings.device, dtype=embeddings.dtype)
    logits = embeddings @ embeddings.T
    predictions = torch.sigmoid(logits)
    positive_count = target.sum()
    total_count = target.new_tensor(target.numel())
    if positive_count <= 0 or positive_count >= total_count:
        raise ValueError("reference weighted BCE requires both positive and negative entries")
    positive_weight = (total_count - positive_count) / positive_count
    normalization = total_count / ((total_count - positive_count) * 2)
    weights = torch.ones_like(target)
    weights[target == 1] = positive_weight
    return normalization * F.binary_cross_entropy(
        predictions.reshape(-1),
        target.reshape(-1),
        weight=weights.reshape(-1),
    )


# 功能：按所选计划给当前训练步的函数映射损失设权重。
def functional_map_schedule_weight(
    schedule: str,
    step: int,
    total_epochs: int,
) -> float:
    if schedule not in FUNCTIONAL_MAP_SCHEDULES:
        raise ValueError(f"unknown functional_map_schedule: {schedule}")
    if total_epochs <= 0:
        raise ValueError("total_epochs must be positive")
    if schedule == "constant" or step <= 0:
        return 1.0
    return max(1.0 * (1.0 - step / total_epochs), 0.5)


# 功能：扩展 GADL 的可选实验配置，保持与基础 train() 相同的调用接口。
class ConfigurableGADL(GADL):
    """GADL with opt-in loss/schedule behavior for reproducibility studies."""

    def __init__(
        self,
        *args,
        reconstruction_mode: str = "unweighted_dense_bce",
        functional_map_schedule: str = "constant",
        activation_mode: str = "planetalign_prelu",
        objective_observer: Optional[Callable] = None,
        **kwargs,
    ):
        if reconstruction_mode not in RECONSTRUCTION_MODES:
            raise ValueError(f"unknown reconstruction_mode: {reconstruction_mode}")
        if functional_map_schedule not in FUNCTIONAL_MAP_SCHEDULES:
            raise ValueError(f"unknown functional_map_schedule: {functional_map_schedule}")
        if activation_mode not in ACTIVATION_MODES:
            raise ValueError(f"unknown activation_mode: {activation_mode}")
        super().__init__(*args, **kwargs)
        self.reconstruction_mode = reconstruction_mode
        self.functional_map_schedule = functional_map_schedule
        self.activation_mode = activation_mode
        self.objective_observer = objective_observer
        self.schedule_history = []

    # 输出结果：创建采用公开参考行为的配置对象，可由调用者覆盖部分参数。
    @classmethod
    def public_reference_compatible(cls, **overrides):
        options = dict(PUBLIC_REFERENCE_COMPATIBLE_PROFILE)
        options.update(overrides)
        return cls(**options)

    # 核心思路：在 ExitStack 管理的范围内替换目标函数和可选激活，再调用父类 GADL.train()。
    # 输出结果：返回父类的矩阵与日志，另记录权重计划；退出范围后恢复被替换的函数。
    def train(
        self,
        dataset,
        gids,
        use_attr: bool = True,
        total_epochs: int = 100,
        save_log: bool = True,
        verbose: bool = True,
    ):
        original_objective = gadl_main.gadl_objective
        objective_call = {"index": 0}
        self.schedule_history = []

        # 输入来源：本函数临时替换 PlanetAlign/algorithms/gadl/main.py 从 model.py 导入的 gadl_objective，父类 GADL.train() 调用时进入这里。
        # 核心思路：先求原始损失，再按配置替换重构项，并将计划权重乘到所有非重构项上。
        # 输出结果：损失字典交回父类训练；另把当前嵌入、损失与已完成轮次传给可选 objective_observer 回调。
        def configured_objective(*args, **kwargs):
            call_index = objective_call["index"]
            objective_call["index"] += 1
            losses = original_objective(*args, **kwargs)
            z1 = kwargs["z1"] if "z1" in kwargs else args[1]
            z2 = kwargs["z2"] if "z2" in kwargs else args[2]
            adjacency1 = kwargs["adjacency1"] if "adjacency1" in kwargs else args[3]
            adjacency2 = kwargs["adjacency2"] if "adjacency2" in kwargs else args[4]

            if call_index == 0:
                step = 0
                state_epoch = 0
                phase = "initial"
            elif call_index <= total_epochs:
                step = call_index
                state_epoch = call_index - 1
                phase = "training"
            else:
                step = total_epochs
                state_epoch = total_epochs
                phase = "final"

            if self.reconstruction_mode == "reference_weighted_bce":
                reconstruction = (
                    reference_weighted_reconstruction_loss(z1, adjacency1)
                    + reference_weighted_reconstruction_loss(z2, adjacency2)
                )
            else:
                reconstruction = losses["reconstruction"]

            schedule_weight = functional_map_schedule_weight(
                self.functional_map_schedule,
                step,
                total_epochs,
            )
            non_reconstruction = losses["total"] - losses["reconstruction"]
            losses = dict(losses)
            losses["reconstruction"] = reconstruction
            losses["functional_map_schedule_weight"] = reconstruction.new_tensor(schedule_weight)
            losses["non_reconstruction_unscaled"] = non_reconstruction
            losses["total"] = reconstruction + schedule_weight * non_reconstruction

            if phase == "training":
                self.schedule_history.append(schedule_weight)
            if self.objective_observer is not None:
                self.objective_observer(
                    state_epoch=state_epoch,
                    phase=phase,
                    call_index=call_index,
                    z1=z1,
                    z2=z2,
                    adjacency1=adjacency1,
                    adjacency2=adjacency2,
                    losses=losses,
                )
            return losses

        # 功能：reference_identity 模式临时用原值替代 PReLU 激活；退出当前 train() 的作用域后恢复。
        def identity_prelu(_module, value):
            return value

        with ExitStack() as stack:
            stack.enter_context(patch.object(gadl_main, "gadl_objective", configured_objective))
            if self.activation_mode == "reference_identity":
                stack.enter_context(patch.object(torch.nn.PReLU, "forward", identity_prelu))
            return super().train(
                dataset,
                gids,
                use_attr=use_attr,
                total_epochs=total_epochs,
                save_log=save_log,
                verbose=verbose,
            )

