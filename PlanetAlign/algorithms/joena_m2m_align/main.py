from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple, Union

import torch

from PlanetAlign.algorithms.base_model import BaseModel
from PlanetAlign.algorithms.joena import JOENA
from PlanetAlign.algorithms.m2m_align import M2MAlign
from PlanetAlign.data import Dataset


class JOENAM2MAlign(BaseModel):
    """Conservative many-to-many refinement of JOENA with M2MAlign.

    This wrapper keeps JOENA as the base node-level aligner and uses its
    similarity matrix as ``init_S`` for M2MAlign.  It does not modify either
    algorithm's core implementation.  The default M2MAlign ``alpha=0.9`` keeps
    most of JOENA's ranking signal and applies only a light group-aware lift.
    """

    def __init__(
        self,
        joena_alpha: float = 0.7,
        joena_gamma_p: float = 1e-2,
        joena_init_lambda: float = 1.0,
        joena_hid_dim: int = 128,
        joena_out_dim: int = 128,
        joena_lr: float = 1e-4,
        m2m_alpha: float = 0.9,
        m2m_tau: float = 0.95,
        m2m_overlap_slack: float = 0.10,
        m2m_lambda_struct: float = 0.5,
        m2m_beta: float = 0.6,
        m2m_max_group_size: int = 4,
        m2m_n_iter: int = 2,
        m2m_smooth_source: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(dtype=dtype)
        self.joena_kwargs: Dict[str, Any] = {
            "alpha": joena_alpha,
            "gamma_p": joena_gamma_p,
            "init_lambda": joena_init_lambda,
            "hid_dim": joena_hid_dim,
            "out_dim": joena_out_dim,
            "lr": joena_lr,
            "dtype": dtype,
        }
        self.m2m_kwargs: Dict[str, Any] = {
            "alpha": m2m_alpha,
            "tau": m2m_tau,
            "overlap_slack": m2m_overlap_slack,
            "lambda_struct": m2m_lambda_struct,
            "beta": m2m_beta,
            "max_group_size": m2m_max_group_size,
            "n_iter": m2m_n_iter,
            "smooth_source": m2m_smooth_source,
            "dtype": dtype,
        }
        self.joena_: JOENA | None = None
        self.refiner_: M2MAlign | None = None
        self.base_S: torch.Tensor | None = None
        self.timing_: Dict[str, float] = {}

    def train(
        self,
        dataset: Dataset,
        gids: Union[List[int], Tuple[int, ...]],
        use_attr: bool = True,
        total_epochs: int = 10,
        save_log: bool = True,
        verbose: bool = True,
    ):
        self.check_inputs(dataset, gids, plain_method=False, use_attr=use_attr, pairwise=True, supervised=True)
        gid1, gid2 = int(gids[0]), int(gids[1])
        expected_shape = (
            int(dataset.pyg_graphs[gid1].num_nodes),
            int(dataset.pyg_graphs[gid2].num_nodes),
        )

        self.joena_ = JOENA(**self.joena_kwargs).to(self.device)
        t0 = time.perf_counter()
        self.joena_.train(
            dataset=dataset,
            gids=gids,
            use_attr=use_attr,
            total_epochs=total_epochs,
            save_log=save_log,
            verbose=verbose,
        )
        joena_time = time.perf_counter() - t0
        if self.joena_.S is None:
            raise RuntimeError("JOENA did not produce self.S")
        base_s = self.joena_.S.detach().to(self.dtype).cpu()
        if tuple(base_s.shape) != expected_shape:
            raise AssertionError(f"JOENA S shape {tuple(base_s.shape)} != expected {expected_shape}")
        if not torch.isfinite(base_s).all():
            raise FloatingPointError("JOENA S contains NaN or Inf")
        self.base_S = base_s

        self.refiner_ = M2MAlign(**self.m2m_kwargs).to(self.device)
        t1 = time.perf_counter()
        refined_s, logger = self.refiner_.train(
            dataset=dataset,
            gids=gids,
            use_attr=use_attr,
            save_log=save_log,
            verbose=verbose,
            init_S=base_s,
        )
        refine_time = time.perf_counter() - t1
        refined_s = refined_s.detach().to(self.dtype).cpu()
        if tuple(refined_s.shape) != expected_shape:
            raise AssertionError(f"refined S shape {tuple(refined_s.shape)} != expected {expected_shape}")
        if not torch.isfinite(refined_s).all():
            raise FloatingPointError("refined S contains NaN or Inf")

        self.S = refined_s
        self.timing_ = {
            "joena_time_s": float(joena_time),
            "m2m_refine_time_s": float(refine_time),
            "total_time_s": float(joena_time + refine_time),
        }
        return self.S, logger
