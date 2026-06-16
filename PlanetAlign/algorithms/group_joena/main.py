from __future__ import annotations

import os
import time
import warnings
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np
import psutil
import torch

from PlanetAlign.algorithms.base_model import BaseModel
from PlanetAlign.data import Dataset
from PlanetAlign.metrics import hits_ks_scores, many_to_many_scores, mrr_score
from PlanetAlign.utils import get_anchor_pairs

from .assignment import SoftGroupAssignment
from .config import GroupJOENAConfig
from .encoder import NodeEncoder, build_node_input
from .diagnostics import (
    assignment_diagnostics,
    prediction_diagnostics,
    score_diagnostics,
    transport_diagnostics,
)
from .group_aligner import GroupAligner
from .inference import (
    group_prediction_from_alignment,
    node_scores_from_groups,
    predict_entities_from_group_alignment,
)
from .losses import (
    assignment_entropy,
    internal_cohesion_loss,
    reconstruction_loss,
    separation_loss,
    supervised_group_balanced_loss,
)
from .quotient_graph import QuotientGraphBuilder
from .types import GroupAlignmentPrediction


EntityMap = Dict[str, Dict[str, List[int]]]


class GroupJOENA(BaseModel):
    """Group-mediated many-to-many alignment based on JOENA components.

    The model discovers soft node groups on both graphs, builds split-invariant
    quotient graphs, aligns those quotient graphs with a JOENA-style transport
    solver, and exposes node scores through ``S = U_s @ T @ U_t.T``.
    """

    def __init__(self, config: Optional[GroupJOENAConfig] = None, **kwargs: Any):
        if config is not None and kwargs:
            raise ValueError("pass either config or keyword parameters, not both")
        self.config = config if config is not None else GroupJOENAConfig(**kwargs)
        super().__init__(dtype=torch.float32)
        self.source_assignments: Optional[torch.Tensor] = None
        self.target_assignments: Optional[torch.Tensor] = None
        self.group_alignment: Optional[torch.Tensor] = None
        self.source_quotient = None
        self.target_quotient = None
        self.loss_history: List[Dict[str, float]] = []
        self.prediction_: Optional[GroupAlignmentPrediction] = None
        self.num_groups_: Optional[int] = None
        self.oracle_num_groups_: bool = False
        self.training_supervision_: Optional[str] = None
        self.encoder_: Optional[NodeEncoder] = None
        self.assign_src_: Optional[SoftGroupAssignment] = None
        self.assign_tgt_: Optional[SoftGroupAssignment] = None
        self.quotient_builder_: Optional[QuotientGraphBuilder] = None
        self.group_aligner_: Optional[GroupAligner] = None
        self.last_gradient_norms_: Dict[str, float] = {}

    def train(
        self,
        dataset: Dataset,
        gids: Union[Tuple[int, int], List[int]],
        use_attr: bool = True,
        total_epochs: Optional[int] = None,
        gt_entities: Optional[Mapping[str, Mapping[str, Iterable[int]]]] = None,
        train_entities: Optional[Mapping[str, Mapping[str, Iterable[int]]]] = None,
        num_groups: Optional[int] = None,
        oracle_num_groups: bool = False,
        save_log: bool = True,
        verbose: bool = True,
    ):
        """Compatibility wrapper for PlanetAlign's ``BaseModel.train`` API."""

        return self.fit(
            dataset=dataset,
            gids=gids,
            use_attr=use_attr,
            total_epochs=total_epochs,
            gt_entities=gt_entities,
            train_entities=train_entities,
            num_groups=num_groups,
            oracle_num_groups=oracle_num_groups,
            save_log=save_log,
            verbose=verbose,
        )

    def fit(
        self,
        dataset: Dataset,
        gids: Union[Tuple[int, int], List[int]],
        use_attr: bool = True,
        total_epochs: Optional[int] = None,
        gt_entities: Optional[Mapping[str, Mapping[str, Iterable[int]]]] = None,
        train_entities: Optional[Mapping[str, Mapping[str, Iterable[int]]]] = None,
        num_groups: Optional[int] = None,
        oracle_num_groups: bool = False,
        save_log: bool = True,
        verbose: bool = True,
    ):
        """Fit Group-JOENA without using evaluation GT by default.

        ``gt_entities`` is accepted only as a deprecated compatibility alias for
        explicit training supervision when ``supervised_weight > 0``. It is not
        used to infer ``num_groups``. For benchmark oracle-K runs, pass
        ``num_groups=len(test_or_all_entities)`` together with
        ``oracle_num_groups=True`` so the run is labeled honestly.
        """

        if gt_entities is not None:
            warnings.warn(
                "GroupJOENA.fit/train no longer uses gt_entities for group count or inference. "
                "Use train_entities for supervised loss and pass num_groups with "
                "oracle_num_groups=True for oracle-K experiments.",
                UserWarning,
            )
            if train_entities is None and self.config.supervised_weight > 0:
                train_entities = gt_entities
        if self.config.supervised_weight > 0 and train_entities is None:
            raise ValueError("supervised_weight > 0 requires explicit train_entities")

        self.check_inputs(dataset, gids, plain_method=False, use_attr=use_attr, pairwise=True, supervised=True)
        device = self._resolve_device()
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        gid1, gid2 = int(gids[0]), int(gids[1])
        graph1, graph2 = dataset.pyg_graphs[gid1], dataset.pyg_graphs[gid2]
        n1, n2 = int(graph1.num_nodes), int(graph2.num_nodes)
        group_count = self._resolve_num_groups(num_groups, train_entities, dataset, gid1, gid2, n1, n2, oracle_num_groups)
        self.num_groups_ = group_count
        self.oracle_num_groups_ = bool(oracle_num_groups)
        self.training_supervision_ = "train_entities" if train_entities is not None else None
        epochs = int(total_epochs or self.config.max_epochs)

        process = psutil.Process(os.getpid())
        logger = self.init_training_logger(
            dataset,
            use_attr,
            additional_headers=[
                "memory",
                "infer_time",
                "align_loss",
                "structure_loss",
                "cohesion_loss",
                "reconstruction_loss",
                "sparsity_loss",
                "separation_loss",
                "supervised_loss",
                "valid_src_groups",
                "valid_tgt_groups",
                "mean_group_mass",
                "max_group_mass",
                "assignment_entropy",
                "transport_peak",
                "has_nan",
            ],
            save_log=save_log,
        )

        train_pairs = get_anchor_pairs(dataset.train_data, gid1, gid2)
        test_pairs = get_anchor_pairs(dataset.test_data, gid1, gid2)
        x1 = build_node_input(graph1, train_pairs[:, 0], use_attr, self.dtype, device)
        x2 = build_node_input(graph2, train_pairs[:, 1], use_attr, self.dtype, device)
        if x1.shape[1] != x2.shape[1]:
            raise ValueError(
                "GroupJOENA reuses JOENA's shared MLP encoder and requires equal input dimensions; "
                f"got {x1.shape[1]} and {x2.shape[1]}"
            )

        encoder = NodeEncoder(x1.shape[1], self.config.hidden_dim, self.config.out_dim, self.dtype).to(device)
        assign_src = SoftGroupAssignment(
            group_count,
            self.config.out_dim,
            method=self.config.assignment_method,
            temperature=self.config.assignment_temperature,
            seed=self.config.seed,
            dtype=self.dtype,
        ).to(device)
        assign_tgt = SoftGroupAssignment(
            group_count,
            self.config.out_dim,
            method=self.config.assignment_method,
            temperature=self.config.assignment_temperature,
            seed=self.config.seed + 1,
            dtype=self.dtype,
        ).to(device)
        quotient_builder = QuotientGraphBuilder(
            aggregation=self.config.quotient_aggregation,
            soft_or_gamma=self.config.soft_or_gamma,
            eps=self.config.eps,
        ).to(device)
        group_aligner = GroupAligner(
            alpha=self.config.group_alignment_alpha,
            gamma_p=self.config.gamma_p,
            in_iter=self.config.sinkhorn_in_iter,
            out_iter=self.config.sinkhorn_out_iter,
            eps=self.config.eps,
            dtype=self.dtype,
        ).to(device)
        self.encoder_ = encoder
        self.assign_src_ = assign_src
        self.assign_tgt_ = assign_tgt
        self.quotient_builder_ = quotient_builder
        self.group_aligner_ = group_aligner

        with torch.no_grad():
            h1_init, h2_init = encoder(x1, x2)
            assign_src.reset_from_embeddings(h1_init)
            assign_tgt.reset_from_embeddings(h2_init)

        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(assign_src.parameters()) + list(assign_tgt.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        graph1 = graph1.to(device)
        graph2 = graph2.to(device)
        train_t0 = time.time()
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            encoder.train()
            optimizer.zero_grad()
            h1, h2 = encoder(x1, x2)
            u1, u2 = assign_src(h1), assign_tgt(h2)
            q1 = quotient_builder(graph1, h1, u1)
            q2 = quotient_builder(graph2, h2, u2)
            align_result = group_aligner(q1, q2)
            scores = node_scores_from_groups(u1, align_result.transport, u2)

            loss_terms = self._loss_terms(graph1, graph2, u1, u2, q1, q2, align_result, scores, train_entities)
            total_loss = sum(
                loss_terms[key]
                for key in (
                    "align_loss",
                    "cohesion_loss",
                    "reconstruction_loss",
                    "sparsity_loss",
                    "separation_loss",
                    "supervised_loss",
                )
            )
            if not torch.isfinite(total_loss):
                raise FloatingPointError(f"GroupJOENA loss became non-finite at epoch {epoch}: {loss_terms}")
            total_loss.backward()
            self.last_gradient_norms_ = self._gradient_norms(encoder, assign_src, assign_tgt)
            optimizer.step()

            with torch.no_grad():
                self.S = scores.detach().cpu()
                self.source_assignments = u1.detach().cpu()
                self.target_assignments = u2.detach().cpu()
                self.group_alignment = align_result.transport.detach().cpu()
                self.source_quotient = q1
                self.target_quotient = q2

            if epoch % self.config.eval_interval == 0 or epoch == epochs:
                hits = hits_ks_scores(self.S, test_pairs, mode="mean") if len(test_pairs) > 0 else {}
                mrr = mrr_score(self.S, test_pairs, mode="mean") if len(test_pairs) > 0 else 0.0
                ent = 0.5 * (
                    float(assignment_entropy(u1).detach().cpu().item())
                    + float(assignment_entropy(u2).detach().cpu().item())
                )
                has_nan = int(
                    torch.isnan(scores).any().item()
                    or torch.isnan(u1).any().item()
                    or torch.isnan(u2).any().item()
                    or torch.isnan(align_result.transport).any().item()
                )
                loss_log = {key: float(value.detach().cpu().item()) for key, value in loss_terms.items()}
                loss_log["total_loss"] = float(total_loss.detach().cpu().item())
                self.loss_history.append(loss_log)
                logger.log(
                    epoch=epoch,
                    loss=float(total_loss.detach().cpu().item()),
                    epoch_time=time.time() - t0,
                    hits=hits,
                    mrr=mrr,
                    memory=round(process.memory_info().rss / 1024 ** 3, 4),
                    infer_time=round(time.time() - train_t0, 4),
                    align_loss=round(loss_log["align_loss"], 6),
                    structure_loss=round(loss_log["structure_loss"], 6),
                    cohesion_loss=round(loss_log["cohesion_loss"], 6),
                    reconstruction_loss=round(loss_log["reconstruction_loss"], 6),
                    sparsity_loss=round(loss_log["sparsity_loss"], 6),
                    separation_loss=round(loss_log["separation_loss"], 6),
                    supervised_loss=round(loss_log["supervised_loss"], 6),
                    valid_src_groups=int(q1.valid_mask.sum().detach().cpu().item()),
                    valid_tgt_groups=int(q2.valid_mask.sum().detach().cpu().item()),
                    mean_group_mass=round(float((q1.masses.mean() + q2.masses.mean()).detach().cpu().item() / 2), 4),
                    max_group_mass=round(float(torch.maximum(q1.masses.max(), q2.masses.max()).detach().cpu().item()), 4),
                    assignment_entropy=round(ent, 6),
                    transport_peak=round(float(align_result.transport.max().detach().cpu().item()), 6),
                    has_nan=has_nan,
                    verbose=verbose,
                )

        return self.S.detach(), logger

    def _loss_terms(
        self,
        graph1,
        graph2,
        u1: torch.Tensor,
        u2: torch.Tensor,
        q1,
        q2,
        align_result,
        scores: torch.Tensor,
        train_entities: Optional[Mapping[str, Mapping[str, Iterable[int]]]],
    ) -> Dict[str, torch.Tensor]:
        zero = scores.new_zeros(())
        align_loss = align_result.alignment_loss * self.config.alignment_weight
        structure_loss = align_result.structure_loss
        cohesion = (
            internal_cohesion_loss(graph1, u1, self.config.cohesion_threshold, self.config.eps)
            + internal_cohesion_loss(graph2, u2, self.config.cohesion_threshold, self.config.eps)
        ) * 0.5 * self.config.cohesion_weight
        recon = (
            reconstruction_loss(graph1, u1, q1.adjacency, self.config.reconstruction_neg_ratio, self.config.seed, self.config.eps)
            + reconstruction_loss(graph2, u2, q2.adjacency, self.config.reconstruction_neg_ratio, self.config.seed + 1, self.config.eps)
        ) * 0.5 * self.config.reconstruction_weight
        sparse = (assignment_entropy(u1, self.config.eps) + assignment_entropy(u2, self.config.eps)) * 0.5
        sparse = sparse * self.config.sparsity_weight
        sep = (separation_loss(q1.embeddings, q1.valid_mask, self.config.separation_margin)
               + separation_loss(q2.embeddings, q2.valid_mask, self.config.separation_margin)) * 0.5
        sep = sep * self.config.separation_weight
        supervised = zero
        if train_entities is not None and self.config.supervised_weight > 0:
            supervised = supervised_group_balanced_loss(scores, train_entities) * self.config.supervised_weight
        return {
            "align_loss": align_loss,
            "structure_loss": structure_loss.detach(),
            "cohesion_loss": cohesion,
            "reconstruction_loss": recon,
            "sparsity_loss": sparse,
            "separation_loss": sep,
            "supervised_loss": supervised,
        }

    def predict_many_to_many(
        self,
        gt_entities: Mapping[str, Mapping[str, Iterable[int]]],
        target_size_mode: str = "group",
    ) -> EntityMap:
        if self.source_assignments is None or self.target_assignments is None or self.group_alignment is None:
            raise RuntimeError("Model is not trained yet, call fit()/train() before predict_many_to_many()")
        prediction = predict_entities_from_group_alignment(
            gt_entities=gt_entities,
            src_assignments=self.source_assignments,
            tgt_assignments=self.target_assignments,
            transport=self.group_alignment,
            target_size_mode=target_size_mode,
        )
        self.prediction_ = prediction
        return prediction.entities

    def align(
        self,
        min_transport: Optional[float] = None,
    ) -> GroupAlignmentPrediction:
        """Return the GT-free group alignment prediction."""

        if self.source_assignments is None or self.target_assignments is None or self.group_alignment is None:
            raise RuntimeError("Model is not trained yet, call fit()/train() before align()")
        self.prediction_ = group_prediction_from_alignment(
            src_assignments=self.source_assignments,
            tgt_assignments=self.target_assignments,
            transport=self.group_alignment,
            min_transport=min_transport,
        )
        return self.prediction_

    def test_many_to_many(
        self,
        gt_entities: Mapping[str, Mapping[str, Iterable[int]]],
        metrics: Optional[Iterable[str]] = None,
        target_size_mode: str = "group",
    ) -> Dict[str, float]:
        pred = self.predict_many_to_many(gt_entities, target_size_mode=target_size_mode)
        return many_to_many_scores(gt_entities, pred, metrics=metrics)

    def diagnostics(
        self,
        pred_entities: Optional[Mapping[str, Mapping[str, Iterable[int]]]] = None,
        gt_entities: Optional[Mapping[str, Mapping[str, Iterable[int]]]] = None,
    ) -> Dict[str, object]:
        """Return collapse and alignment diagnostics for the latest fit."""

        if self.source_assignments is None or self.target_assignments is None or self.group_alignment is None or self.S is None:
            raise RuntimeError("Model is not trained yet, call fit()/train() before diagnostics()")
        pred = pred_entities
        if pred is None:
            if self.prediction_ is None:
                pred = self.align().entities
            else:
                pred = self.prediction_.entities
        return {
            "num_groups": self.num_groups_,
            "oracle_num_groups": self.oracle_num_groups_,
            "training_supervision": self.training_supervision_,
            "source_assignment": assignment_diagnostics(
                self.source_assignments,
                configured_group_count=self.num_groups_,
                majority_threshold=self.config.collapse_majority_threshold,
                min_non_empty_ratio=self.config.min_non_empty_group_ratio,
                uniform_row_max_multiplier=self.config.uniform_row_max_multiplier,
                eps=self.config.eps,
            ),
            "target_assignment": assignment_diagnostics(
                self.target_assignments,
                configured_group_count=self.num_groups_,
                majority_threshold=self.config.collapse_majority_threshold,
                min_non_empty_ratio=self.config.min_non_empty_group_ratio,
                uniform_row_max_multiplier=self.config.uniform_row_max_multiplier,
                eps=self.config.eps,
            ),
            "transport": transport_diagnostics(
                self.group_alignment,
                effective_threshold=self.config.transport_effective_threshold,
                eps=self.config.eps,
            ),
            "scores": score_diagnostics(
                self.S,
                pred_entities=pred,
                gt_entities=gt_entities,
                threshold=self.config.score_density_threshold,
            ),
            "prediction": prediction_diagnostics(pred, gt_entities=gt_entities),
            "gradient_norms": dict(self.last_gradient_norms_),
        }

    def _resolve_device(self) -> torch.device:
        device = torch.device(self.config.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("GroupJOENA config requested CUDA, but CUDA is not available")
        self.device = device
        return device

    def _resolve_num_groups(
        self,
        num_groups: Optional[int],
        train_entities: Optional[Mapping[str, Mapping[str, Iterable[int]]]],
        dataset: Dataset,
        gid1: int,
        gid2: int,
        n1: int,
        n2: int,
        oracle_num_groups: bool,
    ) -> int:
        if num_groups is not None:
            k = int(num_groups)
        elif self.config.num_groups is not None:
            k = int(self.config.num_groups)
        elif oracle_num_groups and train_entities is not None:
            k = len(train_entities)
        else:
            k = int(get_anchor_pairs(dataset.train_data, gid1, gid2).shape[0])
            warnings.warn(
                "GroupJOENA num_groups was not provided; falling back to the "
                "number of one-to-one training anchors as an estimated-K "
                "heuristic. If you intentionally use the true entity count, "
                "pass num_groups and oracle_num_groups=True.",
                UserWarning,
            )
        if k < 1:
            raise ValueError("GroupJOENA could not resolve a positive num_groups")
        if k > n1 or k > n2:
            raise ValueError(f"num_groups={k} cannot exceed graph sizes ({n1}, {n2})")
        return k

    @staticmethod
    def _module_grad_norm(module: torch.nn.Module) -> float:
        sq_sum = 0.0
        for param in module.parameters():
            if param.grad is None:
                continue
            sq_sum += float(param.grad.detach().pow(2).sum().cpu().item())
        return float(sq_sum ** 0.5)

    def _gradient_norms(
        self,
        encoder: NodeEncoder,
        assign_src: SoftGroupAssignment,
        assign_tgt: SoftGroupAssignment,
    ) -> Dict[str, float]:
        return {
            "encoder": self._module_grad_norm(encoder),
            "assignment_src": self._module_grad_norm(assign_src),
            "assignment_tgt": self._module_grad_norm(assign_tgt),
            "source_prototypes": 0.0 if assign_src.prototypes.grad is None else float(assign_src.prototypes.grad.detach().norm().cpu().item()),
            "target_prototypes": 0.0 if assign_tgt.prototypes.grad is None else float(assign_tgt.prototypes.grad.detach().norm().cpu().item()),
        }
