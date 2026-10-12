# 文件作用：把算法结果统一成节点配对分数表 S，并用训练时已知的节点对选择分数表的版本。
# 数据来源：scripts/run_quotient_compare.py → main() 给 train_base_S() 传 Dataset，给 arbitrate_sharpen() 传矩阵、图与 anchors。
# 核心依赖：PARROT 分支调用 PlanetAlign/algorithms/parrot.py → PARROT.train()；候选比较调用 PlanetAlign/m2m/quotient.py → quotient_decode()。
# 结果去向：分数表返回入口，再由入口交给 PlanetAlign/m2m/quotient.py → evaluate_quotient_blind()。
# 阅读建议：先看 train_base_S() 怎样创建算法并提取 S，再看 arbitrate_sharpen() 怎样选择版本。

"""
把不同基础对齐算法统一成相似度矩阵 S，再用训练锚点选择合适的矩阵处理方式或基础算法。

Base-aligner layer for the M2M quotient system.

QuotientDecode is aligner-agnostic: it consumes a similarity matrix ``S`` and
never touches the aligner internals. Which aligner produces ``S`` is therefore
a *hyperparameter of the representation*, and — like evidence selection and the
sharpening temperature — it can be chosen blind on the training anchors instead
of hand-picked per dataset.

Two families of ``S`` reach the decoder:

* **coupling-type** (JOENA, PARROT, IsoRank, FINAL): an OT/consistency matrix
  whose rows already live on a comparable scale. The profile-cosine evidence is
  used raw.
* **embedding-cosine-type** (BRIGHT, REGAL, ...): ``cos(z_i, z_j)``. Empirically
  the raw cosine is too flat for profile clustering; a row-softmax
  ``softmax(S / T)`` sharpens it. ``T`` is not tuned on the metric — it is
  arbitrated by chance-corrected anchor agreement, exactly like every other
  blind choice in the system (arenas post-mortem, docs §5.11.1).

The lesson that motivated this module: probing an *incomplete* roster at
*quick* budgets condemned four datasets ("infeasible"); adding PARROT and honest
budgets rescued all four. The base is now a first-class, arbitrated axis so that
mistake cannot recur silently — :func:`arbitrate_base` scores every base on the
anchors and reports the ranking.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

import PlanetAlign
from PlanetAlign.m2m.quotient import (
    entity_anchor_agreement,
    evaluate_quotient_blind,
    quotient_decode,
)
from PlanetAlign.utils import pairwise_cosine_similarity

GIDS_DEFAULT = (0, 1)

# Temperatures probed for embedding-cosine sharpening. ``None`` = raw S. The
# grid brackets the values the arbiter has historically selected (BRIGHT@arenas
# chose 0.005); raw is always a candidate so a coupling-type S is left untouched
# when sharpening does not help.
SHARPEN_TEMPS: Tuple[Optional[float], ...] = (None, 0.5, 0.1, 0.05, 0.01, 0.005)


# 功能：为每个基础算法提供创建方式、训练参数和结果提取方式，供 train_base_S() 使用。
# 输出结果：配置中的 mode 区分直接矩阵、self.S 和嵌入向量，避免混淆算法返回值。
def base_configs(seed: int = 0) -> Dict[str, Dict[str, Any]]:
    """Base-aligner recipes. ``mode`` says how to read ``S`` off the trained model.

    ``kind`` records the S family so callers can default the sharpening policy:
    ``coupling`` bases skip sharpening by default, ``embedding`` bases enable it.
    """
    A = PlanetAlign.algorithms
    return {
        "IsoRank":   dict(factory=lambda: A.IsoRank(alpha=0.4),
                          kwargs=dict(use_attr=False, total_epochs=100),
                          mode="S", kind="coupling"),
        "FINAL":     dict(factory=lambda: A.FINAL(alpha=0.9),
                          kwargs=dict(use_attr=True, total_epochs=100),
                          mode="self.S", kind="coupling"),
        "PARROT":    dict(factory=lambda: A.PARROT(alpha=0.5),
                          kwargs=dict(use_attr=True),
                          mode="self.S", kind="coupling"),
        "REGAL":     dict(factory=lambda: A.REGAL(),
                          kwargs=dict(use_attr=True),
                          mode="embs-cos", kind="embedding"),
        "BRIGHT":    dict(factory=lambda: A.BRIGHT(),
                          kwargs=dict(use_attr=True, total_epochs=100),
                          mode="embs-cos", kind="embedding"),
        "JOENA":     dict(factory=lambda: A.JOENA(alpha=0.7),
                          kwargs=dict(use_attr=True, total_epochs=100),
                          mode="self.S", kind="coupling"),
        "GADL":      dict(factory=lambda: A.GADL(seed=seed),
                          kwargs=dict(use_attr=True, total_epochs=100),
                          mode="self.S", kind="embedding"),
    }


# 输入来源：train_base_S() 传入训练后的算法对象、train() 返回值和配置中的 mode。
# 输出结果：统一返回 CPU float32 矩阵；嵌入模式调用 PlanetAlign/utils/distance.py → pairwise_cosine_similarity() 转成节点分数。
def _extract_s(algo, ret, mode: str, gids=GIDS_DEFAULT) -> torch.Tensor:
    if not isinstance(ret, tuple):
        ret = (ret,)
    if mode == "self.S":
        return algo.S.detach().to(torch.float32).cpu()
    if mode == "S":
        return ret[0].detach().to(torch.float32).cpu()
    if mode == "embs-cos":
        return pairwise_cosine_similarity(ret[0].detach().to(torch.float32).cpu(),
                                          ret[1].detach().to(torch.float32).cpu())
    if mode == "embs-dot-dict":
        d = ret[0]
        return (d[gids[0]].detach().to(torch.float32).cpu()
                @ d[gids[1]].detach().to(torch.float32).cpu().T)
    raise ValueError(f"unknown extract mode {mode!r}")


# 流程位置：入口 scripts/run_quotient_compare.py → main() 没有找到矩阵缓存时调用，负责启动基础算法。
# 输入来源：入口传入 Dataset；gids 选出要比较的两张图。
# 功能：按 base_configs() 创建算法并调用 train()，再用 _extract_s() 取出分数表。
# 输出结果：S 交回入口成为 S_raw；分组与评分由入口安排，不在这里执行。
def train_base_S(name: str,
                 dataset,
                 gids=GIDS_DEFAULT,
                 seed: int = 42,
                 use_attr: Optional[bool] = None) -> torch.Tensor:
    """Train base aligner ``name`` and return its similarity matrix ``S``.

    ``use_attr=None`` follows the config but downgrades to ``False`` when a graph
    has no node features (weak-attribute datasets: arenas, phone-email, italy,
    foursquare).
    """
    cfgs = base_configs(seed=seed)
    if name not in cfgs:
        raise KeyError(f"unknown base {name!r}; known: {sorted(cfgs)}")
    cfg = cfgs[name]
    kw = dict(cfg["kwargs"])
    g_src, g_tgt = dataset.pyg_graphs[gids[0]], dataset.pyg_graphs[gids[1]]
    has_attr = all(g.x is not None for g in (g_src, g_tgt))
    if use_attr is not None:
        kw["use_attr"] = use_attr
    if kw.get("use_attr") and not has_attr:
        kw["use_attr"] = False
    torch.manual_seed(seed)
    algo = cfg["factory"]().to("cpu")
    ret = algo.train(dataset=dataset, gids=list(gids), save_log=False, verbose=False, **kw)
    return _extract_s(algo, ret, cfg["mode"], gids)


# 功能：按行执行 softmax，使每个节点的高分匹配更突出；temp 为 None 时保留原矩阵。
# 输出结果：候选矩阵交给 arbitrate_sharpen() 比较。
def sharpen(S: torch.Tensor, temp: Optional[float]) -> torch.Tensor:
    """Row-softmax sharpening ``softmax(S / T)``; ``temp=None`` returns ``S``."""
    if temp is None:
        return S
    return torch.softmax(S / float(temp), dim=1)


# 输入来源：入口或 arbitrate_base() 传入 S、两侧图和训练锚点 anchors，不接收真实分组答案。
# 核心思路：试用原表和突出高分的版本，调用 PlanetAlign/m2m/quotient.py → quotient_decode() 分组，再用 entity_anchor_agreement() 检查已知节点对。
# 输出结果：选中的 S、处理温度和比较表返回调用方；实验入口接着调用 evaluate_quotient_blind()。
def arbitrate_sharpen(S: torch.Tensor,
                      g_src,
                      g_tgt,
                      anchors: torch.Tensor,
                      use_attr: bool = True,
                      temps: Tuple[Optional[float], ...] = SHARPEN_TEMPS,
                      decode_kwargs: Optional[Dict[str, Any]] = None,
                      ) -> Tuple[torch.Tensor, Optional[float], List[Tuple[Optional[float], float, float]]]:
    """Pick the sharpening temperature blind, by chance-corrected anchor agreement.

    For each candidate ``T`` (including raw), decode the entity map and score it
    with :func:`entity_anchor_agreement` — the same task-supervision signal used
    everywhere else in the system, never the entity GT. Returns
    ``(S_best, temp_best, table)`` where ``table`` is
    ``[(temp, corrected, raw), ...]`` for logging. Ties in corrected agreement
    prefer *less* transformation (raw first, then larger ``T``) — Occam on the
    representation.
    """
    decode_kwargs = dict(decode_kwargs or {})
    n2 = int(g_tgt.num_nodes)
    table: List[Tuple[Optional[float], float, float]] = []
    best = None  # (corrected, tie_rank, temp, S_cand)
    for rank, T in enumerate(temps):
        S_cand = sharpen(S, T)
        pred, _ = quotient_decode(S_cand, g_src, g_tgt, use_attr=use_attr,
                                  anchors=anchors, **decode_kwargs)
        corrected, raw = entity_anchor_agreement(pred, anchors, n2)
        corr = -1.0 if corrected != corrected else corrected  # NaN -> worst
        table.append((T, corrected, raw))
        key = (corr, -rank)  # higher corrected wins; ties -> earlier (less sharp)
        if best is None or key > best[0]:
            best = (key, T, S_cand)
    return best[2], best[1], table


# 功能：逐一训练并处理候选基础算法，用训练锚点决定采用哪一个。
# 核心思路：调用 train_base_S()、arbitrate_sharpen()，再用 PlanetAlign/m2m/quotient.py → entity_anchor_agreement() 排序。
# 输出结果：返回算法名称、矩阵和比较表；gt_entities 只传给 evaluate_quotient_blind() 报告分数，不参与排序。
def arbitrate_base(dataset,
                   bases: List[str],
                   gt_entities,
                   gids=GIDS_DEFAULT,
                   seed: int = 42,
                   use_attr: Optional[bool] = None,
                   temps: Tuple[Optional[float], ...] = SHARPEN_TEMPS,
                   decode_kwargs: Optional[Dict[str, Any]] = None,
                   verbose: bool = False,
                   ) -> Dict[str, Any]:
    """Train every base, sharpen-arbitrate each, and rank bases by anchor agreement.

    The winning base/temperature is chosen entirely on the training anchors
    (chance-corrected entity agreement). ``gt_entities`` is used only to *report*
    the blind MicroF1 that the chosen configuration would score — it never enters
    the selection. Returns a dict with the chosen base, its S, and the full table.
    """
    from PlanetAlign.utils import get_anchor_pairs

    decode_kwargs = dict(decode_kwargs or {})
    g_src, g_tgt = dataset.pyg_graphs[gids[0]], dataset.pyg_graphs[gids[1]]
    has_attr = all(g.x is not None for g in (g_src, g_tgt))
    anchors = get_anchor_pairs(dataset.train_data, gids[0], gids[1])
    n2 = int(g_tgt.num_nodes)

    rows: List[Dict[str, Any]] = []
    for name in bases:
        S = train_base_S(name, dataset, gids=gids, seed=seed, use_attr=use_attr)
        S_best, T, tbl = arbitrate_sharpen(S, g_src, g_tgt, anchors,
                                           use_attr=has_attr, temps=temps,
                                           decode_kwargs=decode_kwargs)
        pred, _ = quotient_decode(S_best, g_src, g_tgt, use_attr=has_attr,
                                  anchors=anchors, **decode_kwargs)
        corrected, raw = entity_anchor_agreement(pred, anchors, n2)
        blind = evaluate_quotient_blind(S_best, gt_entities, g_src, g_tgt,
                                        metrics=["MicroF1"], use_attr=has_attr,
                                        anchors=anchors, **decode_kwargs)["MicroF1"]
        rows.append(dict(base=name, temp=T, corrected=corrected, raw=raw,
                         blind_microf1=blind, S=S_best))
        if verbose:
            print(f"  [base {name:<8}] T={str(T):<6} anchor corrected={corrected:.4f} "
                  f"raw={raw:.4f} | blind MicroF1={blind:.4f}")

    ranked = sorted(rows, key=lambda r: (r["corrected"], r["raw"]), reverse=True)
    chosen = ranked[0]
    return dict(chosen=chosen["base"], temp=chosen["temp"], S=chosen["S"],
                corrected=chosen["corrected"], blind_microf1=chosen["blind_microf1"],
                table=[{k: v for k, v in r.items() if k != "S"} for r in ranked])
