"""Base-aligner layer for the M2M quotient system.

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
from PlanetAlign.m2m_quotient import (
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


def base_configs() -> Dict[str, Dict[str, Any]]:
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
    }


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
    cfgs = base_configs()
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


def sharpen(S: torch.Tensor, temp: Optional[float]) -> torch.Tensor:
    """Row-softmax sharpening ``softmax(S / T)``; ``temp=None`` returns ``S``."""
    if temp is None:
        return S
    return torch.softmax(S / float(temp), dim=1)


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
