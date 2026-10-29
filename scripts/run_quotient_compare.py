# 文件作用：把读图、算法训练、节点分组和评分接起来，比较几种分组办法。
# 数据来源：main() 读取 .pt 的图与训练节点对，以及实体 JSON 中的真实分组答案 gt。
# 核心依赖：--base PARROT 时先 train_base_S()/arbitrate_sharpen()，再调用 PlanetAlign/m2m/quotient.py → quotient_decode() 预测。
# 结果去向：PlanetAlign/m2m/core.py → evaluate_predictions() 用 gt 评分，main() 写 scores、info 到 JSON，不保存 pred。
# 阅读建议：按 CODE_MAP.md 找 main() 中的 Dataset、args.base 分支和 quotient(full)；其他选项见 CODE_MAP_DETAILS.md。

"""Head-to-head M2M readout comparison over the SAME trained aligner.

Trains JOENA once per dataset, freezes its similarity matrix ``S``, then scores
every readout on that same matrix under the blind protocol. Because the
representation is held fixed, any metric difference is attributable to the
readout alone — the controlled experiment behind ``docs/m2m_quotient_align_design.md``.

Readouts
--------
- ``attr-blind``       : attribute-cohesion grouping + greedy match (m2m_blind default)
- ``profile-greedy``   : fixed-tau profile grouping + greedy match (the earliest readout)
- ``quotient``         : full QuotientDecode (Otsu taus, anti-chaining, Hungarian, 2 iters)
- ablations           : quotient minus one component at a time

Usage
-----
    python scripts/run_quotient_compare.py \
        --root data/m2m_overlap_0.05 --dataset douban_m2m
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "planetalign-matplotlib"))

import torch

from PlanetAlign.algorithms import JOENA
from PlanetAlign.data import Dataset
from PlanetAlign.m2m.blind import evaluate_blind, discover_groups_by_profile, decode_entity_map
from PlanetAlign.m2m import align_prediction_to_ground_truth, evaluate_predictions, use_full_anchor_supervision
from PlanetAlign.m2m.quotient import quotient_decode
from PlanetAlign.metrics import InvalidPredictionError, MSF1_PROTOCOL, many_to_many_scores
from PlanetAlign.utils import get_anchor_pairs

GIDS = [0, 1]
M2M_METRICS = ["Macro Set-F1", "Micro Set-F1", "AUIS", "AUCS"]


# 核心思路：调用 PlanetAlign/m2m/blind.py → discover_groups_by_profile() 固定阈值分组，再 decode_entity_map() 贪心匹配。
# 输出结果：MSF1 用原始预测、其他指标用编号适配结果评分；分数字典和两侧组数返回 main()。
def profile_greedy_scores(S, gt, g_src, g_tgt, tau=0.1, *, allow_overlap=False):
    """The earliest readout: fixed-tau profile union-find + greedy match."""
    src_groups = discover_groups_by_profile(S, g_src, tau=tau)
    tgt_groups = discover_groups_by_profile(S.T.contiguous(), g_tgt, tau=tau)
    pred = decode_entity_map(S, src_groups, tgt_groups)
    aligned = align_prediction_to_ground_truth(gt, pred)
    scores = many_to_many_scores(gt, pred, metrics=M2M_METRICS, aligned_pred_entities=aligned,
                                allow_overlap=allow_overlap)
    return scores, (len(src_groups), len(tgt_groups))


# 功能：加载数据、取得 S，比较分组办法，最后保存评分和过程记录；--base PARROT 选择主导读路线。
# 输入输出：图和训练锚点用于训练与预测，gt 只用于评分；结果 JSON 格式由 add() 和末尾写入代码决定。
# 调用关系：预测调用 PlanetAlign/m2m/quotient.py → quotient_decode()；评分调用 PlanetAlign/m2m/core.py → evaluate_predictions()。
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/m2m_overlap_0.05"))
    p.add_argument("--dataset", default="douban_m2m")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.2)
    p.add_argument("--skip-ablations", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--model", choices=["joena", "joena-pc", "auto", "ensemble"], default="joena")
    p.add_argument("--base", default=None,
                   help="use a base aligner (PARROT/BRIGHT/JOENA/...) instead of the "
                        "joena-family model; S is sharpen-arbitrated blind on anchors")
    p.add_argument("--pc-lambda-unif", type=float, default=1.0)
    p.add_argument("--pc-lambda-align", type=float, default=1.0)
    p.add_argument("--pc-no-break-symmetry", action="store_true")
    p.add_argument("--pc-temp", type=float, default=0.1)
    p.add_argument("--pc-group-marginals", action="store_true")
    p.add_argument("--pc-marginal-rho", type=float, default=0.5)
    p.add_argument("--no-full-anchors", action="store_true",
                   help="reproduce the old double-split protocol (~4%% supervision)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    ds = Dataset(root=args.root, name=args.dataset, train_ratio=args.train_ratio, seed=args.seed)
    if not args.no_full_anchors:
        # M2M anchor_links already IS the original 20% train split; entity GT is
        # built from the original test split (disjoint, verified all datasets).
        use_full_anchor_supervision(ds)
    # 数据来源：gt 是 JSON 中的真实分组，提前读入供评分；PARROT 和 quotient_decode() 不接收它。
    with (args.root / f"{args.dataset}_gt_many2many.json").open() as handle:
        gt_payload = json.load(handle)
    gt = gt_payload["entities"]
    allow_overlap = gt_payload.get("metadata", {}).get("overlap_ratio", 0.0) > 0
    g_src, g_tgt = ds.pyg_graphs[GIDS[0]], ds.pyg_graphs[GIDS[1]]
    use_attr = all(g.x is not None for g in (g_src, g_tgt))
    print(f"dataset={args.dataset} nodes={[g_src.num_nodes, g_tgt.num_nodes]} "
          f"entities={len(gt)} use_attr={use_attr}")

    # 功能：按模型名创建 JOENA 或 PlanetAlign/m2m/contrastive.py → JOENAPC，复用缓存或训练。
    # 输出结果：CPU float32 的 S 交回 main()，可直接解码或参与三个分支的矩阵平均。
    def get_S(model_name: str) -> torch.Tensor:
        # "joena-pc-gm" (an auto branch) forces group marginals at the frozen
        # rho; plain "joena-pc" follows the CLI flags (defaults = frozen recipe).
        use_gm = args.pc_group_marginals or model_name == "joena-pc-gm"
        if model_name == "joena":
            tag = ""
        else:
            tag = (f"_pc-u{args.pc_lambda_unif}-a{args.pc_lambda_align}"
                   f"-t{args.pc_temp}{'-nosym' if args.pc_no_break_symmetry else ''}"
                   f"{f'-gm{args.pc_marginal_rho}' if use_gm else ''}")
        fa = "" if args.no_full_anchors else "_fa"
        cache = Path("logs/m2m_diag/S_cache") / f"{args.dataset}{tag}_e{args.epochs}_s{args.seed}{fa}.pt"
        if cache.exists():
            s_ = torch.load(cache, map_location="cpu", weights_only=True)
            print(f"[{model_name}] loaded cached S {list(s_.shape)}")
            return s_
        t0 = time.perf_counter()
        if model_name == "joena":
            aligner = JOENA(alpha=0.7).to("cpu")
        else:
            from PlanetAlign.m2m.contrastive import JOENAPC
            aligner = JOENAPC(alpha=0.7,
                              lambda_unif=args.pc_lambda_unif,
                              lambda_align=args.pc_lambda_align,
                              temp=args.pc_temp,
                              break_symmetry=not args.pc_no_break_symmetry,
                              group_marginals=use_gm,
                              marginal_rho=args.pc_marginal_rho).to("cpu")
        torch.manual_seed(args.seed)
        aligner.train(dataset=ds, gids=GIDS, use_attr=use_attr,
                      total_epochs=args.epochs, save_log=False, verbose=False)
        s_ = aligner.S.detach().to(torch.float32).cpu()
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(s_, cache)
        print(f"[{model_name}] trained in {time.perf_counter()-t0:.1f}s; cached")
        return s_

    # 训练锚点：允许训练和预测使用的已知节点对，来自 ds.train_data，不是 gt。
    anchors = get_anchor_pairs(ds.train_data, GIDS[0], GIDS[1])
    model_tag = "" if args.model == "joena" else f"_{args.model.replace('-', '_')}"

    if args.base is not None:
        # Base-aligner axis: train an alternative base (PARROT wins the
        # weak-attribute datasets; docs §5.11.1), then pick raw vs softmax(S/T)
        # blind on the training anchors. The base becomes a first-class arbitrated
        # hyperparameter of the representation, not a per-dataset hand-pick.
        from PlanetAlign.m2m.base import train_base_S, arbitrate_sharpen
        model_tag = f"_base-{args.base}"
        fa = "" if args.no_full_anchors else "_fa"
        cache = Path("logs/m2m_diag/S_cache") / f"{args.dataset}_base-{args.base}_e-full_s{args.seed}{fa}.pt"
        if cache.exists():
            S_raw = torch.load(cache, map_location="cpu", weights_only=True)
            print(f"[base {args.base}] loaded cached S {list(S_raw.shape)}")
        else:
            t0 = time.perf_counter()
            S_raw = train_base_S(args.base, ds, gids=GIDS, seed=args.seed)
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(S_raw, cache)
            print(f"[base {args.base}] trained in {time.perf_counter()-t0:.1f}s; cached")
        S, T_pick, tbl = arbitrate_sharpen(S_raw, g_src, g_tgt, anchors, use_attr=use_attr)
        print(f"  [base {args.base}] sharpen-arbitrated T={T_pick} "
              f"(corrected agreements: {[(str(t), round(c, 3)) for t, c, _ in tbl]})")
    elif args.model == "ensemble":
        # Selector-free system: average the three branches' couplings (same
        # scale, ~doubly-stochastic) and decode once. Multi-seed measurement:
        # matches the oracle branch on douban/cora/pems08 without any
        # arbitration; airport (one strongly dominant branch) trails by ~0.06.
        S = torch.stack([get_S(m) for m in ["joena", "joena-pc", "joena-pc-gm"]]).mean(dim=0)
        print("  [ensemble] averaged 3 branch couplings")
    elif args.model == "auto":
        # Aligner-level arbitration over three branches, train anchors only.
        # Rule: (>= 20 anchors) keep branches whose chance-corrected agreement
        # is within 0.02 of the best; among those take the strict max raw
        # agreement; if raw also ties, prefer the most refined branch (the
        # anchors then genuinely cannot distinguish them). < 20 anchors ->
        # conservative default = joena (no validated blind signal there;
        # measured: 6 anchors mis-rank, and train-anchor MRR anti-correlates).
        from PlanetAlign.m2m.quotient import entity_anchor_agreement
        BRANCHES = ["joena", "joena-pc", "joena-pc-gm"]   # refined last
        branches = {}
        for m in BRANCHES:
            S_m = get_S(m)
            # 预测：只传矩阵、图和训练锚点，不传真实答案。
            pred, info = quotient_decode(S_m, g_src, g_tgt, use_attr=use_attr, anchors=anchors)
            # 评估：gt 只参与评分，分支选择仍只看下方的训练锚点一致性。
            scores = evaluate_predictions(gt, pred, metrics=M2M_METRICS, allow_overlap=allow_overlap)
            corr, raw = entity_anchor_agreement(pred, anchors, int(g_tgt.num_nodes))
            branches[m] = (S_m, scores, corr, raw)
            print(f"  [branch {m:<12}] Micro Set-F1={scores['Micro Set-F1']:.4f} "
                  f"anchor corrected={corr:.4f} raw={raw:.4f}")
        if anchors.shape[0] < 20:
            chosen = "joena"
        else:
            best_corr = max(b[2] for b in branches.values())
            tied = [m for m in BRANCHES if branches[m][2] >= best_corr - 0.02]
            best_raw = max(branches[m][3] for m in tied)
            tied = [m for m in tied if branches[m][3] >= best_raw - 1e-9]
            chosen = tied[-1]                              # most refined among full ties
        S = branches[chosen][0]
        print(f"  [auto] chose {chosen}")
    else:
        S = get_S(args.model)

    diag_pairs = get_anchor_pairs(ds.test_data if ds.test_data.numel() else ds.train_data,
                                  GIDS[0], GIDS[1])
    hits1 = float(__import__('PlanetAlign').metrics.hits_ks_scores(
        S, diag_pairs, ks=[1], mode="mean").get(1, 0.0))
    tag_h = "in-sample " if not ds.test_data.numel() else ""
    print(f"{tag_h}Hits@1={hits1:.4f}")
    anchors = get_anchor_pairs(ds.train_data, GIDS[0], GIDS[1])
    print(f"train_anchors={anchors.shape[0]}")

    records: List[Dict[str, Any]] = []

    # 功能：将当前解码方式的指标、额外信息和耗时追加到 records，并打印主要分数。
    def add(name: str, scores: Dict[str, float], extra: Dict[str, Any] = None, dt: float = 0.0):
        rec = {"readout": name, "status": "ok", **scores, "decode_s": round(dt, 2), **(extra or {})}
        records.append(rec)
        print(f"  {name:<28} Micro Set-F1={scores['Micro Set-F1']:.4f} "
              f"Macro Set-F1={scores['Macro Set-F1']:.4f} "
              f"AUIS={scores['AUIS']:.4f} AUCS={scores['AUCS']:.4f} ({dt:.1f}s)")

    def record_readout(name, evaluate, started):
        try:
            scores, extra = evaluate()
        except InvalidPredictionError as exc:
            records.append({"readout": name, "status": "invalid_prediction", "error": str(exc),
                            "decode_s": round(time.perf_counter() - started, 2)})
            print(f"  {name:<28} invalid_prediction: {exc}")
            return
        add(name, scores, extra, time.perf_counter() - started)

    # 1) attribute-cohesion blind (the diagnostic's default readout)
    t = time.perf_counter()
    record_readout("attr-blind", lambda: (
        evaluate_blind(S, gt, g_src, g_tgt, metrics=M2M_METRICS, use_attr=use_attr,
                       allow_overlap=allow_overlap), {}), t)

    # 2) fixed-tau profile-greedy readout (tau 0.1)
    t = time.perf_counter()
    def profile_scores():
        sc, counts = profile_greedy_scores(S, gt, g_src, g_tgt, tau=0.1, allow_overlap=allow_overlap)
        return sc, {"groups": counts}
    record_readout("profile-greedy(tau=0.1)", profile_scores, t)

    # 3) QuotientDecode 主线：先预测，再评估；计时仍覆盖两步。
    t = time.perf_counter()
    # 预测阶段：PlanetAlign/m2m/quotient.py → quotient_decode()；不使用真实答案 gt。
    pred, info = quotient_decode(S, g_src, g_tgt, use_attr=use_attr, anchors=anchors)
    # 评估阶段：PlanetAlign/m2m/core.py → evaluate_predictions()；用 gt 对应编号并评分。
    record_readout("quotient(full)", lambda: (
        evaluate_predictions(gt, pred, metrics=M2M_METRICS, allow_overlap=allow_overlap),
        {"info": info}), t)

    if not args.skip_ablations:
        for name, kw in [
            ("quotient - hungarian",    dict(matcher="greedy", anchors=anchors)),
            ("quotient - anti-chain",   dict(anti_chaining=False, anchors=anchors)),
            ("quotient - iteration",    dict(max_iters=1, anchors=anchors)),
            ("quotient - otsu(tau=.1)", dict(tau=0.1, anchors=anchors)),
            ("quotient - evidence-sel", dict(evidence_selection=False, anchors=anchors)),
            ("quotient - anchors",      dict(anchors=None)),
            ("quotient - null-cand",    dict(null_candidate=False, anchors=anchors)),
            ("quotient + global-cand",  dict(global_candidate=True, anchors=anchors)),
            ("quotient + overlap(.35)", dict(overlap_expand_tau=0.35, anchors=anchors)),
            ("quotient + fgw(a.1)",     dict(matcher="fgw", fgw_alpha=0.1, anchors=anchors)),
        ]:
            t = time.perf_counter()
            # 每个对照配置也遵循先预测、后评分的顺序；参数与主线保持原来的设置。
            pred, info = quotient_decode(S, g_src, g_tgt, use_attr=use_attr, **kw)
            record_readout(name, lambda: (
                evaluate_predictions(gt, pred, metrics=M2M_METRICS, allow_overlap=allow_overlap),
                {"info": info}), t)

    out = args.out or Path(f"logs/m2m_diag/quotient_compare_{args.dataset}{model_tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"dataset": args.dataset, "root": str(args.root), "seed": args.seed,
                   "epochs": args.epochs, "msf1_protocol": MSF1_PROTOCOL,
                   "allow_overlap": allow_overlap, "records": records}, f, ensure_ascii=False, indent=2,
                  default=str)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
