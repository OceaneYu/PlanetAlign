# 文件作用：比较 PARROT 与三个 JOENA 分支的缓存矩阵平均，检查训练锚点能否选出更好的结果。
# 数据来源：main() 加载 Dataset，ensemble_S() 读取 scripts/run_quotient_compare.py 生成的三个 full-anchor 缓存。
# 核心依赖：PlanetAlign/m2m/base.py 的基础训练与矩阵处理；PlanetAlign/m2m/quotient.py 的预测、锚点一致性和评分。
# 结果去向：各数据集与种子的比较记录写入 --out，供 scripts/aggregate_unified_grid.py 读取。

"""Head-to-head: PARROT vs the JOENA-family ensemble, arbitrated on anchors.

For each strong-attribute dataset, reconstruct the 3-branch ensemble coupling
from the cached component matrices, score it (anchor agreement + blind MicroF1),
and put it beside PARROT. The question the unified system must answer blind:
does ``argmax(corrected anchor agreement)`` pick the base with the higher blind
MicroF1? A ``MISMATCH`` marks a dataset where anchor agreement saturates and
cannot discriminate (the §5.8 failure mode, now at the base level).
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "planetalign-matplotlib"))

import torch

from PlanetAlign.data import Dataset
from PlanetAlign.m2m import evaluate_predictions, use_full_anchor_supervision
from PlanetAlign.m2m.base import train_base_S, arbitrate_sharpen
from PlanetAlign.m2m.quotient import entity_anchor_agreement, quotient_decode
from PlanetAlign.utils import get_anchor_pairs

GIDS = (0, 1)
CACHE = Path("logs/m2m_diag/S_cache")
STRONG = ["douban_m2m", "cora_m2m", "airport_m2m", "pems08_m2m", "ppi_m2m"]


# 输出结果：三个分支缓存齐全时返回平均矩阵，否则返回 None；不在这里训练缺失分支。
def ensemble_S(dataset_name: str, seed: int) -> torch.Tensor | None:
    """Mean of the 3 cached branch couplings (joena, joena-pc, joena-pc-gm), _fa."""
    tags = ["", "_pc-u1.0-a1.0-t0.1", "_pc-u1.0-a1.0-t0.1-gm0.5"]
    mats = []
    for t in tags:
        p = CACHE / f"{dataset_name}{t}_e10_s{seed}_fa.pt"
        if not p.exists():
            return None
        mats.append(torch.load(p, map_location="cpu", weights_only=True))
    return torch.stack(mats).mean(dim=0)


# 功能：按选项做矩阵温度选择，预测实体并分别计算锚点一致性与真实实体 MicroF1。
def score(S, g_src, g_tgt, anchors, gt, has_attr):
    S_best, T, _ = arbitrate_sharpen(S, g_src, g_tgt, anchors, use_attr=has_attr)
    pred, _ = quotient_decode(S_best, g_src, g_tgt, use_attr=has_attr, anchors=anchors)
    corrected, raw = entity_anchor_agreement(pred, anchors, int(g_tgt.num_nodes))
    mf1 = evaluate_predictions(gt, pred, metrics=["MicroF1"])["MicroF1"]
    return corrected, mf1, T


# 功能：逐数据集/种子训练 PARROT，加载集成缓存，用 score() 比较并保存结果。
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/m2m_no_overlap"))
    p.add_argument("--datasets", nargs="+", default=STRONG)
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--out", type=Path, default=Path("logs/m2m_diag/parrot_vs_ensemble.json"))
    args = p.parse_args()

    hdr = (f"{'dataset':<14}{'seed':>5}{'PARROT corr':>13}{'PARROT mf1':>12}"
           f"{'ENS corr':>10}{'ENS mf1':>9}   anchor-pick / oracle-pick")
    print(hdr + "\n" + "-" * len(hdr))
    n_mismatch = 0
    results = []
    for name in args.datasets:
        for seed in args.seeds:
            ds = Dataset(root=args.root, name=name, train_ratio=0.2, seed=seed)
            use_full_anchor_supervision(ds)
            gt = json.load(open(args.root / f"{name}_gt_many2many.json"))["entities"]
            g_src, g_tgt = ds.pyg_graphs[0], ds.pyg_graphs[1]
            has_attr = all(g.x is not None for g in (g_src, g_tgt))
            anchors = get_anchor_pairs(ds.train_data, 0, 1)

            S_p = train_base_S("PARROT", ds, seed=seed)
            pc, pm, _ = score(S_p, g_src, g_tgt, anchors, gt, has_attr)

            S_e = ensemble_S(name, seed)
            if S_e is None:
                print(f"{name:<14}{seed:>5}{pc:>13.4f}{pm:>12.4f}{'(no cache)':>19}")
                results.append(dict(dataset=name, seed=seed, parrot_corr=round(pc, 4),
                                    parrot_mf1=round(pm, 4), ens=None))
                continue
            ec, em, _ = score(S_e, g_src, g_tgt, anchors, gt, has_attr)

            apick = "PARROT" if pc >= ec else "ENSEMBLE"
            opick = "PARROT" if pm >= em else "ENSEMBLE"
            flag = "" if apick == opick else "  <-- MISMATCH"
            if apick != opick:
                n_mismatch += 1
            print(f"{name:<14}{seed:>5}{pc:>13.4f}{pm:>12.4f}{ec:>10.4f}{em:>9.4f}   "
                  f"{apick} / {opick}{flag}")
            results.append(dict(dataset=name, seed=seed, parrot_corr=round(pc, 4),
                                parrot_mf1=round(pm, 4), ens_corr=round(ec, 4),
                                ens_mf1=round(em, 4), anchor_pick=apick, oracle_pick=opick))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nmismatches: {n_mismatch}\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
