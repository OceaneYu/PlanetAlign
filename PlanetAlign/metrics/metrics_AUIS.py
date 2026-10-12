# 文件作用：计算 AUIS，观察真实组的成员是否仍集中在预测组中。
# 数据来源：PlanetAlign/metrics/many_to_many.py → many_to_many_scores() 调用 auis_score() 并传入实体字典。
# 核心处理：将 src/tgt 节点转换为带侧标记的集合，统计真实组与全部预测组的重叠；不要求组编号一致。
# 阅读建议：先读 auis_score() → _auis_score_details()；weighted 版本是按组大小加权的补充诊断。

import json
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
import re
import matplotlib.pyplot as plt

"""
Alignment Unit Integrity Score (AUIS) 指标实现

思想：
- GT 中每个 entity 视为一个真实群组 g
- Pred 中每个 entity 视为一个预测簇 C_j
- 对每个 g，统计它在所有预测簇中的分布 a_j^(g) = |g ∩ C_j|
- R(g) = sum_j (a_j^(g))^2
- AUIS(g) = (R(g) - N_g) / (N_g^2 - N_g),  当 N_g > 1
- 最终输出：
    1) auis: 所有 group 的简单平均
    2) weighted_auis: 按 group 大小 N_g 加权平均

AUIS 衡量 GT alignment unit 在所有预测簇中的集中程度。
"""


Node = Tuple[str, int]  # 例如 ("src", 13748), ("tgt", 13391)


def load_entities(path: Path) -> Dict[str, Dict[str, List[int]]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "entities" not in data:
        raise ValueError(f"{path} 缺少 'entities' 字段")
    return data["entities"]


def infer_network_keys(gt_entities: Dict[str, Dict[str, List[int]]]) -> List[str]:
    for _, v in gt_entities.items():
        keys = [k for k, vv in v.items() if isinstance(vv, list)]
        if not keys:
            raise ValueError("entities 中没有 list 字段（例如 src/tgt）")
        return sorted(keys)
    raise ValueError("GT entities 为空")


def entity_to_node_set(entity: Dict[str, List[int]], network_keys: List[str]) -> Set[Node]:
    """
    把一个 entity 转成带网络侧标识的节点集合。
    例如：
      {"src":[1,2], "tgt":[3]}
    ->
      {("src",1), ("src",2), ("tgt",3)}

    为什么要带上 "src"/"tgt"？
    因为 src 的 100 和 tgt 的 100 不是同一个节点。
    """
    nodes: Set[Node] = set()
    for k in network_keys:
        for nid in entity.get(k, []):
            nodes.add((k, int(nid)))
    return nodes


def build_group_sets(
    entities: Dict[str, Dict[str, List[int]]],
    network_keys: List[str]
) -> Dict[str, Set[Node]]:
    """
    把 entities 转成：
      eid -> 节点集合
    """
    out: Dict[str, Set[Node]] = {}
    for eid, item in entities.items():
        out[eid] = entity_to_node_set(item, network_keys)
    return out


# 功能：按真实组统计与预测组的重叠，返回每组详情、宏平均值和按组大小加权值。
def _auis_score_details(
    gt_entities: Dict[str, Dict[str, List[int]]],
    pred_entities: Dict[str, Dict[str, List[int]]],
    network_keys: List[str] = None,
) -> Dict[str, Any]:
    """
    计算 AUIS

    对每个 GT 群组 g:
      a_j^(g) = | g ∩ C_j |
      R(g)    = Σ_j (a_j^(g))^2
      N_g    = |g|
      AUIS(g) = (R(g)-N_g)/(N_g^2-N_g), N_g>1; 否则记为1

    最终返回：
      - auis
      - weighted_auis
      - 每个 group 的详细信息
    """
    if network_keys is None:
        network_keys = infer_network_keys(gt_entities)

    gt_groups = build_group_sets(gt_entities, network_keys)
    pred_clusters = build_group_sets(pred_entities, network_keys)

    if not gt_groups:
        return {
            "auis": 0.0,
            "weighted_auis": 0.0,
            "num_groups": 0.0,
            "total_gt_nodes": 0.0,
            "per_group": {},
            "network_keys": network_keys,
        }

    per_group: Dict[str, Any] = {}
    macro_vals = []
    weighted_num = 0.0
    weighted_den = 0.0
    total_gt_nodes = 0

    for gid, g_nodes in gt_groups.items():
        N_g = len(g_nodes)
        total_gt_nodes += N_g

        counts_by_pred: Dict[str, int] = {}
        integrity_raw = 0

        # 遍历所有预测簇，看 GT group 在哪些簇里出现，以及各出现多少次
        for pid, pred_nodes in pred_clusters.items():
            overlap = len(g_nodes & pred_nodes)
            if overlap > 0:
                counts_by_pred[pid] = overlap
                integrity_raw += overlap * overlap

        # 归一化
        if N_g <= 1:
            auis_g = 1.0
        else:
            denom = N_g * N_g - N_g
            auis_g = (integrity_raw - N_g) / denom if denom != 0 else 1.0
            # 数值安全截断
            auis_g = max(0.0, min(1.0, auis_g))

        macro_vals.append(auis_g)
        weighted_num += auis_g * N_g
        weighted_den += N_g

        per_group[gid] = {
            "N_g": float(N_g),
            "IntegritySum_g": float(integrity_raw),
            "AUIS_g": float(auis_g),
            "NumTouchedPredClusters": float(len(counts_by_pred)),
            "CountsByPred": counts_by_pred,
        }

    auis = sum(macro_vals) / len(macro_vals) if macro_vals else 0.0
    weighted_auis = weighted_num / weighted_den if weighted_den != 0 else 0.0

    return {
        "auis": float(auis),
        "weighted_auis": float(weighted_auis),
        "num_groups": float(len(gt_groups)),
        "total_gt_nodes": float(total_gt_nodes),
        "per_group": per_group,
        "network_keys": network_keys,
    }


# 功能：批量读取预测文件并计算 AUIS，将宏平均与加权分数写为曲线和 CSV。
def evaluate_pred_folder(
    gt_path: Path,
    pred_dir: Path,
    out_png: Path,
    file_prefix: str = "pred_err_"
) -> None:
    """
    遍历 pred_dir 下所有 pred_err_*.json，
    计算每个预测文件的 AUIS，并按误差率画图。
    """
    gt_entities = load_entities(gt_path)
    network_keys = infer_network_keys(gt_entities)

    pred_files = sorted(
        [p for p in pred_dir.iterdir() if p.name.startswith(file_prefix) and p.suffix == ".json"]
    )
    if not pred_files:
        raise ValueError(f"{pred_dir} 下找不到 {file_prefix}*.json")

    xs = []  # 误差率
    ys_macro = []  # macro AUIS
    ys_weighted = []  # weighted AUIS
    records = []

    print(f"GT: {gt_path}")
    print(f"Pred dir: {pred_dir}")
    print(f"Networks: {network_keys}")
    print("-" * 100)

    for pf in pred_files:
        pred_entities = load_entities(pf)
        res = _auis_score_details(gt_entities, pred_entities, network_keys)

        # 从文件名 pred_err_0.2.json 解析出 0.2
        m = re.search(r"pred_err_([0-9.]+)\.json", pf.name)
        err = float(m.group(1)) if m else float("nan")

        xs.append(err)
        ys_macro.append(res["auis"])
        ys_weighted.append(res["weighted_auis"])

        records.append({
            "file": pf.name,
            "error_rate": err,
            "auis": res["auis"],
            "weighted_auis": res["weighted_auis"],
            "num_groups": res["num_groups"],
            "total_gt_nodes": res["total_gt_nodes"],
        })

        print(
            f"{pf.name:20s}  "
            f"macro={res['auis']:.4f}  "
            f"weighted={res['weighted_auis']:.4f}  "
            f"groups={int(res['num_groups'])}  "
            f"nodes={int(res['total_gt_nodes'])}"
        )

    # 按误差率排序
    pairs = sorted(zip(xs, ys_macro, ys_weighted), key=lambda t: t[0])
    xs, ys_macro, ys_weighted = zip(*pairs)

    # 保存 csv
    out_csv = out_png.with_suffix(".csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("error_rate,auis,weighted_auis\n")
        for x, y1, y2 in zip(xs, ys_macro, ys_weighted):
            f.write(f"{x:.10f},{y1:.10f},{y2:.10f}\n")

    # 画图：一张图两条线，不指定颜色
    plt.figure()
    plt.plot(xs, ys_macro, marker="o", label="Macro AUIS")
    plt.plot(xs, ys_weighted, marker="s", label="Weighted AUIS")
    plt.xlabel("Error rate")
    plt.ylabel("AUIS")
    plt.title("AUIS vs Error rate")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved csv: {out_csv}")
    print(f"Saved plot: {out_png}")


# 输出结果：返回 AUIS 的宏平均标量，供统一指标入口汇总。
def auis_score(
    gt_entities: Dict[str, Dict[str, List[int]]],
    pred_entities: Dict[str, Dict[str, List[int]]],
) -> float:
    """Compute the macro Alignment Unit Integrity Score."""
    return _auis_score_details(gt_entities, pred_entities)["auis"]


# 输出结果：返回按真实组节点数加权的 AUIS，作为显式请求的补充指标。
def weighted_auis_score(
    gt_entities: Dict[str, Dict[str, List[int]]],
    pred_entities: Dict[str, Dict[str, List[int]]],
) -> float:
    """Compute the existing size-weighted AUIS diagnostic."""
    return _auis_score_details(gt_entities, pred_entities)["weighted_auis"]


if __name__ == "__main__":
    BASE = Path(__file__).resolve().parent
    ROOT = BASE.parent

    DATASETS = ["Facebook-Twitter", "Arxiv1-Arxiv2", "DBLP1-DBLP2"]

    # 统一图输出目录
    PLOT_DIR = ROOT / "outputs" / "plots" / "m2m_auis"
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    for DATASETNAME in DATASETS:
        # GT 从 data-many2many 读取
        DATASET_DIR = ROOT / "data" / "data-many2many" / DATASETNAME
        GT_PATH = DATASET_DIR / "gt_many2many.json"

        # 预测从 outputs/pred_sims/<DATASETNAME> 读取
        PRED_DIR = ROOT / "outputs" / "pred_sims" / DATASETNAME

        # 图输出到统一目录
        OUT_PNG = PLOT_DIR / f"AUIS_curve_{DATASETNAME}.png"

        print(f"\n=== Evaluating dataset: {DATASETNAME} ===")
        evaluate_pred_folder(GT_PATH, PRED_DIR, OUT_PNG)
        print(f"Saved plot: {OUT_PNG}")
