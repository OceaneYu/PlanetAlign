import json
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
import re
import matplotlib.pyplot as plt

"""
M2M-SGS 指标实现

思想：
- GT 中每个 entity 视为一个真实群组 g
- Pred 中每个 entity 视为一个预测簇 C_j
- 对每个 g，统计它在所有预测簇中的分布 a_j^(g) = |g ∩ C_j|
- SGS(g) = sum_j (a_j^(g))^2
- M2M-SGS(g) = (SGS(g) - N_g) / (N_g^2 - N_g),  当 N_g > 1
- 最终输出：
    1) macro_m2m_sgs: 所有 group 的简单平均
    2) weighted_m2m_sgs: 按 group 大小 N_g 加权平均

注意：
这里不能像 ACS 一样靠 entity id 对齐，因为 M2M-SGS 衡量的是
“GT 群组在所有预测簇中的集中程度”。
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


def m2m_sgs_score(
    gt_entities: Dict[str, Dict[str, List[int]]],
    pred_entities: Dict[str, Dict[str, List[int]]],
    network_keys: List[str] = None,
) -> Dict[str, Any]:
    """
    计算 M2M-SGS

    对每个 GT 群组 g:
      a_j^(g) = | g ∩ C_j |
      SGS(g) = Σ_j (a_j^(g))^2
      N_g    = |g|
      M2M-SGS(g) = (SGS(g)-N_g)/(N_g^2-N_g), N_g>1; 否则记为1

    最终返回：
      - macro_m2m_sgs
      - weighted_m2m_sgs
      - 每个 group 的详细信息
    """
    if network_keys is None:
        network_keys = infer_network_keys(gt_entities)

    gt_groups = build_group_sets(gt_entities, network_keys)
    pred_clusters = build_group_sets(pred_entities, network_keys)

    if not gt_groups:
        return {
            "macro_m2m_sgs": 0.0,
            "weighted_m2m_sgs": 0.0,
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
        sgs_raw = 0

        # 遍历所有预测簇，看 GT group 在哪些簇里出现，以及各出现多少次
        for pid, pred_nodes in pred_clusters.items():
            overlap = len(g_nodes & pred_nodes)
            if overlap > 0:
                counts_by_pred[pid] = overlap
                sgs_raw += overlap * overlap

        # 归一化
        if N_g <= 1:
            m2m_sgs_g = 1.0
        else:
            denom = N_g * N_g - N_g
            m2m_sgs_g = (sgs_raw - N_g) / denom if denom != 0 else 1.0
            # 数值安全截断
            m2m_sgs_g = max(0.0, min(1.0, m2m_sgs_g))

        macro_vals.append(m2m_sgs_g)
        weighted_num += m2m_sgs_g * N_g
        weighted_den += N_g

        per_group[gid] = {
            "N_g": float(N_g),
            "SGS_g": float(sgs_raw),
            "M2M_SGS_g": float(m2m_sgs_g),
            "NumTouchedPredClusters": float(len(counts_by_pred)),
            "CountsByPred": counts_by_pred,
        }

    macro_m2m_sgs = sum(macro_vals) / len(macro_vals) if macro_vals else 0.0
    weighted_m2m_sgs = weighted_num / weighted_den if weighted_den != 0 else 0.0

    return {
        "macro_m2m_sgs": float(macro_m2m_sgs),
        "weighted_m2m_sgs": float(weighted_m2m_sgs),
        "num_groups": float(len(gt_groups)),
        "total_gt_nodes": float(total_gt_nodes),
        "per_group": per_group,
        "network_keys": network_keys,
    }


def evaluate_pred_folder(
    gt_path: Path,
    pred_dir: Path,
    out_png: Path,
    file_prefix: str = "pred_err_"
) -> None:
    """
    遍历 pred_dir 下所有 pred_err_*.json，
    计算每个预测文件的 M2M-SGS，并按误差率画图。
    """
    gt_entities = load_entities(gt_path)
    network_keys = infer_network_keys(gt_entities)

    pred_files = sorted(
        [p for p in pred_dir.iterdir() if p.name.startswith(file_prefix) and p.suffix == ".json"]
    )
    if not pred_files:
        raise ValueError(f"{pred_dir} 下找不到 {file_prefix}*.json")

    xs = []  # 误差率
    ys_macro = []  # macro M2M-SGS
    ys_weighted = []  # weighted M2M-SGS
    records = []

    print(f"GT: {gt_path}")
    print(f"Pred dir: {pred_dir}")
    print(f"Networks: {network_keys}")
    print("-" * 100)

    for pf in pred_files:
        pred_entities = load_entities(pf)
        res = m2m_sgs_score(gt_entities, pred_entities, network_keys)

        # 从文件名 pred_err_0.2.json 解析出 0.2
        m = re.search(r"pred_err_([0-9.]+)\.json", pf.name)
        err = float(m.group(1)) if m else float("nan")

        xs.append(err)
        ys_macro.append(res["macro_m2m_sgs"])
        ys_weighted.append(res["weighted_m2m_sgs"])

        records.append({
            "file": pf.name,
            "error_rate": err,
            "macro_m2m_sgs": res["macro_m2m_sgs"],
            "weighted_m2m_sgs": res["weighted_m2m_sgs"],
            "num_groups": res["num_groups"],
            "total_gt_nodes": res["total_gt_nodes"],
        })

        print(
            f"{pf.name:20s}  "
            f"macro={res['macro_m2m_sgs']:.4f}  "
            f"weighted={res['weighted_m2m_sgs']:.4f}  "
            f"groups={int(res['num_groups'])}  "
            f"nodes={int(res['total_gt_nodes'])}"
        )

    # 按误差率排序
    pairs = sorted(zip(xs, ys_macro, ys_weighted), key=lambda t: t[0])
    xs, ys_macro, ys_weighted = zip(*pairs)

    # 保存 csv
    out_csv = out_png.with_suffix(".csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("error_rate,macro_m2m_sgs,weighted_m2m_sgs\n")
        for x, y1, y2 in zip(xs, ys_macro, ys_weighted):
            f.write(f"{x:.10f},{y1:.10f},{y2:.10f}\n")

    # 画图：一张图两条线，不指定颜色
    plt.figure()
    plt.plot(xs, ys_macro, marker="o", label="Macro M2M-SGS")
    plt.plot(xs, ys_weighted, marker="s", label="Weighted M2M-SGS")
    plt.xlabel("Error rate")
    plt.ylabel("M2M-SGS")
    plt.title("M2M-SGS vs Error rate")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved csv: {out_csv}")
    print(f"Saved plot: {out_png}")
# 直接调用 m2m_sgs_score 计算 macro M2M-SGS，一个明确的包装函数，方便外部调用。
def sgs_score(
    gt_entities: Dict[str, Dict[str, List[int]]],
    pred_entities: Dict[str, Dict[str, List[int]]],
) -> float:
    """
    默认把 SGS 定义为 macro_m2m_sgs，便于总控脚本统一调用。
    """
    return m2m_sgs_score(gt_entities, pred_entities)["macro_m2m_sgs"]

# 直接调用 m2m_sgs_score 计算 weighted M2M-SGS，一个明确的包装函数，方便外部调用。
def sgs_weighted_score(
    gt_entities: Dict[str, Dict[str, List[int]]],
    pred_entities: Dict[str, Dict[str, List[int]]],
) -> float:
    """
    若后续想比较 weighted 版本，可单独调用。
    """
    return m2m_sgs_score(gt_entities, pred_entities)["weighted_m2m_sgs"]

if __name__ == "__main__":
    BASE = Path(__file__).resolve().parent
    ROOT = BASE.parent

    DATASETS = ["Facebook-Twitter", "Arxiv1-Arxiv2", "DBLP1-DBLP2"]

    # 统一图输出目录
    PLOT_DIR = ROOT / "outputs" / "plots" / "m2m_sgs"
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    for DATASETNAME in DATASETS:
        # GT 从 data-many2many 读取
        DATASET_DIR = ROOT / "data" / "data-many2many" / DATASETNAME
        GT_PATH = DATASET_DIR / "gt_many2many.json"

        # 预测从 outputs/pred_sims/<DATASETNAME> 读取
        PRED_DIR = ROOT / "outputs" / "pred_sims" / DATASETNAME

        # 图输出到统一目录
        OUT_PNG = PLOT_DIR / f"M2M_SGS_curve_{DATASETNAME}.png"

        print(f"\n=== Evaluating dataset: {DATASETNAME} ===")
        evaluate_pred_folder(GT_PATH, PRED_DIR, OUT_PNG)
        print(f"Saved plot: {OUT_PNG}")