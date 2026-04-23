import json
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
import re
import matplotlib.pyplot as plt

"""
M2M-EGS 指标实现

思想：
- GT 中每个 entity 视为一个真实群组 g
- Pred 中每个 entity 视为一个预测簇 C_j
- 对每个 g，统计它在多少个预测簇中出现：
      EG(g) = |{j : |g ∩ C_j| > 0}|
- 再归一化得到 M2M-EGS(g)

最终输出：
    1) macro_m2m_egs: 所有 group 的简单平均
    2) weighted_m2m_egs: 按 group 大小 N_g 加权平均
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
    entities -> eid 到节点集合的映射
    """
    out: Dict[str, Set[Node]] = {}
    for eid, item in entities.items():
        out[eid] = entity_to_node_set(item, network_keys)
    return out


def m2m_egs_score(
    gt_entities: Dict[str, Dict[str, List[int]]],
    pred_entities: Dict[str, Dict[str, List[int]]],
    network_keys: List[str] = None,
) -> Dict[str, Any]:
    """
    计算 M2M-EGS

    对每个 GT 群组 g:
      a_j^(g) = | g ∩ C_j |
      EG(g)   = |{j : a_j^(g) > 0}|
      N_g     = |g|
      M       = 预测簇总数

      M2M-EGS(g) = (min(N_g, M) - EG(g)) / (min(N_g, M) - 1), 当 min(N_g, M) > 1
                   1, 其他情况

    最终返回：
      - macro_m2m_egs
      - weighted_m2m_egs
      - 每个 group 的详细信息
    """
    if network_keys is None:
        network_keys = infer_network_keys(gt_entities)

    gt_groups = build_group_sets(gt_entities, network_keys)
    pred_clusters = build_group_sets(pred_entities, network_keys)

    M = len(pred_clusters)

    if not gt_groups:
        return {
            "macro_m2m_egs": 0.0,
            "weighted_m2m_egs": 0.0,
            "num_groups": 0.0,
            "total_gt_nodes": 0.0,
            "num_pred_clusters": float(M),
            "per_group": {},
            "network_keys": network_keys,
        }

    per_group: Dict[str, Any] = {}
    macro_vals = []
    weighted_num = 0.0
    weighted_den = 0.0
    total_gt_nodes = 0

    for gid, g_nodes in gt_groups.items(): # 遍历每个 GT 群组
        N_g = len(g_nodes) # 群组大小
        total_gt_nodes += N_g # 统计总的 GT 节点数 ，所有的群组大小加在一起

        counts_by_pred: Dict[str, int] = {} # 统计 g 在每个预测簇中的重叠大小
        exposed = 0

        # 看 g 出现在多少个预测簇里
        for pid, pred_nodes in pred_clusters.items(): # 遍历每个预测簇
            overlap = len(g_nodes & pred_nodes) # 计算 g 与 预测簇的重叠大小
            if overlap > 0: # 如果有重叠，说明 g 在这个预测簇里被“暴露”了
                counts_by_pred[pid] = overlap # 记录 g 在这个预测簇里的重叠大小
                exposed += 1 # 统计 g 被暴露的次数，即出现在多少个预测簇里

        max_exposed = min(N_g, M) # g 最多只能被暴露 min(N_g, M) 次，因为预测簇总数是 M，g 的大小是 N_g

        if max_exposed <= 1:
            m2m_egs_g = 1.0
        else:
            m2m_egs_g = (max_exposed - exposed) / (max_exposed - 1) # 归一化，得到 M2M-EGS(g)
            m2m_egs_g = max(0.0, min(1.0, m2m_egs_g))

        macro_vals.append(m2m_egs_g) # 收集每个 group 的 M2M-EGS，后续计算 macro 平均
        weighted_num += m2m_egs_g * N_g # 加权累加 M2M-EGS * group 大小，后续计算 weighted 平均
        weighted_den += N_g # 累加 group 大小，后续计算 weighted 平均

        per_group[gid] = {
            "N_g": float(N_g),
            "EG_g": float(exposed),
            "MaxExposed_g": float(max_exposed),
            "M2M_EGS_g": float(m2m_egs_g),
            "CountsByPred": counts_by_pred,
        }

    macro_m2m_egs = sum(macro_vals) / len(macro_vals) if macro_vals else 0.0
    weighted_m2m_egs = weighted_num / weighted_den if weighted_den != 0 else 0.0

    return {
        "macro_m2m_egs": float(macro_m2m_egs),
        "weighted_m2m_egs": float(weighted_m2m_egs),
        "num_groups": float(len(gt_groups)),
        "total_gt_nodes": float(total_gt_nodes),
        "num_pred_clusters": float(M),
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
    计算每个预测文件的 M2M-EGS，并按误差率画图。
    """
    gt_entities = load_entities(gt_path)
    network_keys = infer_network_keys(gt_entities)

    pred_files = sorted(
        [p for p in pred_dir.iterdir() if p.name.startswith(file_prefix) and p.suffix == ".json"]
    )
    if not pred_files:
        raise ValueError(f"{pred_dir} 下找不到 {file_prefix}*.json")

    xs = []  # 误差率
    ys_macro = []  # macro M2M-EGS
    ys_weighted = []  # weighted M2M-EGS

    print(f"GT: {gt_path}")
    print(f"Pred dir: {pred_dir}")
    print(f"Networks: {network_keys}")
    print("-" * 100)

    for pf in pred_files:
        pred_entities = load_entities(pf)
        res = m2m_egs_score(gt_entities, pred_entities, network_keys)

        # 从文件名 pred_err_0.2.json 解析出 0.2
        m = re.search(r"pred_err_([0-9.]+)\.json", pf.name)
        err = float(m.group(1)) if m else float("nan")

        xs.append(err)
        ys_macro.append(res["macro_m2m_egs"])
        ys_weighted.append(res["weighted_m2m_egs"])

        print(
            f"{pf.name:20s}  "
            f"macro={res['macro_m2m_egs']:.4f}  "
            f"weighted={res['weighted_m2m_egs']:.4f}  "
            f"groups={int(res['num_groups'])}  "
            f"nodes={int(res['total_gt_nodes'])}  "
            f"pred_clusters={int(res['num_pred_clusters'])}"
        )

    # 按误差率排序
    pairs = sorted(zip(xs, ys_macro, ys_weighted), key=lambda t: t[0])
    xs, ys_macro, ys_weighted = zip(*pairs)

    # 保存 csv
    out_csv = out_png.with_suffix(".csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("error_rate,macro_m2m_egs,weighted_m2m_egs\n")
        for x, y1, y2 in zip(xs, ys_macro, ys_weighted):
            f.write(f"{x:.10f},{y1:.10f},{y2:.10f}\n")

    # 画图
    plt.figure()
    plt.plot(xs, ys_macro, marker="o", label="Macro M2M-EGS")
    plt.plot(xs, ys_weighted, marker="s", label="Weighted M2M-EGS")
    plt.xlabel("Error rate")
    plt.ylabel("M2M-EGS")
    plt.title("M2M-EGS vs Error rate")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved csv: {out_csv}")
    print(f"Saved plot: {out_png}")

# 直接调用 m2m_egs_score 计算 macro M2M-EGS，一个明确的包装函数，方便外部调用。
def egs_score(
    gt_entities: Dict[str, Dict[str, List[int]]],
    pred_entities: Dict[str, Dict[str, List[int]]],
) -> float:
    """
    默认把 EGS 定义为 macro_m2m_egs，便于总控脚本统一调用。
    """
    return m2m_egs_score(gt_entities, pred_entities)["macro_m2m_egs"]

# 直接调用 m2m_egs_score 计算 weighted M2M-EGS，一个明确的包装函数，方便外部调用。
def egs_weighted_score(
    gt_entities: Dict[str, Dict[str, List[int]]],
    pred_entities: Dict[str, Dict[str, List[int]]],
) -> float:
    """
    若后续想比较 weighted 版本，可单独调用。
    """
    return m2m_egs_score(gt_entities, pred_entities)["weighted_m2m_egs"]

if __name__ == "__main__":
    BASE = Path(__file__).resolve().parent
    ROOT = BASE.parent

    DATASETS = ["Facebook-Twitter", "Arxiv1-Arxiv2", "DBLP1-DBLP2"]

    # 统一图输出目录
    PLOT_DIR = ROOT / "outputs" / "plots" / "m2m_egs"
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    for DATASETNAME in DATASETS:
        # GT 从 data-many2many 读取
        DATASET_DIR = ROOT / "data" / "data-many2many" / DATASETNAME
        GT_PATH = DATASET_DIR / "gt_many2many.json"

        # 预测从 outputs/pred_sims/<DATASETNAME> 读取
        PRED_DIR = ROOT / "outputs" / "pred_sims" / DATASETNAME

        # 图输出到统一目录
        OUT_PNG = PLOT_DIR / f"M2M_EGS_curve_{DATASETNAME}.png"

        print(f"\n=== Evaluating dataset: {DATASETNAME} ===")
        evaluate_pred_folder(GT_PATH, PRED_DIR, OUT_PNG)
        print(f"Saved plot: {OUT_PNG}")