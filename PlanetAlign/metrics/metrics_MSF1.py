# 文件作用：计算双侧测试实体成员恢复的 MSF1，对全部真实 unit 的匹配 F1 取平均。
# 数据来源：PlanetAlign/metrics/many_to_many.py → many_to_many_scores() 传入 GT 与原始预测，不使用贪心编号匹配结果。
# 核心处理：仅纳入含测试节点的预测组，保留其全部成员，以最大总 F1 做全局一对一匹配。
# 阅读建议：先看 macro_set_f1() 的评分范围、匹配和分母；协议与边界见 docs/MSF1.md。

import json
from collections import defaultdict
from math import fsum
from pathlib import Path
from typing import Dict, List, Tuple

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching


MSF1_PROTOCOL = "MSF1-v2-K"


class InvalidPredictionError(ValueError):
    """A prediction violates the non-overlapping benchmark contract."""


def load_entities(path: Path) -> Dict[str, Dict[str, List[int]]]:
    """
    读取 json 文件，返回 entities 字典：
    entities[eid] = {"src":[...], "tgt":[...]}
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "entities" not in data:
        raise ValueError(f"{path} 缺少 'entities' 字段")
    return data["entities"]


def safe_div(numer: float, denom: float) -> float:
    """安全除法：分母为 0 时返回 0"""
    return numer / denom if denom != 0 else 0.0


def f1_from_sets(pred_set: List[int], gt_set: List[int]) -> Tuple[float, float, float]:
    """
    对单个 u（单个 entity）计算：
    Precision = |P ∩ G| / |P|
    Recall    = |P ∩ G| / |G|
    F1        = 2PR / (P+R)

    返回 (F1, Precision, Recall)

    约定：
    - 若 pred 为空，则 Precision=0
    - 若 gt 为空，则 Recall=0（通常你的 gt 不会空）
    - 若 P+R=0，则 F1=0
    """
    P = set(pred_set)
    G = set(gt_set)
    inter = len(P & G)

    precision = safe_div(inter, len(P))
    recall = safe_div(inter, len(G))
    f1 = safe_div(2 * precision * recall, (precision + recall))

    return f1, precision, recall


# 功能：把两侧成员转成带图侧标记的集合，并按成员排序；不依赖实体编号，不合并重复预测组。
def _unit_sets(entities, *, require_disjoint=False, label="prediction"):
    units = []
    owner = {}
    for eid, item in entities.items():
        unit = {(side, int(node)) for side in ("src", "tgt") for node in item.get(side, [])}
        if require_disjoint:
            for side, node in sorted(unit):
                key = (side, node)
                if key in owner:
                    error = InvalidPredictionError if label == "prediction" else ValueError
                    raise error(
                        f"Invalid {label} for non-overlapping benchmark: "
                        f"{side} node {node} belongs to units {owner[key]!r} and {eid!r}. "
                        "Use allow_overlap=True only for an overlapping benchmark."
                    )
                owner[key] = eid
        units.append(unit)
    return sorted(units, key=lambda nodes: tuple(sorted(nodes)))


# 功能：以双侧成员 F1 做全局最优匹配，再除以固定真实组数 K。
# 输入输出：输入完整测试 GT 和原始预测；返回 MSF1 及参与、忽略、未匹配真实/预测组的数量。
# 评分范围：包含任一测试节点的预测组才参与，但组内范围外成员仍保留并计入 FP。
def macro_set_f1(
    gt_entities: Dict[str, Dict[str, List[int]]],
    pred_entities: Dict[str, Dict[str, List[int]]],
    *,
    allow_overlap: bool = False,
) -> Dict[str, float]:
    """MSF1-v2-K: maximum matched bilateral member F1 / number of GT units.

    GT must completely describe the test units. Its side-tagged node union
    defines the test scope. A prediction is scored iff it touches this union;
    all its members remain in the pair F1 denominator, including background
    and training nodes. Empty or wholly out-of-scope predictions are ignored.
    Empty GT or no scored predictions yields zero. Empty GT units contribute
    zero. By default, each same-side node may belong to at most one unit,
    in both GT and the full prediction (validated BEFORE scope filtering).
    Invalid predictions raise InvalidPredictionError; no groups are repaired
    or dropped. Repeated members within one unit are still set-valued.

    Overlapping benchmarks must explicitly pass allow_overlap=True. That
    mode preserves overlapping and duplicate units; it is not the official
    non-overlapping submission contract and does not prevent duplicate gaming.
    """
    gt_units = _unit_sets(gt_entities, require_disjoint=not allow_overlap, label="ground truth")
    raw_pred_units = _unit_sets(pred_entities, require_disjoint=not allow_overlap)
    test_nodes = set().union(*gt_units)
    pred_units = [unit for unit in raw_pred_units if unit & test_nodes]
    num_gt, num_pred = len(gt_units), len(pred_units)
    matched_scores = []

    if num_gt and num_pred:
        # 只记录有交集的组对，避免为上万个实体创建巨大的全零矩阵。
        gt_by_node = defaultdict(list)
        for i, unit in enumerate(gt_units):
            for node in unit:
                gt_by_node[node].append(i)
        overlaps = defaultdict(int)
        for j, unit in enumerate(pred_units):
            for node in unit:
                for i in gt_by_node.get(node, ()):
                    overlaps[i, j] += 1
        pair_f1 = {
            (i, j): 2.0 * count / (len(gt_units[i]) + len(pred_units[j]))
            for (i, j), count in overlaps.items()
        }

        rows, cols, costs = [], [], []
        for (i, j), score in pair_f1.items():
            rows.append(i)
            cols.append(j)
            costs.append(2.0 - score)
        # 每个真实组都有专属的零分虚拟组，允许漏预测；固定每行匹配一次。
        # 最小化总 (2-F1) 等价于最大化总 F1；成本保持正数，兼容稀疏匹配器。
        for i in range(num_gt):
            rows.append(i)
            cols.append(num_pred + i)
            costs.append(2.0)
        cost_matrix = csr_matrix((costs, (rows, cols)), shape=(num_gt, num_pred + num_gt))
        matched_rows, matched_cols = min_weight_full_bipartite_matching(cost_matrix)
        matched_scores = [pair_f1[i, j] for i, j in zip(matched_rows, matched_cols) if j < num_pred]

    return {
        "MSF1": fsum(matched_scores) / num_gt if num_gt else 0.0,
        "NumEntities": float(num_gt),
        "NumPredEntities": float(num_pred),
        "IgnoredPredEntities": float(len(raw_pred_units) - num_pred),
        "MissingPredEntities": float(num_gt - len(matched_scores)),
        "ExtraPredEntities": float(num_pred - len(matched_scores)),
    }


# 流程位置：独立文件夹评估入口，逐个读取 pred_err_*.json 并打印宏平均分数。
def evaluate_folder(
    gt_path: Path,
    pred_dir: Path,
    pattern_prefix: str = "pred_err_",
    *,
    allow_overlap: bool = False,
) -> None:
    """
    批量评测 pred_dir 下的 pred_err_*.json
    并按文件名顺序打印结果
    """
    gt_entities = load_entities(gt_path)

    pred_files = sorted([p for p in pred_dir.iterdir() if p.name.startswith(pattern_prefix) and p.suffix == ".json"])
    if not pred_files:
        raise ValueError(f"{pred_dir} 下没有找到 {pattern_prefix}*.json")

    print(f"GT: {gt_path}")
    print(f"Pred dir: {pred_dir}")
    print("-" * 60)

    for pf in pred_files:
        pred_entities = load_entities(pf)
        res = macro_set_f1(gt_entities, pred_entities, allow_overlap=allow_overlap)
        print(f"{pf.name:20s}  {MSF1_PROTOCOL}={res['MSF1']:.4f}  scored={res['NumPredEntities']:.0f}  ignored={res['IgnoredPredEntities']:.0f}")

# 功能：仅返回 MSF1-v2 标量，供 PlanetAlign/metrics/many_to_many.py 的指标注册表调用。
def macro_f1_score(
    gt_entities: Dict[str, Dict[str, List[int]]],
    pred_entities: Dict[str, Dict[str, List[int]]],
    *,
    allow_overlap: bool = False,
) -> float:
    return macro_set_f1(gt_entities, pred_entities, allow_overlap=allow_overlap)["MSF1"]

if __name__ == "__main__":
    BASE = Path(__file__).resolve().parent
    ROOT = BASE.parent
    DATASETS = ["Arxiv1-Arxiv2", "DBLP1-DBLP2", "Facebook-Twitter"]
    # DATASETNAME = "Facebook-Twitter" # 你也可以改成 "Arxiv1-Arxiv2" 或 "DBLP1-DBLP2"
    for DATASETNAME in DATASETS:
        DATASET_DIR = ROOT / "data" / "data-many2many" / DATASETNAME
        GT_PATH = DATASET_DIR / "gt_many2many.json"

        PRED_DIR = ROOT / "outputs" / "pred_sims" / DATASETNAME
        evaluate_folder(GT_PATH, PRED_DIR)
