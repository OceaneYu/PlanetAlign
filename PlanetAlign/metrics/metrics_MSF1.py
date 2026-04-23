import json
from pathlib import Path
from typing import Dict, List, Tuple, Any


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


def macro_set_f1(
    gt_entities: Dict[str, Dict[str, List[int]]],
    pred_entities: Dict[str, Dict[str, List[int]]],
    key: str = "tgt",
) -> Dict[str, float]:
    """
    计算 Macro-averaged Set F1 (MSF1)：
      MSF1 = (1/|U|) * sum_u F1(u)

    同时也给出宏平均 precision / recall（可选，用于分析）

    key 默认用 "tgt"（评测目标侧集合）
    """
    eids = sorted(gt_entities.keys())
    n = len(eids)

    f1_list = []
    p_list = []
    r_list = []

    missing_pred = 0

    for eid in eids:
        gt_set = gt_entities[eid].get(key, [])

        if eid not in pred_entities:
            # 预测缺失：当作空预测
            pred_set = []
            missing_pred += 1
        else:
            pred_set = pred_entities[eid].get(key, [])

        f1, p, r = f1_from_sets(pred_set, gt_set)
        f1_list.append(f1)
        p_list.append(p)
        r_list.append(r)

    msf1 = sum(f1_list) / n if n > 0 else 0.0
    mp = sum(p_list) / n if n > 0 else 0.0
    mr = sum(r_list) / n if n > 0 else 0.0

    return {
        "MSF1": msf1,
        "MacroPrecision": mp,
        "MacroRecall": mr,
        "NumEntities": float(n),
        "MissingPredEntities": float(missing_pred),
    }


def evaluate_folder(
    gt_path: Path,
    pred_dir: Path,
    pattern_prefix: str = "pred_err_",
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
        res = macro_set_f1(gt_entities, pred_entities, key="tgt")
        print(f"{pf.name:20s}  MSF1={res['MSF1']:.4f}  MP={res['MacroPrecision']:.4f}  MR={res['MacroRecall']:.4f}")

# 直接调用 macro_set_f1 计算 Macro Set F1，一个明确的包装函数，方便外部调用。
def macro_f1_score(
    gt_entities: Dict[str, Dict[str, List[int]]],
    pred_entities: Dict[str, Dict[str, List[int]]],
) -> float:
    return macro_set_f1(gt_entities, pred_entities, key="tgt")["MSF1"]

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