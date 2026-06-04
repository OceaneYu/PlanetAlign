import json
from pathlib import Path
from typing import Dict, List, Any
import re
import matplotlib.pyplot as plt

"""
Concentration Accuracy Score (ACS)指标实现
靠entity id 作为索引，对齐预测结果和groundtruth
"""

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


def concentration_accuracy_score(
    gt_entities: Dict[str, Dict[str, List[int]]],
    pred_entities: Dict[str, Dict[str, List[int]]],
    network_keys: List[str] = None,
) -> Dict[str, Any]:
    """
    新版指标：Normalized Concentration Accuracy (ACS)

    a_{T,i} = |P_i(T) ∩ G_i(T)|
    SG      = Σ_i Σ_T a_{T,i}^2
    Max*    = Σ_i Σ_T |G_i(T)|^2
    ACS    = SG / Max*

    性质：
    - pred == gt  -> ACS = 1
    - pred 与 gt 完全无交集 -> ACS = 0
    """
    if network_keys is None:
        network_keys = infer_network_keys(gt_entities)

    eids = sorted(gt_entities.keys())
    if not eids:
        return {"SG": 0.0, "MaxStar": 0.0, "ACS": 0.0, "network_keys": network_keys}

    SG = 0
    MaxStar = 0
    missing_pred_entities = 0

    for eid in eids:
        gt_item = gt_entities[eid]
        pred_item = pred_entities.get(eid, None)
        if pred_item is None:
            missing_pred_entities += 1
            pred_item = {}  # 当作预测全空

        for k in network_keys:
            G = set(gt_item.get(k, []))
            P = set(pred_item.get(k, []))
            a = len(P & G)
            g = len(G)

            SG += a * a
            MaxStar += g * g

    ACS = (SG / MaxStar) if MaxStar != 0 else 0.0

    return {
        "SG": float(SG),
        "MaxStar": float(MaxStar),
        "ACS": float(ACS),
        "network_keys": network_keys,
        "MissingPredEntities": float(missing_pred_entities),
        "NumEntities": float(len(eids)),
    }


def evaluate_pred_folder(gt_path: Path, pred_dir: Path, out_png: Path, file_prefix: str = "pred_err_") -> None:
    gt_entities = load_entities(gt_path)
    network_keys = infer_network_keys(gt_entities)

    pred_files = sorted([p for p in pred_dir.iterdir() if p.name.startswith(file_prefix) and p.suffix == ".json"])
    if not pred_files:
        raise ValueError(f"{pred_dir} 下找不到 {file_prefix}*.json")

    xs = []  # 误差率
    ys = []  # 指标分数

    print(f"GT: {gt_path}")
    print(f"Pred dir: {pred_dir}")
    print(f"Networks: {network_keys}")
    print("-" * 80)

    for pf in pred_files:
        pred_entities = load_entities(pf)
        res = concentration_accuracy_score(gt_entities, pred_entities, network_keys)

        # 从文件名 pred_err_0.2.json 解析出 0.2
        m = re.search(r"pred_err_([0-9.]+)\.json", pf.name)
        err = float(m.group(1)) if m else float("nan")

        xs.append(err)
        ys.append(res["ACS"])

        print(f"{pf.name:20s}  ACS={res['ACS']:.4f}  SG={res['SG']:.0f}  Max*={res['MaxStar']:.0f}")

    # 按误差率排序（保证曲线从 0->1）
    pairs = sorted(zip(xs, ys), key=lambda t: t[0])
    xs, ys = zip(*pairs)

    # 画图：不指定颜色，按默认即可
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Error rate")
    plt.ylabel("ACS")
    plt.title("ACS vs Error rate")
    plt.grid(True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved plot: {out_png}")

def acs_score(
    gt_entities: Dict[str, Dict[str, List[int]]],
    pred_entities: Dict[str, Dict[str, List[int]]],
) -> float:
    return concentration_accuracy_score(gt_entities, pred_entities)["ACS"]


if __name__ == "__main__":
    from pathlib import Path

    BASE = Path(__file__).resolve().parent
    ROOT = BASE.parent

    DATASETS = ["Facebook-Twitter", "Arxiv1-Arxiv2", "DBLP1-DBLP2"]

    # 统一的图输出目录（不放在各数据集文件夹里）
    PLOT_DIR = ROOT / "outputs" / "plots" / "acs"
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    for DATASETNAME in DATASETS:
        # GT 仍然从 data-many2many 里读
        DATASET_DIR = ROOT / "data" / "data-many2many" / DATASETNAME
        GT_PATH = DATASET_DIR / "gt_many2many.json"

        # 预测仍然从 outputs/pred_sims/<DATASETNAME> 里读
        PRED_DIR = ROOT / "outputs" / "pred_sims" / DATASETNAME

        # 图统一输出到 PLOT_DIR，文件名带数据集名
        OUT_PNG = PLOT_DIR / f"ACS_curve_{DATASETNAME}.png"

        print(f"\n=== Evaluating dataset: {DATASETNAME} ===")
        evaluate_pred_folder(GT_PATH, PRED_DIR, OUT_PNG)
        print(f"Saved plot: {OUT_PNG}")
