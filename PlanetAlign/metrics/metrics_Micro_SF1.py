import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any

import matplotlib.pyplot as plt
"""
严格按 eid 一一对应，同 ID 才算交集，不同 ID 完全不匹配
定义：
1. GT 中每个 entity 是一个真实 many-to-many group
2. 对于每个 entity，计算它的预测集合和真实集合的交集、差集，得到 TP/FP/FN
3. 在所有 entity 上累计 TP/FP/FN，统一计算 Precision/Recall/F1
4. 最终输出：F1、Precision、Recall 等指标"""

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


def count_tp_fp_fn(pred_set: List[int], gt_set: List[int]) -> Tuple[int, int, int]:
    """
    对单个 entity 的预测集合和真实集合，计算：
      TP = |P ∩ G|
      FP = |P - G|
      FN = |G - P|
    """
    P = set(pred_set)
    G = set(gt_set)

    tp = len(P & G)
    fp = len(P - G)
    fn = len(G - P)

    return tp, fp, fn


def global_set_f1(
    gt_entities: Dict[str, Dict[str, List[int]]],
    pred_entities: Dict[str, Dict[str, List[int]]],
    key: str = "tgt",
) -> Dict[str, Any]:
    """
    全局 F1：
    先在所有 entity 上累计 TP / FP / FN，
    再统一计算 Precision / Recall / F1。
    """
    eids = sorted(gt_entities.keys())

    tp_total = 0
    fp_total = 0
    fn_total = 0
    missing_pred = 0

    for eid in eids:
        gt_set = gt_entities[eid].get(key, [])

        if eid not in pred_entities:
            pred_set = []
            missing_pred += 1
        else:
            pred_set = pred_entities[eid].get(key, [])

        tp, fp, fn = count_tp_fp_fn(pred_set, gt_set)
        tp_total += tp
        fp_total += fp
        fn_total += fn

    precision = safe_div(tp_total, tp_total + fp_total)
    recall = safe_div(tp_total, tp_total + fn_total)
    f1 = safe_div(2 * precision * recall, precision + recall)

    return {
        "F1": float(f1),
        "Precision": float(precision),
        "Recall": float(recall),
        "NumEntities": float(len(eids)),
        "MissingPredEntities": float(missing_pred),
        "TP": float(tp_total),
        "FP": float(fp_total),
        "FN": float(fn_total),
    }


def evaluate_pred_folder(
    gt_path: Path,
    pred_dir: Path,
    out_png: Path,
    file_prefix: str = "pred_err_"
) -> None:
    """
    批量评测 pred_dir 下的 pred_err_*.json
    保存：
      - out_png: F1 曲线图
      - out_png.with_suffix(".csv"): 数值结果表
    """
    gt_entities = load_entities(gt_path)

    pred_files = sorted(
        [p for p in pred_dir.iterdir() if p.name.startswith(file_prefix) and p.suffix == ".json"]
    )
    if not pred_files:
        raise ValueError(f"{pred_dir} 下找不到 {file_prefix}*.json")

    rows = []

    print(f"GT: {gt_path}")
    print(f"Pred dir: {pred_dir}")
    print("-" * 120)

    for pf in pred_files:
        pred_entities = load_entities(pf)
        res = global_set_f1(gt_entities, pred_entities, key="tgt")

        m = re.search(r"pred_err_([0-9.]+)\.json", pf.name)
        err = float(m.group(1)) if m else float("nan")

        rows.append({
            "error_rate": err,
            "file": pf.name,
            "f1": res["F1"],
            "precision": res["Precision"],
            "recall": res["Recall"],
            "tp": int(res["TP"]),
            "fp": int(res["FP"]),
            "fn": int(res["FN"]),
            "num_entities": int(res["NumEntities"]),
            "missing_pred_entities": int(res["MissingPredEntities"]),
        })

        print(
            f"{pf.name:20s}  "
            f"F1={res['F1']:.4f}  "
            f"P={res['Precision']:.4f}  "
            f"R={res['Recall']:.4f}  "
            f"TP={int(res['TP'])}  "
            f"FP={int(res['FP'])}  "
            f"FN={int(res['FN'])}"
        )

    rows = sorted(rows, key=lambda x: x["error_rate"])

    out_csv = out_png.with_suffix(".csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("error_rate,file,f1,precision,recall,tp,fp,fn,num_entities,missing_pred_entities\n")
        for row in rows:
            f.write(
                f"{row['error_rate']:.10f},"
                f"{row['file']},"
                f"{row['f1']:.10f},"
                f"{row['precision']:.10f},"
                f"{row['recall']:.10f},"
                f"{row['tp']},"
                f"{row['fp']},"
                f"{row['fn']},"
                f"{row['num_entities']},"
                f"{row['missing_pred_entities']}\n"
            )

    xs = [row["error_rate"] for row in rows]
    ys_f1 = [row["f1"] for row in rows]

    plt.figure()
    plt.plot(xs, ys_f1, marker="o", label="Global F1")
    plt.xlabel("Error rate")
    plt.ylabel("F1")
    plt.title("Global F1 vs Error rate")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved csv: {out_csv}")
    print(f"Saved plot: {out_png}")

# 直接调用 global_set_f1 计算 Micro F1，一个明确的包装函数，方便外部调用。
def micro_f1_score(
    gt_entities: Dict[str, Dict[str, List[int]]],
    pred_entities: Dict[str, Dict[str, List[int]]],
) -> float:
    return global_set_f1(gt_entities, pred_entities, key="tgt")["F1"]

if __name__ == "__main__":
    BASE = Path(__file__).resolve().parent
    ROOT = BASE.parent

    DATASETS = ["Facebook-Twitter", "Arxiv1-Arxiv2", "DBLP1-DBLP2"]

    PLOT_DIR = ROOT / "outputs" / "plots" / "global_f1"
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    for DATASETNAME in DATASETS:
        DATASET_DIR = ROOT / "data" / "data-many2many" / DATASETNAME
        GT_PATH = DATASET_DIR / "gt_many2many.json"

        PRED_DIR = ROOT / "outputs" / "pred_sims" / DATASETNAME

        OUT_PNG = PLOT_DIR / f"GlobalF1_curve_{DATASETNAME}.png"

        print(f"\n=== Evaluating dataset: {DATASETNAME} ===")
        evaluate_pred_folder(GT_PATH, PRED_DIR, OUT_PNG)
        print(f"Saved plot: {OUT_PNG}")