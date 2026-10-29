# 文件作用：按数据集和算法批量训练，再用统一 QuotientDecode 评估多对多实体匹配。
# 核心依赖：PlanetAlign/algorithms/ 的 train()；PlanetAlign/m2m/quotient.py → quotient_decode()；PlanetAlign/m2m/core.py → evaluate_predictions()。
# 核心处理：evaluate_algorithm() 生成矩阵并校验，真实实体只交给评分；oracle 诊断需单独开启。
# 结果去向：_write_outputs() 写 JSON/CSV，main() 打印批次摘要。
# 阅读建议：main() → _algo_configs() → evaluate_algorithm() → _extract_s()，先区分各算法返回格式。

"""Run many-to-many benchmarks on selected datasets and algorithms.

The script generalizes the old Douban-only runners. It loads generated
``*_m2m.pt`` datasets plus their ``*_gt_many2many.json`` files, trains selected
PlanetAlign algorithms, converts each similarity matrix into many-to-many
groups with the blind quotient decoder, and writes JSON/CSV summaries.

Recommended first sweep:

    python scripts/run_m2m_experiments.py \
      --m2m-root data/m2m_no_overlap \
      --datasets pems08_m2m airport_m2m cora_m2m \
      --algorithms all \
      --profile quick

Run all datasets in a directory:

    python scripts/run_m2m_experiments.py \
      --m2m-root data/m2m_overlap_0.05 \
      --datasets all \
      --algorithms all \
      --profile quick

Run selected algorithms:

    python scripts/run_m2m_experiments.py \
      --m2m-root data/m2m_no_overlap \
      --datasets pems08_m2m \
      --algorithms IsoRank FINAL JOENA M2MAlign TGAE

Run every algorithm except especially slow ones:

    python -u scripts/run_m2m_experiments.py \
      --m2m-root data/m2m_no_overlap \
      --datasets pems08_m2m airport_m2m cora_m2m \
      --algorithms all \
      --exclude HOT SLOTAlign
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "planetalign-matplotlib"))

import PlanetAlign
from PlanetAlign.data import Dataset
from PlanetAlign.m2m.diagnostics import (
    ORACLE_GROUPED_TOPK_PROTOCOL,
    oracle_grouped_topk,
    oracle_grouped_topk_target_metrics,
)
from PlanetAlign.m2m import evaluate_predictions
from PlanetAlign.m2m.quotient import quotient_decode
from PlanetAlign.metrics import InvalidPredictionError, MSF1_PROTOCOL
from PlanetAlign.utils import get_anchor_pairs, pairwise_cosine_similarity


GIDS = [0, 1]
RECOMMENDED_DATASETS = ["pems08_m2m", "airport_m2m", "cora_m2m"]
M2M_METRICS = ["Macro Set-F1", "Micro Set-F1", "AUIS", "AUCS"]
CSV_FIELDS = [
    "dataset",
    "algorithm",
    "status",
    "profile",
    "time_s",
    "m2m_protocol",
    "msf1_protocol",
    "allow_overlap",
    "Blind-Macro Set-F1",
    "Blind-Micro Set-F1",
    "Blind-AUIS",
    "Blind-AUCS",
    "oracle_protocol",
    "OracleSrc+GT-K-Target-MicroF1",
    "OracleSrc+GT-K-Target-MSF1",
    "OracleSrc+GT-K-Target-Recall@GT-K",
    "oracle_diagnostic_error",
    "error",
]


def _sim_from_embeddings(emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
    emb1 = emb1.detach().to(torch.float32).cpu()
    emb2 = emb2.detach().to(torch.float32).cpu()
    return pairwise_cosine_similarity(emb1, emb2)


def _sim_from_inner_product(emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
    return emb1.detach().to(torch.float32).cpu() @ emb2.detach().to(torch.float32).cpu().T


# 输入来源：evaluate_algorithm() 传入训练对象、train() 返回值和配置 mode。
# 输出结果：从 self.S、直接返回矩阵或嵌入提取 CPU float32 的 S，供统一解码。
def _extract_s(algo: Any, ret: Tuple[Any, ...], mode: str, dataset: Dataset, gids: List[int]) -> torch.Tensor:
    gid1, gid2 = gids
    if mode == "self.S":
        s = algo.S
        if s is None:
            raise RuntimeError("algorithm did not populate self.S")
        return s.detach().to(torch.float32).cpu()
    if mode == "S":
        return ret[0].detach().to(torch.float32).cpu()
    if mode == "embs-cos":
        return _sim_from_embeddings(ret[0], ret[1])
    if mode == "embs-dot-dict":
        emb_dict = ret[0]
        return _sim_from_inner_product(emb_dict[gid1], emb_dict[gid2])
    raise ValueError(f"unknown sim extraction mode: {mode}")


def _has_node_attributes(dataset: Dataset, gids: List[int]) -> bool:
    return all(dataset.pyg_graphs[gid].x is not None for gid in gids)


def _quick(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return dict(kwargs)


def _full(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return dict(kwargs)


# 功能：按实验 profile 定义算法工厂、训练参数和结果模式；显式跳过未实现或不适用接口。
def _algo_configs(profile: str) -> Dict[str, Dict[str, Any]]:
    if profile == "quick":
        return {
            "IsoRank": {
                "factory": lambda: PlanetAlign.algorithms.IsoRank(alpha=0.4),
                "train_kwargs": _quick({"use_attr": False, "total_epochs": 10}),
                "mode": "S",
            },
            "FINAL": {
                "factory": lambda: PlanetAlign.algorithms.FINAL(alpha=0.9),
                "train_kwargs": _quick({"use_attr": True, "total_epochs": 10}),
                "mode": "self.S",
            },
            "IONE": {
                "factory": lambda: PlanetAlign.algorithms.IONE(out_dim=64),
                "train_kwargs": _quick({"use_attr": False, "total_epochs": 5}),
                "mode": "S",
            },
            "REGAL": {
                "factory": lambda: PlanetAlign.algorithms.REGAL(),
                "train_kwargs": _quick({"use_attr": True}),
                "mode": "embs-cos",
            },
            "CrossMNA": {
                "factory": lambda: PlanetAlign.algorithms.CrossMNA(),
                "train_kwargs": _quick({"use_attr": False, "total_epochs": 10}),
                "mode": "embs-dot-dict",
            },
            "NetTrans": {
                "factory": lambda: PlanetAlign.algorithms.NetTrans(),
                "train_kwargs": _quick({"use_attr": True, "total_epochs": 5}),
                "mode": "embs-cos",
            },
            "BRIGHT": {
                "factory": lambda: PlanetAlign.algorithms.BRIGHT(),
                "train_kwargs": _quick({"use_attr": True, "total_epochs": 10}),
                "mode": "embs-cos",
            },
            "NeXtAlign": {
                "factory": lambda: PlanetAlign.algorithms.NeXtAlign(),
                "train_kwargs": _quick({"use_attr": True, "total_epochs": 5}),
                "mode": "embs-cos",
            },
            "PARROT": {
                "factory": lambda: PlanetAlign.algorithms.PARROT(alpha=0.5),
                "train_kwargs": _quick({"use_attr": True}),
                "mode": "self.S",
            },
            "SLOTAlign": {
                "factory": lambda: PlanetAlign.algorithms.SLOTAlign(bases=4),
                "train_kwargs": _quick({"use_attr": True, "total_epochs": 10, "joint_epochs": 5}),
                "mode": "self.S",
            },
            "WLAlign": {
                "factory": lambda: PlanetAlign.algorithms.WLAlign(),
                "train_kwargs": _quick({"use_attr": False, "total_epochs": 5, "struct_epochs": 10}),
                "mode": "embs-cos",
            },
            "WAlign": {
                "factory": lambda: PlanetAlign.algorithms.WAlign(),
                "train_kwargs": _quick({"use_attr": True, "total_epochs": 5}),
                "mode": "embs-cos",
            },
            "HOT": {
                "factory": lambda: PlanetAlign.algorithms.HOT(alpha=0.5, lp=0.1),
                "train_kwargs": _quick({"use_attr": True, "in_iters": 2, "out_iters": 5}),
                "mode": "self.S",
            },
            "JOENA": {
                "factory": lambda: PlanetAlign.algorithms.JOENA(alpha=0.7),
                "train_kwargs": _quick({"use_attr": True, "total_epochs": 10}),
                "mode": "self.S",
            },
            "M2MAlign": {
                "factory": lambda: PlanetAlign.algorithms.M2MAlign(),
                "train_kwargs": _quick({"use_attr": True}),
                "mode": "self.S",
            },
            "TGAE": {
                "factory": lambda: PlanetAlign.algorithms.TGAE(
                    num_hidden_layers=3,
                    hidden_dim=16,
                    output_dim=16,
                    lr=1e-3,
                ),
                "train_kwargs": _quick({"use_attr": True, "total_epochs": 5, "eval_interval": 5}),
                "mode": "self.S",
            },
            "DualMatch": {"skip": "train() is a stub in this repository"},
            "MEAformer": {"skip": "train() is a stub in this repository"},
        }

    if profile == "full":
        return {
            "IsoRank": {
                "factory": lambda: PlanetAlign.algorithms.IsoRank(alpha=0.4),
                "train_kwargs": _full({"use_attr": False, "total_epochs": 50}),
                "mode": "S",
            },
            "FINAL": {
                "factory": lambda: PlanetAlign.algorithms.FINAL(alpha=0.9),
                "train_kwargs": _full({"use_attr": True, "total_epochs": 50}),
                "mode": "self.S",
            },
            "IONE": {
                "factory": lambda: PlanetAlign.algorithms.IONE(out_dim=100),
                "train_kwargs": _full({"use_attr": False, "total_epochs": 30}),
                "mode": "S",
            },
            "REGAL": {
                "factory": lambda: PlanetAlign.algorithms.REGAL(),
                "train_kwargs": _full({"use_attr": True}),
                "mode": "embs-cos",
            },
            "CrossMNA": {
                "factory": lambda: PlanetAlign.algorithms.CrossMNA(),
                "train_kwargs": _full({"use_attr": False, "total_epochs": 100}),
                "mode": "embs-dot-dict",
            },
            "NetTrans": {
                "factory": lambda: PlanetAlign.algorithms.NetTrans(),
                "train_kwargs": _full({"use_attr": True, "total_epochs": 30}),
                "mode": "embs-cos",
            },
            "BRIGHT": {
                "factory": lambda: PlanetAlign.algorithms.BRIGHT(),
                "train_kwargs": _full({"use_attr": True, "total_epochs": 100}),
                "mode": "embs-cos",
            },
            "NeXtAlign": {
                "factory": lambda: PlanetAlign.algorithms.NeXtAlign(),
                "train_kwargs": _full({"use_attr": True, "total_epochs": 30}),
                "mode": "embs-cos",
            },
            "PARROT": {
                "factory": lambda: PlanetAlign.algorithms.PARROT(alpha=0.5),
                "train_kwargs": _full({"use_attr": True}),
                "mode": "self.S",
            },
            "SLOTAlign": {
                "factory": lambda: PlanetAlign.algorithms.SLOTAlign(bases=4),
                "train_kwargs": _full({"use_attr": True, "total_epochs": 200, "joint_epochs": 50}),
                "mode": "self.S",
            },
            "WLAlign": {
                "factory": lambda: PlanetAlign.algorithms.WLAlign(),
                "train_kwargs": _full({"use_attr": False, "total_epochs": 30, "struct_epochs": 50}),
                "mode": "embs-cos",
            },
            "WAlign": {
                "factory": lambda: PlanetAlign.algorithms.WAlign(),
                "train_kwargs": _full({"use_attr": True, "total_epochs": 20}),
                "mode": "embs-cos",
            },
            "HOT": {
                "factory": lambda: PlanetAlign.algorithms.HOT(alpha=0.5, lp=0.1),
                "train_kwargs": _full({"use_attr": True, "in_iters": 5, "out_iters": 30}),
                "mode": "self.S",
            },
            "JOENA": {
                "factory": lambda: PlanetAlign.algorithms.JOENA(alpha=0.7),
                "train_kwargs": _full({"use_attr": True, "total_epochs": 50}),
                "mode": "self.S",
            },
            "M2MAlign": {
                "factory": lambda: PlanetAlign.algorithms.M2MAlign(),
                "train_kwargs": _full({"use_attr": True}),
                "mode": "self.S",
            },
            "TGAE": {
                "factory": lambda: PlanetAlign.algorithms.TGAE(
                    num_hidden_layers=4,
                    hidden_dim=16,
                    output_dim=16,
                    lr=1e-3,
                ),
                "train_kwargs": _full({"use_attr": True, "total_epochs": 20, "eval_interval": 5}),
                "mode": "self.S",
            },
            "DualMatch": {"skip": "train() is a stub in this repository"},
            "MEAformer": {"skip": "train() is a stub in this repository"},
        }

    raise ValueError(f"unknown profile: {profile}")


def _available_dataset_names(root: Path) -> List[str]:
    return sorted(path.stem for path in root.glob("*_m2m.pt"))


# 功能：检查请求的数据集名，未指定时扫描 *_m2m.pt。
def _resolve_datasets(root: Path, requested: Optional[List[str]]) -> List[str]:
    available = _available_dataset_names(root)
    if not requested:
        requested = list(RECOMMENDED_DATASETS)
    if requested == ["all"]:
        return available
    missing = [name for name in requested if name not in available]
    if missing:
        raise FileNotFoundError(f"missing datasets in {root}: {', '.join(missing)}")
    return requested


# 功能：按名称、别名与排除列表选择算法，未知名称报错。
def _resolve_algorithms(
    configs: Dict[str, Dict[str, Any]],
    requested: Optional[List[str]],
    excluded: Optional[List[str]],
) -> List[str]:
    available = list(configs.keys())
    if not requested or requested == ["all"]:
        resolved = available
    else:
        lowered = {name.lower(): name for name in available}
        resolved = []
        missing = []
        for raw in requested:
            name = lowered.get(raw.lower())
            if name is None:
                missing.append(raw)
            else:
                resolved.append(name)
        if missing:
            raise ValueError(f"unknown algorithms: {', '.join(missing)}")

    if excluded:
        lowered = {name.lower(): name for name in available}
        exclude_set = set()
        missing_excluded = []
        for raw in excluded:
            name = lowered.get(raw.lower())
            if name is None:
                missing_excluded.append(raw)
            else:
                exclude_set.add(name)
        if missing_excluded:
            raise ValueError(f"unknown excluded algorithms: {', '.join(missing_excluded)}")
        resolved = [name for name in resolved if name not in exclude_set]
    return resolved


# 输出结果：读取基准真实实体 JSON 的完整负载，main() 再从 entities 字段取实体。
def _load_gt(root: Path, dataset_name: str) -> Dict[str, Any]:
    gt_path = root / f"{dataset_name}_gt_many2many.json"
    if not gt_path.exists():
        raise FileNotFoundError(f"ground-truth JSON not found: {gt_path}")
    with open(gt_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _flatten_metrics(prefix: str, metrics: Dict[str, float]) -> Dict[str, float]:
    return {f"{prefix}{key}": value for key, value in metrics.items()}


# 输入来源：main() 传入已经加载的 Dataset、真实实体与一个算法配置。
# 核心思路：训练后检查 S，交给 quotient_decode() 预测，再由 evaluate_predictions() 用真实答案评分。
# 输出结果：返回状态、耗时和带 Blind- 前缀的指标；可选 oracle 分数另行标记，异常不终止整批。
def evaluate_algorithm(
    dataset_name: str,
    dataset: Dataset,
    gt_entities: Dict[str, Dict[str, List[int]]],
    algo_name: str,
    config: Dict[str, Any],
    profile: str,
    seed: int,
    verbose: bool,
    auto_disable_attr: bool,
    include_oracle_diagnostic: bool = False,
    allow_overlap: bool = False,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "dataset": dataset_name,
        "algorithm": algo_name,
        "status": "pending",
        "profile": profile,
        "msf1_protocol": MSF1_PROTOCOL,
        "allow_overlap": allow_overlap,
    }

    if "skip" in config:
        record.update(status="skipped", error=config["skip"])
        return record

    torch.manual_seed(seed)
    train_kwargs = dict(config["train_kwargs"])
    if auto_disable_attr and train_kwargs.get("use_attr") and not _has_node_attributes(dataset, GIDS):
        train_kwargs["use_attr"] = False
        record["attr_note"] = "use_attr was disabled because this dataset has no node attributes"

    started = time.perf_counter()
    try:
        algo = config["factory"]().to("cpu")
        ret = algo.train(
            dataset=dataset,
            gids=GIDS,
            save_log=False,
            verbose=verbose,
            **train_kwargs,
        )
        if not isinstance(ret, tuple):
            ret = (ret,)
        s = _extract_s(algo, ret, config["mode"], dataset, GIDS)
        algo.S = s.to(algo.device)

        expected_shape = (
            int(dataset.pyg_graphs[GIDS[0]].num_nodes),
            int(dataset.pyg_graphs[GIDS[1]].num_nodes),
        )
        if tuple(s.shape) != expected_shape:
            raise ValueError(f"similarity shape {tuple(s.shape)} does not match graph sizes {expected_shape}")
        if not torch.isfinite(s).all():
            raise ValueError("similarity contains NaN or infinite values")

        anchors = get_anchor_pairs(dataset.train_data, GIDS[0], GIDS[1])
        # 预测：矩阵、图和训练锚点决定分组，不接收真实答案。
        pred, quotient_info = quotient_decode(
            s,
            dataset.pyg_graphs[GIDS[0]],
            dataset.pyg_graphs[GIDS[1]],
            use_attr=_has_node_attributes(dataset, GIDS),
            anchors=anchors,
        )
        # 评估：只在预测完成后使用真实答案；输出字段仍沿用原来的 Blind- 前缀。
        blind_scores = evaluate_predictions(gt_entities, pred, metrics=M2M_METRICS,
                                            allow_overlap=allow_overlap)

        logged_train_kwargs = dict(train_kwargs)

        record.update(
            {
                "status": "ok",
                "time_s": round(time.perf_counter() - started, 4),
                "train_kwargs": logged_train_kwargs,
                "m2m_protocol": "blind_quotient_decode",
                "quotient_decode": quotient_info,
                **_flatten_metrics("Blind-", blind_scores),
            }
        )

        if include_oracle_diagnostic:
            try:
                oracle_pred = oracle_grouped_topk(s, gt_entities)
                record["oracle_protocol"] = ORACLE_GROUPED_TOPK_PROTOCOL
                record.update(oracle_grouped_topk_target_metrics(gt_entities, oracle_pred))
            except Exception as oracle_exc:  # keep the blind result valid.
                record["oracle_diagnostic_error"] = f"{type(oracle_exc).__name__}: {oracle_exc}"

    except Exception as exc:  # pragma: no cover - intentionally keeps long sweeps alive.
        logged_train_kwargs = dict(train_kwargs)
        record.update(
            {
                "status": "invalid_prediction" if isinstance(exc, InvalidPredictionError) else "error",
                "time_s": round(time.perf_counter() - started, 4),
                "train_kwargs": logged_train_kwargs,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
    return record


# 功能：将本批 records 与运行参数写 JSON，并把可比较字段整理为 CSV，返回输出路径。
def _write_outputs(records: List[Dict[str, Any]], out_dir: Path, tag: str, args: argparse.Namespace) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{tag}.json"
    csv_path = out_dir / f"{tag}.csv"
    payload = {
        "args": {
            "m2m_root": str(args.m2m_root),
            "datasets": args.datasets,
            "algorithms": args.algorithms,
            "exclude": args.exclude,
            "profile": args.profile,
            "seed": args.seed,
            "include_oracle_diagnostic": args.include_oracle_diagnostic,
        },
        "records": records,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in CSV_FIELDS})
    return json_path, csv_path


def _print_summary(records: List[Dict[str, Any]]) -> None:
    print("\nSummary")
    print("=" * 96)
    header = f"{'dataset':<18}{'algorithm':<14}{'status':<9}{'Macro Set-F1':>15}{'Micro Set-F1':>15}{'AUIS':>9}{'AUCS':>9}{'time(s)':>9}"
    print(header)
    print("-" * len(header))
    for r in records:
        if r.get("status") == "ok":
            print(
                f"{r['dataset']:<18}{r['algorithm']:<14}{r['status']:<9}"
                f"{r.get('Blind-Macro Set-F1', 0):>15.4f}"
                f"{r.get('Blind-Micro Set-F1', 0):>15.4f}{r.get('Blind-AUIS', 0):>9.4f}"
                f"{r.get('Blind-AUCS', 0):>9.4f}"
                f"{r.get('time_s', 0):>9.1f}"
            )
        else:
            err = str(r.get("error", ""))[:40]
            print(f"{r['dataset']:<18}{r['algorithm']:<14}{r['status']:<9}{err}")


# 功能：解析基准目录、数据集、算法与实验 profile 等批次选项。
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run M2M experiments on selected datasets and algorithms.")
    parser.add_argument("--m2m-root", type=Path, default=Path("data/m2m_no_overlap"))
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(RECOMMENDED_DATASETS),
        help="Dataset names, or 'all'. Default: pems08_m2m airport_m2m cora_m2m.",
    )
    parser.add_argument("--algorithms", nargs="+", default=["all"], help="Algorithm names, or 'all'.")
    parser.add_argument("--exclude", nargs="+", default=None, help="Algorithm names to exclude from the run.")
    parser.add_argument("--profile", choices=["quick", "full"], default="quick")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.2)
    parser.add_argument("--out-dir", type=Path, default=Path("logs/m2m_sweeps"))
    parser.add_argument("--tag", default=None, help="Output filename stem. Defaults to root/profile/dataset summary.")
    parser.add_argument("--verbose", action="store_true", help="Print per-epoch algorithm logs.")
    parser.add_argument(
        "--no-auto-disable-attr",
        action="store_true",
        help="Do not turn use_attr off automatically on datasets without node attributes.",
    )
    parser.add_argument(
        "--include-oracle-diagnostic",
        action="store_true",
        help=(
            "Also report target-ranking diagnostics conditioned on ground-truth "
            "source groups and target cardinalities. Disabled by default."
        ),
    )
    parser.add_argument("--list-algorithms", action="store_true")
    parser.add_argument("--list-datasets", action="store_true")
    return parser.parse_args()


# 核心思路：解析并检查清单，逐数据集加载图和实体，逐算法调用 evaluate_algorithm()。
# 输出结果：保存结果、打印摘要，以退出码报告是否有运行错误。
def main() -> int:
    args = parse_args()
    args.m2m_root = args.m2m_root.expanduser()
    configs = _algo_configs(args.profile)

    if args.list_algorithms:
        print("\n".join(configs.keys()))
        return 0
    if args.list_datasets:
        print("\n".join(_available_dataset_names(args.m2m_root)))
        return 0

    dataset_names = _resolve_datasets(args.m2m_root, args.datasets)
    algorithm_names = _resolve_algorithms(configs, args.algorithms, args.exclude)
    print(f"M2M root   : {args.m2m_root}")
    print(f"Datasets   : {', '.join(dataset_names)}")
    print(f"Algorithms : {', '.join(algorithm_names)}")
    print(f"Profile    : {args.profile}")
    print(f"Oracle diag: {'enabled' if args.include_oracle_diagnostic else 'disabled'}")

    records: List[Dict[str, Any]] = []
    for dataset_name in dataset_names:
        print("\n" + "=" * 96)
        print(f"Dataset: {dataset_name}")
        print("=" * 96)
        dataset = Dataset(root=args.m2m_root, name=dataset_name, train_ratio=args.train_ratio, seed=args.seed)
        gt_payload = _load_gt(args.m2m_root, dataset_name)
        gt_entities = gt_payload["entities"]
        print(
            f"nodes={[g.num_nodes for g in dataset.pyg_graphs]} "
            f"entities={len(gt_entities)} "
            f"overlap={gt_payload.get('metadata', {}).get('overlap_ratio')}"
        )

        for algo_name in algorithm_names:
            print(f"[run] {dataset_name} / {algo_name}")
            record = evaluate_algorithm(
                dataset_name=dataset_name,
                dataset=dataset,
                gt_entities=gt_entities,
                algo_name=algo_name,
                config=configs[algo_name],
                profile=args.profile,
                seed=args.seed,
                verbose=args.verbose,
                auto_disable_attr=not args.no_auto_disable_attr,
                include_oracle_diagnostic=args.include_oracle_diagnostic,
                allow_overlap=gt_payload.get("metadata", {}).get("overlap_ratio", 0.0) > 0,
            )
            records.append(record)
            status = record["status"]
            if status == "ok":
                print(
                    f"      ok {record['time_s']:.1f}s "
                    f"Macro Set-F1={record['Blind-Macro Set-F1']:.4f} "
                    f"Micro Set-F1={record['Blind-Micro Set-F1']:.4f} "
                    f"AUIS={record['Blind-AUIS']:.4f} "
                    f"AUCS={record['Blind-AUCS']:.4f}"
                )
            else:
                print(f"      {status}: {record.get('error', '')}")

    tag = args.tag
    if tag is None:
        root_tag = args.m2m_root.name.replace("/", "_")
        ds_tag = "recommended3" if dataset_names == RECOMMENDED_DATASETS else f"{len(dataset_names)}datasets"
        algo_tag = "all" if args.algorithms == ["all"] else f"{len(algorithm_names)}algorithms"
        tag = f"{root_tag}_{ds_tag}_{algo_tag}_{args.profile}"

    json_path, csv_path = _write_outputs(records, args.out_dir, tag, args)
    _print_summary(records)
    print(f"\nJSON results: {json_path}")
    print(f"CSV results : {csv_path}")

    errors = sum(1 for record in records if record.get("status") == "error")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
