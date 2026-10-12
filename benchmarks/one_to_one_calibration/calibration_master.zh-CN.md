# 一对一基线校准总表

本记录冻结了 M2M 总表之前已完成的一对一校准审计结果。不包含任何新执行的实验、超参数搜索或对缺失结果的推断。

## 记录元数据

- 生成日期：2026-08-17
- 审计时仓库 HEAD：`f72d2e0785dee4c608068dba4168689f8b2a6f61`
- 范围：PLANETALIGN 评测的 14 种一对一方法，以及已准备好的 GADL 基线
- 参考协议：20% 训练锚点、普通 argsort 排名、双向平均 Hits@K/MRR；除非某行另有说明
- PLANETALIGN 参考聚合：5 个数据集划分，每个划分 5 次运行；超参数搜索使用论文规定的固定预算
- 当前共享评估器注意事项：`BaseModel.test` 默认采用乐观的并列处理，而 PLANETALIGN 论文结果使用普通 argsort
- 结果来源政策：复用 PLANETALIGN 表格和可靠的已有日志；已完成的仅终端 sanity 结果会明确标注；缺失值保持缺失

## 一对一校准总表

`Gap` 表示 Hits@1 的“当前结果减论文结果”，单位为百分点。PLANETALIGN 表 5 的行比较了基线官方实现与对应 PLANETALIGN 实现在论文统一协议下的结果。

| 方法 | 参考来源 | 数据集 | 指标 | 论文结果 | 当前结果 | 差值 | 协议差异 | 校准状态 | M2M 准入 |
|---|---|---|---|---:|---:|---:|---|---|---|
| IsoRank | PLANETALIGN 表 8 | PeMS08 | 双向普通 Hits@1/Hits@10/MRR | 0.2537 / 0.7353 / 0.4197 | 0.253676 / 0.735294 / 0.419663 | 约 0.00 个百分点 | 已完成的审计运行使用 `data`、20% 锚点、seed 0、普通图、`alpha=0.4`、100 个 epoch、普通 argsort。同一矩阵在当前乐观默认设置下的 Hits@1 为 0.264706。 | `CALIBRATED_WITH_PROTOCOL_GAP` | `READY_FOR_M2M` |
| FINAL | PLANETALIGN 表 8 | PeMS08 | 双向 Hits@1 | 0.1985 | 0.209559 | +1.11 个百分点 | 已有当前结果为 seed 0、普通图、`alpha=0.5`、float64、乐观并列处理；论文使用普通 argsort，并聚合多个划分和运行。Phone-Email 与 Italy 结果也与论文接近。 | `CALIBRATED_WITH_PROTOCOL_GAP` | `READY_FOR_M2M` |
| IONE | PLANETALIGN 表 8 | PeMS08 | 双向普通 Hits@1 | 0.3824 +/- 0.0252 | 缺失 | 缺失 | 没有可靠的当前一对一结果。实现默认执行 100 个外层 epoch 乘以 100,000 次基础更新；缩短运行不作为校准证据。 | `UNCHECKED` | `PENDING_CALIBRATION` |
| REGAL | PLANETALIGN 表 5 | 18 个数据集平均值 | 双向 Hits@1 | 0.079（官方） | 0.080（PLANETALIGN 实现） | +0.10 个百分点 | 直接比较官方实现与 PLANETALIGN 实现。当前评估器默认采用乐观并列处理；当前 seed-0 的 Phone-Email 与 Italy 日志支持该已发布复现结果。 | `CALIBRATED_WITH_PROTOCOL_GAP` | `READY_FOR_M2M` |
| CrossMNA | PLANETALIGN 表 5 | 18 个数据集平均值 | 双向 Hits@1 | 0.220（官方） | 0.222（PLANETALIGN 实现） | +0.20 个百分点 | 直接比较官方实现与 PLANETALIGN 实现；当前共享评估器默认设置不同于论文时期的普通 argsort。 | `CALIBRATED_WITH_PROTOCOL_GAP` | `READY_FOR_M2M` |
| NetTrans | PLANETALIGN 表 5 | 18 个数据集平均值 | 双向 Hits@1 | 0.373（官方） | 0.374（PLANETALIGN 实现） | +0.10 个百分点 | 直接比较官方实现与 PLANETALIGN 实现；当前共享评估器默认设置不同于论文时期的普通 argsort。 | `CALIBRATED_WITH_PROTOCOL_GAP` | `READY_FOR_M2M` |
| WAlign | PLANETALIGN 表 5 | 18 个数据集平均值 | 双向 Hits@1 | 0.271（官方） | 0.270（PLANETALIGN 实现） | -0.10 个百分点 | 直接比较官方实现与 PLANETALIGN 实现；当前共享评估器默认设置不同于论文时期的普通 argsort。 | `CALIBRATED_WITH_PROTOCOL_GAP` | `READY_FOR_M2M` |
| BRIGHT | PLANETALIGN 表 5 | 18 个数据集平均值 | 双向 Hits@1 | 0.362（官方） | 0.362（PLANETALIGN 实现） | 0.00 个百分点 | 直接比较官方实现与 PLANETALIGN 实现；当前共享评估器默认设置不同于论文时期的普通 argsort。 | `CALIBRATED_WITH_PROTOCOL_GAP` | `READY_FOR_M2M` |
| NeXtAlign | PLANETALIGN 表 5 | 18 个数据集平均值 | 双向 Hits@1 | 0.391（官方） | 0.391（PLANETALIGN 实现） | 0.00 个百分点 | 直接比较官方实现与 PLANETALIGN 实现；当前共享评估器默认设置不同于论文时期的普通 argsort。 | `CALIBRATED_WITH_PROTOCOL_GAP` | `READY_FOR_M2M` |
| WLAlign | PLANETALIGN 表 5 | 18 个数据集平均值 | 双向 Hits@1 | 0.328（官方） | 0.322（PLANETALIGN 实现） | -0.60 个百分点 | 直接比较官方实现与 PLANETALIGN 实现。差异仍在合理的复现范围内；当前评估器默认采用乐观并列处理。 | `CALIBRATED_WITH_PROTOCOL_GAP` | `READY_FOR_M2M` |
| PARROT | PLANETALIGN 表 8 | PeMS08 | 双向普通 Hits@1/Hits@10/MRR | 0.7647 / 0.9228 / 0.8209 | 0.757353 / 0.930147 / 0.816960 | -0.73 个百分点 | 已完成的审计运行使用 `data`、20% 锚点、seed 0、普通图和项目固定默认配置；未进行调参。 | `CALIBRATED_WITH_PROTOCOL_GAP` | `READY_FOR_M2M` |
| SLOTAlign | PLANETALIGN 表 5 | 18 个数据集平均值 | 双向 Hits@1 | 0.200（官方） | 0.200（PLANETALIGN 实现） | 按显示精度为 0.00 个百分点 | 直接比较官方实现与 PLANETALIGN 实现；当前共享评估器默认设置不同于论文时期的普通 argsort。 | `CALIBRATED_WITH_PROTOCOL_GAP` | `READY_FOR_M2M` |
| HOT | PLANETALIGN 表 5 | 18 个数据集平均值 | 双向 Hits@1 | 0.173（官方） | 0.172（PLANETALIGN 实现） | -0.10 个百分点 | 表 5 支持该实现。表 8 中 PeMS08 的 MRR 单元格与 Hits@1 内部不一致，因此不作为校准证据。 | `CALIBRATED_WITH_PROTOCOL_GAP` | `READY_FOR_M2M` |
| JOENA | PLANETALIGN 表 5 和表 9 | 18 个数据集平均值 / Douban | 双向 Hits@1 | 0.583（官方） / 0.6542 +/- 0.0474（Douban） | 0.583（PLANETALIGN 实现） / 0.63184（已完成审计的 Douban 运行） | 0.00 个百分点 / -2.24 个百分点 | JOENA 论文中的 0.761 是源到目标方向，不可与 PLANETALIGN 的双向平均值直接比较。现有校准已足够。 | `CALIBRATED_WITH_PROTOCOL_GAP` | `READY_FOR_M2M` |
| GADL（官方实现） | GADL 论文 | Douban | 源到目标公开乐观 Hits@1 | 0.5331 | 0.537865 +/- 0.003748 | +0.48 个百分点 | 无监督、图一致的公开代码复现。公开代码配置与论文声明的配置不同。 | `CALIBRATED_WITH_PROTOCOL_GAP` | 不分配；仅作参考行 |
| GADL（PlanetAlign 集成） | GADL 论文 | Douban | 源到目标乐观 Hits@1 | 0.5331 | 0.319320 +/- 0.003347 | -21.38 个百分点 | 并列处理只能解释差距中的很小一部分；图/谱预处理和训练语义与公开实现不同。 | `SUSPICIOUS` | `DO_NOT_USE` |

## 覆盖范围总结

- 已审计的主要方法：15 个（PLANETALIGN 的 14 种方法加 GADL）。
- 当前存在的实现：15/15；`NOT_IMPLEMENTED`：0。
- 有证据表明性能处于合理范围内的算法实现：13 个。
- `CALIBRATED_WITH_PROTOCOL_GAP`：IsoRank、FINAL、REGAL、CrossMNA、NetTrans、WAlign、BRIGHT、NeXtAlign、WLAlign、PARROT、SLOTAlign、HOT、JOENA。
- `SUSPICIOUS`：当前 PlanetAlign 的 GADL 集成。
- `UNCHECKED`：IONE。
- 有可靠当前结果但仍等待校准决策的方法：无。GADL 已完成诊断，并有意保持为可疑状态。
- PLANETALIGN 表 5 提供了 REGAL、CrossMNA、NetTrans、WAlign、BRIGHT、NeXtAlign、WLAlign、SLOTAlign、HOT 和 JOENA 的官方代码对比。
- GADL 有独立的官方代码检出版本和图一致复现结果。
- IsoRank、FINAL、IONE 和 PARROT 的官方代码可用性未单独重新审计；这不应被解读为没有官方代码。

## 最小重跑清单

生成本记录时没有重跑任何实验。未来唯一仍需要进行的校准运行是：

| 方法 | 重跑原因 | 数据集 | 配置 | Seeds | 预期目的 |
|---|---|---|---|---|---|
| IONE | 没有可靠的当前一对一结果 | PeMS08 | `data`、20% 锚点、普通图、`out_dim=100`、`total_epochs=100`、`base_epochs=100000`；普通双向 Hits@1/Hits@10/MRR | 首先运行 seed 0；只有当 seed 0 结果不明确时才增加 1 和 2 | 判断当前实现是否处于已发布结果 0.3824 +/- 0.0252 的合理范围内 |

GADL 不在重跑清单中：已有多 seed 和受控配置证据已经证明存在实现差距。重复运行未改变的实现不会增加校准信息。

## 可疑实现

### GADL（当前 PlanetAlign 集成）

- 参考行为：论文中的 Douban 源到目标 Hits@1 为 0.5331；图一致的公开代码复现为 0.537865 +/- 0.003748。
- 当前行为：乐观 Hits@1 为 0.319320 +/- 0.003347；平均并列 Hits@1 为 0.291294 +/- 0.003748。
- 已确认的实现/协议差异：
  - PlanetAlign 实现使用 PReLU，而公开实现使用恒等激活。
  - 使用稠密、无权重的 BCE，而不是加权重构 BCE。
  - 使用常数 functional-map 调度，而不是线性下限调度。
  - functional-map 初始化、谱基归一化和损失权重映射存在差异。
  - 论文使用全图预处理，而公开基线使用兼容基线的图预处理。
  - 论文声明的 `hidden_dim=256`、6 层和 `lr=1e-3` 与公开代码的 `hidden_dim=512`、3 层和 `lr=1e-4` 不一致。
- 实验支持：隔离的公开实现兼容 PlanetAlign 封装仅达到 0.362254 +/- 0.005978。并列策略约解释 2.8 个百分点，而不是约 21 个百分点的主要差距。
- 建议的最小修复：保持当前实现和日志不变；增加一个隔离的官方行为适配器/配置，导出公开实现的相似度矩阵。在允许 M2M 准入前，要求进行三 seed 的 Douban 校准。
- 影响：仅涉及未来的 GADL 结果和缓存。现有 GADL 日志仍是诊断/复现工件，不得重新标记为有效基线。

### 参考表注意事项

PLANETALIGN 表 8 中 HOT 的 PeMS08 MRR 值小于其 Hits@1 值，对于同一组排序预测而言这在数学上是不可能的。该单元格被视为参考表内部不一致，而不是 HOT 实现存在缺陷的证据。

## M2M 准入决策

| 基线 | 决策 | 原因 |
|---|---|---|
| IsoRank | `READY_FOR_M2M` | 精确复现普通 argsort 的 PeMS08 sanity 结果；必须显式固定评估器默认设置。 |
| FINAL | `READY_FOR_M2M` | 现有结果处于已发布范围内，且并列/划分差异已有解释。 |
| IONE | `PENDING_CALIBRATION` | 没有可靠的当前一对一结果。 |
| REGAL | `READY_FOR_M2M` | 官方实现与 PLANETALIGN 的复现已完成校准；适配器必须保留 `emb1 @ emb2.T`。 |
| CrossMNA | `READY_FOR_M2M` | 官方实现与 PLANETALIGN 的复现已完成校准。 |
| NetTrans | `READY_FOR_M2M` | 官方实现与 PLANETALIGN 的复现已完成校准。 |
| WAlign | `READY_FOR_M2M` | 官方实现与 PLANETALIGN 的复现已完成校准。 |
| BRIGHT | `READY_FOR_M2M` | 官方实现与 PLANETALIGN 的复现已完成校准。 |
| NeXtAlign | `READY_FOR_M2M` | 官方实现与 PLANETALIGN 的复现已完成校准。 |
| WLAlign | `READY_FOR_M2M` | 官方实现与 PLANETALIGN 的复现处于合理范围内。 |
| PARROT | `READY_FOR_M2M` | PeMS08 sanity 复现结果在一个百分点以内，且未调参。 |
| SLOTAlign | `READY_FOR_M2M` | 官方结果与 PLANETALIGN 结果在显示精度下匹配。 |
| HOT | `READY_FOR_M2M` | 表 5 的官方对比支持该实现；异常的表 8 MRR 单元格被排除。 |
| JOENA | `READY_FOR_M2M` | 现有官方对比和直接 Douban 校准已经足够。 |
| GADL（当前 PlanetAlign 集成） | `DO_NOT_USE` | 在评估器和配置控制之后，仍存在较大的未解释实现差距。 |

`READY_FOR_M2M` 仅表示一对一骨干通过了校准。不允许因此复用旧的 M2M 数值，也不代表可以更改任何特定数据集的读出方式或配置。

## 证据清单

- PLANETALIGN 论文：`/Users/yukexin/Documents/akeyan/papers/15073_PLANETALIGN_A_Comprehens.pdf`
- JOENA 论文：`/Users/yukexin/Documents/akeyan/papers/Joint_Optimal_Transport_And_Embedding_for_na.pdf`
- GADL 论文：`/Users/yukexin/Documents/akeyan/papers/dual-pass_spectral_encoding.pdf`
- FINAL/REGAL 现有结果：`.reference/Alpine/planetalign_benchmark/results/existing_baselines_all_seed0.json`
- GADL 官方图一致结果：`logs/gadl_one_to_one/gadl_reference_graph_consistent.json`
- GADL 当前集成结果：`logs/gadl_one_to_one/douban_full_training_seeds_0_1_2.json`
- GADL 公开实现兼容配置结果：`logs/gadl_one_to_one/douban_public_reference_compatible_seeds_0_1_2.json`
- GADL 配置定义：`PlanetAlign/algorithms/gadl/profiles.py` 和 `PlanetAlign/algorithms/gadl/paper_profile.py`
- 当前共享评估器默认设置：`PlanetAlign/algorithms/base_model.py`
- IsoRank、PARROT 和直接 JOENA sanity 数值是在已完成审计中较早生成的，使用 `save_log=false`；没有创建或覆盖原始日志。

## 变更范围

为生成本校准记录，没有修改任何算法、评估器、指标定义、配置、缓存或原始日志。
