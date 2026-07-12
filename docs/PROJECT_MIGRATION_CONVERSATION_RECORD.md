# PlanetAlign M2M 项目状态与迁移记录

> 更新时间：2026-07-13（Asia/Shanghai）
> 工作区：`/Users/yukexin/Documents/akeyan/code/PlanetAlign`
> 当前分支：`migrate-m2m-from-backup`
> 记录时提交：`5648bf8`
> 环境：macOS Darwin 25.5.0 arm64，Python 3.13.12
> 说明：本文是便于迁移与交接的当前状态记录，覆盖最终算法、验证结果、已解决问题、源码位置、复现命令与后续工作。历史演进（曾用的 JOENA 集成、被证伪的分支仲裁等）保留在 `docs/m2m_quotient_align_design.md` 的 §5.x 实录中，本文只反映**当前**系统。

---

## 1. 项目目标与最终系统

目标：在从一对一数据合成的多对多（M2M）基准上，做出一个统一、盲评、无逐集调参的多对多图对齐系统，并把"一对一方法为何失败"讲清楚（每条结论有实验 + 原论文理论双证）。

**最终系统（当前）= PARROT 基座 + QuotientDecode 读出**：

```text
PARROT(G1, G2, 训练锚点)        # 位置感知一致性正则 OT；确定性，秒级
        │  S  (节点相似度矩阵)
arbitrate_sharpen(S)            # raw vs softmax(S/T)，按机会校正锚点一致率盲选
        │
QuotientDecode(S)               # 商图读出（基座无关）
        │
    实体映射 → 盲评 ACS / MSF1 / MicroF1 / M2M-SGS / M2M-EGS
```

一句话核心：**一对一排他性先验没有错，而是被放在了错误的节点层**。M2M 的正确学习对象是"每图一个等价类划分 + 商图间一一匹配"；分组证据早已藏在任何 1-1 对齐器的 `S` 里（同实体节点的 `S` 行近共线）。

**一个盲信号贯穿四层**：分组证据选择、合并阈值、锐化温度、基座选择，全部由机会校正的训练锚点组级一致率 + 奥卡姆仲裁；真值实体从不进预测路径。

> 历史说明：早期"最终系统"是 `JOENA + JOENA-PC + JOENA-PC-GM` 三分支耦合平均 + QuotientDecode。多种子全网格重验证（§5.13）表明 **PARROT 是更强的统一基座**（7/9 数据集直接最优、确定性、快约 20×），已取代集成成为默认；JOENA 家族集成保留为可选精修（`--model ensemble/auto`）。

---

## 2. 最终结果（盲评 MicroF1，修正协议，4 种子；PARROT 确定性 std=0）

| 数据集 | 起点（最佳 1-1 读出） | **最终系统** | 备注 |
|---|---:|---:|---|
| douban | 0.318 | **0.730** | 旧集成 0.694，PARROT +0.036 |
| cora | 0.893 | **0.981**（集成可盲选中 0.992） | 属性天花板 0.991 |
| airport | 0.27 | **0.798** | |
| pems08 | ~0.29 | **0.601** | 集成 0.643 为诚实残差（锚点饱和，无法盲选中） |
| ppi | — | **0.930** | |
| arenas | 0.003 | **0.851** | 留出 1-1 实体 Hits@1=0.981 |
| phone-email | 0.000 | **0.424** | **盲评反超泄漏式适配器 0.230** |
| italy | 0.000 | **0.372** | 同上（弱基座三集盲评全部反超泄漏适配器） |
| foursquare | 0.051 | **0.404** | 同上 |

强属性五集均值：PARROT 0.808 vs 旧 JOENA 集成 0.789（+0.019），且四个弱属性集全部被 PARROT 复活。同一基座 + 同一读出、零逐集调参，9 个数据集全部非平凡。

> 复现：`scripts/validate_base_selection.py` + `scripts/compare_parrot_vs_ensemble.py` + `scripts/aggregate_unified_grid.py`；详见设计文档 §5.13。

---

## 3. 已解决的关键问题（曾是 P0/P1）

本记录上一版（提交 `a37be5b`，2026-07-11）列出的实验协议隐患，现已全部修复：

1. **训练锚点二次切分（名义 20% 实为 4%）** → `PlanetAlign.m2m.use_full_anchor_supervision(dataset)`：M2M 的 `anchor_links` 本就是原始 20% 训练切分，实体真值来自原始测试切分（已验证 9 数据集监督∩评测=∅）。所有脚本默认走修正协议（缓存标签 `_fa`）。见 §5.11。
2. **`algo.test()` Hits 测错集合** → 改测构成 M2M 实体的原始测试锚点；全量锚点协议下用 held-out 1-1 实体 Hits 诊断。
3. **跨基座不公平对比** → 受控消融固定同一个 `S`；基座作为一等**可仲裁轴**（`m2m_base.py`）盲选，不再手工挑。见 §5.13。
4. **四数据集"不可解"误判** → 系探测缺陷（probe 名单缺 PARROT + quick 预算），非数据缺陷；补全后四集全部复活，"拆分摧毁结构信号"的机理判断撤回，基准 v2 提议撤回。见 §5.11.1。
5. **flickr-lastfm 生成器泄漏** → 训练锚点对 (4227,11939) 漏进测试实体 e62；根因是原始锚点表重复行 + 生成器只在测试集内去重。已修：跨集合端点排除 + 构建末端硬护栏（泄漏即抛错）+ 回归测试；两 root 复扫 18/18 干净。见 §5.12。
6. **缓存键碰撞（overlap/no-overlap 同名）** → 缓存键含数据集名 + 协议标签 `_fa` + 种子 + epochs；overlap/no-overlap 走不同 root 目录，不再串用。

仍未复现原论文 JOENA 的 Arenas 0.987（缺论文最优超参 + 5×5 划分设置）——但这不影响系统，因为 **PARROT 在本地精确复现原版 Arenas 0.988**，已作为统一基座。

---

## 4. 后续工作（当前优先级）

- **(a) FGW / srGW 联合块结构方向**（已立项未测，见 §7-5 与记忆）：当前分组与匹配解耦，正确对象是 `S` 的联合块结构。候选：商图 Fused Gromov-Wasserstein 匹配（替代匈牙利 + 线性邻居平滑，后者是其一阶近似）、半松弛 GW 分区发现（多对一坍缩天然表达 M2M）。**警示**：组级 UOT 已证伪（排他性承重），FGW 须保持精确边际、只加结构代价项。POT 0.9.6 已内置求解器。
- **(b) 稀疏基座**：8 个大图（`n≈24k`）稠密 `O(n²)` 不可行（PARROT 同样稠密），需稀疏/低秩/分块。
- **(c) 通用组内不连通兄弟发现**：不依赖相邻/完全相同属性/固定生成规则。
- **(d) 谱指纹接入仲裁**：`erank/n≈1 ⇒ 置换锁死 ⇒ 不信饱和的锚点一致率`，用于标记 pems08 型基座仲裁盲区。

---

## 5. 关键源码位置（当前）

| 作用 | 文件 |
|---|---|
| QuotientDecode（核心读出） | `PlanetAlign/m2m_quotient.py` |
| 基座轴（PARROT 等）+ 锐化仲裁 | `PlanetAlign/m2m_base.py` |
| JOENA-PC / 组级边际（训练侧） | `PlanetAlign/m2m_contrastive.py` |
| 盲协议与分组发现 | `PlanetAlign/m2m_blind.py` |
| M2M 通用接口（`use_full_anchor_supervision` 等） | `PlanetAlign/m2m.py` |
| M2M 数据生成 + 泄漏硬护栏 | `PlanetAlign/utils/many2many_builder.py` |
| 指标 | `PlanetAlign/metrics/`（`metrics_ACS.py`、`metrics_MSF1.py`、`metrics_Micro_SF1.py`、`metrics_m2m_SGS.py`、`metrics_m2m_EGS.py`、`many_to_many.py`） |
| 主实验 runner | `scripts/run_quotient_compare.py` |
| 基座探测 / 基座选择网格 | `scripts/probe_base_aligners.py`、`validate_base_selection.py`、`compare_parrot_vs_ensemble.py`、`aggregate_unified_grid.py` |
| 读出变体配对多种子扫描 | `scripts/sweep_readout_variants.py` |
| M2M 批量构建 | `scripts/build_m2m_benchmarks.py` |
| 诊断（1-1 失败 / RMT 谱） | `scripts/diagnose_m2m_failure.py`、`scripts/diagnose_rmt.py` |
| 设计与实验实录 | `docs/m2m_quotient_align_design.md` |
| 1-1 失败分析 | `docs/m2m_why_one_to_one_fails.md` |
| 论文素材版总结 | `docs/m2m_research_summary_for_paper.md` |

> 已在整理中删除的历史文件（被 QuotientDecode 取代，正式路径不依赖）：`PlanetAlign/m2m_decode.py`（旧 JOENAGroupDecode）、`scripts/run_joena_group_decode.py`、`scripts/run_joena_m2m_refinement.py`、根目录 `run_m2m_benchmark.py`/`run_m2m_tgae.py`/`run_m2m_tgae_perturb.py` 及其测试。

---

## 6. 关键复现命令

### 单元测试（52 项，每项守护一条设计声明）

```bash
python -m unittest discover -s tests -p "test_*.py"
```

### 构建 M2M 基准

```bash
python scripts/build_m2m_benchmarks.py \
  --input-root data --output-root data/m2m_no_overlap \
  --overlap-ratio 0 --keep-going
```

### 最终统一系统（PARROT 基座 + QuotientDecode）

```bash
python scripts/run_quotient_compare.py \
  --root data/m2m_no_overlap --dataset ppi_m2m --base PARROT
```

### 全网格基座选择重验证（§5.13）

```bash
python scripts/validate_base_selection.py --seeds 42 0 1 2 --bases PARROT
python scripts/compare_parrot_vs_ensemble.py --seeds 42 0 1 2
python scripts/aggregate_unified_grid.py
```

### 可选精修：JOENA 家族集成 / 锚点仲裁

```bash
python scripts/run_quotient_compare.py \
  --root data/m2m_no_overlap --dataset cora_m2m --model auto
```

---

## 7. 迁移清单

必须复制（源码 + 文档；数据可重建）：

```text
PlanetAlign/    scripts/    tests/    docs/
data/*.pt                      # 1-1 源数据（用于重建 M2M）
```

数据（生成产物，被 `.gitignore` 忽略）可选复制或用 `build_m2m_benchmarks.py` 重建：

```text
data/m2m_no_overlap/    data/m2m_overlap_0.05/
```

日志与缓存（`logs/` 全部 gitignore，体积大且可重算）按需复制：

```text
logs/m2m_diag/*.json    logs/m2m_diag/S_cache/    logs/m2m_diag/multiseed/
```

迁移后自检：

```bash
git status --short
python -m unittest discover -s tests -p "test_*.py"     # 期望 52 passing
python scripts/run_quotient_compare.py --help
python -c "from PlanetAlign import m2m, m2m_base, m2m_quotient, m2m_blind, m2m_contrastive; print('OK')"
```

确认：PyTorch / PyG / SciPy / POT 可用；`data/m2m_no_overlap/*_gt_many2many.json` 与 `.pt` 成对存在；M2M 源码已提交（当前分支已全部提交并推送至 PR #2）。
