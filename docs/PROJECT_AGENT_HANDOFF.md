# PlanetAlign 多对多图对齐项目：智能体交接全景

> 更新时间：2026-08-06（Asia/Shanghai）  
> 工作区：`/Users/yukexin/Documents/akeyan/code/PlanetAlign`  
> 当前分支：`migrate-m2m-from-backup`  
> 当前提交：`f72d2e0`（`feat: Fused Gromov-Wasserstein quotient matcher (option) + srGW grouping probe`）  
> 远端状态：当前分支跟踪 `origin/migrate-m2m-from-backup`；相对 `upstream/main` 为 0 behind / 28 ahead  
> 本文用途：让新的研究或编码智能体在提出修改建议前，迅速掌握当前算法、实验协议、可信结果、历史证伪、工程风险和修改优先级。

---

## 0. 先读结论

1. 当前主系统是 **PARROT + QuotientDecode**，不是早期的 JOENA 三分支集成。
2. 核心创新是把一对一排他性从节点层移动到实体组（商图）层：先发现每图的实体等价类，再在商图间做全局一一匹配。
3. 当前论文级主结果采用 `data/m2m_no_overlap`、完整原始 20% 训练锚点、盲实体评测；不能把旧 4% 双重切分结果、overlap 结果、一对一 Hits@1 和实体 MicroF1 混在一起。
4. 数据生成器已修复 Flickr-LastFM 的跨 train/test 端点泄漏，并有硬护栏和回归测试；两套 root 的当前清单均记录了修复 amendment。
5. PARROT 在九个当前主评测数据集上统一运行，7/9 直接最优，确定性、速度快；JOENA-PC/GM 和三分支集成只应视为可选研究分支。
6. FGW 商图匹配已实现并验证，但只应是可选项：固定 `alpha=0.1` 对九集平均约 `+0.0034`，收益主要来自 Douban `+0.0238`，训练锚点无法可靠选择匹配器。
7. 最大研究缺口是外部效度和规模：目前基准来自同一种“一对一图节点拆分”生成族，且稠密 `S`/OT 路径为 `O(n²)`，大图尚未进入最终网格。
8. 最大工程风险是**缓存缺少数据 root/数据哈希/完整配置**。同名 overlap 与 no-overlap 数据可能读取同一缓存；旧迁移文档关于该问题“已修复”的表述与当前代码不一致。
9. README、模块注释和 CLI 默认值存在状态漂移：README 说默认 PARROT，`m2m/quotient.py` 顶部仍说 ensemble，主 runner 默认仍是 overlap + JOENA。
10. 当前 Conda 环境中 56 项单元测试全部通过；系统 `/usr/bin/python3` 缺少 PyTorch，不能用于本项目。

---

## 1. 项目是什么

本仓库是上游 [PlanetAlign](https://github.com/yq-leo/PlanetAlign) 的研究分支。上游目标是统一评测一对一网络对齐算法；本分支增加了完整的多对多（many-to-many, M2M）网络对齐研究管线。

M2M 中，一个现实实体可在任一图中对应多个节点：

```text
图 1：{Alice-1, Alice-2}
图 2：{Alice-a, Alice-b, Alice-c}

正确输出：{Alice-1, Alice-2} ↔ {Alice-a, Alice-b, Alice-c}
```

传统一对一方法直接对节点相似度矩阵 `S` 做 argmax/一一指派，会让同实体节点互相竞争。本项目的核心判断是：

> 一对一排他性先验本身没有错，但放错了层级。节点层应允许多对多；把同实体节点压缩成组后，实体组之间重新适合一对一全局匹配。

---

## 2. 当前主算法

### 2.1 端到端流程

```text
输入：两个图 G1/G2、节点属性（若有）、训练锚点 A
  ↓
PARROT：产生节点级相似度/耦合矩阵 S
  ↓
arbitrate_sharpen：在 raw S 与 softmax(S/T) 中用训练锚点盲选
  ↓
QuotientDecode：
  1. 将 S 的行/列视为节点“对齐画像”
  2. 在每张图内生成候选实体分区
  3. 用训练锚点选择分区证据
  4. 对另一侧分组池化画像，交替细化
  5. 构造组间得分矩阵 T
  6. 默认用矩形 Hungarian 在商图层一一匹配
  ↓
输出：实体映射 {pred_entity: {src:[...], tgt:[...]}}
  ↓
盲评：Macro Set-F1 / Micro Set-F1 / AUIS / AUCS
```

### 2.2 QuotientDecode 的组件

- **Alignment profile**：源节点用 `S` 的行、目标节点用 `S` 的列表示。共指节点应关注对侧同一区域，因此画像近共线。
- **自适应阈值**：Otsu 阈值替代固定 `tau`。
- **随机零分布候选**：用随机节点对画像余弦的高分位估计无信号上界，补救 Otsu 过高。
- **属性候选**：属性强时可形成更可靠的实体组，但弱属性图可能严重过合并。
- **全局候选**：寻找图内不相邻、但属性相容且画像接近的兄弟节点；有足够训练锚点时才自动启用。
- **反链式合并**：average-linkage/组质心门控，防止 `A~B, B~C` 把不相似的 A/C 串成一组。
- **异常成员驱逐**：只对由 profile 证据形成的组执行 leave-self-out 驱逐，避免用错误证据拆属性组。
- **交替细化**：用对侧实体组池化 `S`，将节点画像从节点维降至组维，再重新分组。
- **商图匹配**：默认 Hungarian；可选 FGW；greedy 仅用于消融。
- **模糊边界扩展**：`overlap_expand_tau` 可让节点加入第二实体，默认关闭。

核心实现：

- `PlanetAlign/m2m/quotient.py::quotient_decode`
- `PlanetAlign/m2m/quotient.py::evaluate_quotient_blind`
- `PlanetAlign/m2m/base.py::train_base_S`
- `PlanetAlign/m2m/base.py::arbitrate_sharpen`

### 2.3 为什么当前基座是 PARROT

早期系统是 `JOENA + JOENA-PC + JOENA-PC-GM` 三个耦合矩阵求平均，再运行 QuotientDecode。后续 4 seeds × 9 datasets 重验证表明：

- PARROT 在九集中的七集直接更好；
- 能恢复 JOENA 几乎失效的 Arenas、Phone-Email、Italy、Foursquare；
- 对固定数据与配置是确定性的；
- 单基座远快于训练三个 JOENA 分支；
- 强属性五集均值约 `0.808`，高于旧集成约 `0.789`。

因此当前默认研究结论应写为：

```text
PARROT + QuotientDecode
```

JOENA 集成只在 Cora 和 PeMS08 上仍有局部优势，但训练锚点无法在 PeMS08 上识别该优势。

### 2.4 JOENA-PC 的正确定位

`PlanetAlign/m2m/contrastive.py` 实现：

- 传播属性、传播 RWR、log-degree 等对称破缺特征；
- profile uniformity；
- 自举正对；
- 可选 group-level Sinkhorn marginals。

它对解释 Douban 的重复耦合行非常重要，但不是当前统一系统必需组件。已有消融显示主要增益来自**输入对称破缺**，profile 对比损失只提供小幅附加收益。建议：

- 论文中作为“表示质量决定读出证据可用性”的机制研究保留；
- 工程默认路径不依赖它；
- 不应把三分支 ensemble 再描述成当前默认系统。

### 2.5 FGW / srGW 当前状态

最新提交 `f72d2e0` 新增：

- `matcher="fgw"` 和 `fgw_alpha`；
- 商图邻接作为结构代价、池化 `S` 作为特征代价；
- FGW 软计划经 Hungarian 取整，仍保持商图层精确排他性；
- `tests/test_m2m_fgw.py`；
- `scripts/test_fgw_matcher.py`。

固定 `alpha=0.1` 相对 Hungarian 的 MicroF1 变化：

| 数据集 | Δ MicroF1 |
|---|---:|
| douban | +0.0238 |
| cora | +0.0018（固定 α 表）；日志中的锚点挑选结果为 +0.0005 |
| airport | +0.0017 |
| pems08 | 0.0000 |
| ppi | +0.0004 |
| arenas | +0.0041 |
| phone-email | +0.0020 |
| italy | -0.0018 |
| foursquare-twitter | -0.0016 |

结论：FGW 有清晰机制和小幅真实收益，但不能由训练锚点可靠盲选，暂不应设为默认。

srGW 分组只做过探测：Italy `+0.015`、Arenas `-0.021`。它替换而非补充 PARROT 的 `S`，且仍为稠密 `O(n²)`，因此未集成。

---

## 3. 数据与监督协议

### 3.1 数据集

两个生成 regime：

- `data/m2m_no_overlap`：实体边界不模糊，主论文结果使用该 root；
- `data/m2m_overlap_0.05`：约 5% 模糊实体成员重叠，用于鲁棒性/边界实验。

每个 root 有 18 套 `.pt + _gt_many2many.json`：

| 数据集 | 新图节点数 | 实体数 | 当前主网格 |
|---|---:|---:|---|
| airport | 1563 / 1554 | 952 | 是 |
| arenas | 1488 / 1495 | 908 | 是 |
| cora | 3595 / 3553 | 2167 | 是 |
| douban | 4265 / 1460 | 895 | 是 |
| foursquare-twitter | 5840 / 5618 | 1288 | 是 |
| italy | 1166 / 1319 | 302 | 是 |
| pems08 | 223 / 230 | 136 | 是 |
| phone-email | 1347 / 1317 | 800 | 是 |
| ppi | 4577 / 4600 | 2784 | 是 |
| sacchcere | 5441 / 1834 | 1070 | 否，原因未在当前主表解释清楚 |
| acm-dblp | 11896 / 11907 | 5060 | 否，规模 |
| arxiv | 24775 / 24753 | 15018 | 否，规模 |
| dbp15k fr-en | 24467 / 24791 | 12000 | 否，规模 |
| dbp15k ja-en | 24620 / 24578 | 12000 | 否，规模 |
| dbp15k zh-en | 24194 / 24370 | 12000 | 否，规模 |
| flickr-lastfm | 13115 / 15582 | 361 | 否，规模 |
| flickr-myspace | 6802 / 10807 | 214 | 否，规模 |
| ggi | 13752 / 13674 | 8323 | 否，规模 |

README 的“9 个小图 + 8 个大图”只计到 17 个，实际遗漏了 SacchCere 的归类。后续应明确它为何未进入最终九集网格。

### 3.2 生成参数

清单记录的标准配置：

```text
source train_ratio = 0.2
seed = 42
entity type ratios = 1-1:0.7, 1-many:0.1, many-1:0.1, many-many:0.1
max_expansion = 4
internal_density = 0.8
anchor_source = test
train_anchor_source = train
overlap_ratio = 0 或 0.05
```

含义：原一对一数据的训练锚点保存到生成数据的 `anchor_links`；原测试锚点的实体端点用于生成 M2M 测试实体。

### 3.3 20% → 4% 双重切分问题

通用 `Dataset` 加载后会按 `train_ratio` 再切一次 `anchor_links`。但生成数据中的 `anchor_links` 已经是原始 20% 训练划分，再切一次会变成约 4% 总监督。

M2M 正式协议必须在加载后立即调用：

```python
use_full_anchor_supervision(dataset)
```

它只做：

```python
dataset.train_data = dataset._anchor_links.clone()
dataset.test_data = dataset._anchor_links[:0]
```

这恢复原定 20%，不是增加新监督。此时 `dataset.test_data` 为空，所以普通 `algo.test()` 的节点 Hits 不再是合法 held-out 结果；主评测必须是盲实体评测。

### 3.4 泄漏修复

历史问题：原 Flickr-LastFM 锚点表存在重复 pair `(4227, 11939)`，两份重复行被随机切到 train/test，旧生成器只在测试集合内部去重，使该训练锚点直接进入测试实体 `e62`。

当前修复：

- 构造测试实体前排除所有与训练锚点端点碰撞的 pair；
- overlap 插入同样禁止引入训练端点；
- 构建结束执行硬不变量检查，发现交叉即抛错；
- `tests/test_m2m_builder_leakage.py` 覆盖重复 pair 和 overlap 再引入；
- 当前两套清单均含 2026-07-12 amendment，Flickr-LastFM 实体数从 362 变为 361。

“overlap 0.05”指测试实体之间的模糊成员重叠，不是允许训练节点与测试实体重叠。

### 3.5 数据文件的版本风险

`data/**/*.pt` 和 `data/**/*.json` 被 Git 忽略。当前约 987 MB 数据不会随提交自动迁移。其他智能体应注意：

- Git 历史不能证明本地生成数据未被手工替换；
- 两套 manifest 的部分 `output_root/pt_path/json_path` 仍写旧目录名 `data/m2m_overlap`，而实际目录是 `data/m2m_overlap_0.05`；
- 若重建数据，应保存清单、数据哈希和生成提交 SHA；
- 不要在没有备份当前忽略文件的情况下直接覆盖。

---

## 4. 评测协议与指标

### 4.1 “盲评”的定义

预测路径可使用：

- 两张输入图；
- 节点属性；
- 训练锚点；
- 基座输出的 `S`。

预测路径不得使用：

- GT 的源实体分组；
- GT 的目标实体规模；
- GT 实体数；
- 根据测试 MicroF1 挑阈值、基座、温度或 matcher。

`evaluate_quotient_blind` 先在完全不读取 GT 实体的情况下预测，再在评测阶段将预测实体 ID 与 GT 实体 ID 对齐并计算指标。实体 ID 对齐使用成员集合重叠，只是标签置换适配，不应被误写成预测算法的一部分。

### 4.2 指标

- `Micro Set-F1`：汇总目标侧成员关系后计算，较受大 alignment unit 影响；
- `Macro Set-F1`：逐 alignment unit 集合 F1 的宏平均，更重视小 alignment unit；
- `AUIS`：真实 alignment unit 的完整性；
- `AUCS`：真实 alignment unit 的紧凑性。

注意：一对一 `Hits@1/MRR` 与实体 `MicroF1` 不是同一个任务，不能直接写成同一列“准确率”。完整锚点监督后，训练锚点上的 Hits 只能标为 in-sample 诊断；held-out 节点结果必须明确来自原测试实体节点或另建划分。

### 4.3 已知评测陷阱

诊断函数 `m2m_diagnostics.oracle_grouped_topk` 会读取 GT 源分组和 GT 目标集合大小，因此即使 `S` 无信息也能输出正确基数。它只用于显式 oracle 诊断，不能用于正式结果；正式 runner 默认走 blind QuotientDecode。

---

## 5. 当前可信实验结果

### 5.1 主结果：PARROT + QuotientDecode

协议：`m2m_no_overlap`、修正后的完整 20% 训练锚点、盲 MicroF1；PARROT 对固定数据为确定性，四种子结果相同。

| 数据集 | 主系统 MicroF1 | 备注 |
|---|---:|---|
| douban | 0.7296 | 旧集成约 0.6944 |
| cora | 0.9805 | JOENA 集成约 0.9922，可选但非统一默认 |
| airport | 0.7980 | PARROT 更好 |
| pems08 | 0.6005 | 集成均值约 0.6428，但锚点不能可靠选中 |
| ppi | 0.9301 | PARROT 明显更好 |
| arenas | 0.8509 | 原“不可解”判断已撤回 |
| phone-email | 0.4237 | 弱信号经实体池化放大 |
| italy | 0.3725 | 同上 |
| foursquare-twitter | 0.4042 | 同上 |

首选证据文件：

- `logs/m2m_diag/parrot_all_s42.json`
- `logs/m2m_diag/parrot_weak_s012.json`
- `logs/m2m_diag/parrot_vs_ensemble_all.json`
- `docs/m2m_quotient_align_design.md` §5.13

### 5.2 FGW 结果

首选证据文件：

- `logs/m2m_diag/fgw_matcher.json`
- `docs/m2m_quotient_align_design.md` §5.14

结论：固定 `alpha=0.1` 7胜2负，平均小幅提升；不能根据训练锚点安全选择 Hungarian/FGW 或 α，因此不设默认。

### 5.3 JOENA-PC 必要性结论

- 对研究机理：有必要保留，证明输入对称会产生跨无关节点的重复画像；
- 对最终主系统：没有必要，PARROT 更简单、更快、九集更统一；
- 对 contrastive loss 本身：现有消融显示增益主要来自输入对称破缺，损失项贡献较小；
- 对 PeMS08：真正显著的是 group marginals，不是单独 PC；
- 对 Airport：PC 往往降低结果。

### 5.4 不应直接作为最终证据的日志

`logs/` 中包含大量不同历史阶段结果，约 4.6 GB，不能按文件名直接合并：

- `runtime_all_no_overlap_*ensemble*`：来自 PARROT 成为默认前的旧三分支集成阶段；其中 Arenas/Phone/Italy/Foursquare 数字已被后续 PARROT 实验推翻；
- 部分 `quotient_compare_*.json`：可能来自 overlap root、旧约 4% 双重切分协议或旧读出默认值；必须检查 JSON 内的 `root`、缓存标签和提交时期；
- `sweep_readout_variants.py` 使用固定缓存集合，历史上主要服务旧集成读出扫描，不能自动视为当前 PARROT 主系统证据；
- `diagnose_m2m_failure.json` 是失败机理诊断，不是最终系统主表；
- 任何没有记录 git SHA、数据 root、完整配置和缓存来源的结果，最多作为探索性证据。

---

## 6. 代码地图

### 6.1 M2M 核心

| 文件 | 作用 |
|---|---|
| `PlanetAlign/m2m/core.py` | EntityMap、加载/保存、预测实体ID评测对齐、`use_full_anchor_supervision` |
| `PlanetAlign/m2m/blind.py` | 最早的盲分组/解码基线 |
| `PlanetAlign/m2m/quotient.py` | QuotientDecode、候选分区、商图打分、Hungarian/FGW、评测包装 |
| `PlanetAlign/m2m/base.py` | 基座注册、训练并提取 `S`、锐化仲裁、基座仲裁实验 |
| `PlanetAlign/m2m/contrastive.py` | JOENA-PC、对称破缺、profile损失、group marginals |
| `PlanetAlign/utils/many2many_builder.py` | 节点拆分生成器、overlap插入、泄漏护栏 |
| `PlanetAlign/metrics/` | 五类M2M指标 |

### 6.2 正式实验脚本

| 脚本 | 用途 | 是否写文件 |
|---|---|---|
| `scripts/build_m2m_benchmarks.py` | 重建M2M数据 | 是，写 `data/` |
| `scripts/run_quotient_compare.py` | 主runner、同S读出消融、基座/集成入口 | 是，写缓存和JSON |
| `scripts/validate_base_selection.py` | 基座网格 | 是，写JSON |
| `scripts/compare_parrot_vs_ensemble.py` | PARROT vs旧集成 | 是，写JSON |
| `scripts/aggregate_unified_grid.py` | 聚合现有结果 | 主要读，打印 |
| `scripts/test_fgw_matcher.py` | Hungarian/线性/FGW对照 | 是，写JSON |
| `scripts/run_m2m_experiments.py` | 多算法批量基准 | 是，写 `logs/m2m_sweeps` |

### 6.3 诊断脚本

- `scripts/diagnose_m2m_failure.py`：一对一失败模式；
- `scripts/diagnose_rmt.py`：`S` 的谱与随机零分布；
- `scripts/probe_base_aligners.py`：基座可用性探测；
- `scripts/sweep_readout_variants.py`：旧集成 `S` 上的读出配对扫描。

### 6.4 文档阅读顺序

1. 本文：当前事实和风险；
2. `README.md`：快速入口，但注意状态漂移；
3. `docs/m2m_research_summary_for_paper.md`：论文素材；
4. `docs/m2m_quotient_align_design.md`：完整研究演进、消融和证伪；
5. `docs/m2m_why_one_to_one_fails.md`：失败机理；
6. `docs/m2m_research_interface.md`：API/评测接口；
7. `docs/PROJECT_MIGRATION_CONVERSATION_RECORD.md`：旧迁移记录，记录时 HEAD 早于当前，不应独立作为真相源。

---

## 7. 测试与环境

### 7.1 当前验证结果

2026-08-06 使用：

```text
/opt/miniconda3/bin/python3  Python 3.13.12
torch                       2.11.0
torch_geometric             2.7.0
POT                         0.9.6.post1
numpy                       2.4.3
```

运行：

```bash
env PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/planetalign-mpl \
  /opt/miniconda3/bin/python3 -m unittest discover -s tests -p 'test_*.py' -v
```

结果：**56 tests passed**。

系统 `/usr/bin/python3` 为 Python 3.9.6，当前没有 `torch`，直接运行会在测试收集阶段报 `ModuleNotFoundError`。这属于环境错误，不是代码测试失败。

### 7.2 环境可复现性缺口

- `pyproject.toml`/`requirements.txt` 只列依赖名，没有版本上限或 lockfile；
- PyTorch 额外索引写死 CUDA 11.8，但当前机器是 macOS/Conda 环境；
- 没有仓库内 `.venv`、`environment.yml` 或 `uv.lock`；
- README 的简单 `pip install -e .` 不保证得到已验证组合。

建议新增可复现环境文件和最小/完整两套依赖，并在 CI 固定至少一个 Python/PyTorch/POT 组合。

---

## 8. 已证伪或撤回的方向

后续智能体不要把以下历史阶段结论重新当成当前事实：

1. **“Arenas/Phone-Email/Italy/Foursquare 不可解”**：已撤回。原因是探测基座列表缺 PARROT且训练预算不诚实。
2. **“JOENA 三分支集成是最终系统”**：已被 PARROT 主系统取代。
3. **分支级训练锚点仲裁可靠**：多种子证伪；训练锚点只测一对一切片，可能看不见M2M收益。
4. **组级 unbalanced OT**：系统性失败；放松商图排他性会导致弃配/坍缩。
5. **邻域池化 profile 对比损失**：四个数据集总体失败，代码未保留。
6. **SVD硬阈值去噪/一跳平滑可以普适改进**：效果混合，不作默认。
7. **FGW可由锚点盲选成为统一默认**：证伪；目前仅固定参数可选项。
8. **旧 Flickr-LastFM 数据无泄漏**：错误；现已修复并重建，旧数据/旧结果需要排除。

负面结果是论文素材，不建议删除历史记录，但应与当前默认路径明确分层。

---

## 9. 当前代码和文档的不一致/风险

### P0：会污染实验结论

#### 9.1 缓存键没有包含数据 root

`scripts/run_quotient_compare.py` 当前缓存形式：

```text
logs/m2m_diag/S_cache/{dataset}_{model/config}_e{epochs}_s{seed}_fa.pt
```

键中没有：

- `data/m2m_no_overlap` vs `data/m2m_overlap_0.05`；
- 数据文件哈希；
- 生成清单哈希；
- 代码提交 SHA；
- 完整基座超参数；
- 部分解码配置。

因此同名数据在两个 root 间运行时可能静默复用错误 `S`。`docs/PROJECT_MIGRATION_CONVERSATION_RECORD.md` 声称该问题已修复，但当前实现并未把 root 加入键，必须重新核查并修正。

#### 9.2 最终系统的入口不统一

- README：主系统 `PARROT + QuotientDecode`；
- `run_quotient_compare.py` 默认：`root=m2m_overlap_0.05`、`model=joena`；
- `m2m/quotient.py` 模块顶部：仍称 shipped default 为三分支 ensemble；
- `run_m2m_experiments.py`：主要是通用基线 runner，并不等于最终系统入口。

其他智能体容易运行“默认命令”却得到非论文主系统结果。建议建立一个唯一 canonical CLI/config，并让 README、代码默认和测试都引用它。

#### 9.3 实验产物缺乏完整 provenance

JSON 多数没有同时记录：git SHA、数据哈希、manifest amendment、root、基座配置、锐化候选、decode配置、指标版本和依赖版本。旧结果与新结果只能靠文件名/文档语境区分，风险较高。

### P1：接口和叙事不完整

#### 9.4 FGW 接口未完整贯通

`quotient_decode` 支持 `matcher="fgw"`，但 `JOENAQuotientDecode.__init__` 仍断言 matcher 只能是 `hungarian/greedy`，也没有暴露 `fgw_alpha`。FGW 当前主要通过函数级接口和实验脚本使用。

#### 9.5 SacchCere 未归类

它的节点规模 `5441/1834` 接近当前可行区间，也已有旧 runtime 记录，但不在最终九集网格；README 的数量统计因此少一个。需要明确是信号质量、运行预算、属性问题还是遗漏。

#### 9.6 overlap manifest 路径过期

实际目录为 `data/m2m_overlap_0.05`，manifest 内仍记录 `data/m2m_overlap`。虽然元数据参数正确，但迁移/校验脚本可能按旧路径找文件。

#### 9.7 “一个锚点信号贯穿四层”表述过强

同一 `S` 内的分区证据和锐化温度仲裁有支持；跨 `S` 的基座/分支仲裁和 Hungarian/FGW matcher 仲裁都出现系统性失败。当前最终基座实际上是固定 PARROT，不是每次靠锚点安全选出。

#### 9.8 公开wrapper与当前主系统脱节

`JOENAQuotientDecode` 仍以 JOENA 为内置训练器；当前主系统需要手动组合 `train_base_S("PARROT")`、`arbitrate_sharpen` 和 `evaluate_quotient_blind`。可以考虑新增清晰的 `QuotientAlign`/`PARROTQuotientAlign` 公共入口，但应避免复制逻辑。

### P2：扩展性和科研完整性

#### 9.9 稠密复杂度

主要基座和 `S` 为稠密 `O(n1*n2)`；JOENA-PC 还有稠密传播/画像计算。八个较大数据集未进入最终网格。建议优先研究 top-k稀疏 `S`、分块/低秩相似度、候选检索后商图解码。

#### 9.10 外部效度

18套M2M数据均来自同一节点拆分生成框架。当前结论尚未在自然产生的多身份、多账号、合并/拆分实体数据上验证。

#### 9.11 单元测试以逻辑/玩具不变量为主

56项测试能守护接口和局部机制，但没有由版本化小型真实数据驱动的端到端数值回归。由于主数据和日志被忽略，CI无法自动发现主结果漂移。

---

## 10. 建议修改优先级

### 第一优先级：先把实验变得不可串用

1. 设计统一 `ExperimentSpec`，至少包含：数据root、数据哈希、manifest哈希、协议版本、基座及超参、seed、监督模式、decode配置、metric版本、git SHA。
2. 以该 spec 的稳定哈希作为缓存和结果目录键；旧缓存默认只读、显式迁移，不静默命中。
3. 添加缓存 shape + dataset fingerprint 校验。
4. 建立一个 canonical 主系统命令，默认就是 `no_overlap + full anchors + PARROT + QuotientDecode + Hungarian`。
5. 为主九集保存一份只读、带 provenance 的最终结果索引，而不是让智能体遍历整个 `logs/` 猜测。

### 第二优先级：统一接口和文档

1. 更新 `m2m/quotient.py` 顶部的旧 ensemble-default 描述；
2. 统一 README、runner 默认、迁移记录和设计文档中的“当前系统”；
3. 将 FGW 参数贯通公共 wrapper，或明确标注其仅为函数级研究选项；
4. 修正 overlap manifest 路径和数据集计数；
5. 解释 SacchCere 未进入最终网格的原因；
6. 将旧日志标记为 `legacy/`，禁止主聚合脚本误读。

### 第三优先级：增强科学证据

1. 对每个比较固定同一个 `S`，分别报告 grouping/readout 改进和 base 改进；
2. stochastic模型至少 4 seeds，报告均值、方差和配对差值；
3. PARROT确定性结果不应伪装成四个独立样本；
4. 指标同时报告 Macro Set-F1/Micro Set-F1/AUIS/AUCS，不只选最有利指标；
5. 为训练锚点参与的所有选择明确“训练监督选择”，避免称为无监督；
6. 增加自然M2M数据或第二种独立生成机制；
7. 为FGW固定α的收益做跨生成seed/overlap regime验证；
8. 将JOENA-PC定位为机制消融，而非默认复杂度来源。

### 第四优先级：规模化

1. 稀疏候选矩阵/ANN召回；
2. blockwise profile计算；
3. 稀疏商图构造和匹配；
4. 大图上先测候选召回上界，再做完整M2M指标；
5. 把大图失败区分为OOM、运行时间、基座无信号和读出失败。

---

## 11. 对后续智能体的操作规则

在提出或实施修改前：

1. 先运行 `git status --short`，不要覆盖用户未提交修改；
2. 明确使用哪个数据root，不允许省略；
3. 检查缓存是否与root和协议一致；当前最好禁用/隔离旧缓存后再做正式实验；
4. 明确是否调用 `use_full_anchor_supervision`；除复现旧bug外，正式实验必须调用；
5. 不使用测试GT分组、GT实体数或GT集合大小参与预测；
6. 不把一对一Hits与M2M MicroF1混为同一指标；
7. 不把探索性单seed结果升级为最终结论；
8. 不从旧日志文件名推断协议，必须读取内容和产生它的代码版本；
9. 不重新生成或覆盖忽略的数据/日志，除非用户明确授权并已保存manifest与哈希；
10. 修改算法后先跑56项单测，再做最小配对实验，最后才扩展网格。

建议测试命令：

```bash
env PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/planetalign-mpl \
  /opt/miniconda3/bin/python3 -m unittest discover -s tests -p 'test_*.py' -v
```

当前主系统显式命令：

```bash
/opt/miniconda3/bin/python3 scripts/run_quotient_compare.py \
  --root data/m2m_no_overlap \
  --dataset ppi_m2m \
  --base PARROT \
  --skip-ablations
```

注意：该脚本会读写 `logs/m2m_diag/S_cache` 和结果JSON；正式复现前应先解决缓存provenance问题。

---

## 12. 可直接分派给智能体的独立任务

### 任务A：缓存与结果溯源修复（最高优先）

目标：所有缓存和结果不可跨root、跨协议、跨配置静默复用。  
边界：不改算法数学，只改spec、cache key、校验和迁移策略。  
验收：同一dataset名在两个root产生不同键；篡改数据或配置必定cache miss；结果JSON含完整provenance。

### 任务B：统一当前默认入口

目标：README命令、CLI默认和公共Python API都指向 PARROT + QuotientDecode。  
边界：保留JOENA/ensemble作为显式legacy/ablation选项。  
验收：无参数或最小参数运行不会落到旧overlap+JOENA协议；帮助文本显示完整协议。

### 任务C：实验资产索引

目标：从4.6GB日志中生成一个只读索引，标记 authoritative / legacy / exploratory / incompatible。  
边界：不删除原始日志。  
验收：每个主表数字都能定位到结果文件、git SHA、数据root和配置。

### 任务D：SacchCere审计

目标：解释它为何不在九集网格。  
边界：先只读已有日志和缓存；如需重跑，另行征得授权。  
验收：给出“规模/基座/指标/协议/遗漏”中的明确原因，并修正README计数。

### 任务E：小型端到端回归fixture

目标：新增一个可版本化的小图fixture，覆盖生成→完整锚点→PARROT或固定S→QuotientDecode→指标。  
边界：不依赖被Git忽略的主数据。  
验收：CI能发现协议、缓存、预测路径和指标漂移。

### 任务F：稀疏化预研

目标：在不降低候选真匹配召回的前提下，把稠密S换成top-k/分块表示。  
边界：先报告候选召回和内存，不直接宣称最终M2M提升。  
验收：至少在一个当前大图上完成可运行的候选生成和profile近似误差分析。

---

## 13. 修改建议应如何评价

任何新建议至少回答以下问题：

1. 它修复的是基座表示、图内分组、商图打分、商图匹配，还是评测协议？
2. 是否固定了同一个 `S` 做配对比较？如果没有，收益可能来自基座变化。
3. 是否读取了训练锚点？若读取，应称为监督/半监督仲裁，而非无监督。
4. 是否触碰GT实体、实体数或实体大小？若是，正式盲评无效。
5. 是否跨数据root或复用了不带root的缓存？
6. 是否只提升一个数据集，却伤害其他体制？
7. 是否有盲信号能在不知道测试GT时决定启用？
8. 增益是否大于引入的运行成本、随机性和维护复杂度？
9. 是“keep-if-better”，还是仅“keep-if-novel”？两者应分开表述。
10. 能否解释已有负面结果，而不是重复已经证伪的路线？

---

## 14. 当前最稳妥的研究叙事

论文故事建议保持简洁：

1. 多种一对一方法在M2M上看似因不同机制失败；
2. 共同根因是节点级排他性先验放错层级；
3. 正确对象是每图实体等价类 + 商图间一一匹配；
4. 同实体节点在基座相似度矩阵中具有共线画像；
5. QuotientDecode从现有 `S` 恢复实体分组并在商图层匹配；
6. PARROT提供统一、稳定的上游 `S`；
7. JOENA-PC说明表示对称会限制画像证据，是辅助机理贡献；
8. FGW说明实体级结构一致性可进一步丰富商图匹配，但当前仍是可选项；
9. 诚实报告边界：锚点仲裁跨层失效、PeMS08残差、稠密扩展性和合成数据外部效度。

不建议把所有历史组件同时包装成最终模型。当前最强卖点不是模块数量，而是“排他性应放到实体层”这一统一解释，以及一个简单的 `PARROT + QuotientDecode` 系统。

---

## 15. 事实来源优先级

发生冲突时，建议按以下顺序判断：

1. 当前代码和当前数据清单；
2. 带完整协议字段的当前结果JSON；
3. `docs/m2m_quotient_align_design.md` 后部较新的 §5.11–§5.14；
4. `docs/m2m_research_summary_for_paper.md`；
5. README；
6. 旧迁移记录和早期日志。

所有高影响修改建议都应先用当前代码重新验证相关事实，而不是仅引用文档中的历史数字。
