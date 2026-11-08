# Cross-Graph Pair Evaluation（CGP-v1）

CGP 将标准 Precision / Recall / F1 用于跨图节点对应关系评价，不提出三个新公式。它衡量预测的源节点—目标节点关系，而 MSF1 继续评价真实 unit 的成员恢复。本次只支持非重叠 Benchmark；MSF1、其他指标、训练与解码算法均不修改。

## 输入与关系

输入完整测试 GT 和模型原始预测：

```python
entities[eid] = {"src": [...], "tgt": [...]}
```

每个双侧 unit 表示 `src × tgt`。对所有 unit 取关系并集：

```text
R_GT   = union_i (S_i × T_i)
R_Pred = union_j (S_hat_j × T_hat_j)
```

关系是有序二元组 `(src_node, tgt_node)`，`(1,2)` 与 `(2,1)` 不同；`(7,7)` 也是一个合法跨图关系。没有同侧节点对。组内重复成员按集合处理，同一关系只计一次。单侧组、空组不产生关系。

不按实体编号对应，不使用 `align_prediction_to_ground_truth()` 的结果，不运行 Hungarian 或贪心组匹配，不修改预测，不重新解码或从相似度矩阵选 argmax。

## 测试范围

```text
E_src = union_i S_i
E_tgt = union_i T_i
R_eval = {(s,t) in R_Pred : s in E_src OR t in E_tgt}
```

条件必须是“或”。只排除两端均在范围外的关系，不预先裁掉组内训练／背景成员。即使一个 unit 触及测试范围，它内部两端均在范围外的关系仍应排除。

例如 GT=`({1},{5})`，预测=`({1,99},{5,200})`：`(1,5)` 为 TP，`(1,200)` 和 `(99,5)` 为 FP，`(99,200)` 不评分，因此 `TP=1, FP=2, FN=0`。

单侧 GT unit 的节点仍进入相应的 E 集合，尽管该 unit 本身不产生真实关系。GT 必须完整描述本次测试范围；指标不会推断缺失的标注。

## 统计与空评测

```text
TP = |R_GT ∩ R_eval|
FP = |R_eval - R_GT|
FN = |R_GT - R_eval|

CGP-P  = TP / (TP + FP)
CGP-R  = TP / (TP + FN)
CGP-F1 = 2TP / (2TP + FP + FN)
```

在单个数据集内汇总关系计数，不做逐 unit Macro 平均。各分母为零时对应分数约定为 0。GT 有真实关系而预测为空或全为单侧组时，三个分数都是 0。

若 GT 根本没有跨图真实关系，`cross_graph_pair_scores()` 抛出 `NoEvaluableRelationsError`，消息包含“无可评估真实关系”。此时不返回可用于性能汇总的数值分数。重评分脚本记录 `status="no_evaluable_gt"`、`NumGTPairs=0`，排除出 `num_evaluated`；不将空评测当作正常的零分结果。

## 非重叠约束和计数实现

复用现有 MSF1 文件中的 `_unit_sets(..., require_disjoint=True)` 校验，未改动该文件。整份预测的每侧节点最多属于一个 unit；校验在关系生成和范围过滤之前执行。重复非空组、部分重叠、纯范围外组复用节点、单侧组复用节点，均抛出原有 `InvalidPredictionError`；不静默修补。空组没有节点，不违反归属约束。

GT 也必须非重叠。重叠 GT 或显式 `allow_overlap=True` 抛出 `UnsupportedCGPOverlapError`，明确说明 CGP-v1 本轮不支持重叠。命令行按 GT 的 `metadata.overlap_ratio` 检查声明的 Benchmark 模式，不因某一子集恰好没有重叠就放宽协议。

正式实现始终使用非重叠条件下的等价整数计数，不展开笛卡尔积：

```text
NumGTPairs = Σ_i |S_i| · |T_i|
NumPredPairs = Σ_j [|S_hat_j| · |T_hat_j|
                   - |S_hat_j - E_src| · |T_hat_j - E_tgt|]
TP = Σ_i,j |S_i ∩ S_hat_j| · |T_i ∩ T_hat_j|
```

为 GT 两侧节点建立所属 unit 的索引，再分别统计每个预测 unit 与 GT 的源侧、目标侧交集大小并相乘。计数部分只遍历节点成员和非零交集，不创建稠密 unit 矩阵；内存随成员数量增长，不随潜在关系数量增长。复用的合法性校验仍有集合排序开销。计数采用 Python 整数，避免 32 位溢出。这些求和只在非重叠约束成立后使用。

## 调用

```python
from PlanetAlign.metrics import cross_graph_pair_scores

details = cross_graph_pair_scores(gt_entities, raw_pred_entities)
# CGP-P, CGP-R, CGP-F1, TP, FP, FN, NumGTPairs, NumPredPairs
```

`NumPredPairs` 指实际参与评分的关系数，而不是原始预测的全部笛卡尔积大小。

```python
from PlanetAlign.m2m import evaluate_predictions
from PlanetAlign.metrics import many_to_many_scores

names = ["CGP-P", "CGP-R", "CGP-F1"]
scores = many_to_many_scores(gt_entities, raw_pred_entities, metrics=names)
scores = evaluate_predictions(gt_entities, raw_pred_entities, metrics=names)
```

三个名字已注册，大小写和下划线按项目现有规则规范化。一次 `many_to_many_scores()` 调用只计算一次 CGP 关系计数。单独请求 CGP 时，`evaluate_predictions()` 跳过所有实体编号适配；与其他指标一同请求时，旧适配仅供旧指标使用，CGP 始终使用原始预测。默认指标列表保持不变，CGP 需显式请求。

## 已保存预测重评分

```bash
python -m scripts.rescore_cgp \
  --gt data/m2m_no_overlap/douban_m2m_gt_many2many.json \
  --predictions logs/m2m_diag/gadl_m2m_backbone_douban_no_overlap_seed42_predictions.json \
  --out logs/cgp_new/douban.json
```

接受原始 entity map、`{"entities": ...}`、或 `{"method": entity_map, ...}`。输入文件缺失时明确报错，不训练、不解码；拒绝覆盖已有输出。结果保存协议 `CGP-v1`、输入路径与 SHA-256、全部计数和三个分数。

结果状态包括 `ok`、`invalid_prediction`、`unsupported_overlap`、`no_evaluable_gt`。后三种不产生 CGP 分数，不纳入正常性能汇总；脚本继续记录其他方法，并以非零退出码提示存在未完成评测的记录。

## 验证

必需小例、额外范围边界、非法复用、API 原始预测传递与单次计算由 [test_cross_graph_pairs.py](../tests/test_cross_graph_pairs.py) 覆盖；1,000 个随机非重叠小例与独立显式集合参考实现逐项核对全部计数和分数。另有 49 亿潜在关系的用例验证大规模计数与范围扣除。

[test_rescore_cgp.py](../tests/test_rescore_cgp.py) 检查无训练／解码／匹配、缺失输入、历史文件保护、空评测排除与重叠模式拒绝。相关回归共 133 项通过，包含此前的 101 项测试；MSF1 等既有预期值不修改。

## 2026-11-08 已保存预测结果

六份原始预测文件均存在并通过非重叠校验。本轮只读这些预测，未重训、未重新解码；四份 PARROT 预测是 2026-10-28 已保存的群组结果，本轮没有读取 S 缓存。

| 数据／方法 | CGP-P | CGP-R | CGP-F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| Douban／GADL | 0.026613 | 0.034393 | 0.030007 | 66 | 2414 | 1853 |
| Douban／JOENA | 0.135461 | 0.362689 | 0.197251 | 696 | 4442 | 1223 |
| Airport／PARROT | 0.606358 | 0.508483 | 0.553124 | 1049 | 681 | 1014 |
| Cora／PARROT | 0.960924 | 0.950540 | 0.955704 | 4574 | 186 | 238 |
| Douban／PARROT | 0.257817 | 0.421053 | 0.319810 | 808 | 2326 | 1111 |
| Arenas／PARROT | 0.762881 | 0.766717 | 0.764794 | 1525 | 474 | 464 |

独立结果保存于 [benchmarks/cgp_review_20261108](../benchmarks/cgp_review_20261108/)，包含 CGP-v1 协议、输入哈希、计数、分数及验证记录，未覆盖历史结果。CGP 与 MSF1 的对象和汇总方式不同，数值不能互换；CGP 对单侧输出不授予跨图关系的部分分数。
