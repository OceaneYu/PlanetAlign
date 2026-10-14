# PlanetAlign 深入阅读

先按 [简明导航](CODE_MAP.md) 走完 PARROT + QuotientDecode；想理解内部实现时，再查这份说明。下面保留原导读中的调用链、数据协议、其他入口和 JOENA 训练细节。

这个仓库比较两张图里的节点是否对应，并在原有节点对齐算法上增加了多对多对齐：一个实体可以在每张图里对应多个节点。这里先沿 README 的 PARROT + QuotientDecode 路线阅读，再看其他算法和实验。

## 1. 从哪里开始

仓库是 Python 库，没有一个运行所有功能的总入口。[PlanetAlign/__init__.py](PlanetAlign/__init__.py) 导出各个包；导入包不等于执行训练。

| 想做什么 | 实际入口 | 下一步看哪里 |
| --- | --- | --- |
| 从一对一数据构建多对多数据 | [scripts/build_m2m_benchmarks.py](scripts/build_m2m_benchmarks.py) → `main()` | `build_one()`，然后 [PlanetAlign/utils/many2many_builder.py](PlanetAlign/utils/many2many_builder.py) → `build_many_to_many_benchmark()` |
| 运行 README 的 PARROT 多对多路线，并比较解码方式 | [scripts/run_quotient_compare.py](scripts/run_quotient_compare.py) → `main()`，传入 `--base PARROT` | [PlanetAlign/m2m/base.py](PlanetAlign/m2m/base.py) → `train_base_S()`、`arbitrate_sharpen()` |
| 批量比较数据集和算法 | [scripts/run_m2m_experiments.py](scripts/run_m2m_experiments.py) → `main()` | `evaluate_algorithm()`、`_extract_s()` |
| 在自己的程序里做一对一对齐 | README 示例：创建 [PlanetAlign/datasets/douban.py](PlanetAlign/datasets/douban.py) → `Douban`，再调用 [PlanetAlign/algorithms/final.py](PlanetAlign/algorithms/final.py) → `FINAL.train()` | [PlanetAlign/algorithms/base_model.py](PlanetAlign/algorithms/base_model.py) → `BaseModel.test()` |

下文的箭头表示执行步骤或数据衔接，不表示相邻两个函数一定直接调用；文字会说明谁发起调用、结果返回哪里。

上述脚本底部的 `if __name__ == "__main__"` 才会调用 `main()`。`run_quotient_compare.py` 不带 `--base` 时默认走 JOENA，并非 PARROT；其内部 `get_S()` 负责 JOENA 系列的训练和缓存。

已有原始 `.pt` 数据时，下面显式指定构建目录，让两步使用同一套数据。当前构建脚本默认输出 `data/m2m`，README 的运行示例使用 `data/m2m_no_overlap`：

```bash
python scripts/build_m2m_benchmarks.py --output-root data/m2m_no_overlap --overlap-ratio 0
python scripts/run_quotient_compare.py --root data/m2m_no_overlap --dataset ppi_m2m --base PARROT
```

脚本还会运行对比解码器和消融实验（逐项移除功能以比较效果）；`--skip-ablations` 只跳过消融部分。数据下载不是这两个脚本的职责，内置数据集类的 `__init__()` 才会在 `download=True` 时下载已整理的 `.pt` 文件。

## 2. 先认识流动的数据

| 数据 | 在代码里的样子 | 含义 |
| --- | --- | --- |
| 图 | `dataset.pyg_graphs` 中的 PyG `Data` | `edge_index` 存边，`x` 存可选节点属性，`num_nodes` 存节点数 |
| 已知对应关系（锚点） | `dataset.train_data`、`dataset.test_data` | 每行是各张图中对应节点的编号；两张图时为 `[m, 2]` |
| 节点相似度 | `S`，形状 `[n_src, n_tgt]` | 第 `i` 行、第 `j` 列表示源节点 `i` 与目标节点 `j` 的匹配分数；有些算法输出的是传输权重 |
| 分组 | `src_groups`、`tgt_groups` | 每组是一个节点编号列表，例如 `[[0, 2], [1]]` |
| 实体集合 | `EntityMap`，如 `{"p0": {"src": [0, 2], "tgt": [1, 3]}}` | 一个实体包含两侧的节点集合；预测编号与真实编号可以不同 |
| 评分 | `scores` 字典 | 指标名称对应分数；最终脚本写入实验结果文件 |

`S.T` 是矩阵转置：源侧按 `S` 的行观察节点，目标侧按 `S.T` 的行观察节点。解码器中的 profile（匹配轮廓）就是这些行向量。

## 3. 数据准备链：一对一 → 多对多

1. [scripts/build_m2m_benchmarks.py](scripts/build_m2m_benchmarks.py) → `main()` 找到原始 `.pt` 文件，逐个调用 `build_one()`。
2. `build_one() → _load_dataset() → PlanetAlign/data/dataset.py → Dataset.__init__()`。`load_dataset()` 将文件里的边、属性和锚点转成 PyG 图；`_validate()` 检查编号与形状；`train_test_split()` 按随机种子拆分锚点。
3. `build_one()` 把这个 `Dataset` 传给 [PlanetAlign/utils/many2many_builder.py](PlanetAlign/utils/many2many_builder.py) → `build_many_to_many_benchmark()`。默认从测试锚点构建实体，保留训练锚点作监督；先移除与训练节点冲突的候选，再决定哪些节点需要拆成多个节点。
4. `build_many_to_many_benchmark() → _build_side()` 重建两侧图，组装真实实体集合；`_apply_overlap()` 按设置添加重叠成员。默认训练模式下还会检查训练节点与评估实体不重叠。
5. 返回 `ManyToManyBenchmark`，由 `build_one() → ManyToManyBenchmark.save()` 写出 `<name>.pt` 和 `<name>_gt_many2many.json`。`main()` 另写构建清单。

这里是**文件传递**：构建脚本先写磁盘，实验脚本随后读取；构建器不会直接调用实验脚本。

## 4. 主执行链：PARROT → 分组匹配 → 评分

### 4.1 加载图和训练锚点

[scripts/run_quotient_compare.py](scripts/run_quotient_compare.py) → `main()`

→ [PlanetAlign/data/dataset.py](PlanetAlign/data/dataset.py) → `Dataset(...)`

→ [PlanetAlign/m2m/core.py](PlanetAlign/m2m/core.py) → `use_full_anchor_supervision(ds)`

生成的多对多 `.pt` 已保存原始训练锚点，`Dataset` 的通用加载又会划分一次。入口默认调用 `use_full_anchor_supervision()`，将 `_anchor_links` 全部作为 `train_data`，清空 `test_data`，避免监督量被再次缩小。这个操作针对生成的多对多基准，不适用于所有原始数据。

`main()` 同时读取真实实体 JSON，取出 `g_src`、`g_tgt`，并调用 [PlanetAlign/utils/anchors.py](PlanetAlign/utils/anchors.py) → `get_anchor_pairs()` 得到两张图的训练节点对。真实实体用于最后评分，不传入基础算法训练。

### 4.2 得到节点相似度 S

在 `--base PARROT` 分支中：

`main() → PlanetAlign/m2m/base.py → train_base_S("PARROT", ds)`

→ `base_configs()` 取训练方案

→ [PlanetAlign/algorithms/parrot.py](PlanetAlign/algorithms/parrot.py) → `PARROT.train()`

→ `get_transport_cost()`

→ `con_prox_pt_opt()`

→ [PlanetAlign/m2m/base.py](PlanetAlign/m2m/base.py) → `_extract_s()`

`PARROT.train()` 从 `ds` 取得图和训练锚点。`get_transport_cost()` 调用 [PlanetAlign/utils/rwr.py](PlanetAlign/utils/rwr.py) → `get_batch_rwr_scores()`，计算相对锚点的结构位置；再用 [PlanetAlign/utils/distance.py](PlanetAlign/utils/distance.py) → `get_normalized_neg_exp_dist()` 比较结构向量和可选属性。`get_product_rwr_mat()` 让两张图的邻接关系参与代价传播。

`con_prox_pt_opt()` 用最优传输（OT，即在两侧节点间分配匹配权重）迭代生成 `self.S`，返回矩阵和日志对象。`_extract_s()` 统一不同算法的返回格式，向入口交回 CPU 上的浮点矩阵。入口会缓存矩阵；缓存存在时跳过训练。

### 4.3 用训练锚点选择矩阵处理方式

`main() → PlanetAlign/m2m/base.py → arbitrate_sharpen(S_raw, g_src, g_tgt, anchors)`

→ `sharpen()` 生成原矩阵或不同温度的行 softmax 候选

→ [PlanetAlign/m2m/quotient.py](PlanetAlign/m2m/quotient.py) → `quotient_decode()` 为每个候选生成实体

→ `entity_anchor_agreement()` 检查实体能否解释训练锚点

返回选中的 `S`、温度和比较表给 `main()`。温度越小通常越突出每行的高分节点；选择过程不使用真实评估实体。

### 4.4 从节点分数预测实体

`main() → PlanetAlign/m2m/quotient.py → quotient_decode(S, g_src, g_tgt, anchors=...)`

`quotient_decode()` 先把 `S` 与 `S.T` 当成两侧匹配轮廓，依次：

1. `_candidate_partitions()` 提出分组候选：按轮廓或属性判断哪些节点可能属于同一实体。`otsu_threshold()` 自动取阈值，`merge_average_linkage()` 检查组整体是否相近，避免只靠中间节点把不相近的组串起来。
2. `_select_partitions()` 用训练锚点选择两侧分组；锚点不足时使用 `partition_contrast()`。`evict_outliers()` 可把轮廓证据组里的异常节点拆出。
3. `pool_columns()` 将另一侧节点的分数按组汇总，作为下一轮分组证据；迭代次数用尽或分组稳定时结束。
4. `quotient_scores()` 把节点矩阵汇总成组与组之间的矩阵 `T`。默认 `hungarian_match()` 做全局一对一组匹配；每组可以含多个节点，所以最终仍是节点层面的多对多对应。
5. 返回预测实体 `pred` 和解码记录 `info`。函数参数中没有真实实体 `gt_entities`。

[PlanetAlign/m2m/blind.py](PlanetAlign/m2m/blind.py) 提供更简单的比较路线：`blind_predict() → discover_groups() → decode_entity_map()`；这里的贪心匹配允许多个源组选中同一目标组。入口中的 `profile_greedy_scores()` 则用固定阈值的 `discover_groups_by_profile()` 分组。

### 4.5 对齐编号、计算指标、保存结果

`main()` 得到 `pred` 和 `info` 后，单独调用 [PlanetAlign/m2m/core.py](PlanetAlign/m2m/core.py) → `evaluate_predictions(gt, pred)`；评分函数不执行解码，内部调用：

[PlanetAlign/m2m/core.py](PlanetAlign/m2m/core.py) → `align_prediction_to_ground_truth(gt, pred)`

→ [PlanetAlign/metrics/many_to_many.py](PlanetAlign/metrics/many_to_many.py) → `many_to_many_scores(gt, aligned)`

→ `PlanetAlign/metrics/metrics_MSF1.py → macro_f1_score()`、`PlanetAlign/metrics/metrics_Micro_SF1.py → micro_f1_score()`、`PlanetAlign/metrics/metrics_AUIS.py → auis_score()`、`PlanetAlign/metrics/metrics_AUCS.py → aucs_score()`。

编号适配器根据节点重叠将任意预测组编号对应到真实编号，这是**评估操作**，不是预测的一部分。Set-F1 比较各实体的目标节点集合；AUIS、AUCS 从不同角度评价分组。`evaluate_predictions()` 只返回 `scores`。

所有实验调用方统一先用 `quotient_decode()` 生成 `pred、info`，再用 `evaluate_predictions()` 规范化实体、对应编号并返回 `scores`；没有组合包装函数。

入口 `main()` 的内部 `add()` 收集各解码方式的分数，最后写入 `--out` 指定的 JSON 或 `logs/m2m_diag/quotient_compare_*.json`。完整监督模式下入口打印的节点 Hits@1 使用训练锚点，是样本内诊断；正式多对多结果来自实体评分。

## 5. 一对一接口与其他目录怎么接入

- [PlanetAlign/datasets/douban.py](PlanetAlign/datasets/douban.py) → `Douban.__init__()` 等内置数据集入口在 `download=True` 时调用 `PlanetAlign/datasets/utils.py → download_file_from_google_drive()`，检查 `.pt` 是否存在后调用父类 `Dataset.__init__()`；实际图加载、校验和锚点划分由 [PlanetAlign/data/dataset.py](PlanetAlign/data/dataset.py) 完成。
- [PlanetAlign/algorithms/base_model.py](PlanetAlign/algorithms/base_model.py) → `BaseModel` 约定 `train()`、`test()`、设备设置、输入检查和日志创建。算法的 `train()` 实现不同，部分返回嵌入向量而非矩阵，所以要通过实验脚本的 `_extract_s()` 或 `PlanetAlign/m2m/base.py → _extract_s()` 适配，不能假定所有返回值相同。
- `BaseModel.test()` 用 [PlanetAlign/metrics/hits.py](PlanetAlign/metrics/hits.py) → `hits_ks_scores()` 和 `PlanetAlign/metrics/mrr.py → mrr_score()` 评估节点排名；两者使用 `PlanetAlign/metrics/ranking.py → pairwise_ranks()` 处理同分情况。这是一对一评估链，与实体集合评分分开。
- [PlanetAlign/logger/train_logger.py](PlanetAlign/logger/train_logger.py) → `TrainingLogger.log()` 接收算法训练循环提供的分数与时间；日志对象自身保存 CSV/JSON；`PlanetAlign/logger/output.py` 格式化和打印评分，`PlanetAlign/logger/plot.py` 绘制曲线。
- `scripts/evaluate_gadl_*.py` 是 GADL 的专门复现与诊断入口，部分读取外部参考实现，不能当成通用主流程。[planetalign-baseline-skills/tools/baselinectl.py](planetalign-baseline-skills/tools/baselinectl.py) 管理隔离参考仓库与复现记录，也不在普通对齐运行链上。
- `tests/` 用小图和固定输入检查接口、指标、数据隔离与算法行为；分布和 VS Code 运行步骤见 [tests/README.md](tests/README.md)。

## 6. 推荐阅读顺序

1. **先看入口的骨架**：[scripts/run_quotient_compare.py](scripts/run_quotient_compare.py) → `main()`。先找到 `Dataset`、`args.base` 分支、`quotient_decode()`、`evaluate_predictions()` 和输出 JSON，暂时跳过其他分支与消融循环。
2. **理解输入格式**：[PlanetAlign/data/dataset.py](PlanetAlign/data/dataset.py) → `Dataset.load_dataset()`、`train_test_split()`，接着读 [PlanetAlign/m2m/core.py](PlanetAlign/m2m/core.py) → `use_full_anchor_supervision()` 和 `normalize_entity_map()`。先分清图、锚点与真实实体三种数据。
3. **看算法如何被包装**：[PlanetAlign/m2m/base.py](PlanetAlign/m2m/base.py) → `train_base_S()`、`base_configs()`、`_extract_s()`。明确“算法 → S → 入口”的数据传递，再读 `arbitrate_sharpen()`。
4. **只跟一个算法深入**：[PlanetAlign/algorithms/parrot.py](PlanetAlign/algorithms/parrot.py) → `PARROT.train()`、`get_transport_cost()`、`con_prox_pt_opt()`；需要时追到 `PlanetAlign/utils/rwr.py` 和 `PlanetAlign/utils/distance.py`。先理解矩阵用途，再研究优化公式。
5. **理解节点如何变成实体**：[PlanetAlign/m2m/quotient.py](PlanetAlign/m2m/quotient.py) → `quotient_decode()`，沿调用依次看 `_candidate_partitions()`、`_select_partitions()`、`pool_columns()`、`quotient_scores()`、`hungarian_match()`。
6. **理解预测如何被评分**：`PlanetAlign/m2m/core.py → evaluate_predictions()` → `align_prediction_to_ground_truth()` → `PlanetAlign/metrics/many_to_many.py → many_to_many_scores()`。配合 [tests/test_m2m_interface.py](tests/test_m2m_interface.py)、[tests/test_many_to_many_metrics.py](tests/test_many_to_many_metrics.py) 的小例子阅读。
7. **最后回看数据怎么生成**：[scripts/build_m2m_benchmarks.py](scripts/build_m2m_benchmarks.py) → `build_one()` → `PlanetAlign/utils/many2many_builder.py → build_many_to_many_benchmark()`、`ManyToManyBenchmark.save()`。这时更容易理解拆分节点、训练/评估隔离和重叠实体。

## 7. 怎样区分调用和传递

| 关系 | 主流程里的实际例子 |
| --- | --- |
| 直接调用 | `PlanetAlign/m2m/base.py → train_base_S()` 创建算法并执行 `algo.train(...)`；PARROT 分支因此直接进入 `PlanetAlign/algorithms/parrot.py → PARROT.train()` |
| 经中间层传递 | `PARROT.train()` 产生 S，`train_base_S() → _extract_s()` 提取并返回给 `scripts/run_quotient_compare.py → main()`，入口随后把矩阵传入解码评估；PARROT 自身不调用实体解码器 |
| 文件传递 | `ManyToManyBenchmark.save()` 写 `.pt` 与实体 JSON，实验入口以后读取；两个脚本可以分开运行 |
| 模块依赖 | `PlanetAlign/algorithms/__init__.py` 导入并导出算法类；这个导入本身不会调用这些类的 `train()` |

文件头的“数据来源”表示输入如何到达当前文件，“结果去向”表示谁接着使用结果。函数旁的注释进一步说明直接调用和返回值，结合 `CODE_MAP.md` 的阅读顺序追踪即可。

## 8. 补充：默认 JOENA 分支的一轮训练

不传 `--base`、保持默认 `--model joena` 时，[scripts/run_quotient_compare.py](scripts/run_quotient_compare.py) 的 `main() → get_S()` 创建 JOENA。缓存命中就直接返回 S；需要训练时才进入 [PlanetAlign/algorithms/joena/main.py](PlanetAlign/algorithms/joena/main.py) → `JOENA.train()`。

先看 `JOENA.train()` 如何准备 `input_emb1/input_emb2`：训练锚点经 [PlanetAlign/utils/rwr.py](PlanetAlign/utils/rwr.py) → `get_batch_rwr_scores()` 变成结构位置特征；启用属性时，再拼接 `graph.x`。之后的一轮循环如下：

| 循环中的代码 | 实际处理位置 | 数据变化与去向 |
| --- | --- | --- |
| `optimizer.zero_grad()` | `JOENA.train()` 的优化器调用点 | 清除上一轮梯度，为本轮计算准备 |
| `model(input_emb1, input_emb2)` | PyTorch 间接调用 [joena/model.py](PlanetAlign/algorithms/joena/model.py) → `MLP.forward()` | 两侧共用网络，输入特征变成归一化节点向量 `out1/out2` |
| `criterion(out1=out1, out2=out2)` | PyTorch 间接调用 [joena/model.py](PlanetAlign/algorithms/joena/model.py) → `FusedGWLoss.forward()` | 向量变成节点/结构代价；调用 [joena/utils.py](PlanetAlign/algorithms/joena/utils.py) → `sinkhorn_stable()` 求 S，再返回 `loss, S, threshold` |
| `loss.backward()` | `JOENA.train()` 的反向传播调用点 | S 在 `no_grad` 中求出；损失通过依赖嵌入的代价计算编码器梯度 |
| `optimizer.step()` | `JOENA.train()` 的优化器调用点 | 按梯度更新 MLP 参数，不在这一步重新生成 S |
| `hits_ks_scores()`、`mrr_score()`、`logger.log()` | `JOENA.train()` 的记录部分 | 使用本轮刚取得的 S 计算节点指标，把分数交给日志对象 |

循环结束后，`JOENA.train()` 保存 `self.S` 并返回最后一次损失函数生成的 S 与日志。`get_S()` 将矩阵转换为 CPU float32 并缓存，`main()` 再把它送入前面介绍的实体解码流程。

遇到 `model(...)` 或 `criterion(...)` 时，不要只搜索同名函数：先找到对象创建处，确认它分别是 `MLP` 或 `FusedGWLoss`，再阅读对应 `forward()`。这里的 `JOENA.train(dataset, ...)` 负责完整训练；循环里的 `model.train()` 只设置 PyTorch 网络为训练模式，训练循环仍由外层代码执行。
