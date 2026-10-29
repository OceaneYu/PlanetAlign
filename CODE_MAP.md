# PlanetAlign：跟着一份数据读代码

这个项目帮你找出两张图里的哪些节点代表同一个对象。图中的节点可以是用户，边可以是用户之间的关系。
例如，同一个人在两张图中各有几个账号，程序需要找出“这几个账号”和“那几个账号”属于同一个人。这就是这里的多对多对齐。
先只跟一条路线：**准备数据 → PARROT 算节点配对分数 → QuotientDecode 把节点分组并匹配 → 对照答案评分**。

## 先找到入口

打开 [scripts/run_quotient_compare.py](scripts/run_quotient_compare.py)，先找 `main()`。它负责把下面各步接起来。
阅读时沿 `if args.base is not None` 分支走，选 `--base PARROT`；不传这个参数时，脚本默认用 JOENA。
下面是运行方式；第一条需要已有 `data/douban.pt`，这里只生成实验数据，不负责下载：

```bash
python scripts/build_m2m_benchmarks.py --datasets douban --output-root data/m2m_no_overlap --overlap-ratio 0
python scripts/run_quotient_compare.py --root data/m2m_no_overlap --dataset douban_m2m --base PARROT --skip-ablations
```

脚本还会跑两种对照办法。初读只关注 `quotient(full)`；`--skip-ablations` 省去另外的逐项删功能对比。

## 先认清这几份数据

| 变量 | 你可以把它理解成什么 | 结构和来源 |
| --- | --- | --- |
| `ds` | 装着图和已知节点对应关系的数据集 | `Dataset` 对象，由 `main()` 读取 `.pt` 创建 |
| `g_src`、`g_tgt` | 要比较的第一张图和第二张图 | `ds.pyg_graphs[0]`、`[1]`；PyG 图对象，`edge_index` 存边，`x` 存可选属性 |
| `anchors` | 训练时已经知道对应的节点对 | `[m, 2]` 整数张量（m 行、2 列的数字表），每行一个节点对；`get_anchor_pairs(ds.train_data, 0, 1)` 提取 |
| `S_raw`、`S` | 节点配对分数表 | `[源节点数, 目标节点数]` 浮点张量（分数表）；`S_raw` 来自 PARROT，`S` 是步骤 4 选出的版本；`S[i,j]` 是节点 i 与 j 的分数 |
| `gt` | 留到评分时用的真实分组答案 | 从 `*_gt_many2many.json` 的 `entities` 读取；字典中每个实体都有 `src`、`tgt` 节点编号列表 |

`anchors` 会用于训练和预测时的分组选择；`gt` 虽在入口提前读入，却不传给 PARROT 或 `quotient_decode()`。两者用途不同。

## 1. 准备数据：先造出可以测试的多对多例子

- **为什么：** 原始基准是一对一对应，需要把部分节点拆成一组，才能测试程序是否找得回这些组。
- **打开哪里：** [scripts/build_m2m_benchmarks.py](scripts/build_m2m_benchmarks.py) 先看 `main()`、`build_one()`；后者把原始 `Dataset` 交给 [PlanetAlign/utils/many2many_builder.py](PlanetAlign/utils/many2many_builder.py) 的 `build_many_to_many_benchmark()`。
- **得到什么：** 构建器扩展两张图、生成真实分组，默认保留训练锚点并排除与其冲突的评估节点；`build_one()` 调用结果对象的 `ManyToManyBenchmark.save()` 写 `.pt` 和答案 JSON。
- **下一站：** 实验入口 `scripts/run_quotient_compare.py → main()` 读取这两个文件。这里靠文件传递数据，构建器不会直接启动训练。

## 2. 读取数据：把文件变成程序能用的图和节点对

- **打开哪里：** 从入口的 `ds = Dataset(...)` 跳到 [PlanetAlign/data/dataset.py](PlanetAlign/data/dataset.py)，先看 `Dataset.__init__()`、`load_dataset()`；边和属性变成 `g_src`、`g_tgt` 所指的图对象。
- **为什么还多一步：** `Dataset` 默认会再次划分锚点；生成的多对多文件已存好训练锚点，所以入口调用 [PlanetAlign/m2m/core.py](PlanetAlign/m2m/core.py) 的 `use_full_anchor_supervision()` 将它们全部放回 `train_data`。
- **交给谁：** 入口调用 [PlanetAlign/utils/anchors.py](PlanetAlign/utils/anchors.py) 的 `get_anchor_pairs()` 得到 `anchors`，再把 `ds` 传给步骤 3；图对象和 `anchors` 留给步骤 4、5。

## 3. PARROT 算分数：先回答“哪个节点像哪个节点”

- **打开哪里：** 入口调用 [PlanetAlign/m2m/base.py](PlanetAlign/m2m/base.py) 的 `train_base_S("PARROT", ds)`。先看它怎样从 `base_configs()` 取 PARROT，再跳到 [PlanetAlign/algorithms/parrot.py](PlanetAlign/algorithms/parrot.py) 的 `PARROT.train()`。
- **做了什么：** `PARROT.train()` 从 `ds` 取两图和训练节点对；`get_transport_cost()` 比较节点信息与图上位置，`con_prox_pt_opt()` 反复调整配对分数。第一遍只看这两个调用及返回值，公式以后再读。
- **交给谁：** PARROT 保存 `self.S`；`train_base_S()` 用 `_extract_s()` 取出矩阵，返回给入口成为 `S_raw`。它还没有生成节点组。有缓存时入口直接读矩阵，跳过训练。

## 4. 选分数表的版本：用已知节点对判断怎样处理更合适

- **打开哪里：** 入口把 `S_raw`、两图和 `anchors` 交给 [PlanetAlign/m2m/base.py](PlanetAlign/m2m/base.py) 的 `arbitrate_sharpen()`。
- **为什么：** 同一张表可以保留原分数，也可以突出每行高分；函数对各版本调用 `quotient_decode()` 试着分组，再用 `entity_anchor_agreement()` 检查能否解释训练节点对，不拿 `gt` 选版本。
- **交给谁：** 返回选中的 `S` 给入口，由入口送到步骤 5。第一次阅读只看输入、候选循环和返回值。

## 5. 多对多解码：把“节点分数”变成“哪一组对应哪一组”

- **打开哪里：** 入口 `main()` 的 `quotient(full)` 段直接调用 [PlanetAlign/m2m/quotient.py](PlanetAlign/m2m/quotient.py) 的 `quotient_decode(S, g_src, g_tgt, anchors=...)`。这是预测阶段，不接收真实答案 `gt`。
- **重点看什么：** `quotient_decode()` 用 `_candidate_partitions()` 提出可能同组的节点，用 `_select_partitions()` 选择两侧分组；`pool_columns()` 把另一侧的分数按组汇总，让下一轮判断更清楚。
- **为什么要匹配组：** `quotient_scores()` 汇总组与组的分数，默认 `hungarian_match()` 在组之间安排不重复的匹配。每组可以有多个节点，因此得到的是节点层面的多对多对应。
- **交给谁：** 返回 `pred`（如 `{"p0": {"src": [0,2], "tgt": [1,3]}}`）和 `info`（组数等过程记录）给 `main()`；下一行才开始评估。

## 6. 评分和保存：最后才拿预测与答案比较

- **打开哪里：** 入口接着调用 [PlanetAlign/m2m/core.py](PlanetAlign/m2m/core.py) 的 `evaluate_predictions(gt, pred)`。MSF1 直接使用原始预测；Micro-SF1、AUIS、AUCS 继续使用 `align_prediction_to_ground_truth()` 的旧编号适配结果。
- **得到什么：** `many_to_many_scores()` 返回 `scores`。新版 MSF1 用双侧成员 F1 全局匹配，分母固定为真实测试 unit 数 K，混入的范围外成员计 FP；纯训练/背景组不评分。默认拒绝同侧节点跨预测组复用；重叠基准显式设置 allow_overlap=True。Micro-SF1、AUIS、AUCS 不变，历史 MSF1 不可直接比较，详见 [评分协议](docs/MSF1.md)。
- **交给谁：** `evaluate_predictions()` 只返回 `scores`；`pred`、`info` 已由预测阶段得到。`main()` 内的 `add()` 收集分数，最后写入 `--out` 或 `logs/m2m_diag/quotient_compare_*.json`，只保存评分与记录，不保存 `pred`。

## 建议这样读，第一遍不用钻进所有辅助函数

1. 先看入口 `main()` 中的 `Dataset`、`args.base` 分支、`quotient(full)` 和 JSON 输出，认清各步在哪里发生。
2. 按步骤 2 找 `Dataset.load_dataset()`、`use_full_anchor_supervision()`、`get_anchor_pairs()`，弄明白图和训练节点对从哪里来。
3. 按步骤 3、4 找 `train_base_S()`、`PARROT.train()`、`arbitrate_sharpen()`，追踪节点分数怎样算出并交回入口。
4. 按步骤 5、6 找 `quotient_decode()`、`evaluate_predictions()`、`align_prediction_to_ground_truth()`、`many_to_many_scores()`，追踪预测怎样交给评分；真实答案只交给评分函数。
5. 最后回看步骤 1 的 `build_one()`、`build_many_to_many_benchmark()`、`ManyToManyBenchmark.save()`，理解实验输入是怎么造出来的。

其他路线先只记名字：JOENA、GADL 等负责提供节点分数；`scripts/run_m2m_experiments.py → main()` 负责批量比较。它们不改变本手册的阅读主线。
想看 PARROT 内部计算、分组阈值、其他算法入口或 JOENA 训练循环，再打开 [CODE_MAP_DETAILS.md](CODE_MAP_DETAILS.md)。
