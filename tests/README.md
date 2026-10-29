# 自动化测试怎么读、怎么运行

测试就是用小图、固定矩阵或算法替身运行代码，再用 `assert...` 检查结果是否符合预期。
每个 `test_...()` 方法算一项测试；`setUp()` 准备样例，`subTest()` 检查同一项测试的不同参数。
它们保护代码行为，不等于重新跑完整数据集，也不能证明所有评价指标都合理。

## 测试分布

当前共 **140 项**测试；本轮覆盖固定 K、非重叠提交校验、显式重叠模式与无需训练的预测重评分。MSF1 的独立穷举测试还包含 400 个随机小例。

| 文件 | 当前数量 | 主要检查 |
|---|---:|---|
| [test_m2m_quotient.py](test_m2m_quotient.py) | 25 | 分组阈值、防止串联合并、组匹配、锚点选择、预测与评分分离 |
| [test_macro_set_f1.py](test_macro_set_f1.py) | 29 | 固定 K、非法重叠/重复、双侧 F1、范围、全局匹配、随机穷举、解码器与 API 隔离 |
| [test_rescore_msf1.py](test_rescore_msf1.py) | 5 | 保存预测的重评分、非法状态、模式校验、保护原始输入与旧结果 |
| [test_m2m_base.py](test_m2m_base.py) | 7 | 矩阵锐化、用训练锚点选矩阵、基础算法配置 |
| [test_m2m_blind.py](test_m2m_blind.py) | 3 | 属性分组、简单解码和不依赖真实答案的预测 |
| [test_m2m_fgw.py](test_m2m_fgw.py) | 4 | FGW 组匹配及特殊参数、矩阵形状 |
| [test_m2m_interface.py](test_m2m_interface.py) | 5 | 实体格式、文件读写、漏预测、实体编号变化 |
| [test_many_to_many_metrics.py](test_many_to_many_metrics.py) | 6 | Set-F1、AUIS、AUCS 的固定值与边界情况 |
| [test_tie_aware_metrics.py](test_tie_aware_metrics.py) | 8 | 节点分数相同时的排名和 Hits/MRR 等指标 |
| [test_m2m_protocol_cleanup.py](test_m2m_protocol_cleanup.py) | 7 | 主入口先预测后评分、GT 不传入预测、oracle 仅显式启用 |
| [test_m2m_builder_leakage.py](test_m2m_builder_leakage.py) | 2 | 训练节点不会泄漏进评估实体 |
| [test_m2m_failure_modes.py](test_m2m_failure_modes.py) | 8 | 一对一解码的局限、矩阵约束和 oracle 诊断边界 |
| [test_alpine.py](test_alpine.py) | 3 | Alpine 的小图运行与资源限制 |
| [test_gadl.py](test_gadl.py) | 17 | GADL 编码器、频谱、过滤规则、训练和输出 |
| [test_gadl_paper_profile.py](test_gadl_paper_profile.py) | 5 | GADL 论文配置、损失和训练步骤 |
| [test_gadl_profiles.py](test_gadl_profiles.py) | 6 | GADL 参考配置、训练安排和可配置参数 |

上轮 Codex 新增的三项都在 `test_m2m_quotient.py`：组合接口等价、改变 GT 不影响预测、整数实体编号兼容。
现在保留并调整前两项，方法名为 `test_prediction_and_scoring_recover_expected_entities()` 和
`test_changing_ground_truth_does_not_change_prediction_or_info()`；第三项随旧接口一起删除。
当时还调整了 `test_m2m_protocol_cleanup.py` 中已有测试；MSF1 的用例独立放在 `test_macro_set_f1.py`，其他指标的数值预期保持不变。

## 在 VS Code 中查看与执行

1. 打开仓库根目录，安装 Microsoft Python 扩展，执行 `Python: Select Interpreter`，选择依赖齐全的项目解释器。
2. 执行 `Python: Configure Tests`，选择 **unittest**，目录选 **tests**，文件模式选 **test_*.py**。
3. 打开左侧烧杯图标的测试面板，展开文件、类和测试方法；点播放按钮运行，点调试按钮可逐步查看变量。
4. 第一次阅读建议打开 `test_m2m_quotient.py → EndToEndToyTest`，先看 `setUp()` 的小图与矩阵，再看断言。

操作依据：[VS Code 官方 Python 测试说明](https://code.visualstudio.com/docs/python/testing)。

在依赖齐全的解释器下，也可以从仓库根目录的终端运行：

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python -m unittest discover -s tests -p 'test_m2m_quotient.py' -v
```

当前机器的 `/opt/miniconda3/bin/python3` 有 PyTorch，但缺少 scikit-learn。本次仅在临时目录补齐缺包，未修改仓库依赖或 Conda 环境。在本机保留这些临时包时，以下命令可复现本次完整测试：

```bash
env PYTHONPATH="/tmp/planetalign-refactor-deps:$PWD" PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/planetalign-refactor-mpl \
  /opt/miniconda3/bin/python3 -m unittest discover -s tests -p 'test_*.py' -v
```

测试面板同样需要依赖齐全的解释器；终端的临时 `PYTHONPATH` 不会自动设置到测试面板。缺包导致导入失败时，先处理环境，再判断是否有代码问题。

2026-10-28 的 MSF1 定稿修正验证使用 `/opt/miniconda3/envs/m2m/bin/python`，该环境依赖齐全，无需上述临时依赖目录。测试范围为 `test_macro_set_f1.py`、`test_rescore_msf1.py`、`test_m2m*.py`、`test_many_to_many_metrics.py`（共 101 项）。
