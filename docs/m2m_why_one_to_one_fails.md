# 为什么一对一图对齐方法不适用于多对多图对齐

### —— 基于 PLANETALIGN 的实证与理论分析

> 本文档中每一条结论都同时给出两类依据：**〔实验〕** 取自在本仓库多对多（many-to-many，M2M）数据集上实际运行算法的结果，可由
> [`scripts/diagnose_m2m_failure.py`](../scripts/diagnose_m2m_failure.py) 复现，原始数据见
> [`logs/m2m_diag/diagnose_m2m_failure.json`](../logs/m2m_diag/diagnose_m2m_failure.json)；**〔理论〕** 取自提出该算法的原始论文。
> 断言式校验见 [`tests/test_m2m_failure_modes.py`](../tests/test_m2m_failure_modes.py)。
> 实验配置：`seed=42`、`train_ratio=0.2`、quick profile（小 epoch 数）；数值为定性证据，量级稳定、可复现，但非调参后的最优值。

---

## 摘要

PLANETALIGN 集成的 18 个网络对齐（network alignment, NA）算法均面向**一对一**（one-to-one, 1-1）设定。本文将它们按原始论文归为三族
——**一致性（consistency）**、**嵌入（embedding）**、**最优传输（OT）**——并在仓库自带的 M2M 基准（Douban、Cora）上逐一运行，
论证这些方法在多对多对齐上不适用的原因，可归为三类：**(A) 表示缺口**、**(B) 目标/归纳偏置错配**、**(C) 监督与评测错配**。
我们进一步指出，三族在 (B) 上呈现**相反方向**的失败：一致性方法因缺乏边际约束而坍缩到枢纽节点（实测单个目标节点被
**最多 3110 个**源节点指向），OT 方法则因质量守恒的边际约束而被强制趋近置换（JOENA 的列边际变异系数实测为 **0.0**，
即严格均匀），二者都无法表达"多个节点同属一个实体"。我们也诚实地指出该结论的边界：当群组可由属性独立恢复且两图近似同构时
（Cora），强 1-1 对齐器叠加属性分组已能取得高分，**问题不在于 1-1 方法不会对齐节点，而在于它们只输出节点级二部相似度、
不提供任何实体/群组机制**；一旦 M2M 结构非平凡（Douban：源图远大于目标图、含模糊边界），这一缺失就在诚实评测下暴露为巨大差距。

---

## 1. 引言与问题定义

### 1.1 一对一网络对齐（现有方法的设定）

给定两张图 $G_1=(V_1,E_1)$、$G_2=(V_2,E_2)$（$|V_1|=n_1$，$|V_2|=n_2$），一对一对齐寻求一个跨图节点对应。
在 PLANETALIGN 中，这一设定被**写死在核心 API**里：所有算法的输出都是单一相似度/传输矩阵

$$
\mathbf{S}\in\mathbb{R}^{n_1\times n_2},\qquad \mathbf{S}(x,y)\ \text{表示 } x\in G_1 \text{ 与 } y\in G_2 \text{ 对齐的可能性}.
$$

> **〔理论〕** 这一形式化与各原论文一致。例如 JOENA 的问题定义（Definition 1）即把对齐写作"映射矩阵
> $\mathbf{S}\in\mathbb{R}^{n_1\times n_2}$，$\mathbf{S}(x,y)$ 表示节点 $x$ 与 $y$ 对齐的可能性"，并以 OT 映射 $\mathbf{S}\in\Pi(\boldsymbol\mu_1,\boldsymbol\mu_2)$
> 求解（[Yu et al., WWW&#39;25](https://arxiv.org/abs/2502.19334)）。

评测端同样是 1-1：`PlanetAlign/algorithms/base_model.py` 的 `test()` 取 `test_pairs (m,2)` 的**一对一**金标，用
`Hits@K`、`MRR` 对每个查询节点的相似度行排序，只检验**唯一**正确伙伴是否进入前 K（`metrics/hits.py`、`metrics/mrr.py`）。
`check_pairwise_input_graphs` 还断言输入**恰好 2 张图**。

### 1.2 多对多图对齐（目标任务）

仓库的 M2M 基准由 [`PlanetAlign/utils/many2many_builder.py`](../PlanetAlign/utils/many2many_builder.py) 经"节点拆分"从 1-1 数据生成，
真值存于 `*_gt_many2many.json`，其结构为**实体（entity）= 双侧节点集合**：

```json
"e1": {"src": [1035, 3906, 3863], "tgt": [677, 1118, 1119]}
```

每个实体是一个跨两图的等价类，满足四条性质：**① 混合粒度**（1-1 / 1-N / N-1 / N-N）；**② 组内内聚**（同一拆分组共享高概率内部边）；
**③ 组级外部结构**（原锚点的邻居被分摊到各拆分节点，单个拆分节点只保留约 $1/k$ 的邻居，连通性在**组级**而非节点级保持）；
**④ 模糊边界**（`overlap_ratio`，少量节点可同属多个实体）。

与 1-1 设定最本质的差异是 **图内共指（intra-graph co-reference）**：如上例，`src` 侧的 3 个节点是**同一张图内**的同一实体。
正确的 M2M 输出是"对每张图的一个划分 + 划分之间的匹配"，而非节点到节点的映射。

> **〔实验〕** 在 Douban M2M 上，895 个实体中 **442 个（49%）** 至少有一侧含多个节点（即需要图内共指）；
> 在 Cora M2M 上为 651/2167（30%）。见表 2。

---

## 2. 现有方法分类

下表按各算法**原始论文**归族（族标签取自仓库各算法 docstring，权威且与论文一致）。**输出形态**与**核心约束**两列是后文失败分析的依据。

| 算法      | 族              | 出处        | 输出形态                                    | 核心约束 / 归纳偏置                       |
| --------- | --------------- | ----------- | ------------------------------------------- | ----------------------------------------- |
| IsoRank   | 一致性          | PNAS 2008   | $\mathbf{S}\in\mathbb{R}^{n_1\times n_2}$ | 对齐一致性不动点（邻居的邻居也对齐）      |
| FINAL     | 一致性          | KDD 2016    | $\mathbf{S}$                              | 属性+拓扑一致性不动点                     |
| IONE      | 嵌入            | IJCAI 2016  | 每节点一向量                                | 锚点监督，输入/输出上下文嵌入             |
| REGAL     | 嵌入            | CIKM 2018   | 每节点一向量                                | 结构身份嵌入（xNetMF）+ 最近邻            |
| CrossMNA  | 嵌入            | WWW 2019    | 每节点一向量                                | 跨网络共享嵌入，锚点对齐                  |
| NetTrans  | 嵌入            | KDD 2020    | 每节点一向量                                | 学习网络到网络的变换                      |
| BRIGHT    | 嵌入            | WWW 2021    | 每节点一向量                                | RWR 位置嵌入 + 共享 GCN                   |
| NeXtAlign | 嵌入            | KDD 2021    | 每节点一向量                                | **一致性与差异性（disparity）平衡** |
| WL-Align  | 嵌入            | TKDE 2023   | 每节点一向量                                | Weisfeiler-Lehman 重标号正则表示          |
| WAlign    | 嵌入            | KDD 2021    | 每节点一向量                                | GCN + Wasserstein 距离判别器              |
| DualMatch | 嵌入            | WWW 2023    | 每节点一向量                                | 时序 KG 实体对齐（仓库内为 stub）         |
| MEAformer | 嵌入            | ACM MM 2023 | 每节点一向量                                | 多模态实体对齐 Transformer（stub）        |
| T-GAE     | 嵌入            | 2023        | 每节点一向量                                | 可迁移图自编码器                          |
| PARROT    | OT              | WWW 2023    | 传输计划$\mathbf{S}\in\Pi$                | 位置感知正则 OT，近端点法（Sinkhorn 型）  |
| SLOTAlign | OT              | ICDE 2023   | 传输计划$\mathbf{S}\in\Pi$                | Gromov-Wasserstein + 结构学习             |
| HOT       | OT              | AAAI 2024   | 多边际耦合                                  | 层次化**多边际** OT                 |
| JOENA †  | OT（+嵌入混合） | WWW 2025    | 传输计划$\mathbf{S}\in\Pi(\mu_1,\mu_2)$   | 嵌入塑造 OT 代价 + Sinkhorn 联合优化      |

> † JOENA 是 OT 与嵌入的**混合**方法（论文标题即 *Joint Optimal Transport and Embedding*）。本文将其归入 OT，理由有二：
> (1) 仓库 docstring 权威标注为 "OT-based method JOENA"；(2) 就失败分析而言，**定义并约束其输出的是 OT 侧**——MLP 嵌入仅在上游
> 塑造代价矩阵（`FusedGWLoss`），最终返回的 `self.S` 是受边际约束的传输计划 $\mathbf{S}\in\Pi(\mu_1,\mu_2)$。让它在 M2M 失败的
> 归纳偏置（质量守恒，实测 colCoV=0.00）来自 OT 侧；其嵌入侧是 1-1 锚点监督、同样不提供群组结构。

要点：**无论哪一族，对外输出都是一个跨图节点级对象**（二部相似度矩阵、或由每节点一向量诱导的二部相似度，或一个 OT 耦合）。
没有任何一族输出"实体/群组"这一对象，也没有任何一族表达图内共指。

---

## 3. 实验设置

- **数据集**：Douban M2M（`data/m2m_overlap_0.05`，含模糊边界）、Cora M2M（`data/m2m_no_overlap`）。二者均带节点属性。
- **代表算法**（每族选 2）：一致性 = IsoRank / FINAL；嵌入 = REGAL / BRIGHT；OT = PARROT / JOENA。
- **流程**：复用 `scripts/run_m2m_experiments.py` 的加载与取 $\mathbf{S}$ 路径，训练后取每个模型暴露的 $\mathbf{S}$，运行五项诊断探针（P1–P5）。
- **指标**：1-1 端用 `Hits@1/MRR`；M2M 端用 `ACS / MSF1 / MicroF1 / M2M-SGS / M2M-EGS`。M2M 指标分两种读出——
  **泄漏式适配器**（`similarity_to_pred_entities`，把真值的源分组与目标规模回灌给方法）与 **盲评协议**（`m2m_blind.evaluate_blind`，
  方法自行发现群组、自行匹配，不看真值）。

**表 2：数据集统计与"一对一读出召回上限"**

| 数据集     | $n_1$ | $n_2$ | 实体数 | 1-1 / 1-N / N-1 / N-N  | 需图内共指 | argmax 读出召回上限 |
| ---------- | ------- | ------- | ------ | ---------------------- | ---------- | ------------------- |
| Douban M2M | 4265    | 1460    | 895    | 453 / 108 / 203 / 131  | 442 (49%)  | **0.821**     |
| Cora M2M   | 3595    | 3553    | 2167   | 1516 / 217 / 217 / 217 | 651 (30%)  | **0.837**     |

> "argmax 读出召回上限"定义：原生流程对每个源节点取 $\arg\max_y \mathbf{S}(x,y)$，故一个实体的 $|src_e|$ 个源节点最多指出
> $|src_e|$ 个不同目标，至多命中其 $|tgt_e|$ 个真值目标中的 $\min(|src_e|,|tgt_e|)$ 个。微平均召回上限 $=\sum_e \min(|src_e|,|tgt_e|)/\sum_e |tgt_e|$，
> **与具体方法无关，连理想 oracle 也受此限**。（天生不可能召回所有目标节点）

---

## 4. 为什么一对一方法不适用：三类失败 × 三族

**表 3：主结果（Douban / Cora）。** `leak`/`blind` 为泄漏式 / 盲评 MicroF1；`maxSrc/Tgt` 为被同一目标节点 argmax 指向的最多源节点数；`colCoV` 为耦合列边际的变异系数（越小越接近"每个目标吸收等量质量"）。ot方法有质量守恒约束，所以会比较小。

| 算法    | 族     | Hits@1        | MicroF1 (leak) | MicroF1 (blind)         | maxSrc/Tgt            | colCoV                |
| ------- | ------ | ------------- | -------------- | ----------------------- | --------------------- | --------------------- |
| IsoRank | 一致性 | 0.031 / 0.027 | 0.037 / 0.041  | 0.043 / 0.041           | **3110 / 2893** | 0.96 / 1.10           |
| FINAL   | 一致性 | 0.260 / 0.627 | 0.250 / 0.488  | 0.088 / 0.513           | 598 / 80              | 2.06 / 1.42           |
| REGAL   | 嵌入   | 0.031 / 0.346 | 0.055 / 0.320  | 0.292 / 0.387           | 36 / 11               | 0.28 / 0.32           |
| BRIGHT  | 嵌入   | 0.187 / 0.736 | 0.153 / 0.704  | 0.383 / 0.699           | 42 / 8                | 0.50 / 0.18           |
| PARROT  | OT     | 0.458 / 0.956 | 0.429 / 0.951  | **0.091** / 0.952 | 12 / 4                | 0.12 / 0.19           |
| JOENA   | OT     | 0.453 / 0.991 | 0.369 / 0.994  | 0.318 / 0.991           | 170 / 5               | **0.00** / 0.00 |

### 4.1 (A) 表示缺口：缺了什么

**论断：1-1 方法的输出是一个跨图二部对象，没有表达图内共指、可变规模实体的任何槽位。**

> **〔实验〕** 所有方法的 $\mathbf{S}$ 形状恒为 $[n_1,n_2]$（表 3 各行 S 形状均为 `[4265,1460]`/`[3595,3553]`），只有跨图块、
> 没有 $[n_1,n_1]$ 的图内块；而 49%（Douban）/30%（Cora）的实体要求"同图多个节点是同一实体"——这在 $\mathbf{S}$ 中**无处可写**。
> 即便不看性能，这是结构性的（见 `tests/test_m2m_failure_modes.py::RepresentationGapTest`）。
>
> **〔理论〕** 各论文的问题设定都把对齐定义为两图间的（软）节点对应：JOENA 的 $\mathbf{S}\in\mathbb{R}^{n_1\times n_2}$、
> OT 族的耦合 $\mathbf{S}\in\Pi$、嵌入族的"每节点一向量 + 最近邻"。没有"等价类/群组"这一对象，自然无法输出它。
> HOT 虽是多图/多边际 OT，但输出仍是节点级耦合而非实体（[Zeng et al., AAAI&#39;24](https://doi.org/10.1609/aaai.v38i15.29605)）。

### 4.2 (B) 目标/归纳偏置错配：多了什么、且有害

三族都带有为 1-1 设计的归纳偏置，在 M2M 上**主动起反作用**，且方向相反。

#### 4.2.1 一致性方法——无边际约束 → 坍缩到枢纽

> **〔实验〕** IsoRank 把 **3110/2893** 个源节点的 argmax 全部指向同一个目标节点（表 3 `maxSrc/Tgt`），列边际变异系数高达
> 0.96–2.06，行熵接近均匀（0.94）。这是退化的枢纽坍缩：解集中到少数高 PageRank 式节点，既非 1-1 也非实体级。Hits@1 仅 0.03。
>
> **〔理论〕** IsoRank/FINAL 求解"对齐一致性"不动点——若 $x\leftrightarrow y$ 则其邻居也应对齐
> （[Singh et al., PNAS&#39;08](https://www.pnas.org/doi/full/10.1073/pnas.0806627105)；[Zhang &amp; Tong, KDD&#39;16](https://dl.acm.org/doi/10.1145/2939672.2939766)）。
> 该不动点假设两图近似为置换关系；性质③的节点拆分把每个拆分点的度打散到约 $1/k$，破坏了邻域一致性，迭代便把质量灌向少数枢纽。
> 目标函数中**没有任何项**奖励"多个节点共同对齐到同一区域"。

#### 4.2.2 OT 方法——边际（质量守恒）约束 → 强制趋近置换，禁止多对一

> **〔实验〕** JOENA 的列边际变异系数实测为 **0.00**（PARROT 为 0.12–0.19），即每个目标列吸收**严格均匀**的质量；
> 对应地 OT 族的 `maxSrc/Tgt` 很小（Cora 上 PARROT=4、JOENA=5，接近 $n_1/n_2$ 下界）。这正是质量守恒的直接观测：
> 没有哪个目标能"多吃"，于是 Douban/Cora 各 203/217 个 **N-1** 实体（多个源 → 一个目标）在结构上无法被实现。
>
> **〔理论〕** OT 把对齐约束在传输多胞形 $\mathbf{S}\in\Pi(\boldsymbol\mu_1,\boldsymbol\mu_2)$ 内，行/列边际固定（JOENA 取均匀 $1/n$，
> [Yu et al., WWW&#39;25](https://arxiv.org/abs/2502.19334)）。这意味着每个源节点的总质量被锁定、每个目标节点能接收的总质量也被锁定——
> 一个源节点无法把全部质量投给一个 $k>1$ 的群组，多个源节点也无法都集中到一个目标。PARROT 的正则 OT
> （[Zeng et al., WWW&#39;23](https://dl.acm.org/doi/10.1145/3543507.3583357)）、SLOTAlign 的 Gromov-Wasserstein
> （[Tang et al., ICDE&#39;23](https://doi.org/10.1109/ICDE55515.2023.00129)）同理；GW 额外奖励**结构同构**（保距匹配），
> 而节点拆分制造的度不对称恰恰被 GW 当作"不一致"而惩罚。
>
> **〔实验·佐证〕** 在节点拆分最重的 Douban（$n_1=4265 \gg n_2=1460$，大量 N-1）上，即便 OT 方法的 1-1 表现尚可
> （PARROT Hits@1=0.458），其盲评 MicroF1 也跌到 **0.091**（PARROT）/0.318（JOENA）。

#### 4.2.3 嵌入方法——每节点一向量 + 最近邻读出 + 差异性正则 → 推开共指节点

> **〔实验〕** REGAL/BRIGHT 的读出是"每节点一向量取最近邻"，本质 1-1；其 `maxSrc/Tgt` 居中（8–42），既不像一致性那样坍缩、
> 也不像 OT 那样被边际锁死，但盲评 MicroF1 在 Douban 仅 0.29–0.38——能借属性捕到部分组信号，却没有等价类结构可言。
>
> **〔理论〕** 嵌入族给每个节点学**一个**向量、用内积/最近邻读出，且由 1-1 锚点监督。NeXtAlign 更直接：其目标是
> 平衡一致性与**差异性（disparity）**，即刻意把近似重复的节点在嵌入空间**推开**
> （[Zhang et al., KDD&#39;21](https://dl.acm.org/doi/abs/10.1145/3447548.3467331)）；WAlign 用 Wasserstein 判别器对齐两图分布、
> 同样以可区分性为目标。而 M2M 的拆分节点恰是"近似重复且应判为同一实体"——差异性正则的方向与之**正相反**。

### 4.3 (C) 监督与评测错配：错了什么

**论断：训练监督是 1-1 锚点、模型从不知道"分组"任务；标准评测只认一个金标；常用 M2M 适配器靠回灌真值才显得"work"。**

> **〔实验·评测泄漏〕** 同一个 $\mathbf{S}$，泄漏式适配器 vs 盲评的 MicroF1 差距巨大：Douban 上 PARROT 由 0.429 跌到
> **0.091**（虚高 4.7×）、FINAL 由 0.250 跌到 0.088（虚高 2.8×）。泄漏式适配器对每个真值实体取 $k=|tgt_e|$ 个目标、
> 并以真值的源分组为查询，等于把"答案的分组与规模"喂回方法。`tests/test_m2m_failure_modes.py::EvaluationLeakTest` 证明：
> 即便 $\mathbf{S}$ 全为常数（零信息），该适配器仍精确吐出 $|tgt_e|$ 个目标——它泄漏的是答案的基数。
>
> **〔实验·读出天花板〕** 原生 `Hits@K`/`argmax` 读出对每个查询只认一个金标；表 2 显示即便理想 oracle，目标集微平均召回也被
> 钉死在 0.82–0.84 以下；实际 Hits@1 低至 0.03（IsoRank）。
>
> **〔理论〕** 各论文的监督信号都是锚点对 + `Hits@K/MRR` 评测（[REGAL, CIKM&#39;18](https://dl.acm.org/doi/10.1145/3269206.3271788)、
> [BRIGHT, WWW&#39;21](https://doi.org/10.1145/3442381.3450053) 等），其语义是"每个节点恰有一个正确伙伴"。M2M 真值是**集合**语义，
> 两者在定义层面就不对齐：方法既没被告知群组任务，评测也无法为一个"集合答案"记分（除非外部补上分组机制）。

---

## 5. 讨论：结论的边界（何时"够用"、何时不够）

诚实地看实验，结论不是"1-1 方法不会对齐节点"，而是**它们只提供节点级二部相似度、不提供实体/群组机制**；这一缺失是否致命，取决于
群组能否被独立恢复、两图是否近似同构。

> **〔实验〕** 在 **Cora**（属性维 1433、两图近似同构、$n_1\approx n_2$）上，盲评流程用属性内聚就能恢复群组，叠加一个强 1-1 对齐器后
> JOENA 盲评 MicroF1 达 **0.991**、PARROT 达 0.952——M2M 任务被"属性分组 + 1-1 匹配"基本解决。
> 而在 **Douban**（含模糊边界、$n_1\gg n_2$、大量 N-1 拆分）上，**所有**代表方法的盲评 MicroF1 都 $\le 0.44$，无一真正解决。

这说明：(A) 表示缺口与 (C) 评测错配是**普适且结构性**的；(B) 归纳偏置之害在 M2M 结构非平凡时才充分暴露。
要做对 M2M，需要在节点级 $\mathbf{S}$ 之上补回三件 1-1 方法天然没有的东西：**图内群组发现、可变规模的集合读出、群组级匹配**——
这也正是本仓库 `m2m_decode` / `m2m_blind` 盲评方向所做的事。

---

## 6. 结论

1. **分类**：18 个算法按原论文归为一致性（2）、嵌入（12）、OT（4）三族；三族对外都只输出**跨图节点级**对象，无一输出实体/群组。
2. **三类失败**：(A) 表示缺口——$\mathbf{S}\in\mathbb{R}^{n_1\times n_2}$ 无法表达图内共指（49%/30% 的实体需要它）；
   (B) 归纳偏置错配——一致性坍缩到枢纽（maxSrc/Tgt 达 3110）、OT 被边际锁成置换（colCoV=0.00，禁止 N-1）、嵌入用差异性推开共指；
   (C) 监督/评测错配——1-1 锚点 + `Hits@K`（召回上限 0.82–0.84、实测 Hits@1 低至 0.03），泄漏式适配器把分数虚高至 4.7×。
3. **边界**：群组可由属性独立恢复且两图近似同构时（Cora），1-1 方法 + 外部分组已足够；M2M 结构非平凡时（Douban）差距在诚实评测下暴露。

---

## 参考文献

- **PLANETALIGN** — Yu et al. *PLANETALIGN: A Comprehensive Python Library for Benchmarking Network Alignment.* arXiv:2505.21366. [https://arxiv.org/abs/2505.21366](https://arxiv.org/abs/2505.21366)
- **IsoRank** — Singh, Xu, Berger. *Global alignment of multiple protein interaction networks…* PNAS 2008. [https://www.pnas.org/doi/full/10.1073/pnas.0806627105](https://www.pnas.org/doi/full/10.1073/pnas.0806627105)
- **FINAL** — Zhang, Tong. *FINAL: Fast Attributed Network Alignment.* KDD 2016. [https://dl.acm.org/doi/10.1145/2939672.2939766](https://dl.acm.org/doi/10.1145/2939672.2939766)
- **IONE** — Liu et al. *Aligning Users Across Social Networks Using Network Embedding.* IJCAI 2016. [https://www.ijcai.org/Proceedings/16/Papers/254.pdf](https://www.ijcai.org/Proceedings/16/Papers/254.pdf)
- **REGAL** — Heimann et al. *REGAL: Representation Learning-based Graph Alignment.* CIKM 2018. [https://dl.acm.org/doi/10.1145/3269206.3271788](https://dl.acm.org/doi/10.1145/3269206.3271788)
- **CrossMNA** — Chu et al. *Cross-Network Embedding for Multi-Network Alignment.* WWW 2019. [https://doi.org/10.1145/3308558.3313499](https://doi.org/10.1145/3308558.3313499)
- **NetTrans** — Zhang et al. *NetTrans: Neural Cross-Network Transformation.* KDD 2020. [https://dl.acm.org/doi/10.1145/3394486.3403141](https://dl.acm.org/doi/10.1145/3394486.3403141)
- **BRIGHT** — Yan et al. *BRIGHT: A Bridging Algorithm for Network Alignment.* WWW 2021. [https://doi.org/10.1145/3442381.3450053](https://doi.org/10.1145/3442381.3450053)
- **NeXtAlign** — Zhang et al. *Balancing Consistency and Disparity in Network Alignment.* KDD 2021. [https://dl.acm.org/doi/abs/10.1145/3447548.3467331](https://dl.acm.org/doi/abs/10.1145/3447548.3467331)
- **WL-Align** — *WL-Align: Weisfeiler-Lehman Relabeling for Aligning Users Across Networks via Regularized Representation Learning.* TKDE 2023. [https://doi.org/10.1109/TKDE.2023.3277843](https://doi.org/10.1109/TKDE.2023.3277843)
- **WAlign** — *Unsupervised Graph Alignment with Wasserstein Distance Discriminator.* KDD 2021. [https://dl.acm.org/doi/10.1145/3447548.3467332](https://dl.acm.org/doi/10.1145/3447548.3467332)
- **PARROT** — Zeng et al. *PARROT: Position-Aware Regularized Optimal Transport for Network Alignment.* WWW 2023. [https://dl.acm.org/doi/10.1145/3543507.3583357](https://dl.acm.org/doi/10.1145/3543507.3583357)
- **SLOTAlign** — Tang et al. *Robust Attributed Graph Alignment via Joint Structure Learning and Optimal Transport.* ICDE 2023. [https://doi.org/10.1109/ICDE55515.2023.00129](https://doi.org/10.1109/ICDE55515.2023.00129)
- **HOT** — Zeng et al. *Hierarchical Multi-Marginal Optimal Transport for Network Alignment.* AAAI 2024. [https://doi.org/10.1609/aaai.v38i15.29605](https://doi.org/10.1609/aaai.v38i15.29605)
- **JOENA** — Yu et al. *Joint Optimal Transport and Embedding for Network Alignment.* WWW 2025. [https://arxiv.org/abs/2502.19334](https://arxiv.org/abs/2502.19334)
- **DualMatch** — *Unsupervised Entity Alignment for Temporal Knowledge Graphs.* WWW 2023. [https://arxiv.org/pdf/2302.00796](https://arxiv.org/pdf/2302.00796)
- **MEAformer** — *MEAformer: Multi-modal Entity Alignment Transformer for Meta Modality Hybrid.* ACM MM 2023. [https://arxiv.org/pdf/2212.14454](https://arxiv.org/pdf/2212.14454)
- **T-GAE** — *T-GAE: Transferable Graph Autoencoder for Network Alignment.*

---

## 附：复现

```bash
# 1) 跑诊断（6 个代表算法 × 2 数据集，输出 5 项探针到 JSON）
python scripts/diagnose_m2m_failure.py

# 2) 失败模式断言
python -m unittest tests.test_m2m_failure_modes -v

# 3) 全算法对照扫表（可选，慢）
python scripts/run_m2m_experiments.py --m2m-root data/m2m_overlap_0.05 \
    --datasets douban_m2m --algorithms all --profile quick
```
