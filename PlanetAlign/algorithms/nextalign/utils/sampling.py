# 文件作用：按当前嵌入和指定采样策略生成 NeXtAlign 的负例节点。
# 数据来源：PlanetAlign/algorithms/nextalign/main.py → NeXtAlign.train() 传入嵌入、节点编号映射与批次节点。
# 结果去向：negative_sampling_exact() 返回负例与概率，供 PlanetAlign/algorithms/nextalign/model.py → Model.loss() 的训练样本使用。

import numpy as np


# 核心思路：在本图或另一侧候选中计算嵌入分数与采样概率，再按策略抽取负例。
# 输出结果：负例编号数组和概率，不更新嵌入。
def negative_sampling_exact(x, N_negs, node1, node_mapping, target_dist_name, graph_name, removed=0, node_mapping2=None):
    """
    Proposed sampling in NextAlign.
    @param x: node embedding matrix
    @param N_negs: number of sampled negative samples
    @param node1: mini-batch of anchor nodes
    @param node_mapping: original node indices to node indices in world-view graph
    @param target_dist_name: what distribution for sampling
    @param graph_name: indicates nodes in which graph
    @param removed: nodes that are not allowed to be sampled
    @param node_mapping2: node mappings for nodes in G2
    @return:
        generate_examples: nodes that are sampled
        probs: sampling probabilities
    """
    node1 = node1.cpu().detach().numpy()
    x = x.cpu().detach().numpy()
    node_mapping = node_mapping.cpu().detach().numpy()
    if node_mapping2 is not None:
        node_mapping2 = node_mapping2.cpu().detach().numpy()
    out_features = x.shape[1]
    if target_dist_name == 'p_n':
        if graph_name == 'g1':
            emb1 = x[node_mapping[node1], 0: out_features//2]
            emb2 = x[node_mapping, 0: out_features//2]
        else:
            emb1 = x[node_mapping[node1], out_features // 2: out_features]
            emb2 = x[node_mapping, out_features // 2: out_features]
        n2 = node_mapping.shape[0]
        inner_products = np.dot(emb1, emb2.T)
        inner_products = sigmoid(inner_products)
        p_probs = np.clip(inner_products, 1e-7, 1e7)

    elif target_dist_name == 'p_dc':
        if graph_name == 'g1':
            emb1 = x[node_mapping[node1], out_features//2: out_features]
            emb2 = x[node_mapping2, out_features//2: out_features]
        else:
            emb1 = x[node_mapping[node1], 0: out_features//2]
            emb2 = x[node_mapping2, 0: out_features//2]
        n2 = node_mapping2.shape[0]
        inner_products = np.dot(emb1, emb2.T)
        inner_products = sigmoid(inner_products)

        p_probs = np.clip(inner_products, 1e-7, 1e7)

    else:
        emb1_1 = x[node_mapping[node1], 0:out_features//2]
        emb1_2 = x[node_mapping[node1], out_features // 2: out_features]
        emb2_1 = x[node_mapping2, 0:out_features//2]
        emb2_2 = x[node_mapping2, out_features // 2: out_features]
        n2 = node_mapping2.shape[0]

        inner_products = np.dot(emb1_1, emb2_2.T) * 1 + np.dot(emb1_2, emb2_1.T) * 1
        inner_products = sigmoid(inner_products)
        p_probs = np.clip(inner_products, 1e-7, 1e7)

    generate_examples = []
    probs = []

    for i in range(len(node1)):
        p_probs[i][node_mapping[node1[i]]] = 0
        if not isinstance(removed, int):
            p_probs[i][removed] = 0
        p = p_probs[i] / np.sum(p_probs[i])
        samples = np.random.choice(n2, N_negs, p=p)
        probs.append(list(p[samples]))
        generate_examples.append(list(samples))

    generate_examples = np.array(generate_examples, dtype=np.int64).reshape((-1,))
    probs = np.array(probs, dtype=np.float32)

    return generate_examples, probs


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
