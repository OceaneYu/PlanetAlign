import PlanetAlign

# 下载并加载 Douban 数据集
dataset = PlanetAlign.datasets.Douban(
    root='data/',
    download=True,
    train_ratio=0.2,
    seed=42
)

# 初始化 FINAL 对齐模型
model = PlanetAlign.algorithms.FINAL(
    alpha=0.9,  # FINAL 特定的超参数
).to('cpu')  # 或 'cuda' 如果你有 GPU

# 初始化日志记录器
logger = PlanetAlign.logger.TrainingLogger(
    log_dir='logs/',
    log_name='final_douban',
    save=True
)

# 训练模型
model.train(
    dataset=dataset,
    gids=[0, 1],  # 要对齐的图的索引
    use_attr=True,  # 如果可用，使用属性
    total_epochs=50
)

# 评估模型
result = model.test(
    dataset=dataset,
    gids=[0, 1],
    metrics=['Hits@1', 'Hits@10', 'MRR']
)

print(result)