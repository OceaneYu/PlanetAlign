import PlanetAlign

# 加载 Douban 数据集
dataset = PlanetAlign.datasets.Douban(
    root='data/',
    download=True,
    train_ratio=0.2,
    seed=42
)

METRICS = ['Hits@1', 'Hits@10', 'MRR']
GIDS = [0, 1]

# ── PARROT ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Running PARROT")
print("="*60)
parrot = PlanetAlign.algorithms.PARROT(alpha=0.5).to('cpu')
parrot.train(dataset=dataset, gids=GIDS, use_attr=True)
result_parrot = parrot.test(dataset=dataset, gids=GIDS, metrics=METRICS)
print("PARROT result:", result_parrot)

# ── JOENA ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Running JOENA")
print("="*60)
joena = PlanetAlign.algorithms.JOENA(alpha=0.7).to('cpu')
joena.train(dataset=dataset, gids=GIDS, use_attr=True, total_epochs=50)
result_joena = joena.test(dataset=dataset, gids=GIDS, metrics=METRICS)
print("JOENA result:", result_joena)

# ── SLOTAlign ────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Running SLOTAlign")
print("="*60)
slotalign = PlanetAlign.algorithms.SLOTAlign(bases=4).to('cpu')
slotalign.train(dataset=dataset, gids=GIDS, use_attr=True, total_epochs=200, joint_epochs=50)
result_slotalign = slotalign.test(dataset=dataset, gids=GIDS, metrics=METRICS)
print("SLOTAlign result:", result_slotalign)

# ── 汇总 ─────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Summary")
print("="*60)
print(f"{'Model':<12} {'Hits@1':>8} {'Hits@10':>9} {'MRR':>8}")
print("-"*40)
for name, r in [("PARROT", result_parrot), ("JOENA", result_joena), ("SLOTAlign", result_slotalign)]:
    print(f"{name:<12} {r['Hits@1']:>8.4f} {r['Hits@10']:>9.4f} {r['MRR']:>8.4f}")
