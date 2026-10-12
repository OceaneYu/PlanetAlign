# 文件作用：导出 PlanetAlign 的数据、算法、指标、工具、日志与多对多子包，供外部程序导入。
# 阅读建议：实际训练入口在 scripts/run_quotient_compare.py → main() 或调用者脚本，整体顺序见 CODE_MAP.md。

from . import data, datasets, algorithms, metrics, utils, logger, m2m
