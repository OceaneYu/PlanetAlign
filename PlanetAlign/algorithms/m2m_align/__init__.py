# 文件作用：从 PlanetAlign/algorithms/m2m_align/main.py 导出 M2MAlign，供包级导入使用。
# 阅读建议：训练实现看导出类的 train()；本文件只连接导入路径，不包含训练循环。

from .main import M2MAlign

__all__ = ['M2MAlign']
