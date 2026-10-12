# 文件作用：将各算法类集中导出，支持 from PlanetAlign.algorithms import PARROT 等写法。
# 结果去向：PlanetAlign/m2m/base.py → base_configs() 和实验脚本从这里获取算法类，再显式调用 train()。
# 阅读建议：导入不代表算法已训练；DualMatch、MEAformer 当前仍是接口占位。

import copy

from .isorank import IsoRank
from .final import FINAL
from .ione import IONE
from .regal import REGAL
from .crossmna import CrossMNA
from .nettrans import NetTrans
from .bright import BRIGHT
from .nextalign import NeXtAlign
from .parrot import PARROT
from .slotalign import SLOTAlign
from .wlalign import WLAlign
from .walign import WAlign
from .hot import HOT
from .joena import JOENA
from .dualmatch import DualMatch
from .meaformer import MEAformer
from .m2m_align import M2MAlign
from .tgae import TGAE
from .gadl import GADL
from .alpine import Alpine

__all__ = [
    'IsoRank',
    'IONE',
    'FINAL',
    'REGAL',
    'CrossMNA',
    'NetTrans',
    'BRIGHT',
    'NeXtAlign',
    'PARROT',
    'SLOTAlign',
    'WLAlign',
    'WAlign',
    'HOT',
    'JOENA',
    'DualMatch',
    'MEAformer',
    'M2MAlign',
    'TGAE',
    'GADL',
    'Alpine'
]

classes = copy.copy(__all__)
