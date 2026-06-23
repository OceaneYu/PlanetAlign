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
from .joena_group_decode import JOENAGroupDecode
from .dualmatch import DualMatch
from .meaformer import MEAformer
from .m2m_align import M2MAlign
from .tgae import TGAE

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
    'JOENAGroupDecode',
    'DualMatch',
    'MEAformer',
    'M2MAlign',
    'TGAE'
]

classes = copy.copy(__all__)
