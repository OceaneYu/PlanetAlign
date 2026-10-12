# 文件作用：集中导出内置基准类，如 PlanetAlign/datasets/douban.py → Douban。
# 阅读建议：各类 __init__() 负责可选下载和文件检查，实际读取 .pt 由 PlanetAlign/data/dataset.py → Dataset 完成。

import copy

from .phone_email import PhoneEmail
from .foursquare_twitter import FoursquareTwitter
from .acm_dblp import ACM_DBLP
from .cora import Cora
from .douban import Douban
from .flickr_myspace import FlickrMySpace
from .flickr_lastfm import FlickrLastFM
from .dbp15k import DBP15K_FR_EN, DBP15K_JA_EN, DBP15K_ZH_EN
from .arenas import Arenas
from .ppi import PPI
from .arxiv import ArXiv
from .sacchcere import SacchCere
from .ggi import GGI
from .airport import Airport
from .pems import PeMS08
from .italy import Italy

__all__ = ['PhoneEmail',
           'FoursquareTwitter',
           'ACM_DBLP',
           'Cora',
           'Douban',
           'FlickrMySpace',
           'FlickrLastFM',
           'DBP15K_FR_EN',
           'DBP15K_JA_EN',
           'DBP15K_ZH_EN',
           'Arenas',
           'PPI',
           'ArXiv',
           'SacchCere',
           'GGI',
           'Airport',
           'PeMS08',
           'Italy']

classes = copy.copy(__all__)
