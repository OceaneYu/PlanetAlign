# 文件作用：约定对齐算法的共同接口，并集中处理输入检查、设备设置、节点评估和日志创建。
# 数据来源：具体算法的 train() 接收 Dataset 后调用本类检查；test() 使用训练后保存的 self.S。
# 核心依赖：PlanetAlign/utils/anchors.py → get_anchor_pairs()，PlanetAlign/metrics/hits.py、PlanetAlign/metrics/mrr.py、PlanetAlign/metrics/ranking.py 的节点排名指标。
# 阅读建议：先看 test() 与 check_inputs()；train() 是待具体算法实现的接口，不包含训练逻辑。

from typing import Union, Optional, List, Tuple
import warnings
import torch

from PlanetAlign.utils import get_anchor_pairs
from PlanetAlign.data import Dataset, BaseData
from PlanetAlign.logger import TrainingLogger
from PlanetAlign.metrics import hits_ks_scores, mean_rank_score, mrr_score


# 功能：供 PlanetAlign/algorithms/ 下的算法继承，统一 dtype、device 和可选相似度状态 self.S。
class BaseModel:
    def __init__(self, dtype: torch.dtype = torch.float32):
        assert dtype in [torch.float32, torch.float64], 'Invalid floating point dtype'
        self.dtype = dtype
        self.device = 'cpu'
        self.S = None

    # 流程位置：训练接口占位，子类需要实现；直接调用会抛出 NotImplementedError。
    def train(self, 
              dataset: Dataset,
              gids: Union[List[int], Tuple[int, ...]],
              *args, **kwargs):
        """
        Parameters
        ----------
        dataset : Dataset
            The dataset containing the graphs to be aligned and the training/test data.
        gids : list[int] or tuple[int, ...]
            The indices of the graphs in the dataset to be aligned.
        """
        raise NotImplementedError

    # 输入来源：调用者传入 Dataset 与图编号；使用 self.S 和 dataset.test_data 中的节点对。
    # 功能：调用 Hits@K、MRR 和可选 MeanRank 指标，按传入策略处理并列分数。
    # 输出结果：一对一节点指标字典；多对多实体评分需使用 PlanetAlign/m2m/ 的接口。
    def test(self,
             dataset: Dataset,
             gids: Union[List[int], Tuple[int, ...]],
             metrics: tuple[str] | list[str] = None,
             tie_policy: str = 'optimistic',
             tie_atol: float = 1e-8,
             tie_rtol: float = 1e-7):
        """
        Parameters
        ----------

        dataset : Dataset
            The dataset containing the graphs to be aligned and the training/test data.
        gids : list[int] or tuple[int, ...]
            The indices of the graphs in the dataset to be aligned.
        metrics : tuple[str] or list[str], optional
            The metrics to be computed after alignment. Default is None, which computes Hits@K (K=1, 10, 30, 50) and MRR metrics.
        """

        assert self.S is not None, 'Model is not trained yet, call train() method first'

        results = {}

        gid1, gid2 = gids
        test_pairs = get_anchor_pairs(dataset.test_data, gid1, gid2)
        if metrics is None:
            hits = hits_ks_scores(
                self.S, test_pairs, mode='mean', tie_policy=tie_policy,
                atol=tie_atol, rtol=tie_rtol
            )
            results.update({f'Hits@{k}': hits[k] for k in hits.keys()})
            results['MRR'] = mrr_score(
                self.S, test_pairs, mode='mean', tie_policy=tie_policy,
                atol=tie_atol, rtol=tie_rtol
            )
        else:
            ks = [int(hit_metric.split('@')[-1]) for hit_metric in metrics if hit_metric.startswith('Hits@')]
            hits = hits_ks_scores(
                self.S, test_pairs, ks=ks, mode='mean',
                tie_policy=tie_policy, atol=tie_atol, rtol=tie_rtol
            )
            results.update({f'Hits@{k}': hits[k] for k in ks if k in hits})
            if 'MRR' in metrics:
                results['MRR'] = mrr_score(
                    self.S, test_pairs, mode='mean', tie_policy=tie_policy,
                    atol=tie_atol, rtol=tie_rtol
                )
            if 'MeanRank' in metrics or 'Mean Rank' in metrics:
                results['MeanRank'] = mean_rank_score(
                    self.S, test_pairs, mode='mean', tie_policy=tie_policy,
                    atol=tie_atol, rtol=tie_rtol
                )

        return results

    # 功能：记录后续训练所用设备并返回对象，具体张量的创建/搬移由算法实现。
    def to(self, device: Union[str, torch.device]) -> 'BaseModel':
        try:
            self.device = torch.device(device)
        except Exception:
            raise ValueError(f"Invalid device '{device}' specified.")
        return self

    # 输出结果：创建 PlanetAlign/logger/train_logger.py → TrainingLogger，按算法、数据集、训练比例和种子组织路径。
    def init_training_logger(self,
                             dataset: Dataset,
                             use_attr: bool,
                             log_dir: str = 'logs',
                             log_name: Optional[str] = None,
                             save_log: bool = True,
                             **kwargs) -> TrainingLogger:
        attr_dir = 'attr' if use_attr else 'plain'
        log_dir = f"{log_dir}/{self.__class__.__name__}/{attr_dir}/{dataset.__class__.__name__}/{dataset.train_ratio}/seed_{dataset.seed}"
        return TrainingLogger(log_dir=log_dir, log_name=log_name, save=save_log, **kwargs)

    # 功能：依次检查图编号、属性可用性和监督需求，供子类 train() 在计算前调用。
    def check_inputs(self, dataset: Dataset,
                     gids: Union[list[int], tuple[int, ...]],
                     plain_method: bool,
                     use_attr: bool,
                     pairwise: bool,
                     supervised: bool):
        if plain_method:
            assert not use_attr, f'{self.__class__.__name__} does not support attributed network alignment, set "use_attr=False"'

        if pairwise:
            self.check_pairwise_input_graphs(dataset, gids)
        else:
            self.check_input_graphs(dataset, gids)

        self.check_usage_of_attributes(dataset, gids, use_attr)
        self.check_supervision(dataset, supervised)

    # 功能：确认数据集类型、图编号范围和编号唯一性。
    @staticmethod
    def check_input_graphs(dataset: Dataset, gids: Union[list[int], tuple[int, ...]]):
        assert isinstance(dataset, Dataset) or isinstance(dataset, BaseData), 'Input dataset must be a PlanetAlign Dataset or BaseData object'
        assert all([0 <= gid < len(dataset.pyg_graphs) for gid in gids]), 'Invalid graph IDs'
        assert len(set(gids)) == len(gids), 'Graph IDs must be unique'

    # 功能：调用 check_input_graphs() 后确认恰好选择两张图。
    def check_pairwise_input_graphs(self, dataset: Dataset, gids: Union[list[int], tuple[int, ...]]):
        self.check_input_graphs(dataset, gids)
        assert len(gids) == 2, 'Exactly two graphs are required for pairwise alignment'

    # 功能：要求使用属性时检查是否存在；关闭已有属性时给出提示。
    def check_usage_of_attributes(self, dataset: Dataset, gids: Union[list[int], tuple[int, ...]], use_attr: bool):
        attributes_are_available = all([dataset.pyg_graphs[gid].x is not None for gid in gids])
        if not use_attr and attributes_are_available:
            warnings.warn(f'Attributes are available in {dataset.__class__.__name__} but not used by {self.__class__.__name__}')
        elif use_attr and not attributes_are_available:
            raise ValueError(f'Attributes are not available for "{dataset.__class__.__name__}" dataset, set use_attr=False')

    # 功能：监督算法要求至少一个训练锚点，否则在训练前报错。
    @staticmethod
    def check_supervision(dataset: Dataset, supervised: bool):
        if supervised and len(dataset.train_data) == 0:
            raise ValueError(f'Supervision is required but no training data is available in "{dataset.__class__.__name__}" dataset')
