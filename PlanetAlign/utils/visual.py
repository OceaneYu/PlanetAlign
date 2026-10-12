# 文件作用：将节点匹配矩阵画成热力图或高分位置散点图，便于人工观察。
# 数据来源：调用者直接传入算法生成的 torch.Tensor 或 NumPy 矩阵。
# 结果去向：按参数显示图像或保存到指定路径；这些函数不参与训练与解码。

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Optional


# 功能：将完整匹配矩阵绘成热力图，可选按行或列归一化。
def plot_alignment_matrix(
        P: Union[torch.Tensor, np.ndarray],
        title: str = "Alignment Matrix",
        x_label: str = "Nodes in Graph 2",
        y_label: str = "Nodes in Graph 1",
        cmap: str = "viridis",
        figsize: tuple[int, int] = (8, 6),
        vmax: Optional[Union[int, float]] = None,
        vmin: Optional[Union[int, float]] = None,
        save_path: Optional[str] = None,
        show: bool = True,
):
    """
    Visualize an alignment matrix as a heatmap.

    Parameters
    ----------
    P : Union[torch.Tensor, np.ndarray]
        The alignment matrix to visualize.
    title : str, optional
        Title of the plot. Default is "Alignment Matrix".
    x_label : str, optional
        Label for the x-axis. Default is "Nodes in Graph 2".
    y_label : str, optional
        Label for the y-axis. Default is "Nodes in Graph 1".
    cmap : str, optional
        Colormap to use for the heatmap. Default is "viridis".
    figsize : tuple[int, int], optional
        Size of the figure. Default is (8, 6).
    vmax : Optional[Union[int, float]], optional
        Maximum value for colormap scaling. Default is None.
    vmin : Optional[Union[int, float]], optional
        Minimum value for colormap scaling. Default is None.
    save_path : Union[str, None], optional
        Path to save the plot image. Default is None (not saving).
    show : bool, optional
        Whether to display the plot. Default is True.
    """

    if not isinstance(P, np.ndarray):
        P = P.detach().cpu().numpy()
    P = P / np.sum(P)

    plt.figure(figsize=figsize)

    # Plot matrix with seaborn heatmap
    sns.heatmap(
        P,
        cmap=cmap,
        vmax=vmax,
        vmin=vmin,
        cbar=True,
        square=True,
        xticklabels=False,
        yticklabels=False,
    )

    plt.title(title, fontsize=14)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    plt.close()


# 功能：找出矩阵中最高的 k 个分数，仅绘制这些节点对的位置与强度。
def plot_alignment_matrix_topk(
    P: Union[torch.Tensor, np.ndarray],
    k: int = 1000,
    title: str = "Top-k Alignment Scores",
    figsize: tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
):
    """
    Visualizes an alignment matrix as a scatter plot of the top-k largest entries.

    Parameters
    ----------
    P : Union[torch.Tensor, np.ndarray]
        The alignment matrix to visualize.
    k : int, optional
        Number of top entries to plot. Default is 1000.
    title : str, optional
        Title of the plot. Default is "Top-k Alignment Scores".
    figsize : tuple[int, int], optional
        Size of the figure. Default is (8, 6).
    save_path : Optional[str], optional
        Path to save the plot image. Default is None (not saving).
    """
    if not isinstance(P, np.ndarray):
        P = P.detach().cpu().numpy()
    P = P / np.sum(P)

    n1, n2 = P.shape
    assert k <= n1 * n2, "k cannot be larger than the total number of entries in P."

    P_flat = P.ravel()
    idx = np.argpartition(P_flat, -k)[-k:]     # indices of top-k entries
    rows, cols = np.unravel_index(idx, (n1, n2))
    values = P_flat[idx]

    plt.figure(figsize=figsize)
    plt.scatter(cols, rows, c=values, s=5, cmap="viridis")
    plt.xlabel("Graph 2 nodes")
    plt.ylabel("Graph 1 nodes")
    plt.title(title)
    plt.colorbar(label="Alignment score")
    plt.gca().invert_yaxis()  # to match matrix orientation

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close()
