"""
工具箱模块 - 提供向量场工具函数
"""
import numpy as np

def add_inward_edge_vectors(grid: np.ndarray, magnitude: float = 1.0) -> None:
    """
    给网格边缘添加一圈向内的向量

    Args:
        grid: 向量网格，形状为 (height, width, 2)
        magnitude: 向量大小
    """
    height, width = grid.shape[:2]

    # 上边缘：向下向量
    grid[0, :, 1] = magnitude

    # 下边缘：向上向量
    grid[height-1, :, 1] = -magnitude

    # 左边缘：向右向量
    grid[:, 0, 0] = magnitude

    # 右边缘：向左向量
    grid[:, width-1, 0] = -magnitude

    # 四个角点：向内向量（对角线方向）
    # 左上角：向右下
    grid[0, 0] = [magnitude, magnitude]
    # 右上角：向左下
    grid[0, width-1] = [-magnitude, magnitude]
    # 左下角：向右上
    grid[height-1, 0] = [magnitude, -magnitude]
    # 右下角：向左上
    grid[height-1, width-1] = [-magnitude, -magnitude]
