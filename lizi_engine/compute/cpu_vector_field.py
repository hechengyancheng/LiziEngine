"""
CPU向量场计算模块 - 提供基于CPU的向量场计算功能
"""
import numpy as np
from typing import Tuple, Union, List, Optional, Any
from ..core.config import config_manager
from ..core.events import Event, EventType, event_bus

class CPUVectorFieldCalculator:
    """CPU向量场计算器"""
    def __init__(self):
        self._event_bus = event_bus
        self._config_manager = config_manager

    def sum_adjacent_vectors(self, grid: np.ndarray, x: int, y: int,
                           self_weight: float = 1.0, neighbor_weight: float = 0.1) -> Tuple[float, float]:
        """
        读取目标 (x,y) 的上下左右四个相邻格子的向量并相加（越界安全）。
        返回 (sum_x, sum_y) 的 tuple。
        """
        if grid is None:
            return (0.0, 0.0)

        h, w = grid.shape[:2]
        sum_x = 0.0
        sum_y = 0.0

        if 0 <= x < w and 0 <= y < h:
            vx, vy = grid[y, x]
            sum_x += vx * self_weight
            sum_y += vy * self_weight

        neighbors = ((0, -1), (0, 1), (-1, 0), (1, 0))
        for dx, dy in neighbors:
            nx = x + dx
            ny = y + dy
            if 0 <= nx < w and 0 <= ny < h:
                vx, vy = grid[ny, nx]
                sum_x += vx * neighbor_weight
                sum_y += vy * neighbor_weight

        return (sum_x, sum_y)

    def update_grid_with_adjacent_sum(self, grid: np.ndarray) -> np.ndarray:
        """
        使用NumPy的向量化操作高效计算相邻向量之和，替换原有的双重循环实现。
        返回修改后的 grid。
        """
        if grid is None or not isinstance(grid, np.ndarray):
            return grid

        h, w = grid.shape[:2]

        # 获取配置参数
        neighbor_weight = self._config_manager.get("vector_neighbor_weight", 0.1)
        self_weight = self._config_manager.get("vector_self_weight", 1.0)

        # 使用向量化操作计算邻居向量之和
        # 创建填充数组来处理边界条件
        padded_grid = np.pad(grid, ((1, 1), (1, 1), (0, 0)), mode='constant')

        # 计算四个方向的邻居贡献
        up_neighbors = padded_grid[2:, 1:-1] * neighbor_weight
        down_neighbors = padded_grid[:-2, 1:-1] * neighbor_weight
        left_neighbors = padded_grid[1:-1, 2:] * neighbor_weight
        right_neighbors = padded_grid[1:-1, :-2] * neighbor_weight

        # 求和邻居贡献
        result = up_neighbors + down_neighbors + left_neighbors + right_neighbors

        # 总是包含自身贡献
        result += grid * self_weight

        # 将结果复制回原网格
        grid[:] = result
        return grid

    def create_vector_grid(self, width: int = 640, height: int = 480, default: Tuple[float, float] = (0, 0)) -> np.ndarray:
        """创建一个 height x width 的二维向量网格"""
        grid = np.zeros((height, width, 2), dtype=np.float32)
        if default != (0, 0):
            grid[:, :, 0] = default[0]
            grid[:, :, 1] = default[1]
        return grid

    def create_radial_pattern(self, grid: np.ndarray, center: Tuple[float, float] = None,
                            radius: float = None, magnitude: float = 1.0) -> np.ndarray:
        """在网格上创建径向向量模式"""
        if grid is None or not isinstance(grid, np.ndarray):
            raise TypeError("grid 必须是 numpy.ndarray 类型")

        h, w = grid.shape[:2]

        # 如果未指定中心，则使用网格中心
        if center is None:
            center = (w // 2, h // 2)

        # 如果未指定半径，则使用网格尺寸的1/4
        if radius is None:
            radius = min(w, h) // 4

        cx, cy = center

        # 创建坐标网格
        y_coords, x_coords = np.mgrid[0:h, 0:w]

        # 计算每个点到中心的距离和方向
        dx = x_coords - cx
        dy = y_coords - cy
        dist = np.sqrt(dx**2 + dy**2)

        # 创建掩码：只处理在半径内且不在中心的点
        mask = (dist < radius) & (dist > 0)

        # 计算径向角度
        angle = np.arctan2(dy, dx)

        # 计算向量大小（从中心向外递减）
        vec_magnitude = magnitude * (1.0 - (dist / radius))

        # 计算向量分量
        vx = vec_magnitude * np.cos(angle)
        vy = vec_magnitude * np.sin(angle)

        # 应用到网格
        grid[mask, 0] += vx[mask]
        grid[mask, 1] += vy[mask]

        return grid

    def create_tangential_pattern(self, grid: np.ndarray, center: Tuple[float, float] = None,
                               radius: float = None, magnitude: float = 1.0) -> np.ndarray:
        """在网格上创建切线向量模式（旋转）"""
        if grid is None or not isinstance(grid, np.ndarray):
            raise TypeError("grid 必须是 numpy.ndarray 类型")

        h, w = grid.shape[:2]

        # 如果未指定中心，则使用网格中心
        if center is None:
            center = (w // 2, h // 2)

        # 如果未指定半径，则使用网格尺寸的1/4
        if radius is None:
            radius = min(w, h) // 4

        cx, cy = center

        # 创建坐标网格
        y_coords, x_coords = np.mgrid[0:h, 0:w]

        # 计算每个点到中心的距离和方向
        dx = x_coords - cx
        dy = y_coords - cy
        dist = np.sqrt(dx**2 + dy**2)

        # 创建掩码：只处理在半径内且不在中心的点
        mask = (dist < radius) & (dist > 0)

        # 计算切线角度（径向角度+90度）
        angle = np.arctan2(dy, dx) + np.pi/2

        # 计算向量大小（从中心向外递减）
        vec_magnitude = magnitude * (1.0 - (dist / radius))

        # 计算向量分量
        vx = vec_magnitude * np.cos(angle)
        vy = vec_magnitude * np.sin(angle)

        # 应用到网格
        grid[mask, 0] += vx[mask]
        grid[mask, 1] += vy[mask]

        return grid
