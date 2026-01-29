"""
CPU向量场计算模块 - 提供基于CPU的向量场计算功能
"""
import numpy as np
from typing import Tuple, Union, List, Optional, Any
from ..core.state import state_manager
from ..core.events import Event, EventType, event_bus

class CPUVectorFieldCalculator:
    """CPU向量场计算器"""
    def __init__(self):
        self._event_bus = event_bus
        self._state_manager = state_manager

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
        neighbor_weight = self._state_manager.get("vector_neighbor_weight", 0.1)
        self_weight = self._state_manager.get("vector_self_weight", 1.0)

        # 使用向量化操作计算邻居向量之和
        # 创建填充数组来处理边界条件
        padded_grid = np.pad(grid, ((1, 1), (1, 1), (0, 0)), mode='edge')

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

    def create_tiny_vector(self, grid: np.ndarray, x: float, y: float, mag: float = 1.0, radius: int = 1) -> None:
        """在指定位置创建一个微小的向量场影响,填充半径范围内的圆形区域"""
        if not hasattr(grid, "ndim"):
            return

        h, w = grid.shape[0], grid.shape[1]

        # 确保坐标在有效范围内
        x = max(0.0, min(w - 1.0, float(x)))
        y = max(0.0, min(h - 1.0, float(y)))

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                dist_sq = dx**2 + dy**2
                if dist_sq <= radius**2:
                    px = x + dx
                    py = y + dy
                    if dist_sq > 0:
                        dist = dist_sq ** 0.5
                        vx = (dx / dist) * mag
                        vy = (dy / dist) * mag
                    else:
                        vx = 0.0
                        vy = 0.0
                    self.add_vector_at_position(grid, px, py, vx, vy)

    def create_tiny_vectors_batch(self, grid: np.ndarray, positions: List[Tuple[float, float, float]], radius: int = 1) -> None:
        """批量创建微小向量影响，用于优化性能

        Args:
            grid: 向量场网格
            positions: 位置列表，每个元素为 (x, y, mag) 元组
            radius: 影响半径
        """
        if not hasattr(grid, "ndim") or not positions:
            return

        for x, y, mag in positions:
            self.create_tiny_vector(grid, x, y, mag, radius)

    def add_vector_at_position(self, grid: np.ndarray, x: float, y: float, vx: float, vy: float) -> None:
        """在浮点坐标处添加向量，使用双线性插值的逆方法，将向量分布到四个最近的整数坐标"""
        if not hasattr(grid, "ndim") or grid.ndim < 3 or grid.shape[2] < 2:
            return

        h, w = grid.shape[0], grid.shape[1]

        # 确保坐标在有效范围内
        x = max(0.0, min(w - 1.0, float(x)))
        y = max(0.0, min(h - 1.0, float(y)))

        # 计算四个最近的整数坐标
        x0 = int(np.floor(x))
        x1 = min(x0 + 1, w - 1)
        y0 = int(np.floor(y))
        y1 = min(y0 + 1, h - 1)

        # 计算插值权重
        wx = x - x0
        wy = y - y0

        # 双线性插值的逆：将向量按权重分布到四个角
        w00 = (1 - wx) * (1 - wy)
        w01 = wx * (1 - wy)
        w10 = (1 - wx) * wy
        w11 = wx * wy

        try:
            grid[y0, x0, 0] += w00 * vx
            grid[y0, x0, 1] += w00 * vy
            grid[y0, x1, 0] += w01 * vx
            grid[y0, x1, 1] += w01 * vy
            grid[y1, x0, 0] += w10 * vx
            grid[y1, x0, 1] += w10 * vy
            grid[y1, x1, 0] += w11 * vx
            grid[y1, x1, 1] += w11 * vy
        except Exception:
            pass

    def fit_vector_at_position(self, grid: np.ndarray, x: float, y: float, radius: float = 1.0) -> Tuple[float, float]:
        """在浮点坐标处拟合向量值，根据半径的加权平均"""
        if not hasattr(grid, "ndim") or grid.ndim < 3 or grid.shape[2] < 2:
            return (0.0, 0.0)

        h, w = grid.shape[0], grid.shape[1]

        # 确保坐标在有效范围内
        x = max(0.0, min(w - 1.0, float(x)))
        y = max(0.0, min(h - 1.0, float(y)))

        x_min = max(0, int(x - radius))
        x_max = min(w - 1, int(x + radius))
        y_min = max(0, int(y - radius))
        y_max = min(h - 1, int(y + radius))
        sum_vx = 0.0
        sum_vy = 0.0
        sum_weight = 0.0
        for ix in range(x_min, x_max + 1):
            for iy in range(y_min, y_max + 1):
                dist = ((ix - x) ** 2 + (iy - y) ** 2) ** 0.5
                if dist <= radius:
                    weight = 1.0 / (dist + 1.0)
                    sum_vx += grid[iy, ix, 0] * weight
                    sum_vy += grid[iy, ix, 1] * weight
                    sum_weight += weight
        if sum_weight > 0:
            return (sum_vx / sum_weight, sum_vy / sum_weight)
        else:
            return (0.0, 0.0)

    def fit_vectors_at_positions_batch(self, grid: np.ndarray, positions: List[Tuple[float, float]], radius: float = 1.0) -> List[Tuple[float, float]]:
        """批量拟合多个位置的向量值，使用CPU并行处理

        Args:
            grid: 向量场网格
            positions: 位置列表，每个元素为 (x, y) 元组
            radius: 拟合半径

        Returns:
            向量列表，每个元素为 (vx, vy) 元组
        """
        if not hasattr(grid, "ndim") or grid.ndim < 3 or grid.shape[2] < 2 or not positions:
            return [(0.0, 0.0)] * len(positions)

        # 使用列表推导式批量处理
        return [self.fit_vector_at_position(grid, x, y, radius) for x, y in positions]
