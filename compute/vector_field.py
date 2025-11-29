"""
向量场计算模块 - 提供向量场计算的核心功能
"""
import numpy as np
from typing import Tuple, Union, List
from core.config import config_manager
from core.events import EventBus, Event, EventType
from core.state import state_manager

class VectorFieldCalculator:
    """向量场计算器"""
    def __init__(self):
        self._event_bus = EventBus()

    def sum_adjacent_vectors(self, grid: np.ndarray, x: int, y: int, include_self: bool = None) -> Tuple[float, float]:
        """
        读取目标 (x,y) 的上下左右四个相邻格子的向量并相加（越界安全）。
        返回 (sum_x, sum_y) 的 tuple。
        """
        # 从配置管理器读取权重参数和平均值开关
        self_weight = config_manager.get("vector_self_weight", 1.0)
        neighbor_weight = config_manager.get("vector_neighbor_weight", 0.1)
        enable_average = config_manager.get("enable_vector_average", False)

        # 如果未指定include_self，则使用配置管理器中的默认值
        if include_self is None:
            include_self = config_manager.get("include_self", False)

        if grid is None:
            return (0.0, 0.0)

        if not isinstance(grid, np.ndarray):
            raise TypeError("grid 必须是 numpy.ndarray 类型")

        h, w = grid.shape[:2]
        sum_x = 0.0
        sum_y = 0.0
        count = 0

        if include_self and 0 <= x < w and 0 <= y < h:
            vx, vy = float(grid[y, x, 0]), float(grid[y, x, 1])
            sum_x += vx * self_weight  # 使用配置管理器中的自身权重
            sum_y += vy * self_weight
            count += 1

        neighbors = ((0, -1), (0, 1), (-1, 0), (1, 0))
        for dx, dy in neighbors:
            nx = x + dx
            ny = y + dy
            if 0 <= nx < w and 0 <= ny < h:
                vx, vy = float(grid[ny, nx, 0]), float(grid[ny, nx, 1])
                sum_x += vx * neighbor_weight  # 使用配置管理器中的邻居权重
                sum_y += vy * neighbor_weight
                count += 1

        # 如果启用平均值功能，则除以有效向量数量
        if enable_average and count > 0:
            sum_x /= count
            sum_y /= count

        return (sum_x, sum_y)

    def update_grid_with_adjacent_sum(self, grid: np.ndarray, include_self: bool = None) -> np.ndarray:
        """
        使用NumPy的向量化操作高效计算相邻向量之和，替换原有的双重循环实现。
        返回修改后的 grid。
        """
        # 如果未指定include_self，则使用配置管理器中的默认值
        if include_self is None:
            include_self = config_manager.get("include_self", False)

        if grid is None or not isinstance(grid, np.ndarray):
            return grid

        h, w = grid.shape[:2]

        # 获取邻居权重
        neighbor_weight = config_manager.get("vector_neighbor_weight", 0.1)
        self_weight = config_manager.get("vector_self_weight", 1.0)
        enable_average = config_manager.get("enable_vector_average", False)
        enable_normalization = config_manager.get("enable_vector_normalization", False)

        # 预分配结果数组以提高性能
        result = np.zeros_like(grid)

        # 使用向量化操作计算邻居向量之和
        # 创建填充数组来处理边界条件
        padded_grid = np.pad(grid, ((1, 1), (1, 1), (0, 0)), mode='constant')

        # 计算四个方向的邻居贡献
        # 上邻居 (y+1, x)
        up_neighbors = padded_grid[2:, 1:-1] * neighbor_weight
        # 下邻居 (y, x)
        down_neighbors = padded_grid[:-2, 1:-1] * neighbor_weight
        # 左邻居 (y, x+1)
        left_neighbors = padded_grid[1:-1, 2:] * neighbor_weight
        # 右邻居 (y, x)
        right_neighbors = padded_grid[1:-1, :-2] * neighbor_weight

        # 求和邻居贡献
        result = up_neighbors + down_neighbors + left_neighbors + right_neighbors

        # 如果包含自身，添加自身贡献
        if include_self:
            result += grid * self_weight

        # 如果需要平均值，计算有效邻居数量并归一化
        if enable_average:
            # 创建有效邻居计数矩阵
            neighbor_count = np.ones((h, w), dtype=np.float32) * 4  # 默认4个邻居

            # 边界处理
            neighbor_count[0, :] -= 1   # 上边界
            neighbor_count[-1, :] -= 1  # 下边界
            neighbor_count[:, 0] -= 1   # 左边界
            neighbor_count[:, -1] -= 1  # 右边界

            # 如果包含自身，邻居数加1
            if include_self:
                neighbor_count += 1

            # 归一化
            #result = result / neighbor_count[:, :, np.newaxis]

        # 如果需要权重归一化
        elif enable_normalization:
            # 计算每个点的有效权重总和
            weight_sum = np.ones((h, w), dtype=np.float32) * 4 * neighbor_weight  # 默认4个邻居

            # 边界处理
            weight_sum[0, :] -= neighbor_weight   # 上边界
            weight_sum[-1, :] -= neighbor_weight  # 下边界
            weight_sum[:, 0] -= neighbor_weight   # 左边界
            weight_sum[:, -1] -= neighbor_weight  # 右边界

            # 如果包含自身，添加自身权重
            if include_self:
                weight_sum += self_weight

            # 确保权重总和大于0，避免除以0
            weight_sum = np.maximum(weight_sum, 0.1)

            # 归一化
            result = result / weight_sum[:, :, np.newaxis]

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
        grid[mask, 0] = vx[mask]
        grid[mask, 1] = vy[mask]

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
        grid[mask, 0] = vx[mask]
        grid[mask, 1] = vy[mask]

        return grid

# 全局向量场计算器实例
vector_calculator = VectorFieldCalculator()

# 便捷函数
def sum_adjacent_vectors(grid: np.ndarray, x: int, y: int, include_self: bool = None) -> Tuple[float, float]:
    """便捷函数：计算相邻向量之和"""
    return vector_calculator.sum_adjacent_vectors(grid, x, y, include_self)

def update_grid_with_adjacent_sum(grid: np.ndarray, include_self: bool = None) -> np.ndarray:
    """便捷函数：更新整个网格"""
    return vector_calculator.update_grid_with_adjacent_sum(grid, include_self)

def create_vector_grid(width: int = 640, height: int = 480, default: Tuple[float, float] = (0, 0)) -> np.ndarray:
    """便捷函数：创建向量网格"""
    return vector_calculator.create_vector_grid(width, height, default)
