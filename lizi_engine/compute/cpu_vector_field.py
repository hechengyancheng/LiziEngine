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

        # 创建掩码：只处理在半径内的点
        mask = (dist <= radius)

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

        # 创建掩码：只处理在半径内的点
        mask = (dist <= radius)

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

    def create_tiny_vector(self, grid: np.ndarray, x: float, y: float, mag: float = 1.0) -> None:
        """在指定位置创建一个微小的向量场影响,只影响位置本身及上下左右四个邻居"""
        if not hasattr(grid, "ndim"):
            return

        h, w = grid.shape[0], grid.shape[1]

        # 确保坐标在有效范围内
        x = max(0.0, min(w - 1.0, float(x)))
        y = max(0.0, min(h - 1.0, float(y)))

        # 只影响当前位置及其上下左右邻居，使用浮点坐标
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if abs(dx) + abs(dy) == 1:  # 上下左右邻居
                    self.add_vector_at_position(grid, x + dx, y + dy, dx * mag, dy * mag)

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

    def fit_vector_at_position(self, grid: np.ndarray, x: float, y: float) -> Tuple[float, float]:
        """在浮点坐标处拟合向量值，使用双线性插值"""
        if not hasattr(grid, "ndim") or grid.ndim < 3 or grid.shape[2] < 2:
            return (0.0, 0.0)

        h, w = grid.shape[0], grid.shape[1]

        # 确保坐标在有效范围内
        x = max(0.0, min(w - 1.0, float(x)))
        y = max(0.0, min(h - 1.0, float(y)))

        # 计算四个最近的整数坐标
        x0 = int(np.floor(x))
        x1 = min(x0 + 1, w - 1)
        y0 = int(np.floor(y))
        y1 = min(y0 + 1, h - 1)

        # 获取四个角的向量值
        v00 = (grid[y0, x0, 0], grid[y0, x0, 1])
        v01 = (grid[y0, x1, 0], grid[y0, x1, 1])
        v10 = (grid[y1, x0, 0], grid[y1, x0, 1])
        v11 = (grid[y1, x1, 0], grid[y1, x1, 1])

        # 计算插值权重
        wx = x - x0
        wy = y - y0

        # 双线性插值
        vx = (1 - wx) * (1 - wy) * v00[0] + wx * (1 - wy) * v01[0] + (1 - wx) * wy * v10[0] + wx * wy * v11[0]
        vy = (1 - wx) * (1 - wy) * v00[1] + wx * (1 - wy) * v01[1] + (1 - wx) * wy * v10[1] + wx * wy * v11[1]

        return (vx, vy)
    
    def fit_vector_at_position_fp32(self, grid: np.ndarray, x: float, y: float) -> Tuple[float, float]:
        """在浮点坐标处拟合向量值，使用双线性插值（单精度浮点版本）"""
        if not hasattr(grid, "ndim") or grid.ndim < 3 or grid.shape[2] < 2:
            return (0.0, 0.0)

        h, w = grid.shape[0], grid.shape[1]

        # 确保坐标在有效范围内，使用单精度
        x_fp32 = np.float32(max(0.0, min(w - 1.0, float(x))))
        y_fp32 = np.float32(max(0.0, min(h - 1.0, float(y))))

        # 计算四个最近的整数坐标
        x0 = int(np.floor(x_fp32))
        x1 = min(x0 + 1, w - 1)
        y0 = int(np.floor(y_fp32))
        y1 = min(y0 + 1, h - 1)

        # 获取四个角的向量值
        v00 = (np.float32(grid[y0, x0, 0]), np.float32(grid[y0, x0, 1]))
        v01 = (np.float32(grid[y0, x1, 0]), np.float32(grid[y0, x1, 1]))
        v10 = (np.float32(grid[y1, x0, 0]), np.float32(grid[y1, x0, 1]))
        v11 = (np.float32(grid[y1, x1, 0]), np.float32(grid[y1, x1, 1]))

        # 计算插值权重，使用单精度
        wx = np.float32(x_fp32 - x0)
        wy = np.float32(y_fp32 - y0)

        # 双线性插值，使用单精度
        vx = np.float32((1 - wx) * (1 - wy) * v00[0] + wx * (1 - wy) * v01[0] + (1 - wx) * wy * v10[0] + wx * wy * v11[0])
        vy = np.float32((1 - wx) * (1 - wy) * v00[1] + wx * (1 - wy) * v01[1] + (1 - wx) * wy * v10[1] + wx * wy * v11[1])

        return (float(vx), float(vy))
