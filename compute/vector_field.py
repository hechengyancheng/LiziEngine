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

    def _get_adjacent_positions(self, x: int, y: int, h: int, w: int, include_self: bool = True) -> List[Tuple[int, int]]:
        """
        获取相邻位置的坐标列表，包括中心点自身（可选）

        参数:
            x: 目标点的x坐标(列索引)
            y: 目标点的y坐标(行索引)
            h: 网格高度
            w: 网格宽度
            include_self: 是否包含中心点自身

        返回:
            List[Tuple[int, int]]: 相邻位置的坐标列表 [(x, y), ...]
        """
        positions = []

        # 上方相邻位置
        if y > 0:
            positions.append((x, y-1))

        # 下方相邻位置
        if y < h - 1:
            positions.append((x, y+1))

        # 左侧相邻位置
        if x > 0:
            positions.append((x-1, y))

        # 右侧相邻位置
        if x < w - 1:
            positions.append((x+1, y))

        # 中心点自身
        if include_self:
            positions.append((x, y))

        return positions

    def sum_adjacent_vectors(self, grid: np.ndarray, x: int, y: int, include_self: bool = None) -> Tuple[float, float]:
        """
        计算目标点及其相邻点的向量和

        该方法计算指定坐标(x, y)处的向量与其上下左右四个相邻向量的总和。
        所有操作都是边界安全的，不会因为越界访问而产生错误。

        参数:
            grid: 二维向量场数组，形状为(height, width, 2)，最后一维表示向量的x和y分量
            x: 目标点的x坐标(列索引)
            y: 目标点的y坐标(行索引)
            include_self: 是否包含目标点自身的向量，None表示从配置文件读取

        返回:
            Tuple[float, float]: 向量总和的(x, y)分量

        示例:
            >>> calculator = VectorFieldCalculator()
            >>> grid = np.zeros((10, 10, 2))
            >>> grid[5, 5] = [1.0, 0.0]  # 设置中心点向量
            >>> grid[4, 5] = [0.0, 1.0]  # 设置上方向量
            >>> sum_x, sum_y = calculator.sum_adjacent_vectors(grid, 5, 5, include_self=True)
            >>> print(f"向量和: ({sum_x}, {sum_y})")  # 输出: 向量和: (1.0, 1.0)
        """
        # 从配置获取是否包含自身向量的设置，如果未指定则默认为True
        if include_self is None:
            include_self = config_manager.get("vector_field.include_self", True)

        # 获取网格尺寸
        h, w = grid.shape[:2]
        sum_x = sum_y = 0.0

        # 获取所有相邻位置
        positions = self._get_adjacent_positions(x, y, h, w, include_self)

        # 计算所有位置的向量和
        for pos_x, pos_y in positions:
            sum_x += float(grid[pos_y, pos_x, 0])
            sum_y += float(grid[pos_y, pos_x, 1])

        return sum_x, sum_y

    def average_adjacent_vectors(self, grid: np.ndarray, x: int, y: int, include_self: bool = None) -> Tuple[float, float]:
        """
        计算目标点及其相邻点的向量平均值

        该方法计算指定坐标(x, y)处的向量与其上下左右四个相邻向量的平均值。
        所有操作都是边界安全的，不会因为越界访问而产生错误。

        参数:
            grid: 二维向量场数组，形状为(height, width, 2)，最后一维表示向量的x和y分量
            x: 目标点的x坐标(列索引)
            y: 目标点的y坐标(行索引)
            include_self: 是否包含目标点自身的向量，None表示从配置文件读取

        返回:
            Tuple[float, float]: 向量平均值的(x, y)分量

        示例:
            >>> calculator = VectorFieldCalculator()
            >>> grid = np.zeros((10, 10, 2))
            >>> grid[5, 5] = [2.0, 0.0]  # 设置中心点向量
            >>> grid[4, 5] = [0.0, 2.0]  # 设置上方向量
            >>> avg_x, avg_y = calculator.average_adjacent_vectors(grid, 5, 5, include_self=True)
            >>> print(f"向量平均值: ({avg_x}, {avg_y}")  # 输出: 向量平均值: (1.0, 1.0)
        """
        # 从配置获取是否包含自身向量的设置，如果未指定则默认为True
        if include_self is None:
            include_self = config_manager.get("vector_field.include_self", True)

        # 使用sum_adjacent_vectors方法计算向量和
        sum_x, sum_y = self.sum_adjacent_vectors(grid, x, y, include_self)

        # 获取网格尺寸
        h, w = grid.shape[:2]

        # 获取所有相邻位置
        positions = self._get_adjacent_positions(x, y, h, w, include_self)
        count = len(positions)

        # 计算平均值，避免除以零
        if count > 0:
            return sum_x / count, sum_y / count
        else:
            return 0.0, 0.0

    def apply_vector_field(self, grid: np.ndarray, x: int, y: int, magnitude: float = None, brush_size: int = None) -> None:
        """
        在指定位置应用向量场
        """
        if magnitude is None:
            magnitude = config_manager.get("vector_field.default_vector_length", 1.0)
        if brush_size is None:
            brush_size = config_manager.get("vector_field.default_brush_size", 20)

        h, w = grid.shape[:2]
        reverse = config_manager.get("vector_field.reverse_vector", False)

        # 确保在网格范围内
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))

        # 计算向量方向和强度
        avg_x, avg_y = self.average_adjacent_vectors(grid, x, y)
        avg_magnitude = np.sqrt(avg_x**2 + avg_y**2)

        if avg_magnitude > 0:
            # 归一化平均向量
            norm_x = avg_x / avg_magnitude
            norm_y = avg_y / avg_magnitude

            # 根据配置决定是否反转向量
            if reverse:
                norm_x = -norm_x
                norm_y = -norm_y

            # 应用向量场
            for dy in range(-brush_size, brush_size + 1):
                for dx in range(-brush_size, brush_size + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        # 计算距离权重
                        dist = np.sqrt(dx**2 + dy**2)
                        if dist <= brush_size:
                            weight = 1.0 - (dist / brush_size)
                            grid[ny, nx, 0] = float(grid[ny, nx, 0]) * (1 - weight) + norm_x * magnitude * weight
                            grid[ny, nx, 1] = float(grid[ny, nx, 1]) * (1 - weight) + norm_y * magnitude * weight

    def create_tangential_pattern(self, grid: np.ndarray, magnitude: float = 0.2, radius_ratio: float = 0.3) -> None:
        """
        在网格上创建切线向量模式（围绕中心点的旋转模式）
        
        参数:
            grid: 向量网格
            magnitude: 向量强度（默认值减小为0.2）
            radius_ratio: 切线模式的半径比例，相对于网格最小边长的比例（默认0.3）
        """
        if grid is None or not isinstance(grid, np.ndarray):
            return
            
        h, w = grid.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        # 计算切线模式的实际半径
        min_dimension = min(h, w)
        pattern_radius = min_dimension * radius_ratio
        
        for y in range(h):
            for x in range(w):
                # 计算从中心到当前点的向量
                dx = x - center_x
                dy = y - center_y
                dist = np.sqrt(dx**2 + dy**2)
                
                # 只在指定半径内创建切线向量
                if 0 < dist <= pattern_radius:
                    # 计算切线方向（垂直于径向）
                    # 切线方向可以通过交换x和y并取反一个分量得到
                    
                    # 根据距离调整向量强度，距离中心越远，向量越小
                    distance_factor = 1.0 - (dist / pattern_radius) * 0.5  # 距离中心最远处的向量强度为50%
                    adjusted_magnitude = magnitude * distance_factor
                    
                    tangent_x = -dy / dist * adjusted_magnitude
                    tangent_y = dx / dist * adjusted_magnitude
                    
                    # 设置向量
                    grid[y, x, 0] = tangent_x
                    grid[y, x, 1] = tangent_y
                else:
                    # 中心点和半径外的点没有切线方向，设为0
                    grid[y, x, 0] = 0
                    grid[y, x, 1] = 0

    def correct_vector_centers(self, grid: np.ndarray, threshold: float = 0.5, min_distance: int = 10) -> List[Tuple[int, int]]:
        """
        获取已记录的向量场中心点，并在每帧不断修正其位置

        优化版本: 使用向量化操作减少嵌套循环，提高性能

        参数:
            grid: 向量网格
            threshold: 用于判断是否更新中心点的向量强度阈值
            min_distance: 中心点之间的最小距离，避免重复识别

        返回:
            修正后的中心点坐标列表 [(x, y), ...] - 支持多个中心点
        """
        if grid is None or not isinstance(grid, np.ndarray):
            return []

        # 获取已记录的向量场中心点列表
        centers = state_manager.get("vector_field_centers", [])

        # 如果没有记录的中心点，返回空列表
        if not centers:
            return []

        # 确保在网格范围内
        h, w = grid.shape[:2]
        valid_centers = []
        updated_centers = []

        # 获取自动修正开关
        auto_correct = config_manager.get("vector_field.auto_correct_centers", True)
        search_radius = config_manager.get("vector_field.center_search_radius", 10)
        max_magnitude = config_manager.get("vector_field.center_max_magnitude", 5.0)

        for center in centers:
            if len(center) >= 2:  # 确保中心点有x,y坐标
                center_x, center_y = center[0], center[1]

                # 确保在网格范围内
                if 0 <= center_x < w and 0 <= center_y < h:
                    # 如果启用自动修正，则根据向量场调整中心点位置
                    if auto_correct:
                        # 确保搜索范围不超出网格边界
                        min_x = max(0, center_x - search_radius)
                        max_x = min(w - 1, center_x + search_radius)
                        min_y = max(0, center_y - search_radius)
                        max_y = min(h - 1, center_y + search_radius)

                        # 提取搜索区域
                        search_region = grid[min_y:max_y+1, min_x:max_x+1]

                        # 计算向量幅度
                        vec_x = search_region[:, :, 0]
                        vec_y = search_region[:, :, 1]
                        magnitudes = np.sqrt(vec_x**2 + vec_y**2)

                        # 限制向量幅度
                        magnitudes = np.minimum(magnitudes, max_magnitude)

                        # 创建坐标网格
                        y_indices, x_indices = np.mgrid[min_y:max_y+1, min_x:max_x+1]

                        # 计算距离权重 - 使用向量化操作
                        distances = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)
                        sigma = search_radius / 2
                        distance_weights = np.exp(-(distances**2) / (2 * sigma**2))

                        # 计算总权重
                        weights = magnitudes * distance_weights
                        total_weight = np.sum(weights)

                        # 计算加权平均位置
                        if total_weight > 0:
                            weighted_x = np.sum(x_indices * weights) / total_weight
                            weighted_y = np.sum(y_indices * weights) / total_weight

                            best_x = int(round(weighted_x))
                            best_y = int(round(weighted_y))

                            # 确保结果在搜索范围内
                            best_x = max(min_x, min(max_x, best_x))
                            best_y = max(min_y, min(max_y, best_y))

                            # 计算位置变化
                            position_change = np.sqrt((best_x - center_x)**2 + (best_y - center_y)**2)

                            # 如果位置变化足够大，则更新中心点
                            if (best_x != center_x or best_y != center_y) and position_change > 0.5:
                                center_x, center_y = best_x, best_y
                                updated_centers.append([center_x, center_y])

                    # 添加到有效中心点列表
                    valid_centers.append((center_x, center_y))

        # 如果有中心点位置被更新，则保存更新后的位置
        if updated_centers:
            state_manager.set("vector_field_centers", valid_centers)

        return valid_centers

# 创建全局向量场计算器实例
vector_calculator = VectorFieldCalculator()

def correct_vector_centers(grid: np.ndarray, threshold: float = 0.5, min_distance: int = 10) -> List[Tuple[int, int]]:
    """便捷函数：修正向量中心点位置"""
    return vector_calculator.correct_vector_centers(grid, threshold, min_distance)
