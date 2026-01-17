
"""
标记系统插件：管理向量场中的标记点
提供标记点的创建、更新和渲染功能。
"""
from typing import List, Dict, Any, Tuple
import numpy as np
from lizi_engine.compute.vector_field import vector_calculator

class MarkerSystem:
    """标记系统，用于管理向量场中的标记点"""

    def __init__(self, app_core):
        self.app_core = app_core
        self.vector_calculator = vector_calculator
        # 标记列表，存储浮点网格坐标 {'x':float,'y':float,'mag':float,'vx':float,'vy':float}
        self.markers = []

    def add_marker(self, x: float, y: float, mag: float = 1.0, vx: float = 0.0, vy: float = 0.0) -> None:
        """添加一个新标记

        Args:
            x: 标记的x坐标（浮点）
            y: 标记的y坐标（浮点）
            mag: 标记的初始幅值（可选）
            vx: 标记的x方向速度（可选）
            vy: 标记的y方向速度（可选）
        """
        marker = {"x": float(x), "y": float(y), "mag": float(mag), "vx": float(vx), "vy": float(vy)}
        self.markers.append(marker)
        self._sync_to_state_manager()

    def clear_markers(self) -> None:
        """清除所有标记"""
        self.markers = []
        self._sync_to_state_manager()

    def get_markers(self) -> List[Dict[str, float]]:
        """获取所有标记

        Returns:
            标记列表
        """
        return list(self.markers)

    def update_markers(self, grid: np.ndarray, dt: float = 1.0, clear_threshold: float = 1e-3) -> None:
        """根据浮点坐标处拟合向量移动标记。

        算法：在标记的浮点坐标处使用双线性插值拟合向量值，将标记按 fitted_v * dt 偏移。

        Args:
            grid: 向量场网格
            dt: 时间步长
            clear_threshold: 清除阈值，低于此拟合向量幅值的标记将被清除
        """
        if not hasattr(grid, "ndim"):
            return

        # 优先从全局状态同步标记（如果其他模块在放置向量场时添加了标记）
        try:
            stored = self.app_core.state_manager.get("markers", None)
            if stored is not None:
                self.markers = list(stored)
        except Exception:
            pass

        # 检查网格维度是否有效
        if grid.ndim < 3 or grid.shape[2] < 2:
            return

        h, w = grid.shape[0], grid.shape[1]
        cell_size = self.app_core.config_manager.get("cell_size", 1.0)

        # 期望 grid 最后一维至少 2，代表 vx, vy
        new_markers = []

        for m in self.markers:
            x = m["x"]
            y = m["y"]
            mag = m["mag"]
            vx = m["vx"]
            vy = m["vy"]
            try:
                # 在浮点坐标处拟合向量值
                fitted_vx, fitted_vy = self.fit_vector_at_position(grid, x, y)

                # 设置标记的速度属性
                if fitted_vx ** 2 + fitted_vy ** 2 > 0.001 ** 2:
                    vx += fitted_vx * 1/mag
                    vy += fitted_vy * 1/mag

                # 限制速度不超过单元格大小
                if (vx ** 2 + vy ** 2) ** 0.5 > cell_size:  # 限制速度不超过单元格大小
                    vx = vx / (vx ** 2 + vy ** 2) ** 0.5 * cell_size
                    vy = vy / (vx ** 2 + vy ** 2) ** 0.5 * cell_size

                # 使用速度更新浮点位置（带反弹后的速度）
                new_x = max(0.0, min(w - 1.0, x + vx * dt))
                new_y = max(0.0, min(h - 1.0, y + vy * dt))

                # 创建微小向量影响
                self.create_tiny_vector(grid, new_x, new_y, mag)

                m["x"] = new_x
                m["y"] = new_y
                m["vx"] = vx
                m["vy"] = vy
                new_markers.append(m)

            except Exception as e:
                # 添加更详细的错误日志
                print(f"Error updating marker at ({x}, {y}): {str(e)}")
                # 保留标记以便后续检查
                new_markers.append(m)
                continue

        # 更新内部标记列表并写回 state_manager 以便界面绘制或外部使用
        self.markers = new_markers
        self._sync_to_state_manager()

    def create_tiny_vector(self, grid: np.ndarray, x: float, y: float, mag: float = 1.0) -> None:
        # 在指定位置创建一个微小的向量场影响,只影响位置本身及上下左右四个邻居
        self.vector_calculator.create_tiny_vector(grid, x, y, mag)

    def add_vector_at_position(self, grid: np.ndarray, x: float, y: float, vx: float, vy: float) -> None:
        # 在指定位置添加一个向量
        self.vector_calculator.add_vector_at_position(grid, x, y, vx, vy)
        

    def fit_vector_at_position(self, grid: np.ndarray, x: float, y: float) -> Tuple[float, float]:
        # 在指定位置拟合一个向量
        return self.vector_calculator.fit_vector_at_position(grid, x, y)
    
    def update_field_and_markers(self, grid: np.ndarray) -> None:
        # 更新向量场和标记
        self.update_markers(grid)
        self.vector_calculator.update_grid_with_adjacent_sum(grid)
        # 再次更新标记
        self.update_markers(grid)

    def _sync_to_state_manager(self) -> None:
        """将标记列表同步到状态管理器"""
        try:
            self.app_core.state_manager.set("markers", list(self.markers))
        except Exception:
            pass

