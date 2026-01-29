
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

    def update_markers(self, grid: np.ndarray, dt: float = 1.0, gravity: float = 0.01, speed_factor: float = 0.9) -> None:
        """根据浮点坐标处拟合向量移动标记。

        算法：在标记的浮点坐标处使用双线性插值拟合向量值，将标记按 fitted_v * dt 偏移。

        Args:
            grid: 向量场网格
            dt: 时间步长
            gravity: 重力加速度
            speed_factor: 速度衰减因子
        """
        if not self._is_valid_grid(grid):
            return

        self._sync_markers_from_state()

        h, w = grid.shape[0], grid.shape[1]
        cell_size = self.app_core.state_manager.get("cell_size", 1.0)

        marker_positions = self._collect_marker_positions()
        fitted_vectors = self._fit_vectors_batch(grid, marker_positions ,radius=2)

        new_markers = []
        for i, marker in enumerate(self.markers):
            updated_marker = self._update_single_marker(marker, fitted_vectors[i], dt, gravity, speed_factor, cell_size, w, h)
            if updated_marker:
                new_markers.append(updated_marker)

        self._update_markers_list(new_markers)

    def _is_valid_grid(self, grid: np.ndarray) -> bool:
        """检查网格是否有效"""
        return hasattr(grid, "ndim") and grid.ndim >= 3 and grid.shape[2] >= 2

    def _sync_markers_from_state(self) -> None:
        """从全局状态同步标记"""
        try:
            stored = self.app_core.state_manager.get("markers", None)
            if stored is not None:
                self.markers = list(stored)
        except Exception:
            pass

    def _collect_marker_positions(self) -> List[Tuple[float, float]]:
        """收集所有标记位置"""
        return [(m["x"], m["y"]) for m in self.markers]

    def _fit_vectors_batch(self, grid: np.ndarray, positions: List[Tuple[float, float]] , radius: int = 1) -> List[Tuple[float, float]]:
        """批量拟合向量值"""
        return self.vector_calculator.fit_vectors_at_positions_batch(grid, positions , radius)

    def _update_single_marker(self, marker: Dict[str, float], fitted_vector: Tuple[float, float], dt: float, gravity: float, speed_factor: float, cell_size: float, w: int, h: int) -> Dict[str, float]:
        """更新单个标记"""
        x, y, mag, vx, vy = marker["x"], marker["y"], marker["mag"], marker["vx"], marker["vy"]
        try:
            fitted_vx, fitted_vy = fitted_vector

            # 更新速度
            vx += fitted_vx / mag
            vy += fitted_vy / mag

            # 限制速度
            vx, vy = self._clamp_velocity(vx, vy, cell_size)

            # 更新位置
            x, y = self._update_position(x, y, vx, vy, dt, w, h)

            # 应用物理效果
            vx, vy = self._apply_physics(vx, vy, gravity, speed_factor, dt)

            marker["x"], marker["y"], marker["vx"], marker["vy"] = x, y, vx, vy
            return marker
        except Exception as e:
            print(f"Error updating marker at ({x}, {y}): {str(e)}")
            return marker

    def _clamp_velocity(self, vx: float, vy: float, cell_size: float) -> Tuple[float, float]:
        """限制速度不超过单元格大小"""
        speed = (vx ** 2 + vy ** 2) ** 0.5
        if speed > cell_size:
            vx = vx / speed * cell_size
            vy = vy / speed * cell_size
        return vx, vy

    def _update_position(self, x: float, y: float, vx: float, vy: float, dt: float, w: int, h: int) -> Tuple[float, float]:
        """更新位置并钳制边界"""
        new_x = max(0.0, min(w - 1.0, x + vx * dt))
        new_y = max(0.0, min(h - 1.0, y + vy * dt))
        return new_x, new_y

    def _apply_physics(self, vx: float, vy: float, gravity: float, speed_factor: float, dt: float) -> Tuple[float, float]:
        """应用重力和摩擦力"""
        vy += gravity * dt
        vx *= speed_factor
        vy *= speed_factor
        return vx, vy

    def _update_markers_list(self, new_markers: List[Dict[str, float]]) -> None:
        """更新标记列表并同步到状态管理器"""
        self.markers = new_markers
        self._sync_to_state_manager()

    def create_tiny_vector(self, grid: np.ndarray, x: float, y: float, mag: float = 1.0 , radius: int = 1) -> None:
        # 在指定位置创建一个微小的向量场影响
        self.vector_calculator.create_tiny_vector(grid, x, y, mag , radius)

    def add_vector_at_position(self, grid: np.ndarray, x: float, y: float, vx: float, vy: float) -> None:
        # 在指定位置添加一个向量
        self.vector_calculator.add_vector_at_position(grid, x, y, vx, vy)
        

    def fit_vector_at_position(self, grid: np.ndarray, x: float, y: float ,radius: float = 1.0) -> Tuple[float, float]:
        # 在指定位置拟合一个向量
        return self.vector_calculator.fit_vector_at_position(grid, x, y , radius)

    def update_field_and_markers(self, grid: np.ndarray, dt: float, gravity: float, speed_factor: float) -> None:
        # 更新向量场和标记
        self.batch_create_tiny_vectors_from_markers(grid, self.markers ,radius=2)
        self.update_markers(grid, dt=dt, gravity=gravity, speed_factor=speed_factor)

    def batch_create_tiny_vectors_from_markers(self, grid: np.ndarray, markers: List[Dict[str, float]], radius: int = 1) -> None:
        """收集标记位置以便批量创建微小向量影响。

        Args:
            grid: 向量场网格
            markers: 标记列表
        """
        tiny_vector_positions = [(m["x"], m["y"], m["mag"]) for m in markers]
        if tiny_vector_positions:
            self.vector_calculator.create_tiny_vectors_batch(grid, tiny_vector_positions, radius)
    def _sync_to_state_manager(self) -> None:
        """将标记列表同步到状态管理器"""
        try:
            self.app_core.state_manager.set("markers", list(self.markers))
        except Exception:
            pass

