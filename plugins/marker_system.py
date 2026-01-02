
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

    def update_markers(self, grid: np.ndarray, move_factor: float = 1.0, clear_threshold: float = 1e-3) -> None:
        """根据浮点坐标处拟合向量移动标记。

        算法：在标记的浮点坐标处使用双线性插值拟合向量值，将标记按 fitted_v * move_factor 偏移。

        Args:
            grid: 向量场网格
            neighborhood: 邻域大小（保留参数以保持兼容性）
            move_factor: 移动因子
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

        # 期望 grid 最后一维至少 2，代表 vx, vy
        new_markers = []

        for m in self.markers:
            x = m.get("x", 0.0)
            y = m.get("y", 0.0)

            try:
                # 计算标记自身及上下左右四个位置的向量合力
                fx, fy = self.compute_force_from_neighbors(grid, x, y, use_fp32=False)

                # 将合力作为速度（可通过 move_factor 缩放）
                m["vx"] += fx * move_factor
                m["vy"] += fy * move_factor

                # 使用速度更新浮点位置，并处理边界反弹
                tentative_x = x + m["vx"]
                tentative_y = y + m["vy"]

                # 处理x方向边界反弹
                if tentative_x < 0.0:
                    m["vx"] = -m["vx"]
                    new_x = 0.0
                elif tentative_x > w - 1.0:
                    m["vx"] = -m["vx"]
                    new_x = w - 1.0
                else:
                    new_x = tentative_x

                # 处理y方向边界反弹
                if tentative_y < 0.0:
                    m["vy"] = -m["vy"]
                    new_y = 0.0
                elif tentative_y > h - 1.0:
                    m["vy"] = -m["vy"]
                    new_y = h - 1.0
                else:
                    new_y = tentative_y

                # 在新位置创建微小向量影响
                self.create_tiny_vector(grid, new_x, new_y, m.get("mag", 1.0))

                m["x"] = new_x
                m["y"] = new_y
                new_markers.append(m)

            except Exception as e:
                print(f"Error updating marker at ({x}, {y}): {str(e)}")
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
    
    def fit_vector_at_position_fp32(self, grid: np.ndarray, x: float, y: float) -> Tuple[float, float]:
        # 在指定位置拟合一个向量
        return self.vector_calculator.fit_vector_at_position_fp32(grid, x, y)

    def compute_force_from_neighbors(self, grid: np.ndarray, x: float, y: float, use_fp32: bool = True) -> Tuple[float, float]:
        """在标记自身和上下左右四个位置拟合向量并相加，返回合力 (fx, fy)。

        说明：对自身以及上下左右四个浮点坐标位置分别调用拟合函数，将得到的向量逐一相加。
        超出边界的位置会被忽略。

        Args:
            grid: 向量场网格
            x: 标记的浮点 x 坐标
            y: 标记的浮点 y 坐标
            use_fp32: 是否使用 fp32 快速拟合接口（默认 True）

        Returns:
            (fx, fy): 合力向量分量
        """
        # 基本校验
        if not hasattr(grid, "ndim"):
            return 0.0, 0.0
        if grid.ndim < 3 or grid.shape[2] < 2:
            return 0.0, 0.0

        h, w = grid.shape[0], grid.shape[1]

        # 采样点：自身 + 上下左右
        sample_positions = [
            (x, y),
            (x - 1.0, y),
            (x + 1.0, y),
            (x, y - 1.0),
            (x, y + 1.0),
        ]

        total_vx = 0.0
        total_vy = 0.0

        for sx, sy in sample_positions:
            # 检查是否在边界内，如果超出则忽略
            if not (0.0 <= sx <= w - 1.0 and 0.0 <= sy <= h - 1.0):
                continue

            try:
                if use_fp32:
                    vx, vy = self.fit_vector_at_position_fp32(grid, sx, sy)
                else:
                    vx, vy = self.fit_vector_at_position(grid, sx, sy)

                total_vx += float(vx)
                total_vy += float(vy)
            except Exception as e:
                # 保持容错，打印错误便于调试
                print(f"Error fitting vector at ({sx}, {sy}): {e}")
                continue

        return total_vx, total_vy

    def _sync_to_state_manager(self) -> None:
        """将标记列表同步到状态管理器"""
        try:
            self.app_core.state_manager.set("markers", list(self.markers))
        except Exception:
            pass

