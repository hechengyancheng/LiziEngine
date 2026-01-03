
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

    def update_markers(self, grid: np.ndarray, move_factor: float = 1.0, damping_factor: float = 0.99) -> None:
        """根据浮点坐标处拟合向量移动标记。

        算法：在标记的浮点坐标处使用双线性插值拟合向量值，将标记按 fitted_v * move_factor 偏移。

        Args:
            grid: 向量场网格
            neighborhood: 邻域大小（保留参数以保持兼容性）
            move_factor: 移动因子
            damping_factor: 速度阻尼因子，用于模拟摩擦（默认0.99）
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

        # 第一阶段：计算所有标记的合力向量（基于当前网格状态）
        forces = []
        for m in self.markers:
            x = m.get("x", 0.0)
            y = m.get("y", 0.0)
            try:
                fx, fy = self.compute_force_from_neighbors(grid, x, y, use_fp32=False, radius=10.0)
                forces.append((fx, fy))
            except Exception as e:
                print(f"Error computing force for marker at ({x}, {y}): {str(e)}")
                forces.append((0.0, 0.0))

        # 第二阶段：根据合力更新所有标记的位置和速度
        new_markers = []
        for i, m in enumerate(self.markers):
            fx, fy = forces[i]

            # 存储合力向量用于渲染
            m["fx"] = fx
            m["fy"] = fy

            # 将合力作为速度（可通过 move_factor 缩放）
            m["vx"] += fx
            m["vy"] += fy

            # 应用速度阻尼因子模拟摩擦
            m["vx"] *= damping_factor
            m["vy"] *= damping_factor

            # 使用速度更新浮点位置，并处理边界反弹
            tentative_x = m["x"] + m["vx"] * move_factor
            tentative_y = m["y"] + m["vy"] * move_factor

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

            m["x"] = new_x
            m["y"] = new_y
            new_markers.append(m)

        # 第三阶段：在所有标记的新位置创建向量场影响
        for m in new_markers:
            try:
                self.create_vector_field_at_position(grid, m["x"], m["y"], m.get("mag", 1.0), radius=10.0)
            except Exception as e:
                print(f"Error creating vector field for marker at ({m['x']}, {m['y']}): {str(e)}")

        # 更新内部标记列表并写回 state_manager 以便界面绘制或外部使用
        self.markers = new_markers
        self._sync_to_state_manager()

    def create_tiny_vector(self, grid: np.ndarray, x: float, y: float, mag: float = 1.0) -> None:
        # 在指定位置创建一个微小的向量场影响,只影响位置本身及上下左右四个邻居
        self.vector_calculator.create_tiny_vector(grid, x, y, mag)

    def add_vector_at_position(self, grid: np.ndarray, x: float, y: float, vx: float, vy: float) -> None:
        # 在指定位置添加一个向量
        self.vector_calculator.add_vector_at_position(grid, x, y, vx, vy)

    def create_vector_field_at_position(self, grid: np.ndarray, x: float, y: float, mag: float = 1.0, radius: float = 1.0) -> None:
        """在指定位置创建圆形向量场，支持浮点坐标。

        在圆形范围内创建径向向外向量场，用于标记的相互作用。

        Args:
            grid: 向量场网格
            x: 中心x坐标（浮点）
            y: 中心y坐标（浮点）
            mag: 向量幅值
            radius: 圆形半径
        """
        if not hasattr(grid, "ndim") or grid.ndim < 3 or grid.shape[2] < 2:
            return

        h, w = grid.shape[0], grid.shape[1]

        # 计算边界框
        min_x = max(0, int(x - radius))
        max_x = min(w - 1, int(x + radius) + 1)
        min_y = max(0, int(y - radius))
        max_y = min(h - 1, int(y + radius) + 1)

        # 遍历边界框内的所有整数网格点
        for ix in range(min_x, max_x):
            for iy in range(min_y, max_y):
                # 检查是否在圆内
                dx = ix - x
                dy = iy - y
                dist_sq = dx**2 + dy**2
                if dist_sq <= radius**2:
                    dist = np.sqrt(dist_sq)
                    if dist > 0:
                        vx = (dx / dist) * mag
                        vy = (dy / dist) * mag
                        self.add_vector_at_position(grid, float(ix), float(iy), vx, vy)
        

    def fit_vector_at_position(self, grid: np.ndarray, x: float, y: float) -> Tuple[float, float]:
        # 在指定位置拟合一个向量
        return self.vector_calculator.fit_vector_at_position(grid, x, y)
    
    def fit_vector_at_position_fp32(self, grid: np.ndarray, x: float, y: float) -> Tuple[float, float]:
        # 在指定位置拟合一个向量
        return self.vector_calculator.fit_vector_at_position_fp32(grid, x, y)

    def compute_force_from_neighbors(self, grid: np.ndarray, x: float, y: float, use_fp32: bool = True, radius: float = 1.0) -> Tuple[float, float]:
        """遍历圆形范围内向量值并相加，返回合力 (fx, fy)。

        说明：对圆形范围内所有网格点分别调用拟合函数，将得到的向量逐一相加。
        超出边界的位置会被忽略。

        Args:
            grid: 向量场网格
            x: 标记的浮点 x 坐标
            y: 标记的浮点 y 坐标
            use_fp32: 是否使用 fp32 快速拟合接口（默认 True）
            radius: 圆形半径（默认 1.0）

        Returns:
            (fx, fy): 合力向量分量
        """
        # 基本校验
        if not hasattr(grid, "ndim"):
            return 0.0, 0.0
        if grid.ndim < 3 or grid.shape[2] < 2:
            return 0.0, 0.0

        h, w = grid.shape[0], grid.shape[1]

        # 计算边界框
        min_x = max(0, int(x - radius))
        max_x = min(w - 1, int(x + radius) + 1)
        min_y = max(0, int(y - radius))
        max_y = min(h - 1, int(y + radius) + 1)

        total_vx = 0.0
        total_vy = 0.0

        # 遍历边界框内的所有整数网格点
        for ix in range(min_x, max_x):
            for iy in range(min_y, max_y):
                # 检查是否在圆内
                dx = ix - x
                dy = iy - y
                dist_sq = dx**2 + dy**2
                if dist_sq <= radius**2:
                    # 检查是否在边界内
                    if not (0.0 <= ix <= w - 1.0 and 0.0 <= iy <= h - 1.0):
                        continue

                    try:
                        if use_fp32:
                            vx, vy = self.fit_vector_at_position_fp32(grid, float(ix), float(iy))
                        else:
                            vx, vy = self.fit_vector_at_position(grid, float(ix), float(iy))

                        total_vx += float(vx)
                        total_vy += float(vy)
                    except Exception as e:
                        # 保持容错，打印错误便于调试
                        print(f"Error fitting vector at ({ix}, {iy}): {e}")
                        continue

        return total_vx, total_vy

    def _sync_to_state_manager(self) -> None:
        """将标记列表同步到状态管理器"""
        try:
            self.app_core.state_manager.set("markers", list(self.markers))
        except Exception:
            pass

