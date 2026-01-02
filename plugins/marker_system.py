
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

    def update_markers(self, grid: np.ndarray, dt: float = 1.0, move_factor: float = 1.0, clear_threshold: float = 1e-3, use_inertia: bool = False, damping: float = 1.0) -> None:
        """根据浮点坐标处拟合向量移动标记。

        改进点：
        - 引入 `dt`（时间步长），使位移与时间尺度明确
        - 对被动示踪子使用 RK2（midpoint）积分以提高精度
        - 可选惯性模式（`use_inertia=True`）将把场视为加速度并保留粒子速度
        - 延后对 `grid` 的修改：先计算所有新位置，再一次性调用 `create_tiny_vector`

        Args:
            grid: 向量场网格
            dt: 时间步长
            move_factor: 速度/加速度缩放因子
            clear_threshold: 清除阈值（暂保留，用于潜在扩展）
            use_inertia: 是否使用惯性粒子模型（场作为加速度）
            damping: 惯性模式下的速度阻尼系数
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
        # 延迟对 grid 的修改，避免同一次更新周期中顺序依赖
        pending_tiny_vectors = []  # list of (x, y, mag)

        for m in self.markers:
            x = m.get("x", 0.0)
            y = m.get("y", 0.0)

            try:
                if not use_inertia:
                    # 被动示踪子：使用 RK2（midpoint）积分
                    v1x, v1y = self.fit_vector_at_position_fp32(grid, x, y)
                    v1x *= move_factor
                    v1y *= move_factor

                    mx = x + 0.5 * dt * v1x
                    my = y + 0.5 * dt * v1y
                    mx = max(0.0, min(w - 1.0, mx))
                    my = max(0.0, min(h - 1.0, my))

                    v2x, v2y = self.fit_vector_at_position_fp32(grid, mx, my)
                    v2x *= move_factor
                    v2y *= move_factor

                    # 简单的局部子步：若单步位移过大则分多子步积分
                    max_disp = np.hypot(v2x, v2y) * dt
                    if max_disp > 1.0:
                        n_sub = int(np.ceil(max_disp))
                        dt_sub = dt / n_sub
                        curx, cury = x, y
                        for _ in range(n_sub):
                            sv1x, sv1y = self.fit_vector_at_position_fp32(grid, curx, cury)
                            sv1x *= move_factor
                            sv1y *= move_factor
                            midx = curx + 0.5 * dt_sub * sv1x
                            midy = cury + 0.5 * dt_sub * sv1y
                            midx = max(0.0, min(w - 1.0, midx))
                            midy = max(0.0, min(h - 1.0, midy))
                            sv2x, sv2y = self.fit_vector_at_position_fp32(grid, midx, midy)
                            sv2x *= move_factor
                            sv2y *= move_factor
                            curx = curx + dt_sub * sv2x
                            cury = cury + dt_sub * sv2y
                        new_x = max(0.0, min(w - 1.0, curx))
                        new_y = max(0.0, min(h - 1.0, cury))
                        vx = (new_x - x) / dt
                        vy = (new_y - y) / dt
                    else:
                        new_x = max(0.0, min(w - 1.0, x + v2x * dt))
                        new_y = max(0.0, min(h - 1.0, y + v2y * dt))
                        vx, vy = v2x, v2y

                    m["vx"] = float(vx)
                    m["vy"] = float(vy)
                    m["x"] = float(new_x)
                    m["y"] = float(new_y)
                    pending_tiny_vectors.append((new_x, new_y, m.get("mag", 1.0)))
                    new_markers.append(m)

                else:
                    # 惯性粒子：将场视为加速度源（a），保留并更新粒子速度
                    ax, ay = self.fit_vector_at_position_fp32(grid, x, y)
                    ax *= move_factor
                    ay *= move_factor
                    pvx = m.get("vx", 0.0) + ax * dt
                    pvy = m.get("vy", 0.0) + ay * dt
                    pvx *= damping
                    pvy *= damping
                    new_x = max(0.0, min(w - 1.0, x + pvx * dt))
                    new_y = max(0.0, min(h - 1.0, y + pvy * dt))
                    m["vx"] = float(pvx)
                    m["vy"] = float(pvy)
                    m["x"] = float(new_x)
                    m["y"] = float(new_y)
                    pending_tiny_vectors.append((new_x, new_y, m.get("mag", 1.0)))
                    new_markers.append(m)

            except Exception as e:
                print(f"Error updating marker at ({x}, {y}): {str(e)}")
                new_markers.append(m)
                continue

        # 一次性把微小向量影响应用到网格，避免本帧内的顺序依赖
        for px, py, pmag in pending_tiny_vectors:
            try:
                self.create_tiny_vector(grid, px, py, pmag)
            except Exception:
                pass

        # 更新内部标记列表并写回 state_manager
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

    def _sync_to_state_manager(self) -> None:
        """将标记列表同步到状态管理器"""
        try:
            self.app_core.state_manager.set("markers", list(self.markers))
        except Exception:
            pass

