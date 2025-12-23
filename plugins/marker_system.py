
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
        # 标记列表，存储浮点网格坐标 {'x':float,'y':float}
        self.markers = []

    def add_marker(self, x: float, y: float, mag: float = 1.0) -> None:
        """添加一个新标记

        Args:
            x: 标记的x坐标（浮点）
            y: 标记的y坐标（浮点）
            mag: 标记的初始幅值（可选）
        """
        marker = {"x": float(x), "y": float(y), "mag": float(mag)}
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

    def update_markers(self, grid: np.ndarray, move_factor: float = 0.2, clear_threshold: float = 1e-3) -> None:
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
                # 在浮点坐标处拟合向量值
                fitted_vx, fitted_vy = self.fit_vector_at_position(grid, x, y)

                # 计算拟合向量的幅值
                fitted_mag = np.sqrt(fitted_vx**2 + fitted_vy**2)
                
                # 如果拟合向量幅值低于阈值，自动移除该标记
                if fitted_mag < clear_threshold:
                    continue
                
                
                # 使用拟合向量作为位移量
                dx = fitted_vx * move_factor
                dy = fitted_vy * move_factor

                # 更新浮点位置
                new_x = max(0.0, min(w - 1.0, x + dx))
                new_y = max(0.0, min(h - 1.0, y + dy))

                # 创建微小向量影响
                self.create_tiny_vector(grid, new_x, new_y, m["mag"])

                m["x"] = new_x
                m["y"] = new_y
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

    def create_tiny_vector(self, grid: np.ndarray, x: float, y: float, mag: float = 1.0, vx: float = 0.0, vy: float = 0.0) -> None:
        # 在指定位置创建一个微小的向量场影响,只影响位置本身及上下左右四个邻居
        if not hasattr(grid, "ndim"):
            return

        h, w = grid.shape[0], grid.shape[1]

        # 确保坐标在有效范围内
        x = max(0.0, min(w - 1.0, float(x)))
        y = max(0.0, min(h - 1.0, float(y)))

        # 计算整数坐标
        cx = int(round(x))
        cy = int(round(y))

        # 只影响当前位置及其上下左右邻居
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if abs(dx) + abs(dy) == 1:  # 上下左右邻居
                    nx = cx + dx
                    ny = cy + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        try:
                            if grid.ndim >= 3 and grid.shape[2] >= 2:
                                grid[ny, nx, 0] += dx * mag
                                grid[ny, nx, 1] += dy * mag
                        except Exception:
                            continue
        #当前位置的向量值为vx,vy
        if grid.ndim >= 3 and grid.shape[2] >= 2:
            try:
                grid[cy, cx, 0] += vx * mag
                grid[cy, cx, 1] += vy * mag
            except Exception:
                pass

    def add_vector_at_position(self, grid: np.ndarray, x: float, y: float, vx: float, vy: float) -> None:
        """在浮点坐标处添加向量，使用双线性插值的逆方法，将向量分布到四个最近的整数坐标

        Args:
            grid: 向量场网格
            x: 浮点x坐标
            y: 浮点y坐标
            vx: 向量x分量
            vy: 向量y分量
        """
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
        """在浮点坐标处拟合向量值，使用双线性插值

        Args:
            grid: 向量场网格
            x: 浮点x坐标
            y: 浮点y坐标

        Returns:
            插值后的向量 (vx, vy)
        """
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

    def _sync_to_state_manager(self) -> None:
        """将标记列表同步到状态管理器"""
        try:
            self.app_core.state_manager.set("markers", list(self.markers))
        except Exception:
            pass
