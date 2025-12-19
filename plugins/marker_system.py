"""
标记系统插件：管理向量场中的标记点
提供标记点的创建、更新和渲染功能。
"""
from typing import List, Dict, Any
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

    def update_markers(self, grid: np.ndarray, neighborhood: int = 5, 
                      move_factor: float = 0.2, clear_threshold: float = 1e-3) -> None:
        """根据周围向量平均方向移动标记以收敛到中心。

        算法：在每个标记的邻域内计算平均向量(mean_v)，将标记按 -mean_v * move_factor 偏移。
        这对径向场有效：向外的平均向量的负方向指向中心。

        Args:
            grid: 向量场网格
            neighborhood: 邻域大小
            move_factor: 移动因子
            clear_threshold: 清除阈值，低于此平均幅值的标记将被清除
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

        h, w = grid.shape[0], grid.shape[1]

        # 期望 grid 最后一维至少 2，代表 vx, vy
        new_markers = []

        for m in self.markers:
            x = m.get("x", 0.0)
            y = m.get("y", 0.0)

            # 整数邻域范围
            cx = int(round(x))
            cy = int(round(y))

            sx = max(0, cx - neighborhood)
            ex = min(w - 1, cx + neighborhood)
            sy = max(0, cy - neighborhood)
            ey = min(h - 1, cy + neighborhood)

            sum_vx = 0.0
            sum_vy = 0.0
            sum_mag = 0.0
            count = 0
            
            for yy in range(sy, ey + 1):
                for xx in range(sx, ex + 1):
                    try:
                        if grid.ndim >= 3 and grid.shape[2] >= 2:
                            vx = float(grid[yy, xx, 0])
                            vy = float(grid[yy, xx, 1])
                        else:
                            # 如果只有一个通道，则无法计算方向，跳过
                            continue
                    except Exception:
                        continue

                    mag = (vx * vx + vy * vy) ** 0.5
                    # 以幅值加权平均，使明显的向量影响更大
                    sum_vx += vx * mag
                    sum_vy += vy * mag
                    sum_mag += mag
                    count += 1

            if count == 0:
                # 没有有效向量，保留标记以便后续检查
                new_markers.append(m)
                continue

            mean_vx = sum_vx / count
            mean_vy = sum_vy / count
            avg_mag = sum_mag / count

            # 如果邻域内平均幅值低于阈值，自动移除该标记
            if avg_mag < clear_threshold:
                # skip adding to new_markers (即删除)
                continue

            # 正方向朝向可能的中心
            dx = mean_vx * move_factor
            dy = mean_vy * move_factor

            # 更新浮点位置
            new_x = max(0.0, min(w - 1.0, x + dx))
            new_y = max(0.0, min(h - 1.0, y + dy))

            # 创建径向模式
            vector_calculator.create_radial_pattern(grid,center=(new_x,new_y), radius=2.0, magnitude=m["mag"])

            m["x"] = new_x
            m["y"] = new_y
            new_markers.append(m)

        # 更新内部标记列表并写回 state_manager 以便界面绘制或外部使用
        self.markers = new_markers
        self._sync_to_state_manager()
        
            

    def _sync_to_state_manager(self) -> None:
        """将标记列表同步到状态管理器"""
        try:
            self.app_core.state_manager.set("markers", list(self.markers))
        except Exception:
            pass
