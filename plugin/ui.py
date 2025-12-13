"""
UI 管理模块：封装用户交互回调、鼠标拖拽与滚轮缩放处理
将与输入相关的逻辑从示例中抽离，便于维护和复用。
"""
from typing import Tuple
import numpy as np
from lizi_engine.input import input_handler, KeyMap, MouseMap


class UIManager:
    def __init__(self, app_core, window, vector_calculator):
        self.app_core = app_core
        self.window = window
        self.vector_calculator = vector_calculator
        self.enable_update = True

        # 保持最后鼠标位置以便拖拽计算（像素坐标）
        self._last_mouse_x = None
        self._last_mouse_y = None
        # 标记列表，存储浮点网格坐标 {'x':float,'y':float}
        self.markers = []

    def register_callbacks(self, grid: np.ndarray, on_space=None, on_r=None, on_g=None, on_c=None, on_u=None):
        self._grid = grid

        def on_space_press():
            # 优先使用外部提供的回调（用于切换模式等），否则做一个默认视图重置。
            if callable(on_space):
                try:
                    on_space()
                    return
                except Exception as e:
                    print(f"[错误] on_space 回调异常: {e}")
            try:
                self.app_core.view_manager.reset_view(grid.shape[1], grid.shape[0])
            except Exception:
                pass

        def on_r_press():
            if callable(on_r):
                try:
                    on_r()
                    return
                except Exception as e:
                    print(f"[错误] on_r 回调异常: {e}")
            self.app_core.view_manager.reset_view(grid.shape[1], grid.shape[0])

        def on_g_press():
            if callable(on_g):
                try:
                    on_g()
                    return
                except Exception as e:
                    print(f"[错误] on_g 回调异常: {e}")
            show_grid = self.app_core.state_manager.get("show_grid", True)
            self.app_core.state_manager.set("show_grid", not show_grid)

        def on_c_press():
            if callable(on_c):
                try:
                    on_c()
                    return
                except Exception as e:
                    print(f"[错误] on_c 回调异常: {e}")
            grid.fill(0.0)

        def on_u_press():
            if callable(on_u):
                try:
                    on_u()
                    return
                except Exception as e:
                    print(f"[错误] on_u 回调异常: {e}")
            self.enable_update = not self.enable_update
            print(f"[示例] 实时更新已{'开启' if self.enable_update else '关闭'}")

        def on_mouse_left_press():
            try:
                mx, my = input_handler.get_mouse_position()

                cam_x = self.app_core.state_manager.get("cam_x", 0.0)
                cam_y = self.app_core.state_manager.get("cam_y", 0.0)
                cam_zoom = self.app_core.state_manager.get("cam_zoom", 1.0)
                viewport_width = self.app_core.state_manager.get("viewport_width", self.window._width)
                viewport_height = self.app_core.state_manager.get("viewport_height", self.window._height)
                cell_size = self.app_core.config_manager.get("cell_size", 1.0)

                world_x = cam_x + (mx - (viewport_width / 2.0)) / cam_zoom
                world_y = cam_y + (my - (viewport_height / 2.0)) / cam_zoom

                gx = int(world_x / cell_size)
                gy = int(world_y / cell_size)

                h, w = grid.shape[:2]
                if gx < 0 or gx >= w or gy < 0 or gy >= h:
                    print(f"[示例] 点击位置超出网格: ({gx}, {gy})")
                    return

                radius = 8
                magnitude = 0.8

                print(f"[示例] 在网格位置放置向量场: ({gx}, {gy}), radius={radius}, mag={magnitude}")

                self.vector_calculator.create_radial_pattern(grid, center=(gx, gy), radius=radius, magnitude=magnitude)

                # 同时创建一个标记，初始放在点击处（浮点位置）
                marker = {"x": float(gx), "y": float(gy)}
                self.markers.append(marker)
                # 将标记写入 state_manager 以便渲染器或外部代码读取
                try:
                    self.app_core.state_manager.set("markers", list(self.markers))
                except Exception:
                    pass

                self.app_core.state_manager.update({"view_changed": True, "grid_updated": True})
            except Exception as e:
                print(f"[错误] 处理鼠标左键按下时发生异常: {e}")

        # 注册键盘和鼠标回调
        input_handler.register_key_callback(KeyMap.SPACE, MouseMap.PRESS, on_space_press)
        input_handler.register_key_callback(KeyMap.R, MouseMap.PRESS, on_r_press)
        input_handler.register_key_callback(KeyMap.G, MouseMap.PRESS, on_g_press)
        input_handler.register_key_callback(KeyMap.C, MouseMap.PRESS, on_c_press)
        input_handler.register_key_callback(KeyMap.U, MouseMap.PRESS, on_u_press)

        input_handler.register_mouse_callback(MouseMap.LEFT, MouseMap.PRESS, on_mouse_left_press)

    def process_mouse_drag(self):
        window = self.window
        if getattr(window, "_mouse_pressed", False):
            x, y = window._mouse_x, window._mouse_y

            dx = x - (self._last_mouse_x if self._last_mouse_x is not None else x)
            dy = y - (self._last_mouse_y if self._last_mouse_y is not None else y)

            cam_zoom = self.app_core.state_manager.get("cam_zoom", 1.0)

            world_dx = dx / cam_zoom
            world_dy = dy / cam_zoom

            cam_x = self.app_core.state_manager.get("cam_x", 0.0) - world_dx
            cam_y = self.app_core.state_manager.get("cam_y", 0.0) - world_dy

            self.app_core.state_manager.update({
                "cam_x": cam_x,
                "cam_y": cam_y,
                "view_changed": True
            })

            self._last_mouse_x = x
            self._last_mouse_y = y
        else:
            # 清除上次位置，避免下次拖拽跳跃
            self._last_mouse_x = None
            self._last_mouse_y = None

    def process_scroll(self):
        window = self.window
        if hasattr(window, "_scroll_y") and window._scroll_y != 0:
            cam_zoom = self.app_core.state_manager.get("cam_zoom", 1.0)
            zoom_speed = 0.5
            cam_zoom += window._scroll_y * zoom_speed
            cam_zoom = max(0.1, min(10.0, cam_zoom))

            self.app_core.state_manager.update({
                "cam_zoom": cam_zoom,
                "view_changed": True
            })

            window._scroll_y = 0

    def update_markers(self, grid: np.ndarray, neighborhood: int = 5, move_factor: float = 0.2, clear_threshold: float = 1e-3):
        """根据周围向量平均方向移动标记以收敛到中心。

        算法：在每个标记的邻域内计算平均向量(mean_v)，将标记按 -mean_v * move_factor 偏移。
        这对径向场有效：向外的平均向量的负方向指向中心。
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

            m["x"] = new_x
            m["y"] = new_y
            new_markers.append(m)

        # 更新内部标记列表并写回 state_manager 以便界面绘制或外部使用
        self.markers = new_markers
        try:
            self.app_core.state_manager.set("markers", list(self.markers))
        except Exception:
            pass
