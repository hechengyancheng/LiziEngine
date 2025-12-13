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
            self.app_core.state_manager.update({"grid_updated": True})

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
                magnitude = 1.0

                print(f"[示例] 在网格位置放置向量场: ({gx}, {gy}), radius={radius}, mag={magnitude}")

                self.vector_calculator.create_radial_pattern(grid, center=(gx, gy), radius=radius, magnitude=magnitude)

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
