"""
UI 管理模块：封装用户交互回调、鼠标拖拽与滚轮缩放处理
将与输入相关的逻辑从示例中抽离，便于维护和复用。
使用 Controller 模块处理业务逻辑。
"""
from typing import Tuple
import numpy as np
from lizi_engine.input import input_handler, KeyMap, MouseMap


class UIManager:
    def __init__(self, app_core, window, controller, marker_system):
        self.app_core = app_core
        self.window = window
        self.controller = controller
        self.marker_system = marker_system

        self.enable_update = True

        # 保持最后鼠标位置以便拖拽计算（像素坐标）
        self._last_mouse_x = None
        self._last_mouse_y = None

        # 左键按下标志
        self._mouse_left_pressed = False

        # 左键按下时选择的标记
        self._selected_marker = None

        # 鼠标按钮状态
        self._mouse_middle_pressed = False

    def register_callbacks(self, grid: np.ndarray, on_space=None, on_r=None, on_g=None, on_c=None, on_u=None, on_v=None, on_f=None):
        self._grid = grid

        def on_space_press():
            # 优先使用外部提供的回调（用于切换模式等）
            if callable(on_space):
                try:
                    on_space()
                    return
                except Exception as e:
                    print(f"[错误] on_space 回调异常: {e}")
            # 无外部回调时不执行默认行为

        def on_r_press():
            if callable(on_r):
                try:
                    on_r()
                    return
                except Exception as e:
                    print(f"[错误] on_r 回调异常: {e}")
            try:
                self.controller.reset_view()
            except Exception as e:
                print(f"[错误] reset_view 异常: {e}")

        def on_g_press():
            if callable(on_g):
                try:
                    on_g()
                    return
                except Exception as e:
                    print(f"[错误] on_g 回调异常: {e}")
            try:
                self.controller.toggle_grid()
            except Exception as e:
                print(f"[错误] toggle_grid 异常: {e}")

        def on_c_press():
            if callable(on_c):
                try:
                    on_c()
                    return
                except Exception as e:
                    print(f"[错误] on_c 回调异常: {e}")
            try:
                self.controller.clear_grid()
            except Exception as e:
                print(f"[错误] clear_grid 异常: {e}")
            try:
                # 标记系统也应清空标记
                self.marker_system.clear_markers()
            except Exception as e:
                print(f"[错误] clear_markers 异常: {e}")

        def on_v_press():
            if callable(on_v):
                try:
                    on_v()
                    return
                except Exception as e:
                    print(f"[错误] on_v 回调异常: {e}")
            try:
                self.controller.switch_vector_field_direction()
            except Exception as e:
                print(f"[错误] switch_vector_field_direction 异常: {e}")

        def on_u_press():
            if callable(on_u):
                try:
                    on_u()
                    return
                except Exception as e:
                    print(f"[错误] on_u 回调异常: {e}")

        def on_f_press():
            if callable(on_f):
                try:
                    on_f()
                    return
                except Exception as e:
                    print(f"[错误] on_f 回调异常: {e}")
            try:
                mx, my = input_handler.get_mouse_position()
                self.controller.place_vector_field(mx, my)
            except Exception as e:
                print(f"[错误] 处理f键按下时发生异常: {e}")

        def on_mouse_left_press():
            try:
                # 设置左键按下标志
                self._mouse_left_pressed = True

                mx, my = input_handler.get_mouse_position()
                self._selected_marker = self.controller.handle_mouse_left_press(mx, my)
            except Exception as e:
                print(f"[错误] 处理鼠标左键按下时发生异常: {e}")

        # 注册键盘和鼠标回调
        input_handler.register_key_callback(KeyMap.SPACE, MouseMap.PRESS, on_space_press)
        input_handler.register_key_callback(KeyMap.R, MouseMap.PRESS, on_r_press)
        input_handler.register_key_callback(KeyMap.G, MouseMap.PRESS, on_g_press)
        input_handler.register_key_callback(KeyMap.C, MouseMap.PRESS, on_c_press)
        input_handler.register_key_callback(KeyMap.U, MouseMap.PRESS, on_u_press)
        input_handler.register_key_callback(KeyMap.V, MouseMap.PRESS, on_v_press)
        input_handler.register_key_callback(KeyMap.F, MouseMap.PRESS, on_f_press)

        input_handler.register_mouse_callback(MouseMap.LEFT, MouseMap.PRESS, on_mouse_left_press)

        # 添加鼠标左键释放的回调
        def on_mouse_left_release():
            # 清除左键按下标志和选定的标记
            self._mouse_left_pressed = False
            self._selected_marker = None

        input_handler.register_mouse_callback(MouseMap.LEFT, MouseMap.RELEASE, on_mouse_left_release)

        # 添加鼠标中键按下和释放的回调
        def on_mouse_middle_press():
            # 设置中键按下标志
            self._mouse_middle_pressed = True

        def on_mouse_middle_release():
            # 清除中键按下标志
            self._mouse_middle_pressed = False

        # 注册鼠标中键回调
        input_handler.register_mouse_callback(MouseMap.MIDDLE, MouseMap.PRESS, on_mouse_middle_press)
        input_handler.register_mouse_callback(MouseMap.MIDDLE, MouseMap.RELEASE, on_mouse_middle_release)

    def process_mouse_drag(self):
        window = self.window
        # 处理鼠标左键持续按下，在标记位置添加向量
        if self._mouse_left_pressed and self._selected_marker is not None:
            try:
                mx, my = window._mouse_x, window._mouse_y
                self.controller.handle_mouse_drag(mx, my, self._selected_marker)
            except Exception as e:
                print(f"[错误] 处理左键持续按下时发生异常: {e}")

        # 只在鼠标中键按下时才允许拖动视图
        if self._mouse_middle_pressed:
            x, y = window._mouse_x, window._mouse_y

            dx = x - (self._last_mouse_x if self._last_mouse_x is not None else x)
            dy = y - (self._last_mouse_y if self._last_mouse_y is not None else y)

            try:
                self.controller.handle_mouse_drag_view(dx, dy)
            except Exception as e:
                print(f"[错误] process_mouse_drag_view 异常: {e}")

            self._last_mouse_x = x
            self._last_mouse_y = y
        else:
            # 清除上次位置，避免下次拖拽跳跃
            self._last_mouse_x = None
            self._last_mouse_y = None

    def process_scroll(self):
        window = self.window
        if hasattr(window, "_scroll_y") and window._scroll_y != 0:
            try:
                self.controller.handle_scroll_zoom(window._scroll_y)
            except Exception as e:
                print(f"[错误] process_scroll 异常: {e}")

            window._scroll_y = 0

    def update_markers(self, grid: np.ndarray, dt: float = 1.0, clear_threshold: float = 1e-3):
        """使用标记系统更新标记位置

        Args:
            grid: 向量场网格
            dt: 时间步长
            clear_threshold: 清除阈值，低于此平均幅值的标记将被清除
        """
        self.marker_system.update_markers(grid, dt, clear_threshold)

