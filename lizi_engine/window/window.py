"""
窗口管理模块 - 提供窗口管理功能
支持Dear PyGui窗口创建和事件处理
"""
import dearpygui.dearpygui as dpg
import numpy as np
from typing import Optional, Callable, Dict, Any, Tuple
from ..core.config import config_manager
from ..core.events import Event, EventType, event_bus, EventHandler, FunctionEventHandler
from ..core.state import state_manager
from ..graphics.renderer import VectorFieldRenderer
from ..input import input_handler

class Window(EventHandler):
    """窗口管理器"""
    def __init__(self, title: str = "LiziEngine", width: int = 800, height: int = 600):
        self._event_bus = event_bus
        self._state_manager = state_manager
        self._config_manager = config_manager
        self._renderer = None

        # 窗口属性
        self._title = title
        self._width = width
        self._height = height
        self._window = None
        self._should_close = False

        # 鼠标状态
        self._mouse_pressed = False
        self._mouse_x = 0
        self._mouse_y = 0
        self._last_mouse_x = 0
        self._last_mouse_y = 0
        self._scroll_y = 0  # 鼠标滚轮Y轴偏移

        # 键盘状态
        self._keys = {}

        # 事件处理器
        self._event_handlers = {}

        # 订阅事件
        self._event_bus.subscribe(EventType.APP_INITIALIZED, self)

    def initialize(self) -> bool:
        """初始化窗口"""
        try:
            # 初始化Dear PyGui
            dpg.create_context()
            dpg.create_viewport(title=self._title, width=self._width, height=self._height)
            dpg.setup_dearpygui()

            # 创建主窗口
            with dpg.window(label=self._title, width=self._width, height=self._height, no_title_bar=True) as self._window:
                # 创建绘图区域
                with dpg.drawlist(width=self._width, height=self._height) as self._drawlist:
                    pass

            # 设置视口回调
            dpg.set_viewport_resize_callback(self._viewport_resize_callback)

            # 设置键盘回调
            with dpg.handler_registry():
                dpg.add_key_press_handler(callback=self._key_press_callback)
                dpg.add_key_release_handler(callback=self._key_release_callback)
                dpg.add_mouse_click_handler(callback=self._mouse_click_callback)
                dpg.add_mouse_release_handler(callback=self._mouse_release_callback)
                dpg.add_mouse_move_handler(callback=self._mouse_move_callback)
                dpg.add_mouse_wheel_handler(callback=self._mouse_wheel_callback)

            # 显示视口
            dpg.show_viewport()

            # 从容器获取渲染器
            try:
                from ..core.container import container
                self._renderer = container.resolve(VectorFieldRenderer)

                # 如果容器中没有渲染器，则创建它
                if self._renderer is None:
                    self._renderer = VectorFieldRenderer()
                    container.register_singleton(VectorFieldRenderer, self._renderer)
            except Exception as e:
                print(f"[窗口] 获取渲染器失败，创建新实例: {e}")
                self._renderer = VectorFieldRenderer()

            # 设置渲染器的绘图列表
            if self._renderer:
                self._renderer.set_drawlist(self._drawlist)

            # 注册事件处理器
            self._register_event_handlers()

            print("[窗口] 初始化成功")
            return True
        except Exception as e:
            print(f"[窗口] 初始化失败: {e}")
            # 清理已初始化的资源
            self._cleanup_on_failure()
            return False

    def _cleanup_on_failure(self) -> None:
        """在初始化失败时清理资源"""
        try:
            dpg.destroy_context()
        except Exception as e:
            print(f"[窗口] 清理失败资源时出错: {e}")

    def _register_event_handlers(self) -> None:
        """注册事件处理器"""
        # 鼠标点击事件
        mouse_click_handler = FunctionEventHandler(
            self._handle_mouse_click, "WindowMouseClickHandler"
        )
        self._event_handlers[EventType.MOUSE_CLICKED] = mouse_click_handler
        self._event_bus.subscribe(EventType.MOUSE_CLICKED, mouse_click_handler)

        # 鼠标移动事件
        mouse_move_handler = FunctionEventHandler(
            self._handle_mouse_move, "WindowMouseMoveHandler"
        )
        self._event_handlers[EventType.MOUSE_MOVED] = mouse_move_handler
        self._event_bus.subscribe(EventType.MOUSE_MOVED, mouse_move_handler)

        # 鼠标滚轮事件
        mouse_scroll_handler = FunctionEventHandler(
            self._handle_mouse_scroll, "WindowMouseScrollHandler"
        )
        self._event_handlers[EventType.MOUSE_SCROLLED] = mouse_scroll_handler
        self._event_bus.subscribe(EventType.MOUSE_SCROLLED, mouse_scroll_handler)

        # 键盘按下事件
        key_press_handler = FunctionEventHandler(
            self._handle_key_press, "WindowKeyPressHandler"
        )
        self._event_handlers[EventType.KEY_PRESSED] = key_press_handler
        self._event_bus.subscribe(EventType.KEY_PRESSED, key_press_handler)

        # 键盘释放事件
        key_release_handler = FunctionEventHandler(
            self._handle_key_release, "WindowKeyReleaseHandler"
        )
        self._event_handlers[EventType.KEY_RELEASED] = key_release_handler
        self._event_bus.subscribe(EventType.KEY_RELEASED, key_release_handler)

    def _viewport_resize_callback(self):
        """视口大小改变回调"""
        width, height = dpg.get_viewport_width(), dpg.get_viewport_height()
        self._width = width
        self._height = height

        # 更新绘图区域大小
        dpg.configure_item(self._drawlist, width=width, height=height)

        # 更新状态
        self._state_manager.set("viewport_width", width)
        self._state_manager.set("viewport_height", height)

        # 发布窗口大小改变事件
        self._event_bus.publish(Event(
            EventType.VIEW_CHANGED,
            {"width": width, "height": height},
            "Window"
        ))

    def _key_press_callback(self, sender, app_data):
        """键盘按下回调"""
        key = app_data
        self._keys[key] = True

        # 使用input模块处理键盘事件
        input_handler.handle_key_event(None, key, 0, 1, 0)  # 模拟GLFW格式

    def _key_release_callback(self, sender, app_data):
        """键盘释放回调"""
        key = app_data
        self._keys[key] = False

        # 使用input模块处理键盘事件
        input_handler.handle_key_event(None, key, 0, 0, 0)  # 模拟GLFW格式

    def _mouse_click_callback(self, sender, app_data):
        """鼠标点击回调"""
        # app_data might be just button (int) or (button, state) tuple
        if isinstance(app_data, tuple):
            button, state = app_data
        else:
            button = app_data
            state = 1 if self._mouse_pressed else 0

        if state == 1:  # 按下
            self._mouse_pressed = True
            self._mouse_x, self._mouse_y = dpg.get_mouse_pos()
            self._last_mouse_x, self._last_mouse_y = self._mouse_x, self._mouse_y
        else:  # 释放
            self._mouse_pressed = False

        # 使用input模块处理鼠标按钮事件
        input_handler.handle_mouse_button_event(None, button, state, 0)

    def _mouse_release_callback(self, sender, app_data):
        """鼠标释放回调"""
        button = app_data
        self._mouse_pressed = False

        # 使用input模块处理鼠标按钮事件
        input_handler.handle_mouse_button_event(None, button, 0, 0)

    def _mouse_move_callback(self, sender, app_data):
        """鼠标移动回调"""
        self._mouse_x, self._mouse_y = dpg.get_mouse_pos()

        # 使用input模块处理鼠标移动事件
        input_handler.handle_cursor_position_event(None, self._mouse_x, self._mouse_y)

    def _mouse_wheel_callback(self, sender, app_data):
        """鼠标滚轮回调"""
        # app_data might be just yoffset (int) or (xoffset, yoffset) tuple
        if isinstance(app_data, tuple):
            xoffset, yoffset = app_data
        else:
            xoffset = 0
            yoffset = app_data

        self._scroll_y = yoffset

        # 使用input模块处理鼠标滚轮事件
        input_handler.handle_scroll_event(None, xoffset, yoffset)

    def _handle_mouse_click(self, event: Event) -> None:
        """处理鼠标点击事件"""
        # 这里可以添加自定义的鼠标点击处理逻辑
        pass

    def _handle_mouse_move(self, event: Event) -> None:
        """处理鼠标移动事件"""
        if self._mouse_pressed:
            # 计算鼠标移动距离
            dx = self._mouse_x - self._last_mouse_x
            dy = self._mouse_y - self._last_mouse_y

            # 更新相机位置
            cam_speed = 0.1
            cam_x = self._state_manager.get("cam_x", 0.0) - dx * cam_speed
            cam_y = self._state_manager.get("cam_y", 0.0) + dy * cam_speed

            self._state_manager.update({
                "cam_x": cam_x,
                "cam_y": cam_y,
                "view_changed": True
            })

            # 更新最后鼠标位置
            self._last_mouse_x = self._mouse_x
            self._last_mouse_y = self._mouse_y

    def _handle_mouse_scroll(self, event: Event) -> None:
        """处理鼠标滚轮事件"""
        # 获取滚轮偏移量
        xoffset = event.data.get("xoffset", 0)
        yoffset = event.data.get("yoffset", 0)

        # 更新相机缩放
        cam_zoom = self._state_manager.get("cam_zoom", 1.0)
        zoom_speed = 0.1
        cam_zoom -= yoffset * zoom_speed

        # 限制缩放范围
        cam_zoom = max(0.1, min(10.0, cam_zoom))

        self._state_manager.update({
            "cam_zoom": cam_zoom,
            "view_changed": True
        })

    def _handle_key_press(self, event: Event) -> None:
        """处理键盘按下事件"""
        key = event.data.get("key")

        # 处理特定按键 (使用Dear PyGui键码)
        if key == 256:  # ESCAPE
            self.should_close = True
        elif key == 82:  # R
            # 重置视图
            self._event_bus.publish(Event(
                EventType.RESET_VIEW,
                {},
                "Window"
            ))
        elif key == 71:  # G
            # 切换网格显示
            self._event_bus.publish(Event(
                EventType.TOGGLE_GRID,
                {},
                "Window"
            ))
        elif key == 67:  # C
            # 清空网格
            self._event_bus.publish(Event(
                EventType.CLEAR_GRID,
                {},
                "Window"
            ))

    def _handle_key_release(self, event: Event) -> None:
        """处理键盘释放事件"""
        pass

    def handle(self, event: Event) -> None:
        """处理事件"""
        if event.type == EventType.APP_INITIALIZED:
            if "width" in event.data and "height" in event.data:
                self._width = event.data["width"]
                self._height = event.data["height"]

    @property
    def should_close(self) -> bool:
        """获取窗口是否应该关闭"""
        return self._should_close

    @should_close.setter
    def should_close(self, value: bool) -> None:
        """设置窗口是否应该关闭"""
        self._should_close = value

    def close(self) -> None:
        """关闭窗口"""
        self.should_close = True

    def update(self) -> None:
        """更新窗口状态"""
        # 处理Dear PyGui事件
        dpg.render_dearpygui_frame()

    def render(self, grid: np.ndarray) -> None:
        """渲染内容"""
        if not self._renderer:
            return

        # 清除绘图区域
        dpg.delete_item(self._drawlist, children_only=True)

        # 获取相机参数
        cam_x = self._state_manager.get("cam_x", 0.0)
        cam_y = self._state_manager.get("cam_y", 0.0)
        cam_zoom = self._state_manager.get("cam_zoom", 1.0)

        # 获取视口大小
        viewport_width = self._state_manager.get("viewport_width", self._width)
        viewport_height = self._state_manager.get("viewport_height", self._height)

        # 渲染背景
        self._renderer.render_background()

        # 渲染标记（如果有）
        try:
            self._renderer.render_markers(
                cell_size=self._config_manager.get("cell_size", 1.0),
                cam_x=cam_x,
                cam_y=cam_y,
                cam_zoom=cam_zoom,
                viewport_width=viewport_width,
                viewport_height=viewport_height
            )
        except Exception:
            # 渲染标记不是关键路径，忽略错误以保证主渲染继续
            pass

        # 渲染向量场
        self._renderer.render_vector_field(
            grid,
            cell_size=self._config_manager.get("cell_size", 1.0),
            cam_x=cam_x,
            cam_y=cam_y,
            cam_zoom=cam_zoom,
            viewport_width=viewport_width,
            viewport_height=viewport_height
        )

        # 渲染网格
        self._renderer.render_grid(
            grid,
            cell_size=self._config_manager.get("cell_size", 1.0),
            cam_x=cam_x,
            cam_y=cam_y,
            cam_zoom=cam_zoom,
            viewport_width=viewport_width,
            viewport_height=viewport_height
        )

    def cleanup(self) -> None:
        """清理资源"""
        dpg.destroy_context()
        print("[窗口] 资源清理完成")
