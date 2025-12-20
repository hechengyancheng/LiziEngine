"""
窗口管理模块 - 提供窗口管理功能
支持OpenGL窗口创建和事件处理
"""
import glfw
import numpy as np
from OpenGL.GL import *
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
            # 初始化GLFW
            if not glfw.init():
                print("[窗口] GLFW初始化失败")
                return False

            # 配置GLFW
            glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

            # 创建窗口
            self._window = glfw.create_window(self._width, self._height, self._title, None, None)

            if not self._window:
                print("[窗口] 窗口创建失败")
                glfw.terminate()
                return False

            # 设置窗口上下文
            glfw.make_context_current(self._window)

            # 设置窗口回调
            glfw.set_framebuffer_size_callback(self._window, self._framebuffer_size_callback)
            glfw.set_key_callback(self._window, self._key_callback)
            glfw.set_mouse_button_callback(self._window, self._mouse_button_callback)
            glfw.set_cursor_pos_callback(self._window, self._cursor_pos_callback)
            glfw.set_scroll_callback(self._window, self._scroll_callback)

            # 初始化OpenGL
            self._init_opengl()

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
            if self._window:
                glfw.destroy_window(self._window)
                self._window = None
            glfw.terminate()
        except Exception as e:
            print(f"[窗口] 清理失败资源时出错: {e}")

    def _init_opengl(self) -> None:
        """初始化OpenGL"""
        # 启用深度测试
        glEnable(GL_DEPTH_TEST)

        # 设置视口
        glViewport(0, 0, self._width, self._height)

        # 设置清除颜色
        glClearColor(0.1, 0.1, 0.1, 1.0)

        # 启用抗锯齿
        if self._config_manager.get("antialiasing", True):
            glEnable(GL_LINE_SMOOTH)
            glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

    def _register_event_handlers(self) -> None:
        """注册事件处理器"""
        # 鼠标点击事件
        self._event_handlers[EventType.MOUSE_CLICKED] = FunctionEventHandler(
            self._handle_mouse_click, "WindowMouseClickHandler"
        )

        # 鼠标移动事件
        self._event_handlers[EventType.MOUSE_MOVED] = FunctionEventHandler(
            self._handle_mouse_move, "WindowMouseMoveHandler"
        )

        # 鼠标滚轮事件
        self._event_handlers[EventType.MOUSE_SCROLLED] = FunctionEventHandler(
            self._handle_mouse_scroll, "WindowMouseScrollHandler"
        )

        # 键盘按下事件
        self._event_handlers[EventType.KEY_PRESSED] = FunctionEventHandler(
            self._handle_key_press, "WindowKeyPressHandler"
        )

        # 键盘释放事件
        self._event_handlers[EventType.KEY_RELEASED] = FunctionEventHandler(
            self._handle_key_release, "WindowKeyReleaseHandler"
        )

    def _framebuffer_size_callback(self, window, width, height):
        """窗口大小改变回调"""
        self._width = width
        self._height = height

        # 更新OpenGL视口
        glViewport(0, 0, width, height)

        # 更新状态
        self._state_manager.set("viewport_width", width)
        self._state_manager.set("viewport_height", height)

        # 发布窗口大小改变事件
        self._event_bus.publish(Event(
            EventType.VIEW_CHANGED,
            {"width": width, "height": height},
            "Window"
        ))

    def _key_callback(self, window, key, scancode, action, mods):
        """键盘事件回调"""
        # 更新键盘状态
        if action == glfw.PRESS:
            self._keys[key] = True
        elif action == glfw.RELEASE:
            self._keys[key] = False
            
        # 使用input模块处理键盘事件
        input_handler.handle_key_event(window, key, scancode, action, mods)

    def _mouse_button_callback(self, window, button, action, mods):
        """鼠标按钮事件回调"""
        if action == glfw.PRESS:
            self._mouse_pressed = True
            self._last_mouse_x, self._last_mouse_y = glfw.get_cursor_pos(window)
        elif action == glfw.RELEASE:
            self._mouse_pressed = False
            
        # 使用input模块处理鼠标按钮事件
        input_handler.handle_mouse_button_event(window, button, action, mods)

    def _cursor_pos_callback(self, window, xpos, ypos):
        """鼠标位置回调"""
        self._mouse_x = xpos
        self._mouse_y = ypos
        
        # 使用input模块处理鼠标移动事件
        input_handler.handle_cursor_position_event(window, xpos, ypos)

    def _scroll_callback(self, window, xoffset, yoffset):
        """鼠标滚轮回调"""
        # 更新Window类的滚轮状态
        self._scroll_y = yoffset
        # 使用input模块处理鼠标滚轮事件
        input_handler.handle_scroll_event(window, xoffset, yoffset)

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

        # 处理特定按键
        if key == glfw.KEY_ESCAPE:
            self.should_close = True
        elif key == glfw.KEY_R:
            # 重置视图
            self._event_bus.publish(Event(
                EventType.RESET_VIEW,
                {},
                "Window"
            ))
        elif key == glfw.KEY_G:
            # 切换网格显示
            self._event_bus.publish(Event(
                EventType.TOGGLE_GRID,
                {},
                "Window"
            ))
        elif key == glfw.KEY_C:
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
        return self._should_close or glfw.window_should_close(self._window)

    @should_close.setter
    def should_close(self, value: bool) -> None:
        """设置窗口是否应该关闭"""
        self._should_close = value

    def close(self) -> None:
        """关闭窗口"""
        self.should_close = True

    def update(self) -> None:
        """更新窗口状态"""
        # 更新GLFW事件
        glfw.poll_events()

    def render(self, grid: np.ndarray) -> None:
        """渲染内容"""
        if not self._window or not self._renderer:
            return

        # 清除屏幕
        self._renderer.render_background()

        # 获取相机参数
        cam_x = self._state_manager.get("cam_x", 0.0)
        cam_y = self._state_manager.get("cam_y", 0.0)
        cam_zoom = self._state_manager.get("cam_zoom", 1.0)

        # 获取视口大小
        viewport_width = self._state_manager.get("viewport_width", self._width)
        viewport_height = self._state_manager.get("viewport_height", self._height)

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



        # 交换缓冲区
        glfw.swap_buffers(self._window)

    def cleanup(self) -> None:
        """清理资源"""
        if self._window:
            glfw.destroy_window(self._window)
            self._window = None

        glfw.terminate()
        print("[窗口] 资源清理完成")
