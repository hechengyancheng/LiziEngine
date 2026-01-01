"""
GUI管理器模块 - 提供Dear PyGui界面管理功能
支持嵌入式OpenGL渲染窗口
"""
import dearpygui.dearpygui as dpg
import numpy as np
from typing import Optional, Callable, Dict, Any, Tuple
import sys
import os
# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lizi_engine.core.config import config_manager
from lizi_engine.core.events import Event, EventType, event_bus, EventHandler, FunctionEventHandler
from lizi_engine.core.state import state_manager
from lizi_engine.graphics.renderer import VectorFieldRenderer
from lizi_engine.input import input_handler
from lizi_engine.compute.vector_field import vector_calculator
from lizi_engine.gui.opengl_embedder import OpenGLEmbedder

class GUIManager(EventHandler):
    """GUI管理器 - 使用Dear PyGui管理界面"""

    def __init__(self, title: str = "LiziEngine GUI", width: int = 1200, height: int = 800):
        self._event_bus = event_bus
        self._state_manager = state_manager
        self._config_manager = config_manager

        # 窗口属性
        self._title = title
        self._width = width
        self._height = height

        # GUI状态
        self._initialized = False
        self._should_close = False

        # OpenGL嵌入相关
        self._opengl_embedder = None
        self._render_width = 800
        self._render_height = 600

        # 渲染器
        self._renderer = None

        # 网格数据
        self._grid = None

        # 插件管理器
        self._controller = None
        self._marker_system = None
        self._ui_manager = None

        # 订阅事件
        self._event_bus.subscribe(EventType.APP_INITIALIZED, self)

    def initialize(self, grid: np.ndarray, controller, marker_system, ui_manager) -> bool:
        """初始化GUI"""
        try:
            # 保存引用
            self._grid = grid
            self._controller = controller
            self._marker_system = marker_system
            self._ui_manager = ui_manager

            # 初始化Dear PyGui
            dpg.create_context()
            dpg.create_viewport(title=self._title, width=self._width, height=self._height)

            # 设置Dear PyGui主题
            self._setup_theme()

            # 创建纹理注册表
            with dpg.texture_registry() as self._texture_registry:
                pass

            # 创建主界面
            self._create_main_interface()

            # 设置回调
            self._setup_callbacks()

            # 显示视口
            dpg.show_viewport()

            # 初始化OpenGL嵌入
            self._init_opengl_embedding()

            # 获取渲染器
            try:
                from ..core.container import container
                self._renderer = container.resolve(VectorFieldRenderer)
                if self._renderer is None:
                    self._renderer = VectorFieldRenderer()
                    container.register_singleton(VectorFieldRenderer, self._renderer)
            except Exception as e:
                print(f"[GUI] 获取渲染器失败，创建新实例: {e}")
                self._renderer = VectorFieldRenderer()

            self._initialized = True
            print("[GUI] 初始化成功")
            return True

        except Exception as e:
            print(f"[GUI] 初始化失败: {e}")
            self._cleanup_on_failure()
            return False

    def _setup_theme(self):
        """设置Dear PyGui主题"""
        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll):
                # 设置全局颜色
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (30, 30, 30))
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (25, 25, 25))
                dpg.add_theme_color(dpg.mvThemeCol_Button, (60, 120, 180))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (80, 140, 200))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (100, 160, 220))
                dpg.add_theme_color(dpg.mvThemeCol_Text, (220, 220, 220))

        dpg.bind_theme(global_theme)

    def _create_main_interface(self):
        """创建主界面"""
        with dpg.window(label=self._title, width=self._width, height=self._height,
                       no_title_bar=True, no_resize=True, no_move=True) as self._main_window:

            # 顶部工具栏
            with dpg.group(horizontal=True):
                dpg.add_button(label="重置视图", callback=self._reset_view_callback)
                dpg.add_button(label="切换网格", callback=self._toggle_grid_callback)
                dpg.add_button(label="清空网格", callback=self._clear_grid_callback)
                dpg.add_button(label="生成切线模式", callback=self._generate_tangential_callback)
                dpg.add_button(label="切换实时更新", callback=self._toggle_update_callback)

            dpg.add_separator()

            # 主内容区域
            with dpg.group(horizontal=True):

                # 左侧控制面板
                with dpg.child_window(width=250, height=-1) as self._control_panel:
                    self._create_control_panel()

                dpg.add_separator()

                # 右侧渲染区域
                with dpg.child_window(width=-1, height=-1) as self._render_panel:
                    # OpenGL渲染区域将在这里嵌入
                    dpg.add_text("OpenGL渲染区域")
                    # 这里将放置OpenGL纹理

    def _create_control_panel(self):
        """创建控制面板"""
        dpg.add_text("向量场控制", color=(150, 200, 255))
        dpg.add_separator()

        # 相机控制
        dpg.add_text("相机控制:")
        with dpg.group(horizontal=True):
            dpg.add_button(label="重置相机", callback=self._reset_camera_callback)
            dpg.add_button(label="居中视图", callback=self._center_view_callback)

        # 缩放控制
        dpg.add_text("缩放:")
        dpg.add_slider_float(label="缩放", default_value=1.0, min_value=0.1, max_value=10.0,
                           callback=self._zoom_callback, tag="zoom_slider")

        dpg.add_separator()

        # 向量场参数
        dpg.add_text("向量场参数:", color=(150, 200, 255))
        dpg.add_slider_float(label="向量缩放", default_value=1.0, min_value=0.1, max_value=5.0,
                           callback=self._vector_scale_callback, tag="vector_scale_slider")
        dpg.add_slider_float(label="线条宽度", default_value=1.0, min_value=0.5, max_value=3.0,
                           callback=self._line_width_callback, tag="line_width_slider")

        dpg.add_separator()

        # 标记系统
        dpg.add_text("标记系统:", color=(150, 200, 255))
        dpg.add_button(label="添加标记", callback=self._add_marker_callback)
        dpg.add_button(label="清空标记", callback=self._clear_markers_callback)

        dpg.add_separator()

        # 状态信息
        dpg.add_text("状态信息:", color=(150, 200, 255))
        dpg.add_text("FPS: 0", tag="fps_text")
        dpg.add_text("网格大小: 64x64", tag="grid_size_text")
        dpg.add_text("标记数量: 0", tag="marker_count_text")

    def _setup_callbacks(self):
        """设置回调函数"""
        # 键盘回调
        with dpg.handler_registry():
            dpg.add_key_press_handler(callback=self._key_press_callback)
            dpg.add_mouse_wheel_handler(callback=self._mouse_wheel_callback)

    def _init_opengl_embedding(self):
        """初始化OpenGL嵌入"""
        # 创建OpenGL嵌入器
        self._opengl_embedder = OpenGLEmbedder(self._render_width, self._render_height)

        # 初始化OpenGL嵌入器
        if not self._opengl_embedder.initialize():
            print("[GUI] OpenGL嵌入器初始化失败")
            return

        # 在渲染面板中添加OpenGL图像
        # 移除占位符文本
        dpg.delete_item(dpg.last_item())

        # 添加OpenGL渲染图像
        dpg_image = self._opengl_embedder.get_dpg_image(parent=self._render_panel)
        if dpg_image:
            # 设置图像位置和大小
            dpg.set_item_pos(dpg_image, (10, 10))
            # 这里可以根据需要调整图像大小

    def _reset_view_callback(self):
        """重置视图回调"""
        if self._controller:
            self._controller.reset_view()

    def _toggle_grid_callback(self):
        """切换网格显示回调"""
        if self._controller:
            self._controller.toggle_grid()

    def _clear_grid_callback(self):
        """清空网格回调"""
        if self._controller:
            self._controller.clear_grid()

    def _generate_tangential_callback(self):
        """生成切线模式回调"""
        if self._grid is not None:
            center = (self._grid.shape[1] // 2, self._grid.shape[0] // 2)
            vector_calculator.create_tangential_pattern(self._grid, center=center, radius=50, magnitude=1.0)
            self._state_manager.update({"view_changed": True, "grid_updated": True})

    def _toggle_update_callback(self):
        """切换实时更新回调"""
        if self._ui_manager:
            self._ui_manager.enable_update = not self._ui_manager.enable_update

    def _reset_camera_callback(self):
        """重置相机回调"""
        self._state_manager.update({
            "cam_x": 0.0,
            "cam_y": 0.0,
            "cam_zoom": 1.0,
            "view_changed": True
        })

    def _center_view_callback(self):
        """居中视图回调"""
        if self._grid is not None:
            center_x = (self._grid.shape[1] * self._config_manager.get("cell_size", 1.0)) / 2.0
            center_y = (self._grid.shape[0] * self._config_manager.get("cell_size", 1.0)) / 2.0
            self._state_manager.update({
                "cam_x": center_x,
                "cam_y": center_y,
                "view_changed": True
            })

    def _zoom_callback(self, sender, app_data):
        """缩放回调"""
        self._state_manager.update({
            "cam_zoom": app_data,
            "view_changed": True
        })

    def _vector_scale_callback(self, sender, app_data):
        """向量缩放回调"""
        self._config_manager.set("vector_scale", app_data)

    def _line_width_callback(self, sender, app_data):
        """线条宽度回调"""
        self._config_manager.set("line_width", app_data)

    def _add_marker_callback(self):
        """添加标记回调"""
        if self._marker_system:
            # 在随机位置添加标记
            x = np.random.randint(0, self._grid.shape[1])
            y = np.random.randint(0, self._grid.shape[0])
            self._marker_system.add_marker(x, y)

    def _clear_markers_callback(self):
        """清空标记回调"""
        if self._marker_system:
            self._marker_system.clear_markers()

    def _key_press_callback(self, sender, app_data):
        """键盘按下回调"""
        key = app_data

        # 处理特定按键
        if key == 256:  # ESCAPE
            self._should_close = True
        elif key == 82:  # R
            self._reset_view_callback()
        elif key == 71:  # G
            self._toggle_grid_callback()
        elif key == 67:  # C
            self._clear_grid_callback()
        elif key == 32:  # SPACE
            self._generate_tangential_callback()
        elif key == 85:  # U
            self._toggle_update_callback()

    def _mouse_wheel_callback(self, sender, app_data):
        """鼠标滚轮回调"""
        # 更新相机缩放
        cam_zoom = self._state_manager.get("cam_zoom", 1.0)
        zoom_speed = 0.1
        cam_zoom -= app_data * zoom_speed
        cam_zoom = max(0.1, min(10.0, cam_zoom))

        self._state_manager.update({
            "cam_zoom": cam_zoom,
            "view_changed": True
        })

        # 更新缩放滑块
        dpg.set_value("zoom_slider", cam_zoom)

    def _cleanup_on_failure(self):
        """初始化失败时的清理"""
        try:
            dpg.destroy_context()
        except Exception as e:
            print(f"[GUI] 清理失败资源时出错: {e}")

    @property
    def should_close(self) -> bool:
        """获取是否应该关闭"""
        return self._should_close or dpg.is_key_pressed(dpg.mvKey_Escape)

    def update(self) -> None:
        """更新GUI"""
        if not self._initialized:
            return

        # 处理Dear PyGui事件
        dpg.render_dearpygui_frame()

        # 更新状态信息
        self._update_status_info()

    def _update_status_info(self):
        """更新状态信息显示"""
        # 更新FPS（简化实现）
        fps = 60  # 这里应该计算实际FPS
        dpg.set_value("fps_text", f"FPS: {fps}")

        # 更新网格大小
        if self._grid is not None:
            grid_size = f"网格大小: {self._grid.shape[1]}x{self._grid.shape[0]}"
            dpg.set_value("grid_size_text", grid_size)

        # 更新标记数量
        if self._marker_system:
            markers = self._marker_system.get_markers()
            marker_count = f"标记数量: {len(markers)}"
            dpg.set_value("marker_count_text", marker_count)

    def render(self) -> None:
        """渲染GUI内容"""
        if not self._initialized or not self._renderer or self._grid is None or not self._opengl_embedder:
            return

        # 检查OpenGL嵌入器是否已初始化
        if not hasattr(self._opengl_embedder, '_initialized') or not self._opengl_embedder._initialized:
            return

        # 开始OpenGL渲染到帧缓冲
        self._opengl_embedder.begin_render()

        # 获取相机参数
        cam_x = self._state_manager.get("cam_x", 0.0)
        cam_y = self._state_manager.get("cam_y", 0.0)
        cam_zoom = self._state_manager.get("cam_zoom", 1.0)

        # 渲染背景
        self._renderer.render_background()

        # 渲染标记
        try:
            self._renderer.render_markers(
                cell_size=self._config_manager.get("cell_size", 1.0),
                cam_x=cam_x,
                cam_y=cam_y,
                cam_zoom=cam_zoom,
                viewport_width=self._render_width,
                viewport_height=self._render_height
            )
        except Exception:
            # 渲染标记不是关键路径，忽略错误
            pass

        # 渲染向量场
        self._renderer.render_vector_field(
            self._grid,
            cell_size=self._config_manager.get("cell_size", 1.0),
            cam_x=cam_x,
            cam_y=cam_y,
            cam_zoom=cam_zoom,
            viewport_width=self._render_width,
            viewport_height=self._render_height
        )

        # 渲染网格
        self._renderer.render_grid(
            self._grid,
            cell_size=self._config_manager.get("cell_size", 1.0),
            cam_x=cam_x,
            cam_y=cam_y,
            cam_zoom=cam_zoom,
            viewport_width=self._render_width,
            viewport_height=self._render_height
        )

        # 结束OpenGL渲染并更新纹理
        self._opengl_embedder.end_render()

    def cleanup(self) -> None:
        """清理GUI资源"""
        if not self._initialized:
            return

        try:
            # 清理OpenGL嵌入资源
            self._cleanup_opengl_embedding()

            # 销毁Dear PyGui上下文
            dpg.destroy_context()

            self._initialized = False
            print("[GUI] 资源清理完成")
        except Exception as e:
            print(f"[GUI] 清理资源时出错: {e}")

    def _cleanup_opengl_embedding(self):
        """清理OpenGL嵌入资源"""
        # 这里将在后续步骤中实现
        pass

    def handle(self, event: Event) -> None:
        """处理事件"""
        if event.type == EventType.APP_INITIALIZED:
            # 处理应用初始化事件
            pass

# 全局GUI管理器实例
gui_manager = GUIManager()
