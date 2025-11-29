
"""
工具栏模块 - 提供用户界面工具栏功能
"""
import numpy as np
import threading
import time
from typing import Dict, Any, Optional, Tuple, List
from core.config import config_manager
from core.events import EventBus, Event, EventType, EventHandler
from core.state import state_manager

class ToolbarEventHandler(EventHandler):
    """工具栏事件处理器"""

    def __init__(self, toolbar):
        self.toolbar = toolbar

    def handle(self, event: Event) -> None:
        """处理事件"""
        if event.type == EventType.MOUSE_CLICKED:
            self.toolbar._on_mouse_clicked(event)
        elif event.type == EventType.KEY_PRESSED:
            self.toolbar._on_key_pressed(event)
        elif event.type == EventType.GRID_UPDATED:
            self.toolbar._on_grid_updated(event)

class Toolbar:
    """工具栏基类"""
    def __init__(self):
        # 使用全局事件总线实例，而不是创建新的实例
        from core.events import event_bus
        self._event_bus = event_bus
        self._state_manager = state_manager

        # 工具栏状态
        self._brush_size = config_manager.get("vector_field.default_brush_size", 1)
        self._magnitude = config_manager.get("vector_field.default_vector_length", 1.0)
        self._reverse_vector = config_manager.get("vector_field.reverse_vector", False)
        self._show_grid = config_manager.get("rendering.show_grid", True)
        self._vector_mode = self._state_manager.get("vector_mode", 0)  # 0=辐射状, 1=单一方向, 2=顺时针旋转

        # UI状态
        self._show_toolbar = True

        # 注册事件处理器
        self._event_handler = ToolbarEventHandler(self)
        self._event_bus.subscribe(EventType.MOUSE_CLICKED, self._event_handler)
        self._event_bus.subscribe(EventType.KEY_PRESSED, self._event_handler)
        self._event_bus.subscribe(EventType.GRID_UPDATED, self._event_handler)

    def _on_mouse_clicked(self, event: Event) -> None:
        """处理鼠标点击事件"""
        # 子类实现具体逻辑
        pass

    def _on_key_pressed(self, event: Event) -> None:
        """处理键盘按键事件"""
        # 子类实现具体逻辑
        pass

    def _on_grid_updated(self, event: Event) -> None:
        """处理网格更新事件"""
        # 子类实现具体逻辑
        pass

    def update_vector(self, grid_x: int, grid_y: int, vector: Tuple[float, float]) -> None:
        """更新指定位置的向量"""
        if self._reverse_vector:
            vector = (-vector[0], -vector[1])

        # 发布向量更新事件
        self._event_bus.publish(Event(
            EventType.VECTOR_UPDATED if hasattr(EventType, 'VECTOR_UPDATED') else EventType.GRID_UPDATED,
            {"grid_x": grid_x, "grid_y": grid_y, "vector": vector},
            "Toolbar"
        ))

    def clear_grid(self) -> None:
        """清空网格"""
        # 发布网格清空事件
        self._event_bus.publish(Event(
            EventType.GRID_CLEARED,
            {},
            "Toolbar"
        ))

    def reset_view(self) -> None:
        """重置视图"""
        # 发布视图重置事件
        self._event_bus.publish(Event(
            EventType.VIEW_RESET,
            {},
            "Toolbar"
        ))

    def toggle_grid(self) -> None:
        """切换网格显示"""
        self._show_grid = not self._show_grid
        config_manager.set("rendering.show_grid", self._show_grid)

        # 更新状态
        self._state_manager.set("show_grid", self._show_grid)

        # 发布网格显示切换事件
        self._event_bus.publish(Event(
            EventType.GRID_TOGGLED if hasattr(EventType, 'GRID_TOGGLED') else EventType.VIEW_CHANGED,
            {"show": self._show_grid},
            "Toolbar"
        ))

    def set_brush_size(self, size: int) -> None:
        """设置画笔大小"""
        self._brush_size = max(1, min(10, size))
        config_manager.set("vector_field.default_brush_size", self._brush_size)

        # 更新状态
        self._state_manager.set("brush_size", self._brush_size)

    def set_magnitude(self, magnitude: float) -> None:
        """设置向量大小"""
        self._magnitude = max(0.1, min(5.0, magnitude))
        config_manager.set("vector_field.default_vector_length", self._magnitude)

        # 更新状态
        self._state_manager.set("magnitude", self._magnitude)

    def toggle_reverse_vector(self) -> None:
        """切换向量方向"""
        self._reverse_vector = not self._reverse_vector
        config_manager.set("vector_field.reverse_vector", self._reverse_vector)

        # 更新状态
        self._state_manager.set("reverse_vector", self._reverse_vector)

    def toggle_vector_mode(self) -> None:
        """切换向量模式（辐射状/单一方向/顺时针旋转）"""
        self._vector_mode = (self._vector_mode + 1) % 3
        
        # 更新状态
        self._state_manager.set("vector_mode", self._vector_mode)
        
        # 发布事件
        self._event_bus.publish(Event(
            EventType.SET_VECTOR_MODE,
            {"mode": self._vector_mode},
            "Toolbar"
        ))
        
    def set_vector_mode(self, mode: int) -> None:
        """设置向量模式"""
        if 0 <= mode <= 2:
            self._vector_mode = mode
            
            # 更新状态
            self._state_manager.set("vector_mode", self._vector_mode)
            
            # 发布事件
            self._event_bus.publish(Event(
                EventType.SET_VECTOR_MODE,
                {"mode": self._vector_mode},
                "Toolbar"
            ))

    def save_grid(self, file_path: str) -> None:
        """保存网格"""
        # 发布网格保存事件
        self._event_bus.publish(Event(
            EventType.GRID_SAVED,
            {"file_path": file_path},
            "Toolbar"
        ))

    def load_grid(self, file_path: str) -> None:
        """加载网格"""
        # 发布网格加载事件
        self._event_bus.publish(Event(
            EventType.GRID_LOADED,
            {"file_path": file_path},
            "Toolbar"
        ))

    def toggle_toolbar(self) -> None:
        """切换工具栏显示"""
        self._show_toolbar = not self._show_toolbar

        # 更新状态
        self._state_manager.set("show_toolbar", self._show_toolbar)

        # 发布工具栏显示切换事件
        self._event_bus.publish(Event(
            EventType.TOOLBAR_TOGGLED if hasattr(EventType, 'TOOLBAR_TOGGLED') else EventType.VIEW_CHANGED,
            {"show": self._show_toolbar},
            "Toolbar"
        ))

    def cleanup(self) -> None:
        """清理工具栏资源"""
        # 取消事件订阅
        self._event_bus.unsubscribe(EventType.MOUSE_CLICKED, self._event_handler)
        self._event_bus.unsubscribe(EventType.KEY_PRESSED, self._event_handler)
        self._event_bus.unsubscribe(EventType.GRID_UPDATED, self._event_handler)

class ImGuiToolbar(Toolbar):
    """基于ImGui的工具栏实现"""
    def __init__(self):
        super().__init__()
        self._imgui_available = False

        # 尝试导入ImGui
        self._imgui = None
        self._glfw_impl = None
        self._imgui_available = False
        
        try:
            import imgui
            from imgui.integrations.glfw import GlfwRenderer
            self._imgui = imgui
            self._GlfwRenderer = GlfwRenderer  # 保存类引用
            self._imgui_available = True
            print("[工具栏] ImGui导入成功")
        except ImportError as e:
            print(f"[工具栏] ImGui导入失败: {e}")
            print("[工具栏] 将使用键盘快捷键和鼠标操作")

    def initialize(self, window) -> bool:
        """初始化ImGui"""
        if not self._imgui_available:
            return False

        try:
            # 初始化ImGui
            self._imgui.create_context()
            self._glfw_impl = self._GlfwRenderer(window)

            # 发布工具栏初始化事件
            self._event_bus.publish(Event(
                EventType.TOOLBAR_INITIALIZED if hasattr(EventType, 'TOOLBAR_INITIALIZED') else EventType.VIEW_CHANGED,
                {},
                "ImGuiToolbar"
            ))

            return True
        except Exception as e:
            print(f"[工具栏] ImGui初始化失败: {e}")
            return False

    def render(self) -> None:
        """渲染工具栏"""
        if not self._imgui_available or not self._show_toolbar:
            return

        # 开始新帧
        self._glfw_impl.new_frame()

        # 创建主窗口
        self._imgui.begin("LiziEngine 控制面板", True)

        # 网格操作
        if self._imgui.collapsing_header("网格操作"):
            if self._imgui.button("清空网格"):
                self.clear_grid()

            if self._imgui.button("重置视图"):
                self.reset_view()

            if self._imgui.checkbox("显示网格", self._show_grid):
                self.toggle_grid()

        # 画笔设置
        if self._imgui.collapsing_header("画笔设置"):
            changed, self._brush_size = self._imgui.slider_int("画笔大小", self._brush_size, 1, 10)
            if changed:
                self.set_brush_size(self._brush_size)

            changed, self._magnitude = self._imgui.slider_float("向量大小", self._magnitude, 0.1, 5.0)
            if changed:
                self.set_magnitude(self._magnitude)

            changed, self._reverse_vector = self._imgui.checkbox("反转向量", self._reverse_vector)
            if changed:
                self.toggle_reverse_vector()
                
            # 向量模式下拉菜单
            mode_names = ["辐射状模式", "单一方向模式", "顺时针旋转模式"]
            changed, current_mode = self._imgui.combo("向量模式", self._vector_mode, mode_names)
            if changed:
                self.set_vector_mode(current_mode)

        # 文件操作
        if self._imgui.collapsing_header("文件操作"):
            if self._imgui.button("保存网格"):
                # 这里应该显示文件保存对话框
                self.save_grid("grid_save.npy")

            if self._imgui.button("加载网格"):
                # 这里应该显示文件打开对话框
                self.load_grid("grid_save.npy")

        # 结束窗口
        self._imgui.end()

        # 渲染ImGui
        self._imgui.render()
        self._glfw_impl.render()

    def _on_mouse_clicked(self, event: Event) -> None:
        """处理鼠标点击事件"""
        if not self._imgui_available:
            return

        # 检查ImGui是否捕获了鼠标
        if self._imgui.is_any_item_hovered() or self._imgui.is_any_window_hovered():
            return

        # 获取事件数据
        button = event.data.get("button")
        action = event.data.get("action")
        grid_x = event.data.get("grid_x", 0)
        grid_y = event.data.get("grid_y", 0)

        # 只处理左键按下事件
        if button == 0 and action == 1:  # GLFW_MOUSE_BUTTON_LEFT and GLFW_PRESS
            # 获取当前网格数据
            from core.app import app_core
            grid = app_core.grid_manager.grid

            if grid is not None and 0 <= grid_y < grid.shape[0] and 0 <= grid_x < grid.shape[1]:
                # 获取当前位置的现有向量
                current_vx = float(grid[grid_y, grid_x, 0])
                current_vy = float(grid[grid_y, grid_x, 1])

                # 计算新向量 - 基于现有向量调整大小
                if self._reverse_vector:
                    # 如果需要反转，则反转现有向量并调整大小
                    new_vx = -current_vx * self._magnitude
                    new_vy = -current_vy * self._magnitude
                else:
                    # 否则，保持方向，只调整大小
                    current_length = np.sqrt(current_vx**2 + current_vy**2)
                    if current_length > 0:
                        # 保持方向，调整大小
                        new_vx = (current_vx / current_length) * self._magnitude
                        new_vy = (current_vy / current_length) * self._magnitude
                    else:
                        # 如果是零向量，则创建一个默认向量
                        new_vx = self._magnitude
                        new_vy = 0

                # 使用统一接口更新向量
                self.update_vector(grid_x, grid_y, (new_vx, new_vy))

    def _on_key_pressed(self, event: Event) -> None:
        """处理键盘按键事件"""
        if not self._imgui_available:
            return

        # 检查ImGui是否捕获了键盘
        if self._imgui.is_any_item_active() or self._imgui.is_any_window_focused():
            return

        # 获取按键代码
        key = event.data.get("key")

        # 处理特定按键
        if key == 71:  # G键
            self.toggle_grid()
        elif key == 82:  # R键
            self.reset_view()
        elif key == 67:  # C键
            self.clear_grid()
        elif key == 84:  # T键
            self.toggle_toolbar()

    def cleanup(self) -> None:
        """清理ImGui工具栏资源"""
        super().cleanup()

        if self._glfw_impl is not None:
            self._glfw_impl.shutdown()
            self._glfw_impl = None

# 全局工具栏实例
toolbar = None

def create_toolbar(toolbar_type: str = "auto") -> Toolbar:
    """创建工具栏实例"""
    global toolbar

    if toolbar_type.lower() == "imgui":
        toolbar = ImGuiToolbar()
    elif toolbar_type.lower() == "simple":
        from .simple_toolbar import SimpleToolbar
        toolbar = SimpleToolbar()
    else:  # auto模式，根据ImGui是否可用自动选择
        try:
            # 尝试导入ImGui检查是否可用
            import imgui
            toolbar = ImGuiToolbar()
        except ImportError:
            from .simple_toolbar import SimpleToolbar
            toolbar = SimpleToolbar()

    return toolbar

def get_toolbar() -> Optional[Toolbar]:
    """获取当前工具栏实例"""
    return toolbar
