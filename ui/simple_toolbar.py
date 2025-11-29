
"""
简单工具栏实现 - 不依赖ImGui，使用键盘快捷键和鼠标操作
"""
import numpy as np
from typing import Dict, Any, Optional, Tuple
from ui.toolbar import Toolbar
from core.events import EventBus, Event, EventType, FunctionEventHandler
from core.state import state_manager

class SimpleToolbar(Toolbar):
    """简单工具栏实现，不依赖ImGui"""
    def __init__(self):
        super().__init__()
        self._show_help = False
        # 初始化工具栏属性 - 从配置中获取初始值
        from core.config import config_manager
        self._brush_size = config_manager.get("vector_field.default_brush_size", 1)
        self._magnitude = config_manager.get("vector_field.default_vector_length", 1.0)
        self._show_grid = config_manager.get("rendering.show_grid", True)
        self._reverse_vector = config_manager.get("vector_field.reverse_vector", False)
        
        # 使用全局事件总线实例
        from core.events import event_bus
        self._event_bus = event_bus
        
        print("[工具栏] 使用简单工具栏（键盘快捷键和鼠标操作）")

    def render(self) -> None:
        """渲染工具栏（在控制台输出帮助信息）"""
        if self._show_help:
            self._print_help()
            self._show_help = False

    def _print_help(self) -> None:
        """打印帮助信息"""
        print("=== LiziEngine 控制面板 ===")
        print("快捷键:")
        print("  G - 切换网格显示")
        print("  R - 重置视图")
        print("  C - 清空网格")
        print("  T - 切换工具栏显示")
        print("  H - 显示此帮助信息")
        print("  +/- - 增加/减少画笔大小")
        print("  </> - 增加/减少向量大小")
        print("  V - 切换向量方向")
        print("鼠标操作:")
        print("  左键点击 - 在网格上绘制向量")
        print("  滚轮 - 缩放视图")
        print("当前设置:")
        print(f"  画笔大小: {self._brush_size}")
        print(f"  向量大小: {self._magnitude}")
        print(f"  显示网格: {self._show_grid}")
        print(f"  反转向量: {self._reverse_vector}")
        print("========================")

    def _on_key_pressed(self, event: Event) -> None:
        """处理键盘按键事件"""
        # 获取按键代码和动作
        key = event.data.get("key")
        action = event.data.get("action")
        
        # 只处理按键按下事件
        if action != 1:  # GLFW_PRESS
            return

        # 处理特定按键
        if key == 71:  # G键
            print(f"[工具栏] G键被按下，切换网格显示")
            self.toggle_grid()
            print(f"[工具栏] toggle_grid调用完成")
        elif key == 82:  # R键
            print(f"[工具栏] R键被按下，重置视图")
            self.reset_view()
            print(f"[工具栏] reset_view调用完成")
        elif key == 67:  # C键
            print(f"[工具栏] C键被按下，清空网格")
            self.clear_grid()
            print(f"[工具栏] clear_grid调用完成")
        elif key == 84:  # T键
            self.toggle_toolbar()
        elif key == 72:  # H键
            self._show_help = True
        elif key == 61 or key == 171:  # +键
            self.set_brush_size(self._brush_size + 1)
        elif key == 45 or key == 173:  # -键
            self.set_brush_size(self._brush_size - 1)
        elif key == 44:  # ,键
            self.set_magnitude(self._magnitude - 0.1)
        elif key == 46:  # .键
            self.set_magnitude(self._magnitude + 0.1)
        elif key == 86:  # V键
            self.toggle_reverse_vector()

        # 如果按下H键，显示帮助信息
        if key == 72:
            self._show_help = True

    def toggle_grid(self) -> None:
        """切换网格显示状态"""
        self._show_grid = not self._show_grid
        # 发送网格显示状态变化事件
        print(f"[工具栏] 准备发布TOGGLE_GRID事件，show={self._show_grid}")
        self._event_bus.publish(Event(EventType.TOGGLE_GRID, {"show": self._show_grid}, "SimpleToolbar"))
        print(f"[工具栏] TOGGLE_GRID事件已发布，网格显示: {'开启' if self._show_grid else '关闭'}")

    def reset_view(self) -> None:
        """重置视图"""
        # 发送重置视图事件
        self._event_bus.publish(Event(EventType.RESET_VIEW, {}, "SimpleToolbar"))
        print("[工具栏] 视图已重置")

    def clear_grid(self) -> None:
        """清空网格"""
        # 发送清空网格事件
        print(f"[工具栏] 准备发布CLEAR_GRID事件")
        self._event_bus.publish(Event(EventType.CLEAR_GRID, {}, "SimpleToolbar"))
        print("[工具栏] CLEAR_GRID事件已发布，网格已清空")

    def toggle_toolbar(self) -> None:
        """切换工具栏显示状态"""
        # 对于简单工具栏，这个功能只是切换帮助显示
        self._show_help = not self._show_help
        print(f"[工具栏] 工具栏显示: {'开启' if self._show_help else '关闭'}")

    def set_brush_size(self, size: int) -> None:
        """设置画笔大小"""
        # 确保画笔大小在合理范围内
        self._brush_size = max(1, min(10, size))
        # 发送画笔大小变化事件
        self._event_bus.publish(Event(EventType.SET_BRUSH_SIZE, {"size": self._brush_size}, "SimpleToolbar"))
        print(f"[工具栏] 画笔大小设置为: {self._brush_size}")

    def set_magnitude(self, magnitude: float) -> None:
        """设置向量大小"""
        # 确保向量大小在合理范围内
        self._magnitude = max(0.1, min(5.0, magnitude))
        # 发送向量大小变化事件
        self._event_bus.publish(Event(EventType.SET_MAGNITUDE, {"magnitude": self._magnitude}, "SimpleToolbar"))
        print(f"[工具栏] 向量大小设置为: {self._magnitude:.1f}")

    def toggle_reverse_vector(self) -> None:
        """切换向量方向"""
        self._reverse_vector = not self._reverse_vector
        # 发送向量方向变化事件
        self._event_bus.publish(Event(EventType.TOGGLE_REVERSE_VECTOR, {"reverse": self._reverse_vector}, "SimpleToolbar"))
        print(f"[工具栏] 向量方向: {'反向' if self._reverse_vector else '正向'}")
