"""
输入处理器 - 处理用户输入事件
"""
import glfw
import numpy as np
from typing import Dict, Any, Optional, Callable, List, Tuple
from ..core.events import Event, EventType, event_bus, EventHandler
from ..core.state import state_manager


class InputHandler(EventHandler):
    """输入处理器，管理键盘和鼠标输入"""

    def __init__(self):
        self._event_bus = event_bus
        self._state_manager = state_manager
        self._key_states = {}  # 存储按键状态
        self._mouse_buttons = {}  # 存储鼠标按钮状态
        self._mouse_position = (0.0, 0.0)  # 当前鼠标位置
        self._mouse_scroll = (0.0, 0.0)  # 当前鼠标滚轮位置
        self._key_callbacks = {}  # 按键回调函数
        self._mouse_callbacks = {}  # 鼠标回调函数

    def register_key_callback(self, key: int, action: int, callback: Callable):
        """注册键盘回调函数

        Args:
            key: GLFW键码
            action: GLFW动作 (PRESS, RELEASE, REPEAT)
            callback: 回调函数
        """
        key_id = f"{key}_{action}"
        self._key_callbacks[key_id] = callback

    def register_mouse_callback(self, button: int, action: int, callback: Callable):
        """注册鼠标回调函数

        Args:
            button: GLFW鼠标按钮
            action: GLFW动作 (PRESS, RELEASE)
            callback: 回调函数
        """
        button_id = f"{button}_{action}"
        self._mouse_callbacks[button_id] = callback

    def is_key_pressed(self, key: int) -> bool:
        """检查按键是否按下

        Args:
            key: GLFW键码

        Returns:
            bool: 按键是否按下
        """
        return self._key_states.get(key, False)

    def is_mouse_button_pressed(self, button: int) -> bool:
        """检查鼠标按钮是否按下

        Args:
            button: GLFW鼠标按钮

        Returns:
            bool: 鼠标按钮是否按下
        """
        return self._mouse_buttons.get(button, False)

    def get_mouse_position(self) -> Tuple[float, float]:
        """获取当前鼠标位置

        Returns:
            Tuple[float, float]: 鼠标位置 (x, y)
        """
        return self._mouse_position

    def get_mouse_scroll(self) -> Tuple[float, float]:
        """获取当前鼠标滚轮位置

        Returns:
            Tuple[float, float]: 鼠标滚轮位置 (x, y)
        """
        return self._mouse_scroll
        
    def reset_mouse_scroll(self) -> None:
        """重置鼠标滚轮位置为(0, 0)"""
        self._mouse_scroll = (0.0, 0.0)

    def handle_key_event(self, window, key: int, scancode: int, action: int, mods: int):
        """处理键盘事件

        Args:
            window: GLFW窗口
            key: 按键代码
            scancode: 扫描码
            action: 动作 (PRESS, RELEASE, REPEAT)
            mods: 修饰键
        """
        # 更新按键状态
        if action == glfw.PRESS:
            self._key_states[key] = True
        elif action == glfw.RELEASE:
            self._key_states[key] = False

        # 触发按键事件
        event_type = EventType.KEY_PRESSED if action == glfw.PRESS else EventType.KEY_RELEASED
        event = Event(
            type=event_type,
            data={
                "key": key,
                "scancode": scancode,
                "action": action,
                "mods": mods
            }
        )
        self._event_bus.publish(event)

        # 执行注册的回调函数
        key_id = f"{key}_{action}"
        if key_id in self._key_callbacks:
            self._key_callbacks[key_id]()

    def handle_mouse_button_event(self, window, button: int, action: int, mods: int):
        """处理鼠标按钮事件

        Args:
            window: GLFW窗口
            button: 鼠标按钮
            action: 动作 (PRESS, RELEASE)
            mods: 修饰键
        """
        # 更新鼠标按钮状态
        if action == glfw.PRESS:
            self._mouse_buttons[button] = True
        elif action == glfw.RELEASE:
            self._mouse_buttons[button] = False

        # 获取当前鼠标位置
        x, y = glfw.get_cursor_pos(window)

        # 触发鼠标点击事件
        event = Event(
            type=EventType.MOUSE_CLICKED,
            data={
                "button": button,
                "action": action,
                "mods": mods,
                "position": (x, y)
            }
        )
        self._event_bus.publish(event)

        # 执行注册的回调函数
        button_id = f"{button}_{action}"
        if button_id in self._mouse_callbacks:
            self._mouse_callbacks[button_id]()

    def handle_cursor_position_event(self, window, x: float, y: float):
        """处理鼠标移动事件

        Args:
            window: GLFW窗口
            x: 鼠标X坐标
            y: 鼠标Y坐标
        """
        # 计算鼠标移动距离
        dx = x - self._mouse_position[0]
        dy = y - self._mouse_position[1]

        # 更新鼠标位置
        self._mouse_position = (x, y)

        # 触发鼠标移动事件
        event = Event(
            type=EventType.MOUSE_MOVED,
            data={
                "position": (x, y),
                "delta": (dx, dy)
            }
        )
        self._event_bus.publish(event)

    def handle_scroll_event(self, window, x: float, y: float):
        """处理鼠标滚轮事件

        Args:
            window: GLFW窗口
            x: 滚轮X偏移
            y: 滚轮Y偏移
        """
        # 更新滚轮位置
        self._mouse_scroll = (x, y)

        # 触发滚轮事件
        event = Event(
            type=EventType.MOUSE_SCROLLED,
            data={
                "offset": (x, y)
            }
        )
        self._event_bus.publish(event)


# 创建全局输入处理器实例
input_handler = InputHandler()
