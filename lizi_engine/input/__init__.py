"""
输入模块 - 提供用户输入处理功能
支持键盘、鼠标等输入设备的处理
"""
from .input_handler import InputHandler, input_handler
from .key_map import KeyMap
from .mouse_map import MouseMap

__all__ = [
    "InputHandler",
    "input_handler",
    "KeyMap",
    "MouseMap"
]
