"""
鼠标映射 - 定义鼠标按钮的映射
"""
# Use GLFW mouse button codes as they are compatible with Dear PyGui


class MouseMap:
    """鼠标映射类，提供鼠标按钮的常量定义"""

    # 鼠标按钮
    LEFT = 0  # GLFW.MOUSE_BUTTON_LEFT
    RIGHT = 1  # GLFW.MOUSE_BUTTON_RIGHT
    MIDDLE = 2  # GLFW.MOUSE_BUTTON_MIDDLE
    _4 = 3  # GLFW.MOUSE_BUTTON_4
    _5 = 4  # GLFW.MOUSE_BUTTON_5
    _6 = 5  # No direct equivalent, using numeric value
    _7 = 6  # No direct equivalent, using numeric value
    _8 = 7  # No direct equivalent, using numeric value

    # 动作
    PRESS = 1  # GLFW.PRESS
    RELEASE = 0  # GLFW.RELEASE
    REPEAT = 2  # GLFW.REPEAT

    @staticmethod
    def get_button_name(button: int) -> str:
        """获取鼠标按钮名称

        Args:
            button: GLFW鼠标按钮

        Returns:
            str: 按钮名称
        """
        button_names = {
            MouseMap.LEFT: "Left Button",
            MouseMap.RIGHT: "Right Button",
            MouseMap.MIDDLE: "Middle Button",
            MouseMap._4: "Button 4",
            MouseMap._5: "Button 5",
            MouseMap._6: "Button 6",
            MouseMap._7: "Button 7",
            MouseMap._8: "Button 8"
        }
        return button_names.get(button, "Unknown Button")

    @staticmethod
    def get_action_name(action: int) -> str:
        """获取动作名称

        Args:
            action: GLFW动作

        Returns:
            str: 动作名称
        """
        action_names = {
            MouseMap.PRESS: "Press",
            MouseMap.RELEASE: "Release",
            MouseMap.REPEAT: "Repeat"
        }
        return action_names.get(action, "Unknown Action")
