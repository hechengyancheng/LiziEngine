import pytest
from unittest.mock import Mock, MagicMock, patch
from lizi_engine.input.input_handler import InputHandler, input_handler
from lizi_engine.core.events import EventType, event_bus
from lizi_engine.core.state import state_manager


class TestInputHandler:
    """测试输入处理器"""

    def setup_method(self):
        """测试前准备"""
        self.input_handler = InputHandler()
        self.mock_window = Mock()

    def test_input_handler_initialization(self):
        """测试输入处理器初始化"""
        assert self.input_handler._key_states == {}
        assert self.input_handler._mouse_buttons == {}
        assert self.input_handler._mouse_position == (0.0, 0.0)
        assert self.input_handler._mouse_scroll == (0.0, 0.0)
        assert self.input_handler._key_callbacks == {}
        assert self.input_handler._mouse_callbacks == {}

    def test_register_key_callback(self):
        """测试注册按键回调"""
        def callback():
            pass

        self.input_handler.register_key_callback(65, 1, callback)  # GLFW_KEY_A, GLFW_PRESS

        assert self.input_handler._key_callbacks["65_1"] == callback

    def test_register_mouse_callback(self):
        """测试注册鼠标回调"""
        def callback():
            pass

        self.input_handler.register_mouse_callback(0, 1, callback)  # GLFW_MOUSE_BUTTON_LEFT, GLFW_PRESS

        assert self.input_handler._mouse_callbacks["0_1"] == callback

    def test_is_key_pressed(self):
        """测试检查按键是否按下"""
        assert not self.input_handler.is_key_pressed(65)

        # 模拟按键按下
        self.input_handler._key_states[65] = True
        assert self.input_handler.is_key_pressed(65)

    def test_is_mouse_button_pressed(self):
        """测试检查鼠标按钮是否按下"""
        assert not self.input_handler.is_mouse_button_pressed(0)

        # 模拟鼠标按钮按下
        self.input_handler._mouse_buttons[0] = True
        assert self.input_handler.is_mouse_button_pressed(0)

    def test_get_mouse_position(self):
        """测试获取鼠标位置"""
        assert self.input_handler.get_mouse_position() == (0.0, 0.0)

        self.input_handler._mouse_position = (100.0, 200.0)
        assert self.input_handler.get_mouse_position() == (100.0, 200.0)

    def test_get_mouse_scroll(self):
        """测试获取鼠标滚轮位置"""
        assert self.input_handler.get_mouse_scroll() == (0.0, 0.0)

        self.input_handler._mouse_scroll = (1.0, -1.0)
        assert self.input_handler.get_mouse_scroll() == (1.0, -1.0)

    def test_reset_mouse_scroll(self):
        """测试重置鼠标滚轮"""
        self.input_handler._mouse_scroll = (1.0, -1.0)
        self.input_handler.reset_mouse_scroll()
        assert self.input_handler.get_mouse_scroll() == (0.0, 0.0)

    @patch('lizi_engine.input.input_handler.event_bus')
    def test_handle_key_event_press(self, mock_event_bus):
        """测试处理按键按下事件"""
        self.input_handler.handle_key_event(self.mock_window, 65, 0, 1, 0)  # A key press

        assert self.input_handler._key_states[65] == True

        # 检查事件是否发布
        mock_event_bus.publish.assert_called_once()
        event = mock_event_bus.publish.call_args[0][0]
        assert event.type == EventType.KEY_PRESSED
        assert event.data["key"] == 65
        assert event.data["action"] == 1

    @patch('lizi_engine.input.input_handler.event_bus')
    def test_handle_key_event_release(self, mock_event_bus):
        """测试处理按键释放事件"""
        # 先按下
        self.input_handler._key_states[65] = True

        self.input_handler.handle_key_event(self.mock_window, 65, 0, 0, 0)  # A key release

        assert self.input_handler._key_states[65] == False

        # 检查事件是否发布
        mock_event_bus.publish.assert_called_once()
        event = mock_event_bus.publish.call_args[0][0]
        assert event.type == EventType.KEY_RELEASED

    def test_handle_mouse_button_event_press(self):
        """测试处理鼠标按钮按下事件"""
        mock_event_bus = Mock()
        self.input_handler._event_bus = mock_event_bus

        with patch('glfw.get_cursor_pos', return_value=(100.0, 200.0)):
            self.input_handler.handle_mouse_button_event(self.mock_window, 0, 1, 0)  # Left button press

        assert self.input_handler._mouse_buttons[0] == True

        # 检查事件是否发布
        mock_event_bus.publish.assert_called_once()
        event = mock_event_bus.publish.call_args[0][0]
        assert event.type == EventType.MOUSE_CLICKED
        assert event.data["button"] == 0
        assert event.data["action"] == 1
        assert event.data["position"] == (100.0, 200.0)

    @patch('lizi_engine.input.input_handler.event_bus')
    def test_handle_mouse_button_event_release(self, mock_event_bus):
        """测试处理鼠标按钮释放事件"""
        # 先按下
        self.input_handler._mouse_buttons[0] = True

        with patch('glfw.get_cursor_pos', return_value=(100.0, 200.0)):
            self.input_handler.handle_mouse_button_event(self.mock_window, 0, 0, 0)  # Left button release

        assert self.input_handler._mouse_buttons[0] == False

    @patch('lizi_engine.input.input_handler.event_bus')
    def test_handle_cursor_position_event(self, mock_event_bus):
        """测试处理鼠标移动事件"""
        self.input_handler._mouse_position = (50.0, 50.0)

        self.input_handler.handle_cursor_position_event(self.mock_window, 100.0, 200.0)

        assert self.input_handler._mouse_position == (100.0, 200.0)

        # 检查事件是否发布
        mock_event_bus.publish.assert_called_once()
        event = mock_event_bus.publish.call_args[0][0]
        assert event.type == EventType.MOUSE_MOVED
        assert event.data["position"] == (100.0, 200.0)
        assert event.data["delta"] == (50.0, 150.0)

    @patch('lizi_engine.input.input_handler.event_bus')
    def test_handle_scroll_event(self, mock_event_bus):
        """测试处理鼠标滚轮事件"""
        self.input_handler.handle_scroll_event(self.mock_window, 1.0, -1.0)

        assert self.input_handler._mouse_scroll == (1.0, -1.0)

        # 检查事件是否发布
        mock_event_bus.publish.assert_called_once()
        event = mock_event_bus.publish.call_args[0][0]
        assert event.type == EventType.MOUSE_SCROLLED
        assert event.data["offset"] == (1.0, -1.0)

    def test_key_callback_execution(self):
        """测试按键回调执行"""
        called = []

        def callback():
            called.append(True)

        self.input_handler.register_key_callback(65, 1, callback)
        self.input_handler.handle_key_event(self.mock_window, 65, 0, 1, 0)

        assert called == [True]

    def test_mouse_callback_execution(self):
        """测试鼠标回调执行"""
        called = []

        def callback():
            called.append(True)

        self.input_handler.register_mouse_callback(0, 1, callback)

        with patch('glfw.get_cursor_pos', return_value=(100.0, 200.0)):
            self.input_handler.handle_mouse_button_event(self.mock_window, 0, 1, 0)

        assert called == [True]

    def test_multiple_key_states(self):
        """测试多个按键状态"""
        # 模拟多个按键
        self.input_handler._key_states = {65: True, 66: False, 67: True}

        assert self.input_handler.is_key_pressed(65)
        assert not self.input_handler.is_key_pressed(66)
        assert self.input_handler.is_key_pressed(67)
        assert not self.input_handler.is_key_pressed(68)

    def test_multiple_mouse_buttons(self):
        """测试多个鼠标按钮状态"""
        # 模拟多个鼠标按钮
        self.input_handler._mouse_buttons = {0: True, 1: False, 2: True}

        assert self.input_handler.is_mouse_button_pressed(0)
        assert not self.input_handler.is_mouse_button_pressed(1)
        assert self.input_handler.is_mouse_button_pressed(2)
        assert not self.input_handler.is_mouse_button_pressed(3)


class TestGlobalInputHandler:
    """测试全局输入处理器"""

    def test_global_input_handler_exists(self):
        """测试全局输入处理器存在"""
        assert input_handler is not None
        assert isinstance(input_handler, InputHandler)


if __name__ == "__main__":
    pytest.main([__file__])
