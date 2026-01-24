import pytest
from unittest.mock import Mock, MagicMock
from plugins.command import Command


class TestCommand:
    """测试命令插件"""

    def setup_method(self):
        """测试前准备"""
        # 创建模拟的控制器
        self.mock_controller = Mock()
        self.mock_controller.reset_view = Mock(return_value=None)
        self.mock_controller.toggle_grid = Mock(return_value=None)
        self.mock_controller.clear_grid = Mock(return_value=None)
        self.mock_controller.switch_vector_field_direction = Mock(return_value=None)
        # 添加带参数的模拟方法
        def mock_set_speed(speed):
            return "Speed set to 10"
        def mock_set_gravity(gravity):
            return 0.5
        def mock_add_marker(x, y):
            pass

        self.mock_controller.set_speed = Mock(side_effect=mock_set_speed)
        self.mock_controller.set_gravity = Mock(side_effect=mock_set_gravity)
        self.mock_controller.add_marker = Mock(side_effect=mock_add_marker)

        # Set signatures for mock methods to match expected parameter counts
        import inspect
        self.mock_controller.set_speed.__signature__ = inspect.signature(mock_set_speed)
        self.mock_controller.set_gravity.__signature__ = inspect.signature(mock_set_gravity)
        self.mock_controller.add_marker.__signature__ = inspect.signature(mock_add_marker)

        # 创建命令实例（使用自定义commands.json）
        import tempfile
        import json
        import os

        # 创建临时commands.json文件
        commands_data = {
            "commands": {
                "reset_view": "reset_view",
                "toggle_grid": "toggle_grid",
                "clear_grid": "clear_grid",
                "switch_direction": "switch_vector_field_direction",
                "set_speed": "set_speed",
                "set_gravity": "set_gravity",
                "add_marker": "add_marker"
            },
            "descriptions": {
                "reset_view": "重置视图到初始状态",
                "toggle_grid": "切换网格显示",
                "clear_grid": "清除网格内容",
                "switch_direction": "切换向量场方向",
                "set_speed": "设置粒子速度 (参数: speed)",
                "set_gravity": "设置重力值 (参数: gravity)",
                "add_marker": "添加标记 (参数: x, y)"
            },
            "no_param_commands": [
                "reset_view",
                "toggle_grid",
                "clear_grid",
                "switch_direction"
            ]
        }

        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(commands_data, self.temp_file)
        self.temp_file.close()

        # 创建命令实例
        self.command = Command(self.mock_controller, self.temp_file.name)

    def test_execute_reset_view(self):
        """测试执行 reset_view 命令"""
        result = self.command.execute("reset_view")
        assert result == "命令 'reset_view' 执行成功"
        self.mock_controller.reset_view.assert_called_once()

    def test_execute_toggle_grid(self):
        """测试执行 toggle_grid 命令"""
        result = self.command.execute("toggle_grid")
        assert result == "命令 'toggle_grid' 执行成功"
        self.mock_controller.toggle_grid.assert_called_once()

    def test_execute_clear_grid(self):
        """测试执行 clear_grid 命令"""
        result = self.command.execute("clear_grid")
        assert result == "命令 'clear_grid' 执行成功"
        self.mock_controller.clear_grid.assert_called_once()

    def test_execute_switch_direction(self):
        """测试执行 switch_direction 命令"""
        result = self.command.execute("switch_direction")
        assert result == "命令 'switch_direction' 执行成功"
        self.mock_controller.switch_vector_field_direction.assert_called_once()

    def test_execute_unknown_command(self):
        """测试执行未知命令"""
        result = self.command.execute("unknown_command")
        assert "错误: 未知命令 'unknown_command'" in result
        assert "可用命令:" in result

    def test_execute_empty_command(self):
        """测试执行空命令"""
        result = self.command.execute("")
        assert result == "错误: 命令为空"

    def test_execute_whitespace_command(self):
        """测试执行空白命令"""
        result = self.command.execute("   ")
        assert result == "错误: 命令为空"

    def test_execute_command_with_extra_args(self):
        """测试执行带额外参数的命令（现在应该拒绝）"""
        result = self.command.execute("reset_view extra arg")
        assert result == "错误: 命令 'reset_view' 不支持参数"
        self.mock_controller.reset_view.assert_not_called()

    def test_execute_case_insensitive(self):
        """测试命令大小写不敏感"""
        result = self.command.execute("RESET_VIEW")
        assert result == "命令 'reset_view' 执行成功"
        self.mock_controller.reset_view.assert_called_once()

    def test_execute_with_exception(self):
        """测试执行命令时发生异常"""
        self.mock_controller.reset_view.side_effect = Exception("Test exception")
        result = self.command.execute("reset_view")
        assert "错误: 执行命令 'reset_view' 时发生异常: Exception: Test exception" in result

    def test_list_commands(self):
        """测试列出可用命令及其描述"""
        result = self.command.list_commands()
        assert "可用命令:" in result
        assert "reset_view (无参数): 重置视图到初始状态" in result
        assert "toggle_grid (无参数): 切换网格显示" in result
        assert "clear_grid (无参数): 清除网格内容" in result
        assert "switch_direction (无参数): 切换向量场方向" in result
        assert "set_speed (需要 1 个参数): 设置粒子速度 (参数: speed)" in result

    def test_commands_dict_structure(self):
        """测试命令字典结构"""
        expected_commands = ['reset_view', 'toggle_grid', 'clear_grid', 'switch_direction', 'set_speed', 'set_gravity', 'add_marker']
        assert list(self.command.commands.keys()) == expected_commands
        for cmd in expected_commands:
            assert callable(self.command.commands[cmd])

    def test_execute_command_with_unsupported_args(self):
        """测试执行不支持参数的命令时传递参数"""
        result = self.command.execute("reset_view arg1 arg2")
        assert "错误: 命令 'reset_view' 不支持参数" in result

    def test_execute_with_specific_exception(self):
        """测试执行命令时发生特定异常"""
        self.mock_controller.reset_view.side_effect = ValueError("Invalid value")
        result = self.command.execute("reset_view")
        assert "错误: 执行命令 'reset_view' 时发生异常: ValueError: Invalid value" in result

    def test_register_command(self):
        """测试动态注册新命令"""
        def new_command():
            pass
        self.command.register_command('new_cmd', new_command, '新命令描述')
        assert 'new_cmd' in self.command.commands
        assert self.command.commands['new_cmd'] == new_command
        assert self.command.descriptions['new_cmd'] == '新命令描述'
        result = self.command.execute("new_cmd")
        assert result == "命令 'new_cmd' 执行成功"

    def test_execute_command_with_int_args(self):
        """测试执行带整数参数的命令"""
        result = self.command.execute("set_speed 10")
        assert result == "命令 'set_speed' 执行成功，结果: Speed set to 10"
        self.mock_controller.set_speed.assert_called_once_with(10)

    def test_execute_command_with_float_args(self):
        """测试执行带浮点数参数的命令"""
        result = self.command.execute("set_gravity 0.5")
        assert result == "命令 'set_gravity' 执行成功，结果: 0.5"
        self.mock_controller.set_gravity.assert_called_once_with(0.5)

    def test_execute_command_with_multiple_args(self):
        """测试执行带多个参数的命令"""
        result = self.command.execute("add_marker 100 200")
        assert result == "命令 'add_marker' 执行成功"
        self.mock_controller.add_marker.assert_called_once_with(100, 200)

    def test_execute_command_with_string_args(self):
        """测试执行带字符串参数的命令"""
        result = self.command.execute("set_speed hello")
        assert result == "命令 'set_speed' 执行成功，结果: Speed set to 10"
        self.mock_controller.set_speed.assert_called_once_with("hello")

    def test_execute_command_with_too_many_args(self):
        """测试执行参数过多的命令"""
        result = self.command.execute("set_speed 10 20")
        assert "错误: 命令 'set_speed' 期望 1 个参数，但提供了 2 个" in result
        self.mock_controller.set_speed.assert_not_called()

    def test_execute_command_with_too_few_args(self):
        """测试执行参数过少的命令"""
        result = self.command.execute("add_marker 100")
        assert "错误: 命令 'add_marker' 期望 2 个参数，但提供了 1 个" in result
        self.mock_controller.add_marker.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__])
