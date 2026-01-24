"""
Command Plugin Demo 示例
演示如何使用指令插件，按“/”键激活指令输入模式
"""
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lizi_engine.core.container import container
from lizi_engine.core.app import AppCore
from lizi_engine.window.window import Window
from lizi_engine.compute.vector_field import vector_calculator
from lizi_engine.core.plugin import UIManager, Controller, MarkerSystem, add_inward_edge_vectors
from lizi_engine.input import input_handler, KeyMap, MouseMap
from plugins.command import Command


class CommandDemoApp:
    """指令插件演示应用"""

    def __init__(self):
        self.app_core = None
        self.window = None
        self.grid = None
        self.controller = None
        self.marker_system = None
        self.ui_manager = None
        self.command_plugin = None

        # 指令输入模式状态
        self.command_mode = False
        self.command_string = ""
        self.was_pressed = {}  # 记录按键的上一次状态

        self._init_app()

    def _init_app(self):
        """初始化应用"""
        print("[指令演示] 启动指令插件演示示例...")

        # 初始化应用核心
        self.app_core = container.resolve(AppCore)
        if self.app_core is None:
            self.app_core = AppCore()
            container.register_singleton(AppCore, self.app_core)
        else:
            if isinstance(self.app_core, type):
                self.app_core = AppCore()
                container.register_singleton(AppCore, self.app_core)

        # 初始化窗口
        self.window = container.resolve(Window)
        if self.window is None:
            self.window = Window("LiziEngine 指令演示", 800, 600)
            container.register_singleton(Window, self.window)

        if not self.window.initialize():
            print("[指令演示] 窗口初始化失败")
            return

        # 获取网格
        self.grid = self.app_core.grid_manager.init_grid(64, 64)

        # 设置示例向量场
        vector_calculator.create_tangential_pattern(self.grid, magnitude=1.0)

        # 初始化视图
        try:
            self.app_core.view_manager.reset_view(self.grid.shape[1], self.grid.shape[0])
        except Exception:
            pass

        # 初始化标记系统
        self.marker_system = MarkerSystem(self.app_core)

        # 初始化控制器
        self.controller = Controller(self.app_core, vector_calculator, self.marker_system, self.grid)

        # 初始化指令插件
        self.command_plugin = Command(self.controller)

        # 初始化 UI 管理器
        self.ui_manager = UIManager(self.app_core, self.window, self.controller, self.marker_system)

        # 注册基本回调
        def _on_space():
            vector_calculator.create_tangential_pattern(self.grid, magnitude=1.0)
            try:
                self.app_core.view_manager.reset_view(self.grid.shape[1], self.grid.shape[0])
            except Exception:
                pass

        self.ui_manager.register_callbacks(self.grid, on_space=_on_space)

        # 注册指令模式切换回调
        input_handler.register_key_callback(KeyMap.SLASH, MouseMap.PRESS, self._toggle_command_mode)

        print("[指令演示] 按'/'键激活指令输入模式，按ESC退出指令模式，按Enter执行指令")

    def _toggle_command_mode(self):
        """切换指令输入模式"""
        if not self.command_mode:
            self.command_mode = True
            self.command_string = ""
            print("[指令模式] 已激活 - 输入指令后按Enter执行，按ESC取消")
        else:
            self.command_mode = False
            print("[指令模式] 已退出")

    def _handle_command_input(self):
        """处理指令输入，支持引号字符串"""
        # 定义按键到字符的映射
        key_to_char = {
            KeyMap.A: 'a', KeyMap.B: 'b', KeyMap.C: 'c', KeyMap.D: 'd', KeyMap.E: 'e',
            KeyMap.F: 'f', KeyMap.G: 'g', KeyMap.H: 'h', KeyMap.I: 'i', KeyMap.J: 'j',
            KeyMap.K: 'k', KeyMap.L: 'l', KeyMap.M: 'm', KeyMap.N: 'n', KeyMap.O: 'o',
            KeyMap.P: 'p', KeyMap.Q: 'q', KeyMap.R: 'r', KeyMap.S: 's', KeyMap.T: 't',
            KeyMap.U: 'u', KeyMap.V: 'v', KeyMap.W: 'w', KeyMap.X: 'x', KeyMap.Y: 'y',
            KeyMap.Z: 'z',
            KeyMap._0: '0', KeyMap._1: '1', KeyMap._2: '2', KeyMap._3: '3', KeyMap._4: '4',
            KeyMap._5: '5', KeyMap._6: '6', KeyMap._7: '7', KeyMap._8: '8', KeyMap._9: '9',
            KeyMap.SPACE: ' ',
            KeyMap.MINUS: '-',
            KeyMap.EQUAL: '=',
            KeyMap.SEMICOLON: ';',
            KeyMap.APOSTROPHE: "'",
            KeyMap.COMMA: ',',
            KeyMap.PERIOD: '.',
            KeyMap.SLASH: '/',
            KeyMap.LEFTBRACKET: '[',
            KeyMap.RIGHTBRACKET: ']',
            KeyMap.BACKSLASH: '\\',
        }

        # 检查Enter键执行指令
        if input_handler.is_key_pressed(KeyMap.ENTER):
            if not self.was_pressed.get(KeyMap.ENTER, False):
                self._execute_command()
            self.was_pressed[KeyMap.ENTER] = True
        else:
            self.was_pressed[KeyMap.ENTER] = False

        # 检查ESC键退出指令模式
        if input_handler.is_key_pressed(KeyMap.ESCAPE):
            if not self.was_pressed.get(KeyMap.ESCAPE, False):
                self.command_mode = False
                print("[指令模式] 已退出")
            self.was_pressed[KeyMap.ESCAPE] = True
        else:
            self.was_pressed[KeyMap.ESCAPE] = False

        # 检查Backspace键删除字符
        if input_handler.is_key_pressed(KeyMap.BACKSPACE):
            if not self.was_pressed.get(KeyMap.BACKSPACE, False) and self.command_string:
                self.command_string = self.command_string[:-1]
                print(f"[指令输入] {self.command_string}_")
            self.was_pressed[KeyMap.BACKSPACE] = True
        else:
            self.was_pressed[KeyMap.BACKSPACE] = False

        # 处理字符输入，支持引号
        in_quotes = False
        quote_char = None
        for key, char in key_to_char.items():
            if input_handler.is_key_pressed(key):
                if not self.was_pressed.get(key, False):
                    # 处理引号开始/结束
                    if char in ('"', "'") and (not in_quotes or char == quote_char):
                        if not in_quotes:
                            in_quotes = True
                            quote_char = char
                        else:
                            in_quotes = False
                            quote_char = None
                        self.command_string += char
                    elif char == ' ' and in_quotes:
                        # 在引号内允许空格
                        self.command_string += char
                    elif char != ' ' or in_quotes:
                        # 添加字符（非空格或在引号内）
                        self.command_string += char
                    print(f"[指令输入] {self.command_string}_")
                self.was_pressed[key] = True
            else:
                self.was_pressed[key] = False

    def _execute_command(self):
        """执行指令"""
        if not self.command_string.strip():
            print("[指令] 指令为空")
            self.command_mode = False
            return

        print(f"[指令] 执行: {self.command_string}")
        result = self.command_plugin.execute(self.command_string)
        print(f"[指令] 结果: {result}")

        self.command_mode = False

    def run(self):
        """运行主循环"""
        while not self.window.should_close:
            # 更新窗口和处理事件
            self.window.update()

            # 清空网格
            self.grid.fill(0.0)

            # 处理鼠标拖动与滚轮
            try:
                self.ui_manager.process_mouse_drag()
            except Exception as e:
                print(f"[错误] 鼠标拖动处理异常: {e}")

            self.ui_manager.process_scroll()

            # 实时更新向量场
            if self.ui_manager.enable_update:
                add_inward_edge_vectors(self.grid, magnitude=0.5)
                try:
                    self.marker_system.update_field_and_markers(self.grid, dt=1.0, gravity=0.0, speed_factor=0.99)
                except Exception as e:
                    print(f"[错误] 更新标记异常: {e}")

            # 处理指令输入
            if self.command_mode:
                self._handle_command_input()

            # 渲染
            self.window.render(self.grid)

            # FPS 限制
            self.app_core.fps_limiter.limit_fps()

        # 清理资源
        print("[指令演示] 清理资源...")
        self.window.cleanup()
        self.app_core.shutdown()

        print("[指令演示] 示例结束")


if __name__ == "__main__":
    app = CommandDemoApp()
    app.run()
