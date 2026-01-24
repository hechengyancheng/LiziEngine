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
from plugins.command import Command, CommandInputHandler


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
        self.command_input_handler = None

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

        # 初始化指令输入处理器
        self.command_input_handler = CommandInputHandler(self.command_plugin)

        # 初始化 UI 管理器
        self.ui_manager = UIManager(self.app_core, self.window, self.controller, self.marker_system, self.command_input_handler)

        # 注册基本回调
        def _on_space():
            vector_calculator.create_tangential_pattern(self.grid, magnitude=1.0)
            try:
                self.app_core.view_manager.reset_view(self.grid.shape[1], self.grid.shape[0])
            except Exception:
                pass

        self.ui_manager.register_callbacks(self.grid, on_space=_on_space)

        # 注册指令模式切换回调
        input_handler.register_key_callback(KeyMap.SLASH, MouseMap.PRESS, self.command_input_handler.get_toggle_callback())

        print("[指令演示] 按'/'键激活指令输入模式，按ESC退出指令模式，按Enter执行指令")



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
                    # 从状态管理器获取重力和速度因子参数
                    gravity = self.app_core.state_manager.get("gravity", 0.0)
                    speed_factor = self.app_core.state_manager.get("speed_factor", 0.99)
                    self.marker_system.update_field_and_markers(self.grid, dt=1.0, gravity=gravity, speed_factor=speed_factor)
                except Exception as e:
                    print(f"[错误] 更新标记异常: {e}")

            # 处理指令输入
            if self.command_input_handler.command_mode:
                self.command_input_handler._handle_command_input()

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
