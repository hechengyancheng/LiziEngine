"""
input模块使用示例
展示如何使用input模块处理键盘和鼠标输入
"""
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lizi_engine.core.app import AppCore
from lizi_engine.window.window import Window
from lizi_engine.input import input_handler, KeyMap, MouseMap


class InputDemoApp:
    """输入示例应用"""

    def __init__(self):
        self._app_core = AppCore()
        self._window = Window()
        self._init_input_handlers()

    def _init_input_handlers(self):
        """初始化输入处理器"""
        # 注册键盘回调
        input_handler.register_key_callback(KeyMap.ESCAPE, MouseMap.PRESS, self._on_escape_press)
        input_handler.register_key_callback(KeyMap.SPACE, MouseMap.PRESS, self._on_space_press)

        # 注册鼠标回调
        input_handler.register_mouse_callback(MouseMap.LEFT, MouseMap.PRESS, self._on_left_click)

    def _on_escape_press(self):
        """ESC键按下回调"""
        print("ESC键被按下，退出应用")
        self.shutdown()
        
    def shutdown(self):
        """关闭应用"""
        self._app_core.shutdown()
        
    def run(self):
        """运行应用"""
        # 初始化窗口
        if not self._window.initialize():
            print("[错误] 窗口初始化失败")
            return
            
        # 创建一个空的网格
        import numpy as np
        grid = np.zeros((10, 10, 2), dtype=np.float32)
            
        # 主循环
        while not self._window.should_close:
            # 更新输入状态
            self.update(0.016)  # 假设60FPS，每帧约16ms
            
            # 渲染
            self._window.render(grid)
            
            # 处理事件
            self._window.update()
            
        # 关闭窗口
        self._window.cleanup()
        self.shutdown()

    def _on_space_press(self):
        """空格键按下回调"""
        print("空格键被按下")

    def _on_left_click(self):
        """鼠标左键点击回调"""
        x, y = input_handler.get_mouse_position()
        print(f"鼠标左键点击位置: ({x:.2f}, {y:.2f})")

    def update(self, dt):
        """更新函数"""
        # 检查按键状态
        if input_handler.is_key_pressed(KeyMap.UP):
            print("上方向键被按下")

        if input_handler.is_key_pressed(KeyMap.DOWN):
            print("下方向键被按下")

        if input_handler.is_key_pressed(KeyMap.LEFT):
            print("左方向键被按下")

        if input_handler.is_key_pressed(KeyMap.RIGHT):
            print("右方向键被按下")

        # 检查鼠标按钮状态
        if input_handler.is_mouse_button_pressed(MouseMap.RIGHT):
            print("鼠标右键被按下")

        # 获取鼠标位置
        x, y = input_handler.get_mouse_position()
        # print(f"鼠标位置: ({x:.2f}, {y:.2f})")

        # 获取鼠标滚轮位置
        scroll_x, scroll_y = input_handler.get_mouse_scroll()
        if scroll_y != 0:
            print(f"鼠标滚轮滚动: {scroll_y:.2f}")


if __name__ == "__main__":
    app = InputDemoApp()
    app.run()
