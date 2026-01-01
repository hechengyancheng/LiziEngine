"""
GUI演示示例 - 使用Dear PyGui和嵌入式OpenGL渲染
"""
import sys
import os
import numpy as np
import time

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lizi_engine.core.app import AppCore
from lizi_engine.core.config import config_manager
from lizi_engine.core.state import state_manager
from lizi_engine.core.events import event_bus, EventType, Event
from lizi_engine.compute.vector_field import vector_calculator
from plugins.controller import Controller
from plugins.marker_system import MarkerSystem
from lizi_engine.gui.gui_manager import gui_manager

def main():
    """主函数"""
    print("启动LiziEngine GUI演示...")

    # 初始化应用核心
    app_core = AppCore()

    # 初始化配置
    config_manager.load_from_file("config.json")

    # 创建网格
    grid_size = config_manager.get("grid_size", 64)
    grid = np.zeros((grid_size, grid_size, 2), dtype=np.float32)

    # 初始化插件
    marker_system = MarkerSystem(app_core)
    controller = Controller(app_core, vector_calculator, marker_system, grid)

    # 初始化GUI管理器
    if not gui_manager.initialize(grid, controller, marker_system, None):
        print("GUI初始化失败")
        return

    # 发送应用初始化事件
    event_bus.publish(Event(EventType.APP_INITIALIZED, {}))

    # 主循环
    print("进入主循环...")
    last_time = time.time()

    try:
        while not gui_manager.should_close:
            current_time = time.time()
            delta_time = current_time - last_time
            last_time = current_time

            # 更新状态
            state_manager.update({"delta_time": delta_time})

            # 更新插件
            controller.update()
            marker_system.update()

            # 更新GUI
            gui_manager.update()

            # 渲染
            gui_manager.render()

            # 简单的帧率控制
            time.sleep(1/60.0)  # 60 FPS

    except KeyboardInterrupt:
        print("收到中断信号，正在退出...")

    finally:
        # 清理资源
        gui_manager.cleanup()
        print("GUI演示结束")

if __name__ == "__main__":
    main()
