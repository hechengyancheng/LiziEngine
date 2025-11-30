
"""
LiziEngine 主入口文件
"""
import sys
import os
import numpy as np

# 确保当前目录在Python路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.controller import app_controller
from core.events import EventBus, Event, EventType
from compute.vector_field import vector_calculator

def example(grid):
    """示例函数 - 在网格上创建一个旋转模式"""
    # 使用向量计算器创建切线向量模式，增加magnitude使向量更明显
    vector_calculator.create_tangential_pattern(grid, magnitude=1.0, radius_ratio=0.4)

    # 通过事件系统通知网格更新
    app_controller._event_bus.publish(Event(
        EventType.GRID_UPDATED,
        {"updates": {}},  # 空的updates字典，表示整个网格已更新
        "RunExample"
    ))

def main():
    """主函数"""
    try:
        print("[LiziEngine] 启动应用程序...")

        # 初始化应用控制器
        if not app_controller.initialize("LiziEngine", 800, 600):
            print("[LiziEngine] 初始化失败")
            return

        # 获取网格
        grid = app_controller._app_core.grid_manager.get_raw_grid()
        if grid is None:
            print("[LiziEngine] 无法获取网格")
            return

        # 设置示例向量场
        print("[LiziEngine] 设置示例向量场...")
        example(grid)

        # 运行应用程序
        print("[LiziEngine] 运行应用程序...")
        app_controller.run()

        print("[LiziEngine] 应用程序已退出")
    except KeyboardInterrupt:
        print("[LiziEngine] 应用程序被用户中断")
        # 确保资源正确释放
        try:
            app_controller.cleanup()
        except:
            pass

if __name__ == "__main__":
    main()
