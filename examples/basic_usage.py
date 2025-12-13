"""
LiziEngine 基本使用示例
演示如何使用LiziEngine创建和渲染向量场
"""
import sys
import os
import numpy as np

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lizi_engine.core.container import container
from lizi_engine.core.app import AppCore
from lizi_engine.window.window import Window
from lizi_engine.compute.vector_field import vector_calculator

def main():
    """主函数"""
    print("[示例] 启动LiziEngine基本使用示例...")

    # 初始化应用核心
    app_core = container.resolve(AppCore)
    if app_core is None:
        # 如果容器中没有 AppCore 实例，创建并注册实例
        app_core = AppCore()
        container.register_singleton(AppCore, app_core)
    else:
        # 如果获取到的是类而不是实例，创建新实例
        if isinstance(app_core, type):
            app_core = AppCore()
            container.register_singleton(AppCore, app_core)

    # 初始化窗口
    window = container.resolve(Window)
    if window is None:
        window = Window("LiziEngine 示例", 800, 600)
        container.register_singleton(Window, window)

    if not window.initialize():
        print("[示例] 窗口初始化失败")
        return

    # 获取网格
    grid = app_core.grid_manager.init_grid(64, 48)

    # 设置示例向量场 - 创建旋转模式
    vector_calculator.create_tangential_pattern(grid, magnitude=1.0)

    # 运行主循环
    print("[示例] 开始主循环...")
    while not window.should_close:
        # 更新窗口
        window.update()

        # 渲染
        window.render(grid)

    # 清理资源
    print("[示例] 清理资源...")
    window.cleanup()
    app_core.shutdown()

    print("[示例] 示例结束")

if __name__ == "__main__":
    main()
