"""
标记计数器示例：每半秒在中心放置一个标记并计数
演示如何使用定时器在向量场中心放置标记点并计数
"""
import sys
import os
import numpy as np
import time

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lizi_engine.core.container import container
from lizi_engine.core.app import AppCore
from lizi_engine.window.window import Window
from lizi_engine.compute.vector_field import vector_calculator
from plugins.ui import UIManager
from plugins.controller import Controller
from plugins.marker_system import MarkerSystem

def main():
    """主函数"""
    print("[示例] 启动标记计数器示例...")

    # 初始化应用核心
    app_core = container.resolve(AppCore)
    if app_core is None:
        app_core = AppCore()
        container.register_singleton(AppCore, app_core)
    else:
        if isinstance(app_core, type):
            app_core = AppCore()
            container.register_singleton(AppCore, app_core)

    # 初始化窗口
    window = container.resolve(Window)
    if window is None:
        window = Window("标记计数器示例", 800, 600)
        container.register_singleton(Window, window)

    if not window.initialize():
        print("[示例] 窗口初始化失败")
        return

    # 获取网格
    grid = app_core.grid_manager.init_grid(640, 480)

    # 设置示例向量场 - 创建旋转模式
    vector_calculator.create_tangential_pattern(grid, magnitude=1.0)

    # 初始化视图
    try:
        app_core.view_manager.reset_view(grid.shape[1], grid.shape[0])
    except Exception:
        pass

    # 运行主循环
    print("[示例] 开始主循环...")
    print("[示例] 每秒在中心放置一个标记并计数")
    print("[示例] 按空格键重新生成切线模式，按G键切换网格显示，按C键清空网格")
    print("[示例] 按U键切换实时更新；用鼠标拖动视图并滚轮缩放")

    # 初始化标记系统
    marker_system = MarkerSystem(app_core)

    # 初始化控制器
    controller = Controller(app_core, vector_calculator, marker_system, grid)

    # 初始化 UI 管理器并注册回调（与 patterns.py 保持一致）
    ui_manager = UIManager(app_core, window, controller, marker_system)

    def _on_space():
        # 空格键：重新生成切线模式并重置视图
        print("[示例] 重新生成切线模式")
        vector_calculator.create_tangential_pattern(grid, magnitude=1.0)
        try:
            app_core.view_manager.reset_view(grid.shape[1], grid.shape[0])
        except Exception:
            pass

    ui_manager.register_callbacks(grid, on_space=_on_space)

    # 标记计数器和定时器
    marker_count = 0
    last_marker_time = time.time()
    center_x = grid.shape[1] / 2.0  # 中心x坐标
    center_y = grid.shape[0] / 2.0  # 中心y坐标

    while not window.should_close:
        # 更新窗口和处理 UI 事件
        window.update()

        # 实时更新向量场（如果启用）
        if ui_manager.enable_update:
            vector_calculator.update_grid_with_adjacent_sum(grid)

        # 处理鼠标拖动与滚轮
        try:
            ui_manager.process_mouse_drag()
        except Exception as e:
            print(f"[错误] 鼠标拖动处理异常: {e}")

        ui_manager.process_scroll()

        # 更新标记位置（可选）
        try:
            ui_manager.update_markers(grid)
        except Exception as e:
            print(f"[错误] 更新标记异常: {e}")

        # 检查是否需要放置新标记（每半秒一次）
        current_time = time.time()
        if current_time - last_marker_time >= 0.50:
            marker_count += 1
            marker_system.add_marker(center_x, center_y, mag=1.0)
            print(f"[计数器] 已放置标记 #{marker_count} 在中心 ({center_x:.1f}, {center_y:.1f})")
            last_marker_time = current_time

        # 渲染
        window.render(grid)

    # 清理资源
    print("[示例] 清理资源...")
    window.cleanup()
    app_core.shutdown()

    print(f"[示例] 示例结束，总共放置了 {marker_count} 个标记")

if __name__ == "__main__":
    main()
