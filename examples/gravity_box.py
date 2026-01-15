"""
LiziEngine 示例：重力场模拟
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
from plugins.toolkit import add_inward_edge_vectors

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
    grid = app_core.grid_manager.init_grid(64, 64)

    # 初始化视图
    try:
        app_core.view_manager.reset_view(grid.shape[1], grid.shape[0])
    except Exception:
        pass

    # 运行主循环
    print("[示例] 开始主循环...")
    print("[示例] 按空格键重新生成切线模式，按G键切换网格显示，按C键清空网格")
    print("[示例] 按U键切换实时更新；用鼠标拖动视图并滚轮缩放")

    # 初始化标记系统
    marker_system = MarkerSystem(app_core)

    # 初始化控制器
    controller = Controller(app_core, vector_calculator, marker_system, grid)

    # 初始化 UI 管理器并注册回调（与 patterns.py 保持一致）
    ui_manager = UIManager(app_core, window, controller, marker_system)

    # 定义回调函数
    def on_space_press():
        """重新生成切线模式并添加边缘向内向量"""
        center = (grid.shape[1] // 2, grid.shape[0] // 2)
        vector_calculator.create_tangential_pattern(grid, center=center, radius=50, magnitude=1.0)
        # 添加边缘向内向量
        add_inward_edge_vectors(grid, magnitude=0.5)
        app_core.state_manager.update({"view_changed": True, "grid_updated": True})
        print("[示例] 已重新生成切线模式并添加边缘向内向量")

    def on_u_press():
        """切换实时更新"""
        ui_manager.enable_update = not ui_manager.enable_update
        status = "启用" if ui_manager.enable_update else "禁用"
        print(f"[示例] 实时更新: {status}")

    # 注册回调
    ui_manager.register_callbacks(grid, on_space=on_space_press, on_u=on_u_press)

    # FPS 限制变量
    target_fps = app_core.config_manager.get("target_fps", 60)
    frame_time = 1.0 / target_fps
    last_time = time.time()

    while not window.should_close:
        # 更新窗口和处理 UI 事件
        window.update()

        #清空网格
        grid.fill(0.0)      

        # 处理鼠标拖动与滚轮
        try:
            ui_manager.process_mouse_drag()
        except Exception as e:
            print(f"[错误] 鼠标拖动处理异常: {e}")

        ui_manager.process_scroll()

        # 实时更新向量场（如果启用）
        if ui_manager.enable_update:
            #vector_calculator.update_grid_with_adjacent_sum(grid)
            add_inward_edge_vectors(grid, magnitude=0.5)

        # 更新标记位置（可选）
        try:
            #给每个标记添加重力向量
            markers = marker_system.get_markers()
            for marker in markers:
                marker['vy'] += 0.01
                # 摩擦力
                marker['vx'] *= 0.95
                marker['vy'] *= 0.95
            ui_manager.update_markers(grid)
            vector_calculator.update_grid_with_adjacent_sum(grid)     
            ui_manager.update_markers(grid)       
        except Exception as e:
            print(f"[错误] 更新标记异常: {e}")

        # 渲染
        window.render(grid)

        # FPS 限制
        current_time = time.time()
        elapsed = current_time - last_time
        if elapsed < frame_time:
            time.sleep(frame_time - elapsed)
        last_time = time.time()

    # 清理资源
    print("[示例] 清理资源...")
    window.cleanup()
    app_core.shutdown()

    print("[示例] 示例结束")

if __name__ == "__main__":
    main()
