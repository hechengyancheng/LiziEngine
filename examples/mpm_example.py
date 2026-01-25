"""
LiziEngine MPM (Material Point Method) 示例
演示物质点法模拟变形材料
"""
import sys
import os
import numpy as np
from typing import List, Dict, Any, Tuple

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lizi_engine.core.container import container
from lizi_engine.core.app import AppCore
from lizi_engine.window.window import Window
from lizi_engine.compute.vector_field import vector_calculator
from lizi_engine.core.plugin import UIManager, Controller, MarkerSystem
from lizi_engine.input import input_handler, MouseMap
from plugins.mpm_system import MPMSystem, MPMParticle



def main():
    """主函数"""
    print("[MPM示例] 启动LiziEngine MPM示例...")

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
        window = Window("LiziEngine MPM 示例", 800, 600)
        container.register_singleton(Window, window)

    if not window.initialize():
        print("[MPM示例] 窗口初始化失败")
        return

    # 获取网格
    grid = app_core.grid_manager.init_grid(64, 64)

    # 初始化视图
    try:
        app_core.view_manager.reset_view(grid.shape[1], grid.shape[0])
    except Exception:
        pass

    # 初始化MPM系统
    mpm_system = MPMSystem(app_core, (64, 64))

    # 添加一些初始粒子
    for i in range(20):
        for j in range(10):
            mpm_system.add_particle(25 + i, 10 + j, mass=1.0)

    # 初始化标记系统（用于显示粒子）
    marker_system = MarkerSystem(app_core)

    # 初始化控制器
    controller = Controller(app_core, vector_calculator, marker_system, grid)

    # 初始化 UI 管理器
    ui_manager = UIManager(app_core, window, controller, marker_system)

    # 运行主循环
    print("[MPM示例] 开始主循环...")
    print("[MPM示例] 按空格键重置粒子，按C键清空，按U键切换更新")
    print("[MPM示例] 鼠标左键添加粒子，右键添加随机粒子")

    def on_space_press():
        """重置粒子"""
        mpm_system.clear_particles()
        for i in range(20):
            for j in range(10):
                mpm_system.add_particle(25 + i, 10 + j, mass=1.0)
        print("[MPM示例] 粒子已重置")

    def on_c_press():
        """清空粒子"""
        mpm_system.clear_particles()
        print("[MPM示例] 粒子已清空")

    def on_u_press():
        """切换实时更新"""
        ui_manager.enable_update = not ui_manager.enable_update
        status = "启用" if ui_manager.enable_update else "禁用"
        print(f"[MPM示例] 实时更新: {status}")

    # 注册回调
    ui_manager.register_callbacks(grid, on_space=on_space_press, on_c=on_c_press, on_u=on_u_press)

    # 用于跟踪鼠标按键状态，避免连续添加粒子
    left_mouse_pressed_last_frame = False

    while not window.should_close:
        # 更新窗口和处理 UI 事件
        window.update()

        # 处理鼠标拖动与滚轮
        try:
            ui_manager.process_mouse_drag()
        except Exception as e:
            print(f"[错误] 鼠标拖动处理异常: {e}")

        ui_manager.process_scroll()

        # 处理鼠标点击添加粒子
        if ui_manager.enable_update:
            # 检查鼠标输入 - 只在按下瞬间添加粒子，避免连续添加
            left_mouse_pressed = input_handler.is_mouse_button_pressed(MouseMap.LEFT)
            if left_mouse_pressed and not left_mouse_pressed_last_frame:
                # 获取鼠标位置并转换为网格坐标
                mouse_x, mouse_y = input_handler.get_mouse_position()
                grid_x, grid_y = controller._screen_to_grid(mouse_x, mouse_y)
                # 检查边界
                if 0 <= grid_x < 64 and 0 <= grid_y < 64:
                    mpm_system.add_particle(grid_x, grid_y, mass=1.0, vx=0.0, vy=0.0)
                    print(f"[MPM示例] 添加粒子于 ({grid_x:.2f}, {grid_y:.2f})")

            # 更新上一帧的鼠标状态
            left_mouse_pressed_last_frame = left_mouse_pressed

            # 执行MPM步骤
            mpm_system.step(grid)

        # 更新标记系统显示粒子
        marker_system.clear_markers()
        for particle in mpm_system.particles:
            marker_system.add_marker(particle.x, particle.y, mag=1.0, vx=particle.vx, vy=particle.vy)

        # 渲染
        window.render(grid)

        # FPS 限制
        app_core.fps_limiter.limit_fps()

    # 清理资源
    print("[MPM示例] 清理资源...")
    window.cleanup()
    app_core.shutdown()

    print("[MPM示例] 示例结束")

if __name__ == "__main__":
    main()
