"""
LiziEngine 向量场模式示例
演示如何创建不同的向量场模式
"""
import sys
import os
import numpy as np
from enum import Enum

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lizi_engine.core.container import container
from lizi_engine.core.app import AppCore
from lizi_engine.window.window import Window
from lizi_engine.compute.vector_field import vector_calculator
from plugin.ui import UIManager

class PatternType(Enum):
    """向量场模式类型"""
    TANGENTIAL = 1  # 切线模式（旋转）
    RADIAL = 2      # 径向模式（发散）
    CUSTOM = 3      # 自定义模式

def create_pattern(grid: np.ndarray, pattern_type: PatternType, **kwargs) -> None:
    """创建指定类型的向量场模式"""
    # 如果没有显式提供 center，则使用网格中心
    center = kwargs.get("center")
    if center is None:
        try:
            h, w = grid.shape[0], grid.shape[1]
            center = (int(w // 2), int(h // 2))
        except Exception:
            center = None

    if pattern_type == PatternType.TANGENTIAL:
        vector_calculator.create_tangential_pattern(
            grid,
            center=center,
            radius=kwargs.get("radius"),
            magnitude=kwargs.get("magnitude", 1.0)
        )
    elif pattern_type == PatternType.RADIAL:
        vector_calculator.create_radial_pattern(
            grid,
            center=center,
            radius=kwargs.get("radius"),
            magnitude=kwargs.get("magnitude", 1.0)
        )
    elif pattern_type == PatternType.CUSTOM:
        # 自定义模式 - 可以在这里添加更多自定义模式
        pass

def main():
    """主函数"""
    print("[示例] 启动LiziEngine向量场模式示例...")

    # 初始化应用核心
    app_core = container.resolve(AppCore)
    if app_core is None or app_core is AppCore:
        app_core = AppCore()
        container.register_singleton(AppCore, app_core)

    # 初始化窗口
    window = container.resolve(Window)
    if window is None:
        window = Window("LiziEngine 向量场模式示例", 800, 600)
        container.register_singleton(Window, window)

    if not window.initialize():
        print("[示例] 窗口初始化失败")
        return

    # 获取网格
    grid = app_core.grid_manager.init_grid(640, 480)

    # 当前模式索引
    current_pattern = 0
    patterns = [
        ("切线模式", PatternType.TANGENTIAL),
        ("径向模式", PatternType.RADIAL)
    ]

    # 设置初始模式
    pattern_name, pattern_type = patterns[current_pattern]
    print(f"[示例] 设置初始模式: {pattern_name}")
    create_pattern(grid, pattern_type)

    # 初始化视图
    app_core.view_manager.reset_view(grid.shape[1], grid.shape[0])

    # 运行主循环
    print("[示例] 开始主循环...")
    print("[示例] 按空格键切换模式，按R键重置视图，按G键切换网格显示，按C键清空网格")
    print("[示例] 按U键切换实时更新")
    print("[示例] 使用鼠标左键拖动视图，使用鼠标滚轮缩放视图")

    # 初始化 UI 管理器并注册回调（回调包含键盘、鼠标左键、清空等行为）
    ui_manager = UIManager(app_core, window, vector_calculator)

    # 定义空格键切换模式回调以保持原有行为
    def _on_space():
        nonlocal current_pattern
        current_pattern = (current_pattern + 1) % len(patterns)
        pattern_name, pattern_type = patterns[current_pattern]
        print(f"[示例] 切换到模式: {pattern_name}")
        create_pattern(grid, pattern_type)
        app_core.view_manager.reset_view(grid.shape[1], grid.shape[0])

    ui_manager.register_callbacks(grid, on_space=_on_space)

    while not window.should_close:
        # 更新窗口
        window.update()

        # 实时更新向量场
        if ui_manager.enable_update:
            # 使用计算模块的update_grid_with_adjacent_sum方法更新整个网格
            # 添加include_self=True参数，确保每个向量都包含自身的值
            vector_calculator.update_grid_with_adjacent_sum(grid, include_self=True)

        # 处理鼠标拖动
        try:
            ui_manager.process_mouse_drag()
        except Exception as e:
            print(f"[错误] 鼠标拖动处理异常: {e}")

        # 处理鼠标滚轮缩放
        ui_manager.process_scroll()

        # 更新标记位置（跟随向量场找到中心）
        try:
            ui_manager.update_markers(grid)
        except Exception as e:
            print(f"[错误] 更新标记异常: {e}")

        # 渲染
        window.render(grid)

    # 清理资源
    print("[示例] 清理资源...")
    window.cleanup()
    app_core.shutdown()

    print("[示例] 示例结束")

if __name__ == "__main__":
    main()
