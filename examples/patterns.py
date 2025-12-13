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

class PatternType(Enum):
    """向量场模式类型"""
    TANGENTIAL = 1  # 切线模式（旋转）
    RADIAL = 2      # 径向模式（发散）
    CUSTOM = 3      # 自定义模式

def create_pattern(grid: np.ndarray, pattern_type: PatternType, **kwargs) -> None:
    """创建指定类型的向量场模式"""
    if pattern_type == PatternType.TANGENTIAL:
        vector_calculator.create_tangential_pattern(
            grid, 
            center=kwargs.get("center"), 
            radius=kwargs.get("radius"), 
            magnitude=kwargs.get("magnitude", 1.0)
        )
    elif pattern_type == PatternType.RADIAL:
        vector_calculator.create_radial_pattern(
            grid, 
            center=kwargs.get("center"), 
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
    grid = app_core.grid_manager.init_grid(64, 48)

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
    print("[示例] 使用鼠标左键拖动视图，使用鼠标滚轮缩放视图")

    # 初始化鼠标滚轮偏移
    mouse_scroll_y = 0

    # 注册鼠标滚轮回调函数
    def scroll_callback(window, xoffset, yoffset):
        nonlocal mouse_scroll_y
        mouse_scroll_y = yoffset

    # 设置鼠标滚轮回调
    glfw.set_scroll_callback(window._window, scroll_callback)

    while not window.should_close:
        # 更新窗口
        window.update()

        # 处理鼠标拖动
        if window._mouse_pressed:
            # 计算鼠标移动距离
            dx = window._mouse_x - window._last_mouse_x
            dy = window._mouse_y - window._last_mouse_y

            # 更新相机位置
            cam_speed = 0.1
            cam_x = app_core.state_manager.get("cam_x", 0.0) - dx * cam_speed
            cam_y = app_core.state_manager.get("cam_y", 0.0) - dy * cam_speed  # 修复上下拖动方向

            app_core.state_manager.update({
                "cam_x": cam_x,
                "cam_y": cam_y,
                "view_changed": True
            })

            # 更新最后鼠标位置
            window._last_mouse_x = window._mouse_x
            window._last_mouse_y = window._mouse_y

        # 处理鼠标滚轮缩放
        if mouse_scroll_y != 0:
            # 更新相机缩放
            cam_zoom = app_core.state_manager.get("cam_zoom", 1.0)
            zoom_speed = 0.2
            cam_zoom += mouse_scroll_y * zoom_speed  # 修复缩放方向

            # 限制缩放范围
            cam_zoom = max(0.1, min(10.0, cam_zoom))

            app_core.state_manager.update({
                "cam_zoom": cam_zoom,
                "view_changed": True
            })

            # 重置滚轮偏移
            mouse_scroll_y = 0

        # 渲染
        window.render(grid)

        # 处理键盘事件
        # 注意：这里简化了键盘事件处理，实际应该通过事件系统处理
        # 检查空格键是否被按下
        if window._keys.get(glfw.KEY_SPACE):
            # 切换到下一个模式
            current_pattern = (current_pattern + 1) % len(patterns)
            pattern_name, pattern_type = patterns[current_pattern]
            print(f"[示例] 切换到模式: {pattern_name}")

            # 创建新模式
            create_pattern(grid, pattern_type)

            # 重置视图
            app_core.view_manager.reset_view(grid.shape[1], grid.shape[0])

            # 清除按键状态，避免连续切换
            window._keys[glfw.KEY_SPACE] = False

        # 处理R键重置视图
        if window._keys.get(glfw.KEY_R):
            app_core.view_manager.reset_view(grid.shape[1], grid.shape[0])
            window._keys[glfw.KEY_R] = False

        # 处理G键切换网格显示
        if window._keys.get(glfw.KEY_G):
            show_grid = app_core.state_manager.get("show_grid", True)
            app_core.state_manager.set("show_grid", not show_grid)
            window._keys[glfw.KEY_G] = False

        # 处理C键清空网格
        if window._keys.get(glfw.KEY_C):
            grid.fill(0.0)
            window._keys[glfw.KEY_C] = False

    # 清理资源
    print("[示例] 清理资源...")
    window.cleanup()
    app_core.shutdown()

    print("[示例] 示例结束")

if __name__ == "__main__":
    # 导入glfw
    import glfw
    main()
