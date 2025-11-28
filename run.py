
"""
LiziEngine 主入口文件
"""
import sys
import os
import numpy as np

# 确保当前目录在Python路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.controller import app_controller
from compute.vector_field import vector_calculator

def example(grid):
    """示例函数 - 在网格上创建一个旋转模式"""
    # 使用向量计算器创建切线向量模式
    vector_calculator.create_tangential_pattern(grid, magnitude=1.0)

    # 将网格设置到应用核心
    app_controller._app_core.grid_manager._grid = grid

    # 标记网格已更新
    app_controller._state_manager.set("grid_updated", True)

def main():
    """主函数"""
    print("[LiziEngine] 启动应用程序...")

    # 初始化应用控制器
    if not app_controller.initialize("LiziEngine", 800, 600):
        print("[LiziEngine] 初始化失败")
        return

    # 获取网格
    grid = app_controller._app_core.grid_manager.grid
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

if __name__ == "__main__":
    main()
