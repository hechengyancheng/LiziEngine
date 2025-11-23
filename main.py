from LiziLib import GRID_640x480
import viewLib
import numpy as np
from OpenGL.GL import *
from config import config
from app_core import app_core

def example(grid):
    # 设置一些初始值作为示例
    # 在中心创建一个旋转模式
    h, w = grid.shape[:2]
    cx, cy = w // 2, h // 2
    radius = min(w, h) // 4

    for y in range(h):
        for x in range(w):
            # 计算到中心的距离和角度
            dx = x - cx
            dy = y - cy
            dist = np.sqrt(dx**2 + dy**2)

            if dist < radius and dist > 0:
                # 创建切线向量（旋转）
                angle = np.arctan2(dy, dx) + np.pi/2
                magnitude = 1.0 - (dist / radius)  # 距离中心越远，向量越小
                grid[y, x] = (magnitude * np.cos(angle), magnitude * np.sin(angle))
    
    # 使用应用核心设置网格
    app_core.set_grid(grid)

def main():
    print("[INFO] 启动 LiziEngine...")
    
    # 初始化网格
    grid = GRID_640x480.copy()
    print(f"[INFO] 已初始化网格，尺寸: {grid.shape}")
    
    # 调用example函数来设置一些初始值
    print("[INFO] 正在设置初始向量场...")
    example(grid)
    print("[INFO] 初始向量场设置完成")

    # 运行 OpenGL 视图，启用工具栏
    show_grid = config.get("show_grid", True)
    use_gpu = config.get("use_gpu_compute", False)
    print(f"[INFO] 显示网格: {show_grid}, 使用GPU计算: {use_gpu}")
    # 不传递toolbar_draw_func，这样将使用新的ImGui工具栏
    try:
        print("[INFO] 启动OpenGL视图...")
        viewLib.run_opengl_view(grid, cell_size=1, show_grid=show_grid)
    except KeyboardInterrupt:
        print("\n[INFO] 程序被用户中断")
        return  # 正常退出，不重新抛出异常
    except Exception as e:
        print(f"\n[ERROR] 程序运行出错: {e}")
        raise
    
    finally:
        # 清理应用核心
        print("[INFO] 清理资源...")
        app_core.state.clear_listeners()
        print("[INFO] 程序已退出")

if __name__ == "__main__":
    main()
