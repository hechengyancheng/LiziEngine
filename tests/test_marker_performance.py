"""
标记性能测试 - 测量不同标记数量的更新时间
"""
import pytest
import time
import random
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lizi_engine.core.container import container
from lizi_engine.core.app import AppCore
from lizi_engine.compute.vector_field import vector_calculator
from plugins.marker_system import MarkerSystem


@pytest.fixture(scope="module")
def setup_performance_test():
    """设置性能测试环境"""
    # 初始化应用核心
    app_core = container.resolve(AppCore)
    if app_core is None or isinstance(app_core, type):
        app_core = AppCore()
        container.register_singleton(AppCore, app_core)

    # 创建网格
    grid = app_core.grid_manager.init_grid(128, 128)

    # 初始化标记系统
    marker_system = MarkerSystem(app_core)

    yield app_core, grid, marker_system

    # 清理资源
    app_core.shutdown()


def test_marker_update_performance(setup_performance_test):
    """测试不同标记数量的标记更新性能"""
    app_core, grid, marker_system = setup_performance_test

    marker_counts = [100, 1000, 10000]
    h, w = grid.shape[:2]

    print("\n=== 标记更新性能测试 ===")
    print(f"网格大小: {w}x{h}")

    for count in marker_counts:
        print(f"\n--- 测试 {count} 个标记 ---")

        # 清除现有标记
        marker_system.clear_markers()

        # 在随机位置添加标记
        for _ in range(count):
            x = random.uniform(0, w - 1)
            y = random.uniform(0, h - 1)
            marker_system.add_marker(x, y)

        # 准备位置用于直接计时
        marker_positions = [(m['x'], m['y']) for m in marker_system.get_markers()]
        tiny_positions = [(m['x'], m['y'], m['mag']) for m in marker_system.get_markers()]

        # 计时 update_markers
        start_time = time.perf_counter()
        marker_system.update_markers(grid, dt=1.0, gravity=0.01, speed_factor=0.9, tiny_vectors=True)
        update_time = time.perf_counter() - start_time

        # 直接计时 fit_vectors_at_positions_batch
        start_time = time.perf_counter()
        fitted_vectors = vector_calculator.fit_vectors_at_positions_batch(grid, marker_positions)
        fit_time = time.perf_counter() - start_time

        # 直接计时 create_tiny_vectors_batch
        start_time = time.perf_counter()
        vector_calculator.create_tiny_vectors_batch(grid, tiny_positions)
        tiny_time = time.perf_counter() - start_time

        print(f"update_markers 时间: {update_time:.4f}s")
        print(f"fit_vectors_at_positions_batch 时间: {fit_time:.4f}s")
        print(f"create_tiny_vectors_batch 时间: {tiny_time:.4f}s")
        print(f"每个标记总时间: {(update_time)/count:.6f}s")
        print(f"每个标记拟合时间: {fit_time/count:.6f}s")

    print("\n=== 性能测试完成 ===")
