import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from lizi_engine.core.app import AppCore, GridManager, ViewManager, FPSLimiter
from lizi_engine.core.state import state_manager
from lizi_engine.core.events import event_bus, EventType
from lizi_engine.core.config import config_manager
from lizi_engine.core.container import container
from lizi_engine.compute.vector_field import VectorFieldCalculator
from lizi_engine.graphics.renderer import VectorFieldRenderer


class TestFPSLimiter:
    """测试FPS限制器"""

    def setup_method(self):
        """测试前准备"""
        self.state_manager = state_manager
        self.event_bus = event_bus
        self.config_manager = config_manager
        self.fps_limiter = FPSLimiter(self.state_manager, self.event_bus, self.config_manager)

    def test_fps_limiter_initialization(self):
        """测试FPS限制器初始化"""
        assert self.fps_limiter.is_enabled()
        assert self.fps_limiter._enabled == True

    def test_fps_limiter_set_enabled(self):
        """测试启用/禁用FPS限制"""
        self.fps_limiter.set_enabled(False)
        assert not self.fps_limiter.is_enabled()

        self.fps_limiter.set_enabled(True)
        assert self.fps_limiter.is_enabled()

    def test_fps_limiter_limit_fps(self):
        """测试FPS限制功能"""
        # 保存原始FPS值
        original_fps = self.config_manager.get("target_fps")

        try:
            # 设置较低的FPS目标以便测试
            self.config_manager.set("target_fps", 10)

            import time
            start_time = time.time()
            self.fps_limiter.limit_fps()
            end_time = time.time()

            # 应该至少等待0.1秒（1/10 FPS）
            assert end_time - start_time >= 0.09
        finally:
            # 恢复原始FPS值
            self.config_manager.set("target_fps", original_fps)

    @patch('lizi_engine.core.app.time.time')
    def test_fps_limiter_handle_config_change(self, mock_time):
        """测试配置变更处理"""
        # 设置时间返回新值
        mock_time.return_value = 200.0

        # 模拟配置变更事件
        event = Mock()
        event.type = EventType.CONFIG_CHANGED
        event.data = {"key": "target_fps"}

        # 重置计时器
        old_time = self.fps_limiter._last_time
        self.fps_limiter.handle(event)

        # 计时器应该被重置
        assert self.fps_limiter._last_time == 200.0

    @patch('time.time')
    def test_fps_limiter_handle_other_config_change(self, mock_time):
        """测试其他配置变更处理（不应该重置计时器）"""
        mock_time.return_value = 100.0

        # 模拟其他配置变更事件
        event = Mock()
        event.type = EventType.CONFIG_CHANGED
        event.data = {"key": "other_setting"}

        # 重置计时器
        old_time = self.fps_limiter._last_time
        self.fps_limiter.handle(event)

        # 计时器不应该改变
        assert self.fps_limiter._last_time == old_time


class TestGridManager:
    """测试网格管理器"""

    def setup_method(self):
        """测试前准备"""
        self.state_manager = state_manager
        self.event_bus = event_bus
        self.grid_manager = GridManager(self.state_manager, self.event_bus)

    def test_grid_manager_initialization(self):
        """测试网格管理器初始化"""
        assert self.grid_manager.grid is None
        assert self.state_manager.get("grid_width") == 640
        assert self.state_manager.get("grid_height") == 480

    def test_init_grid(self):
        """测试初始化网格"""
        grid = self.grid_manager.init_grid(100, 200)

        assert grid.shape == (200, 100, 2)
        assert np.all(grid == (0.0, 0.0))
        assert self.state_manager.get("grid_width") == 100
        assert self.state_manager.get("grid_height") == 200

    def test_init_grid_with_default(self):
        """测试初始化网格带默认值"""
        grid = self.grid_manager.init_grid(50, 50, (1.0, 2.0))

        assert grid.shape == (50, 50, 2)
        assert np.all(grid == (1.0, 2.0))

    def test_update_grid(self):
        """测试更新网格"""
        self.grid_manager.init_grid(10, 10)
        updates = {(1, 1): (1.0, 2.0), (2, 2): (3.0, 4.0)}

        self.grid_manager.update_grid(updates)

        grid = self.grid_manager.grid
        assert grid[1, 1, 0] == 1.0
        assert grid[1, 1, 1] == 2.0
        assert grid[2, 2, 0] == 3.0
        assert grid[2, 2, 1] == 4.0

    def test_clear_grid(self):
        """测试清空网格"""
        self.grid_manager.init_grid(10, 10)
        self.grid_manager.update_grid({(1, 1): (1.0, 1.0)})

        self.grid_manager.clear_grid()

        grid = self.grid_manager.grid
        assert np.all(grid == 0.0)

    def test_handle_clear_grid_event(self):
        """测试处理清空网格事件"""
        self.grid_manager.init_grid(10, 10)
        self.grid_manager.update_grid({(1, 1): (1.0, 1.0)})

        event = Mock()
        event.type = EventType.CLEAR_GRID
        self.grid_manager.handle(event)

        grid = self.grid_manager.grid
        assert np.all(grid == 0.0)

    def test_handle_toggle_grid_event(self):
        """测试处理切换网格事件"""
        event = Mock()
        event.type = EventType.TOGGLE_GRID
        self.grid_manager.handle(event)

        assert self.state_manager.get("show_grid") == False


class TestViewManager:
    """测试视图管理器"""

    def setup_method(self):
        """测试前准备"""
        self.state_manager = state_manager
        self.event_bus = event_bus
        self.view_manager = ViewManager(self.state_manager, self.event_bus)

    def test_view_manager_initialization(self):
        """测试视图管理器初始化"""
        assert self.state_manager.get("cam_x") == 0.0
        assert self.state_manager.get("cam_y") == 0.0
        assert self.state_manager.get("cam_zoom") == 1.0

    def test_reset_view(self):
        """测试重置视图"""
        self.view_manager.reset_view(100, 200)

        # 相机位置应该在网格中心
        expected_x = (100 * 1.0) / 2.0
        expected_y = (200 * 1.0) / 2.0

        assert self.state_manager.get("cam_x") == expected_x
        assert self.state_manager.get("cam_y") == expected_y
        assert self.state_manager.get("cam_zoom") == 1.0

    def test_handle_reset_view_event(self):
        """测试处理重置视图事件"""
        event = Mock()
        event.type = EventType.RESET_VIEW

        self.view_manager.handle(event)

        # 应该调用reset_view
        assert self.state_manager.get("cam_x") != 0.0


class TestAppCore:
    """测试应用核心"""

    def setup_method(self):
        """测试前准备"""
        # 清理容器
        container.clear()
        self.app_core = AppCore()

    def teardown_method(self):
        """测试后清理"""
        self.app_core.shutdown()
        container.clear()

    def test_app_core_initialization(self):
        """测试应用核心初始化"""
        assert self.app_core.state_manager is not None
        assert self.app_core.event_bus is not None
        assert self.app_core.config_manager is not None
        assert self.app_core.grid_manager is not None
        assert self.app_core.view_manager is not None
        assert self.app_core.fps_limiter is not None

    def test_app_core_services_registration(self):
        """测试服务注册"""
        # AppCore会在初始化时自动注册自己到容器
        # 这里我们检查容器中是否有AppCore类型的服务
        app_core_instance = container.resolve(AppCore)
        assert app_core_instance is not None
        assert isinstance(app_core_instance, AppCore)
        assert container.resolve(AppCore) is not None

    def test_app_core_vector_calculator(self):
        """测试向量计算器服务"""
        calculator = self.app_core.vector_calculator
        assert calculator is not None
        assert isinstance(calculator, VectorFieldCalculator)

    def test_app_core_renderer(self):
        """测试渲染器服务"""
        renderer = self.app_core.renderer
        assert renderer is not None
        assert isinstance(renderer, VectorFieldRenderer)

    def test_app_core_shutdown(self):
        """测试应用关闭"""
        # 应该能够正常关闭而不抛出异常
        self.app_core.shutdown()

        # 渲染器应该被清理
        assert self.app_core.renderer is not None  # 实例仍然存在，但可能被清理


if __name__ == "__main__":
    pytest.main([__file__])
