"""
渲染器模块 - 提供向量场的渲染功能
支持Dear PyGui渲染
"""
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import dearpygui.dearpygui as dpg
from ..core.config import config_manager
from ..core.events import Event, EventType, event_bus, EventHandler
from ..core.state import state_manager

class VectorFieldRenderer(EventHandler):
    """向量场渲染器"""
    def __init__(self):
        self._event_bus = event_bus
        self._state_manager = state_manager
        self._config_manager = config_manager

        # 渲染状态
        self._initialized = False

        # 订阅事件
        self._event_bus.subscribe(EventType.APP_INITIALIZED, self)

    def initialize(self) -> None:
        """初始化渲染器"""
        if self._initialized:
            return

        try:
            self._initialized = True
            print("[渲染器] 初始化成功")
        except Exception as e:
            print(f"[渲染器] 初始化失败: {e}")
            raise

    def set_drawlist(self, drawlist) -> None:
        """设置绘图列表"""
        self._drawlist = drawlist

    def render_vector_field(self, grid: np.ndarray, cell_size: float = 1.0,
                           cam_x: float = 0.0, cam_y: float = 0.0, cam_zoom: float = 1.0,
                           viewport_width: int = 800, viewport_height: int = 600) -> None:
        """渲染向量场"""
        if not self._initialized:
            self.initialize()

        if grid is None or not hasattr(self, '_drawlist'):
            return

        # 获取配置
        vector_color = self._config_manager.get("vector_color", [0.2, 0.6, 1.0])
        vector_scale = self._config_manager.get("vector_scale", 1.0)
        line_width = self._config_manager.get("line_width", 1.0)
        render_lines = self._config_manager.get("render_vector_lines", True)

        # 如果关闭渲染向量线条，直接返回
        if not render_lines:
            return

        # 准备顶点数据
        h, w = grid.shape[:2]

        # 使用向量化操作准备顶点数据，避免循环
        # 创建网格坐标
        y_coords, x_coords = np.mgrid[0:h, 0:w]

        # 获取向量分量
        vx = grid[:, :, 0]
        vy = grid[:, :, 1]

        # 创建非零向量掩码
        mask = (np.abs(vx) > 0.001) | (np.abs(vy) > 0.001)

        # 如果没有非零向量，直接返回
        if not np.any(mask):
            return

        # 提取非零向量的坐标和分量
        non_zero_x = x_coords[mask]
        non_zero_y = y_coords[mask]
        non_zero_vx = vx[mask] * vector_scale
        non_zero_vy = vy[mask] * vector_scale

        # 计算起点和终点坐标
        start_x = non_zero_x * cell_size
        start_y = non_zero_y * cell_size
        end_x = start_x + non_zero_vx
        end_y = start_y + non_zero_vy

        # 转换到屏幕坐标
        center_x = viewport_width / 2.0
        center_y = viewport_height / 2.0

        for i in range(len(non_zero_x)):
            sx = (start_x[i] - cam_x) * cam_zoom + center_x
            sy = (start_y[i] - cam_y) * cam_zoom + center_y
            ex = (end_x[i] - cam_x) * cam_zoom + center_x
            ey = (end_y[i] - cam_y) * cam_zoom + center_y

            # 绘制向量线
            dpg.draw_line([sx, sy], [ex, ey],
                         color=[int(vector_color[0]*255), int(vector_color[1]*255), int(vector_color[2]*255)],
                         thickness=line_width,
                         parent=self._drawlist)

    def render_grid(self, grid: np.ndarray, cell_size: float = 1.0,
                   cam_x: float = 0.0, cam_y: float = 0.0, cam_zoom: float = 1.0,
                   viewport_width: int = 800, viewport_height: int = 600) -> None:
        """渲染网格线"""
        if not self._initialized:
            self.initialize()

        if grid is None or not hasattr(self, '_drawlist'):
            return

        # 获取配置
        grid_color = self._config_manager.get("grid_color", [0.3, 0.3, 0.3])
        # 统一从配置管理器读取 show_grid，确保配置文件和运行时配置一致
        show_grid = self._config_manager.get("show_grid" , True)

        if not show_grid :
            return

        h, w = grid.shape[:2]

        # 转换到屏幕坐标
        center_x = viewport_width / 2.0
        center_y = viewport_height / 2.0

        # 水平线
        for y in range(h):
            py = y * cell_size
            sx = (0 - cam_x) * cam_zoom + center_x
            sy = (py - cam_y) * cam_zoom + center_y
            ex = ((w - 1) * cell_size - cam_x) * cam_zoom + center_x
            ey = (py - cam_y) * cam_zoom + center_y

            dpg.draw_line([sx, sy], [ex, ey],
                         color=[int(grid_color[0]*255), int(grid_color[1]*255), int(grid_color[2]*255)],
                         thickness=1.0,
                         parent=self._drawlist)

        # 垂直线
        for x in range(w):
            px = x * cell_size
            sx = (px - cam_x) * cam_zoom + center_x
            sy = (0 - cam_y) * cam_zoom + center_y
            ex = (px - cam_x) * cam_zoom + center_x
            ey = ((h - 1) * cell_size - cam_y) * cam_zoom + center_y

            dpg.draw_line([sx, sy], [ex, ey],
                         color=[int(grid_color[0]*255), int(grid_color[1]*255), int(grid_color[2]*255)],
                         thickness=1.0,
                         parent=self._drawlist)

    def render_background(self) -> None:
        """渲染背景"""
        # 获取配置
        bg_color = self._config_manager.get("background_color", [0.1, 0.1, 0.1])

        # 在Dear PyGui中，背景通过设置drawlist的clear_color来实现
        # 但这里我们不需要显式清除，因为drawlist会自动处理
        pass

    def render_markers(self, cell_size: float = 1.0,
                       cam_x: float = 0.0, cam_y: float = 0.0, cam_zoom: float = 1.0,
                       viewport_width: int = 800, viewport_height: int = 600) -> None:
        """渲染在 state_manager 中注册的标记（点）"""
        if not self._initialized:
            self.initialize()

        if not hasattr(self, '_drawlist'):
            return

        markers = self._state_manager.get("markers", [])
        if not markers:
            return

        # 统一颜色和点大小
        marker_color = self._config_manager.get("marker_color", [1.0, 0.2, 0.2])
        point_size = int(self._config_manager.get("marker_size", 8))

        # 转换到屏幕坐标
        center_x = viewport_width / 2.0
        center_y = viewport_height / 2.0

        for m in markers:
            try:
                gx = float(m.get("x", 0.0))
                gy = float(m.get("y", 0.0))
            except Exception:
                gx = 0.0
                gy = 0.0

            wx = gx * cell_size
            wy = gy * cell_size

            # 转换到屏幕坐标
            sx = (wx - cam_x) * cam_zoom + center_x
            sy = (wy - cam_y) * cam_zoom + center_y

            # 绘制圆点
            dpg.draw_circle([sx, sy], point_size / 2.0,
                           color=[int(marker_color[0]*255), int(marker_color[1]*255), int(marker_color[2]*255)],
                           fill=[int(marker_color[0]*255), int(marker_color[1]*255), int(marker_color[2]*255)],
                           parent=self._drawlist)

    def cleanup(self) -> None:
        """清理渲染器资源"""
        if not self._initialized:
            return

        try:
            self._initialized = False
            print("[渲染器] 资源清理完成")
        except Exception as e:
            print(f"[渲染器] 清理资源时出错: {e}")

    def handle(self, event: Event) -> None:
        """处理事件"""
        if event.type == EventType.APP_INITIALIZED:
            # 处理应用初始化事件
            pass

# 全局向量场渲染器实例
vector_field_renderer = VectorFieldRenderer()

# 便捷函数
def render_vector_field(grid: np.ndarray, cell_size: float = 1.0,
                       cam_x: float = 0.0, cam_y: float = 0.0, cam_zoom: float = 1.0,
                       viewport_width: int = 800, viewport_height: int = 600) -> None:
    """便捷函数：渲染向量场"""
    vector_field_renderer.render_vector_field(grid, cell_size, cam_x, cam_y, cam_zoom, viewport_width, viewport_height)

def render_grid(grid: np.ndarray, cell_size: float = 1.0,
               cam_x: float = 0.0, cam_y: float = 0.0, cam_zoom: float = 1.0,
               viewport_width: int = 800, viewport_height: int = 600) -> None:
    """便捷函数：渲染网格"""
    vector_field_renderer.render_grid(grid, cell_size, cam_x, cam_y, cam_zoom, viewport_width, viewport_height)

def render_background() -> None:
    """便捷函数：渲染背景"""
    vector_field_renderer.render_background()
