"""
Controller 插件：处理复杂的业务逻辑
将 UI 模块中的复杂功能分离出来，便于维护和扩展。
"""
from typing import Tuple
import numpy as np
from lizi_engine.input import input_handler


class Controller:
    def __init__(self, app_core, vector_calculator, marker_system, grid: np.ndarray):
        self.app_core = app_core
        self.vector_calculator = vector_calculator
        self.marker_system = marker_system
        self.grid = grid

        # 向量场方向状态：True表示朝外，False表示朝内
        self.vector_field_direction = True

    def reset_view(self):
        """重置视图"""
        try:
            self.app_core.view_manager.reset_view(self.grid.shape[1], self.grid.shape[0])
        except Exception:
            pass

    def toggle_grid(self):
        """切换网格显示"""
        show_grid = self.app_core.state_manager.get("show_grid", True)
        self.app_core.state_manager.set("show_grid", not show_grid)

    def clear_grid(self):
        """清空网格"""
        self.grid.fill(0.0)

    def switch_vector_field_direction(self):
        """切换向量场方向"""
        self.vector_field_direction = not self.vector_field_direction
        direction = "朝外" if self.vector_field_direction else "朝内"
        print(f"[示例] 向量场方向已切换为: {direction}")

    def place_vector_field(self, mx: float, my: float):
        """在鼠标位置放置向量场"""
        try:
            cam_x = self.app_core.state_manager.get("cam_x", 0.0)
            cam_y = self.app_core.state_manager.get("cam_y", 0.0)
            cam_zoom = self.app_core.state_manager.get("cam_zoom", 1.0)
            viewport_width = self.app_core.state_manager.get("viewport_width", 800)
            viewport_height = self.app_core.state_manager.get("viewport_height", 600)
            cell_size = self.app_core.config_manager.get("cell_size", 1.0)

            world_x = cam_x + (mx - (viewport_width / 2.0)) / cam_zoom
            world_y = cam_y + (my - (viewport_height / 2.0)) / cam_zoom

            gx = world_x / cell_size
            gy = world_y / cell_size

            h, w = self.grid.shape[:2]
            if gx < 0 or gx >= w or gy < 0 or gy >= h:
                print(f"[示例] 点击位置超出网格: ({gx}, {gy})")
                return

            mag = 1
            magnitude = mag if self.vector_field_direction else -mag

            # 同时创建一个标记，初始放在点击处（浮点位置）
            self.marker_system.add_marker(gx, gy, float(magnitude))

            self.app_core.state_manager.update({"view_changed": True, "grid_updated": True})
        except Exception as e:
            print(f"[错误] 处理f键按下时发生异常: {e}")

    def handle_mouse_left_press(self, mx: float, my: float) -> dict:
        """处理鼠标左键按下，返回选中的标记"""
        try:
            cam_x = self.app_core.state_manager.get("cam_x", 0.0)
            cam_y = self.app_core.state_manager.get("cam_y", 0.0)
            cam_zoom = self.app_core.state_manager.get("cam_zoom", 1.0)
            viewport_width = self.app_core.state_manager.get("viewport_width", 800)
            viewport_height = self.app_core.state_manager.get("viewport_height", 600)
            cell_size = self.app_core.config_manager.get("cell_size", 1.0)

            world_x = cam_x + (mx - (viewport_width / 2.0)) / cam_zoom
            world_y = cam_y + (my - (viewport_height / 2.0)) / cam_zoom

            gx = world_x / cell_size
            gy = world_y / cell_size

            h, w = self.grid.shape[:2]
            if gx < 0 or gx >= w or gy < 0 or gy >= h:
                print(f"[示例] 点击位置超出网格: ({gx}, {gy})")
                return None

            # 获取所有标记
            markers = self.marker_system.get_markers()
            if not markers:
                print("[示例] 没有可用的标记")
                return None

            # 找到最近的标记
            min_dist = float('inf')
            closest_marker = None
            for marker in markers:
                marker_x = marker["x"]
                marker_y = marker["y"]
                dist = ((marker_x - gx) ** 2 + (marker_y - gy) ** 2) ** 0.5
                if dist < min_dist:
                    min_dist = dist
                    closest_marker = marker

            if closest_marker is None:
                print("[示例] 未找到最近的标记")
                return None
                   
            #self.app_core.state_manager.update({"view_changed": True, "grid_updated": True})

            return closest_marker
        except Exception as e:
            print(f"[错误] 处理鼠标左键按下时发生异常: {e}")
            return None

    def handle_mouse_drag(self, mx: float, my: float, selected_marker: dict):
        """处理鼠标拖拽时的向量添加"""
        if selected_marker is None:
            return

        try:
            cam_x = self.app_core.state_manager.get("cam_x", 0.0)
            cam_y = self.app_core.state_manager.get("cam_y", 0.0)
            cam_zoom = self.app_core.state_manager.get("cam_zoom", 1.0)
            viewport_width = self.app_core.state_manager.get("viewport_width", 800)
            viewport_height = self.app_core.state_manager.get("viewport_height", 600)
            cell_size = self.app_core.config_manager.get("cell_size", 1.0)

            world_x = cam_x + (mx - (viewport_width / 2.0)) / cam_zoom
            world_y = cam_y + (my - (viewport_height / 2.0)) / cam_zoom

            gx = float(world_x / cell_size)
            gy = float(world_y / cell_size)

            h, w = self.grid.shape[:2]
            if gx >= 0 and gx < w and gy >= 0 and gy < h:
                # 计算从标记到鼠标位置的方向向量
                vx = gx - selected_marker["x"]
                vy = gy - selected_marker["y"]
                # 归一化向量
                vec_len = (vx ** 2 + vy ** 2) ** 0.5
                if vec_len > 0:
                    vx /= vec_len *10
                    vy /= vec_len *10

                # 使用微小向量创建函数
                self.marker_system.add_vector_at_position(self.grid, x=selected_marker["x"], y=selected_marker["y"], vx=vx, vy=vy)

                self.app_core.state_manager.update({"view_changed": True, "grid_updated": True})
        except Exception as e:
            print(f"[错误] 处理左键持续按下时发生异常: {e}")

