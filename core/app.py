
"""
应用核心模块 - 提供应用程序的主要功能
"""
import os
import threading
import numpy as np
from typing import Optional, Dict, Any, Tuple
from .events import EventBus, Event, EventType, EventHandler
from .state import StateManager, state_manager
from compute.vector_field import vector_calculator

class GridManager:
    """网格数据管理器"""
    def __init__(self, state_manager: StateManager, event_bus: EventBus):
        self._state_manager = state_manager
        self._event_bus = event_bus
        self._lock = threading.Lock()
        self._grid = None

        # 初始化网格状态
        self._state_manager.set("grid_width", 640)
        self._state_manager.set("grid_height", 480)
        self._state_manager.set("grid_updated", False)

    @property
    def grid(self) -> Optional[np.ndarray]:
        """获取网格数据的副本"""
        with self._lock:
            return self._grid.copy() if self._grid is not None else None
            
    def get_raw_grid(self) -> Optional[np.ndarray]:
        """获取原始网格数据的引用（用于直接修改）"""
        with self._lock:
            return self._grid

    def init_grid(self, width: int = 640, height: int = 480, default: Tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
        """初始化网格"""
        with self._lock:
            self._grid = np.zeros((height, width, 2), dtype=np.float32)
            if default != (0.0, 0.0):
                self._grid[:, :, 0] = default[0]
                self._grid[:, :, 1] = default[1]

            # 更新状态
            self._state_manager.update({
                "grid_width": width,
                "grid_height": height,
                "grid_updated": True
            })

            # 发布事件
            self._event_bus.publish(Event(
                EventType.GRID_UPDATED,
                {"width": width, "height": height},
                "GridManager"
            ))

            return self._grid.copy()

    def update_grid(self, updates: Dict[Tuple[int, int], Tuple[float, float]]) -> None:
        """更新网格中的特定点"""
        with self._lock:
            if self._grid is None:
                print("[网格管理] 错误: 网格未初始化")
                return

            print(f"[网格管理] 收到更新请求，更新点数: {len(updates)}")
            changed = False
            for (y, x), (vx, vy) in updates.items():
                if 0 <= y < self._grid.shape[0] and 0 <= x < self._grid.shape[1]:
                    self._grid[y, x] = (vx, vy)
                    changed = True
                else:
                    print(f"[网格管理] 警告: 跳过超出范围的点 ({x}, {y})")

            if changed:
                print("[网格管理] 网格数据已更新")
                # 更新状态
                self._state_manager.set("grid_updated", True)

                # 发布事件
                self._event_bus.publish(Event(
                    EventType.GRID_UPDATED,
                    {"updates": updates},
                    "GridManager"
                ))

    def clear_grid(self) -> None:
        """清空网格"""
        with self._lock:
            if self._grid is not None:
                self._grid.fill(0.0)

                # 更新状态
                self._state_manager.set("grid_updated", True, notify=False)

                # 发布事件
                self._event_bus.publish(Event(
                    EventType.GRID_CLEARED,
                    {},
                    "GridManager"
                ))

    def load_grid(self, file_path: str) -> bool:
        """从文件加载网格"""
        try:
            if not os.path.exists(file_path):
                print(f"[网格管理] 文件不存在: {file_path}")
                return False

            loaded_grid = np.load(file_path)

            with self._lock:
                if self._grid is not None and loaded_grid.shape != self._grid.shape:
                    print(f"[网格管理] 网格尺寸不匹配: {loaded_grid.shape} vs {self._grid.shape}")
                    return False

                self._grid = loaded_grid.copy()

                # 更新状态
                self._state_manager.update({
                    "grid_width": loaded_grid.shape[1],
                    "grid_height": loaded_grid.shape[0],
                    "grid_updated": True
                })

                # 发布事件
                self._event_bus.publish(Event(
                    EventType.GRID_LOADED,
                    {"file_path": file_path, "shape": loaded_grid.shape},
                    "GridManager"
                ))

                return True
        except Exception as e:
            print(f"[网格管理] 加载网格失败: {e}")
            return False

    def save_grid(self, file_path: str) -> bool:
        """保存网格到文件"""
        try:
            with self._lock:
                if self._grid is not None:
                    np.save(file_path, self._grid)

                    # 发布事件
                    self._event_bus.publish(Event(
                        EventType.GRID_SAVED,
                        {"file_path": file_path, "shape": self._grid.shape},
                        "GridManager"
                    ))

                    return True
            return False
        except Exception as e:
            print(f"[网格管理] 保存网格失败: {e}")
            return False

class ViewManager:
    """视图管理器"""
    def __init__(self, state_manager: StateManager, event_bus: EventBus):
        self._state_manager = state_manager
        self._event_bus = event_bus

        # 初始化视图状态
        self._state_manager.update({
            "cam_x": 0.0,
            "cam_y": 0.0,
            "cam_zoom": 1.0,
            "view_changed": False,
            "show_grid": True
        })

    def reset_view(self, width: int = 640, height: int = 480) -> None:
        """重置视图到网格中心"""
        cell_size = 1
        cam_x = (width * cell_size) / 2.0
        cam_y = (height * cell_size) / 2.0
        cam_zoom = 1.0

        # 更新状态
        self._state_manager.update({
            "cam_x": cam_x,
            "cam_y": cam_y,
            "cam_zoom": cam_zoom,
            "view_changed": True
        })

        # 发布事件
        self._event_bus.publish(Event(
            EventType.VIEW_RESET,
            {"cam_x": cam_x, "cam_y": cam_y, "cam_zoom": cam_zoom},
            "ViewManager"
        ))
        
    def _reset_view_internal(self, width: int = 640, height: int = 480) -> None:
        """内部重置视图到网格中心，不发布事件"""
        cell_size = 1
        cam_x = (width * cell_size) / 2.0
        cam_y = (height * cell_size) / 2.0
        cam_zoom = 1.0
        
        # 更新状态，不触发事件
        with self._state_manager._lock:
            self._state_manager._state["cam_x"] = cam_x
            self._state_manager._state["cam_y"] = cam_y
            self._state_manager._state["cam_zoom"] = cam_zoom
            self._state_manager._state["view_changed"] = True

class AppCore:
    """应用核心类，整合各个管理器"""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(AppCore, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._state_manager = state_manager
            # 使用全局事件总线实例，而不是创建新的实例
            from .events import event_bus
            self._event_bus = event_bus

            # 初始化各个管理器
            self._grid_manager = GridManager(self._state_manager, self._event_bus)
            self._view_manager = ViewManager(self._state_manager, self._event_bus)
            self._vector_calculator = vector_calculator

            self._initialized = True

            # 发布应用初始化事件
            self._event_bus.publish(Event(
                EventType.APP_INITIALIZED,
                {},
                "AppCore"
            ))

    @property
    def state_manager(self) -> StateManager:
        """获取状态管理器"""
        return self._state_manager

    @property
    def event_bus(self) -> EventBus:
        """获取事件总线"""
        return self._event_bus

    @property
    def grid_manager(self) -> GridManager:
        """获取网格管理器"""
        return self._grid_manager

    @property
    def view_manager(self) -> ViewManager:
        """获取视图管理器"""
        return self._view_manager

    @property
    def vector_calculator(self):
        """获取向量计算器"""
        return self._vector_calculator

    def shutdown(self) -> None:
        """关闭应用核心"""
        # 发布应用关闭事件
        self._event_bus.publish(Event(
            EventType.APP_SHUTDOWN,
            {},
            "AppCore"
        ))

        # 清理资源
        self._state_manager.clear_listeners()
        self._event_bus.clear()

# 全局应用核心实例
app_core = AppCore()
