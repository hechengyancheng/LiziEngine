
"""
应用核心模块，提供统一的事件系统和状态管理
"""

import numpy as np
import os
import threading
import time
from enum import Enum
from typing import Dict, Any, Callable, Optional

class EventType(Enum):
    """事件类型枚举"""
    GRID_UPDATED = "grid_updated"
    VIEW_CHANGED = "view_changed"
    RESET_VIEW = "reset_view"
    CLEAR_GRID = "clear_grid"
    LOAD_GRID = "load_grid"
    SAVE_GRID = "save_grid"
    TOOLBAR_ACTION = "toolbar_action"

class Event:
    """事件类"""
    def __init__(self, event_type: EventType, data: Optional[Dict[str, Any]] = None):
        self.type = event_type
        self.data = data or {}
        self.timestamp = time.time()

class EventHandler:
    """事件处理器接口"""
    def handle(self, event: Event) -> None:
        """处理事件"""
        pass

class AppCoreEventHandler:
    """应用核心事件处理器包装类"""
    
    def __init__(self, callback):
        self.callback = callback
    
    def handle(self, event):
        """处理事件"""
        self.callback(event)

class AppState:
    """应用状态管理类"""
    def __init__(self):
        self._state = {}
        self._lock = threading.Lock()
        self._listeners = []

    def get(self, key: str, default: Any = None) -> Any:
        """获取状态值"""
        with self._lock:
            return self._state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """设置状态值"""
        with self._lock:
            old_value = self._state.get(key)
            self._state[key] = value
            # 通知监听器状态已改变
            if old_value != value:
                self._notify_listeners(key, value)

    def update(self, updates: Dict[str, Any]) -> None:
        """批量更新状态"""
        with self._lock:
            changed = False
            for key, value in updates.items():
                old_value = self._state.get(key)
                if old_value != value:
                    self._state[key] = value
                    changed = True
            # 通知监听器状态已改变
            if changed:
                self._notify_listeners(None, None)

    def add_listener(self, listener: Callable[[str, Any], None]) -> None:
        """添加状态变化监听器"""
        with self._lock:
            if listener not in self._listeners:
                self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[str, Any], None]) -> None:
        """移除状态变化监听器"""
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)
    
    def clear_listeners(self) -> None:
        """清空所有监听器"""
        with self._lock:
            self._listeners.clear()

    def _notify_listeners(self, key: Optional[str], value: Any) -> None:
        """通知所有监听器状态已改变"""
        for listener in self._listeners:
            try:
                listener(key, value)
            except Exception as e:
                print(f"通知监听器时出错: {e}")

class EventBus:
    """事件总线类"""
    def __init__(self):
        self._handlers = {}
        self._lock = threading.Lock()

    def subscribe(self, event_type: EventType, handler) -> None:
        """订阅事件"""
        with self._lock:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            if handler not in self._handlers[event_type]:
                self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: EventType, handler) -> None:
        """取消订阅事件"""
        with self._lock:
            if event_type in self._handlers and handler in self._handlers[event_type]:
                self._handlers[event_type].remove(handler)

    def publish(self, event: Event) -> None:
        """发布事件"""
        with self._lock:
            handlers = self._handlers.get(event.type, [])
            #print(f"发布事件: {event.type}, 处理器数量: {len(handlers)}")
            for handler in handlers:
                try:
                    #print(f"调用处理器: {handler}")
                    handler.handle(event)
                    #print(f"事件处理完成: {event.type}")
                except Exception as e:
                    print(f"处理事件时出错: {e}")

class AppCore:
    """应用核心类，提供统一的事件系统和状态管理"""
    def __init__(self):
        self.state = AppState()
        self.event_bus = EventBus()
        self.grid = None
        self.grid_lock = threading.Lock()
        self.grid_updated_event = threading.Event()

        # 初始化默认状态
        self.state.set("grid_updated", False)
        self.state.set("view_changed", False)
        self.state.set("cam_x", 0)
        self.state.set("cam_y", 0)
        self.state.set("cam_zoom", 1.0)
        self.state.set("show_grid", True)

        # 订阅内部事件
        self.event_bus.subscribe(EventType.GRID_UPDATED, AppCoreEventHandler(self._on_grid_updated))
        self.event_bus.subscribe(EventType.VIEW_CHANGED, AppCoreEventHandler(self._on_view_changed))

    def _on_grid_updated(self, event: Event) -> None:
        """处理网格更新事件"""
        self.grid_updated_event.set()
        self.state.set("grid_updated", True)

    def _on_view_changed(self, event: Event) -> None:
        """处理视图变化事件"""
        self.state.set("view_changed", True)

    def set_grid(self, grid: np.ndarray) -> None:
        """设置网格数据"""
        with self.grid_lock:
            self.grid = grid.copy()
            # 发布网格更新事件
            self.event_bus.publish(Event(EventType.GRID_UPDATED))

    def update_grid(self, updates: Dict[tuple, tuple]) -> None:
        """更新网格数据"""
        with self.grid_lock:
            if self.grid is not None:
                for (y, x), (vx, vy) in updates.items():
                    if 0 <= y < self.grid.shape[0] and 0 <= x < self.grid.shape[1]:
                        self.grid[y, x] = (vx, vy)
                # 发布网格更新事件
                self.event_bus.publish(Event(EventType.GRID_UPDATED))

    def clear_grid(self) -> None:
        """清空网格"""
        with self.grid_lock:
            if self.grid is not None:
                self.grid[:] = np.zeros_like(self.grid)
                # 发布网格更新事件
                self.event_bus.publish(Event(EventType.GRID_UPDATED))

    def reset_view(self) -> None:
        """重置视图"""
        if self.grid is not None:
            cell_size = 1
            cam_x = (self.grid.shape[1] * cell_size) / 2.0
            cam_y = (self.grid.shape[0] * cell_size) / 2.0
            cam_zoom = 1.0

            # 更新状态
            self.state.update({
                "cam_x": cam_x,
                "cam_y": cam_y,
                "cam_zoom": cam_zoom,
                "view_changed": True
            })

            # 发布视图变化事件
            self.event_bus.publish(Event(EventType.VIEW_CHANGED, {
                "cam_x": cam_x,
                "cam_y": cam_y,
                "cam_zoom": cam_zoom
            }))

    def load_grid(self, file_path: str) -> bool:
        """从文件加载网格"""
        try:
            if not os.path.exists(file_path):
                return False

            loaded_grid = np.load(file_path)

            if self.grid is not None and loaded_grid.shape != self.grid.shape:
                return False

            # 设置网格数据
            self.set_grid(loaded_grid)
            return True
        except Exception as e:
            print(f"加载网格失败: {e}")
            return False

    def save_grid(self, file_path: str) -> bool:
        """保存网格到文件"""
        try:
            with self.grid_lock:
                if self.grid is not None:
                    np.save(file_path, self.grid)
                    return True
            return False
        except Exception as e:
            print(f"保存网格失败: {e}")
            return False

# 创建全局应用核心实例
app_core = AppCore()
