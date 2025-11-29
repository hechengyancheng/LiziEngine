
"""
事件系统模块 - 提供发布-订阅模式的事件通信机制
"""
import time
from enum import Enum
from typing import Dict, Any, Callable, Optional, List
import threading

class EventType(Enum):
    """事件类型枚举"""
    # 网格相关事件
    GRID_UPDATED = "grid_updated"
    GRID_UPDATE_REQUEST = "grid_update_request"
    GRID_CLEARED = "grid_cleared"
    GRID_LOADED = "grid_loaded"
    GRID_SAVED = "grid_saved"
    TOGGLE_GRID = "toggle_grid"
    CLEAR_GRID = "clear_grid"

    # 向量相关事件
    VECTOR_UPDATED = "vector_updated"
    SET_MAGNITUDE = "set_magnitude"
    TOGGLE_REVERSE_VECTOR = "toggle_reverse_vector"

    # 视图相关事件
    VIEW_CHANGED = "view_changed"
    VIEW_RESET = "view_reset"
    RESET_VIEW = "reset_view"

    # 工具栏相关事件
    SET_BRUSH_SIZE = "set_brush_size"

    # 应用程序事件
    APP_INITIALIZED = "app_initialized"
    APP_SHUTDOWN = "app_shutdown"

    # GPU计算事件
    GPU_COMPUTE_STARTED = "gpu_compute_started"
    GPU_COMPUTE_COMPLETED = "gpu_compute_completed"
    GPU_COMPUTE_ERROR = "gpu_compute_error"
    
    # 鼠标事件
    MOUSE_CLICKED = "mouse_clicked"
    MOUSE_MOVED = "mouse_moved"
    MOUSE_SCROLLED = "mouse_scrolled"
    
    # 键盘事件
    KEY_PRESSED = "key_pressed"
    KEY_RELEASED = "key_released"

class Event:
    """事件类"""
    def __init__(self, event_type: EventType, data: Optional[Dict[str, Any]] = None, source: Optional[str] = None):
        self.type = event_type
        self.data = data or {}
        self.source = source
        self.timestamp = time.time()

    def __str__(self):
        return f"Event(type={self.type}, source={self.source}, timestamp={self.timestamp})"

class EventHandler:
    """事件处理器接口"""
    def handle(self, event: Event) -> None:
        """处理事件"""
        pass

class EventBus:
    """线程安全的事件总线类"""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(EventBus, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._handlers: Dict[EventType, List[EventHandler]] = {}
            self._lock = threading.Lock()
            self._initialized = True
            self._config = None
            
    def _get_config_value(self, key: str, default: Any) -> Any:
        """获取配置值的辅助方法，避免循环导入"""
        if self._config is None:
            try:
                # 延迟导入配置管理器
                from .config import config_manager
                self._config = config_manager
            except ImportError:
                # 如果导入失败，返回默认值
                return default

        # 支持嵌套配置键
        if '.' in key:
            keys = key.split('.')
            value = self._config._settings
            try:
                for k in keys:
                    value = value[k]
                return value
            except (KeyError, TypeError):
                return default
        else:
            return self._config.get(key, default)

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """订阅事件"""
        with self._lock:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            if handler not in self._handlers[event_type]:
                self._handlers[event_type].append(handler)
                if self._get_config_value("ui.enable_event_output", True):
                    print(f"[事件系统] 订阅事件: {event_type}, 处理器: {handler.__class__.__name__}")

    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """取消订阅事件"""
        with self._lock:
            if event_type in self._handlers and handler in self._handlers[event_type]:
                self._handlers[event_type].remove(handler)
                if self._get_config_value("ui.enable_event_output", True):
                    print(f"[事件系统] 取消订阅事件: {event_type}, 处理器: {handler.__class__.__name__}")

    def publish(self, event: Event) -> None:
        """发布事件"""
        # 检查递归深度
        if not hasattr(self, '_recursion_depth'):
            self._recursion_depth = 0
            
        # 可配置的递归深度限制
        max_recursion_depth = self._get_config_value("ui.max_event_recursion_depth", 10)

        if self._recursion_depth > max_recursion_depth:
            if self._get_config_value("ui.enable_event_output", True):
                print(f"[事件系统] 警告: 事件递归深度超过限制 ({max_recursion_depth})，停止处理事件: {event.type}")
            return
            
        with self._lock:
            handlers = self._handlers.get(event.type, [])

        if self._get_config_value("ui.enable_event_output", True):
            print(f"[事件系统] 发布事件: {event.type}, 处理器数量: {len(handlers)}")
        
        # 增加递归深度
        self._recursion_depth += 1
        
        try:
            for handler in handlers:
                try:
                    handler.handle(event)
                except Exception as e:
                    if self._get_config_value("ui.enable_event_output", True):
                        print(f"[事件系统] 处理事件时出错: {e}")
                    # 触发错误处理事件
                    self._publish_error_event(event, e)
        finally:
            # 减少递归深度
            self._recursion_depth -= 1

    def _publish_error_event(self, original_event: Event, error: Exception) -> None:
        """发布错误事件"""
        try:
            error_event = Event(
                self._get_event_type("GPU_COMPUTE_ERROR", EventType.APP_SHUTDOWN),
                {"original_event": str(original_event), "error": str(error)},
                "EventBus"
            )
            self.publish(error_event)
        except Exception:
            # 如果发布错误事件也失败了，避免无限递归
            pass

    def _get_event_type(self, type_name: str, default_type):
        """安全地获取事件类型"""
        try:
            return getattr(EventType, type_name, default_type)
        except Exception:
            return default_type

    def clear(self) -> None:
        """清除所有事件处理器"""
        with self._lock:
            self._handlers.clear()
            if self._get_config_value("ui.enable_event_output", True):
                print("[事件系统] 已清除所有事件处理器")

# 全局事件总线实例
event_bus = EventBus()

class FunctionEventHandler(EventHandler):
    """将函数包装为事件处理器"""
    def __init__(self, func: Callable[[Event], None], name: Optional[str] = None):
        self.func = func
        self.name = name or func.__name__

    def handle(self, event: Event) -> None:
        """处理事件"""
        self.func(event)

    def __str__(self):
        return f"FunctionEventHandler({self.name})"
