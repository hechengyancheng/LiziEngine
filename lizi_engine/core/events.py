"""
事件系统模块 - 提供发布-订阅模式的事件通信机制
支持异步处理和事件过滤
"""
import time
import asyncio
import threading
from enum import Enum
from typing import Dict, Any, Callable, Optional, List, Union
from dataclasses import dataclass

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
    
    # 配置事件
    CONFIG_CHANGED = "config_changed"
    
    # 异步处理事件
    ASYNC_EVENT_PROCESSED = "async_event_processed"

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

@dataclass
class Event:
    """事件类"""
    type: EventType
    data: Optional[Dict[str, Any]] = None
    source: Optional[str] = None
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

    def __str__(self):
        return f"Event(type={self.type}, source={self.source}, timestamp={self.timestamp})"

class EventHandler:
    """事件处理器接口"""
    def handle(self, event: Event) -> None:
        """处理事件"""
        pass

    def can_handle(self, event: Event) -> bool:
        """检查是否可以处理该事件"""
        return True

class AsyncEventHandler(EventHandler):
    """异步事件处理器"""
    async def handle_async(self, event: Event) -> None:
        """异步处理事件"""
        pass

    def handle(self, event: Event) -> None:
        """同步处理事件，默认实现为调用异步版本"""
        try:
            loop = asyncio.get_running_loop()
            # 如果事件循环正在运行，创建任务但不等待
            loop.create_task(self.handle_async(event))
        except RuntimeError:
            # 如果没有运行中的事件循环，使用 asyncio.run
            asyncio.run(self.handle_async(event))

class EventFilter:
    """事件过滤器接口"""
    def filter(self, event: Event) -> bool:
        """过滤事件，返回True表示事件应该被处理"""
        return True

class EventTypeFilter(EventFilter):
    """基于事件类型的过滤器"""
    def __init__(self, *allowed_types: EventType):
        self.allowed_types = allowed_types

    def filter(self, event: Event) -> bool:
        return event.type in self.allowed_types

class EventSourceFilter(EventFilter):
    """基于事件源的过滤器"""
    def __init__(self, *allowed_sources: str):
        self.allowed_sources = allowed_sources

    def filter(self, event: Event) -> bool:
        return event.source in self.allowed_sources

class CompositeFilter(EventFilter):
    """组合过滤器，支持AND和OR逻辑"""
    def __init__(self, filters: List[EventFilter], logic: str = "AND"):
        self.filters = filters
        self.logic = logic.upper()

    def filter(self, event: Event) -> bool:
        if not self.filters:
            return True

        if self.logic == "AND":
            return all(f.filter(event) for f in self.filters)
        elif self.logic == "OR":
            return any(f.filter(event) for f in self.filters)
        else:
            raise ValueError(f"不支持逻辑操作: {self.logic}")

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

class AsyncFunctionEventHandler(AsyncEventHandler):
    """将异步函数包装为事件处理器"""
    def __init__(self, func: Callable[[Event], Any], name: Optional[str] = None):
        self.func = func
        self.name = name or func.__name__

    async def handle_async(self, event: Event) -> None:
        """异步处理事件"""
        await self.func(event)

    def __str__(self):
        return f"AsyncFunctionEventHandler({self.name})"

class EventBus:
    """线程安全的事件总线类"""
    def __init__(self):
        self._handlers: Dict[EventType, List[EventHandler]] = {}
        self._filters: Dict[EventType, List[EventFilter]] = {}
        self._lock = threading.Lock()
        self._recursion_depth = 0
        self._max_recursion_depth = 10
        self._async_enabled = True

    def subscribe(self, event_type: EventType, handler: EventHandler, 
                 filter: Optional[EventFilter] = None) -> None:
        """订阅事件"""
        with self._lock:
            if event_type not in self._handlers:
                self._handlers[event_type] = []

            if handler not in self._handlers[event_type]:
                self._handlers[event_type].append(handler)

                # 如果有过滤器，也保存过滤器
                if filter is not None:
                    if event_type not in self._filters:
                        self._filters[event_type] = []
                    self._filters[event_type].append(filter)

    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """取消订阅事件"""
        with self._lock:
            if event_type in self._handlers and handler in self._handlers[event_type]:
                self._handlers[event_type].remove(handler)

                # 如果该事件类型的所有处理器都移除了，也移除过滤器
                if event_type in self._filters and not self._handlers[event_type]:
                    self._filters.pop(event_type, None)

    def publish(self, event: Event) -> None:
        """发布事件"""
        # 检查递归深度
        if self._recursion_depth > self._max_recursion_depth:
            print(f"[事件系统] 警告: 事件递归深度超过限制 ({self._max_recursion_depth})，停止处理事件: {event.type}")
            return

        with self._lock:
            handlers = self._handlers.get(event.type, []).copy()
            filters = self._filters.get(event.type, []).copy()

        # 应用过滤器
        if filters:
            for filter in filters:
                if not filter.filter(event):
                    return

        # 增加递归深度
        self._recursion_depth += 1

        try:
            # 同步处理事件
            for handler in handlers:
                try:
                    handler.handle(event)
                except Exception as e:
                    print(f"[事件系统] 处理事件时出错: {e}")
                    # 触发错误处理事件
                    self._publish_error_event(event, e)

            # 如果启用异步处理，发布异步事件
            if self._async_enabled and event.type not in [EventType.ASYNC_EVENT_PROCESSED, EventType.APP_INITIALIZED]:
                async_event = Event(EventType.ASYNC_EVENT_PROCESSED, {
                    "original_event": str(event),
                    "async_enabled": True
                }, "EventBus")
                self.publish(async_event)
        finally:
            # 减少递归深度
            self._recursion_depth -= 1

    async def publish_async(self, event: Event) -> None:
        """异步发布事件"""
        # 检查递归深度
        if self._recursion_depth > self._max_recursion_depth:
            print(f"[事件系统] 警告: 事件递归深度超过限制 ({self._max_recursion_depth})，停止处理事件: {event.type}")
            return

        with self._lock:
            handlers = self._handlers.get(event.type, []).copy()
            filters = self._filters.get(event.type, []).copy()

        # 应用过滤器
        if filters:
            for filter in filters:
                if not filter.filter(event):
                    return

        # 增加递归深度
        self._recursion_depth += 1

        try:
            # 处理异步事件处理器
            async_tasks = []
            for handler in handlers:
                try:
                    if isinstance(handler, AsyncEventHandler):
                        async_tasks.append(handler.handle_async(event))
                    else:
                        handler.handle(event)
                except Exception as e:
                    print(f"[事件系统] 处理事件时出错: {e}")
                    # 触发错误处理事件
                    await self._publish_error_event_async(event, e)

            # 等待所有异步任务完成
            if async_tasks:
                await asyncio.gather(*async_tasks, return_exceptions=True)
        finally:
            # 减少递归深度
            self._recursion_depth -= 1

    def _publish_error_event(self, original_event: Event, error: Exception) -> None:
        """发布错误事件"""
        try:
            error_event = Event(
                EventType.GPU_COMPUTE_ERROR,
                {"original_event": str(original_event), "error": str(error)},
                "EventBus"
            )
            self.publish(error_event)
        except Exception:
            # 如果发布错误事件也失败了，避免无限递归
            pass

    async def _publish_error_event_async(self, original_event: Event, error: Exception) -> None:
        """异步发布错误事件"""
        try:
            error_event = Event(
                EventType.GPU_COMPUTE_ERROR,
                {"original_event": str(original_event), "error": str(error)},
                "EventBus"
            )
            await self.publish_async(error_event)
        except Exception:
            # 如果发布错误事件也失败了，避免无限递归
            pass

    def clear(self) -> None:
        """清除所有事件处理器和过滤器"""
        with self._lock:
            self._handlers.clear()
            self._filters.clear()

    def set_max_recursion_depth(self, depth: int) -> None:
        """设置最大递归深度"""
        self._max_recursion_depth = depth

    def enable_async(self, enabled: bool) -> None:
        """启用或禁用异步事件处理"""
        self._async_enabled = enabled

    def get_handler_count(self, event_type: EventType) -> int:
        """获取指定事件类型的处理器数量"""
        with self._lock:
            return len(self._handlers.get(event_type, []))

# 全局事件总线实例
event_bus = EventBus()
