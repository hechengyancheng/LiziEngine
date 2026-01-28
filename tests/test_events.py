import pytest
import asyncio
from unittest.mock import Mock, MagicMock
from lizi_engine.core.events import (
    Event, EventType, EventHandler, AsyncEventHandler,
    EventFilter, EventTypeFilter, EventSourceFilter, CompositeFilter,
    FunctionEventHandler, AsyncFunctionEventHandler, EventBus, event_bus
)


class TestEvent:
    """测试事件类"""

    def test_event_creation(self):
        """测试事件创建"""
        event = Event(EventType.APP_INITIALIZED, {"key": "value"}, "test_source")

        assert event.type == EventType.APP_INITIALIZED
        assert event.data == {"key": "value"}
        assert event.source == "test_source"
        assert isinstance(event.timestamp, float)

    def test_event_str_representation(self):
        """测试事件字符串表示"""
        event = Event(EventType.APP_INITIALIZED)
        str_repr = str(event)

        assert "Event(type=EventType.APP_INITIALIZED" in str_repr
        assert "timestamp=" in str_repr


class TestEventHandler:
    """测试事件处理器"""

    def test_event_handler_handle(self):
        """测试事件处理器处理事件"""
        handler = EventHandler()
        event = Event(EventType.APP_INITIALIZED)

        # 默认实现不应该抛出异常
        handler.handle(event)

    def test_event_handler_can_handle(self):
        """测试事件处理器是否可以处理事件"""
        handler = EventHandler()
        event = Event(EventType.APP_INITIALIZED)

        assert handler.can_handle(event)


class TestAsyncEventHandler:
    """测试异步事件处理器"""

    def test_async_event_handler_handle(self):
        """测试异步事件处理器同步处理"""
        handler = AsyncEventHandler()
        event = Event(EventType.APP_INITIALIZED)

        # 应该能够正常处理
        handler.handle(event)

    @pytest.mark.asyncio
    async def test_async_event_handler_handle_async(self):
        """测试异步事件处理器异步处理"""
        handler = AsyncEventHandler()
        event = Event(EventType.APP_INITIALIZED)

        # 应该能够正常处理
        await handler.handle_async(event)


class TestEventFilters:
    """测试事件过滤器"""

    def test_event_type_filter(self):
        """测试事件类型过滤器"""
        filter = EventTypeFilter(EventType.APP_INITIALIZED, EventType.APP_SHUTDOWN)

        event1 = Event(EventType.APP_INITIALIZED)
        event2 = Event(EventType.GRID_UPDATED)

        assert filter.filter(event1)
        assert not filter.filter(event2)

    def test_event_source_filter(self):
        """测试事件源过滤器"""
        filter = EventSourceFilter("source1", "source2")

        event1 = Event(EventType.APP_INITIALIZED, source="source1")
        event2 = Event(EventType.APP_INITIALIZED, source="source3")

        assert filter.filter(event1)
        assert not filter.filter(event2)

    def test_composite_filter_and(self):
        """测试组合过滤器AND逻辑"""
        type_filter = EventTypeFilter(EventType.APP_INITIALIZED)
        source_filter = EventSourceFilter("test")
        composite = CompositeFilter([type_filter, source_filter], "AND")

        event1 = Event(EventType.APP_INITIALIZED, source="test")
        event2 = Event(EventType.APP_INITIALIZED, source="other")
        event3 = Event(EventType.GRID_UPDATED, source="test")

        assert composite.filter(event1)
        assert not composite.filter(event2)
        assert not composite.filter(event3)

    def test_composite_filter_or(self):
        """测试组合过滤器OR逻辑"""
        type_filter = EventTypeFilter(EventType.APP_INITIALIZED)
        source_filter = EventSourceFilter("test")
        composite = CompositeFilter([type_filter, source_filter], "OR")

        event1 = Event(EventType.APP_INITIALIZED, source="other")
        event2 = Event(EventType.GRID_UPDATED, source="test")
        event3 = Event(EventType.GRID_UPDATED, source="other")

        assert composite.filter(event1)
        assert composite.filter(event2)
        assert not composite.filter(event3)


class TestFunctionEventHandlers:
    """测试函数事件处理器"""

    def test_function_event_handler(self):
        """测试函数事件处理器"""
        called = []

        def callback(event):
            called.append(event.type)

        handler = FunctionEventHandler(callback, "test_handler")
        event = Event(EventType.APP_INITIALIZED)

        handler.handle(event)

        assert called == [EventType.APP_INITIALIZED]
        assert str(handler) == "FunctionEventHandler(test_handler)"

    @pytest.mark.asyncio
    async def test_async_function_event_handler(self):
        """测试异步函数事件处理器"""
        called = []

        async def callback(event):
            called.append(event.type)

        handler = AsyncFunctionEventHandler(callback, "test_handler")
        event = Event(EventType.APP_INITIALIZED)

        await handler.handle_async(event)

        assert called == [EventType.APP_INITIALIZED]
        assert str(handler) == "AsyncFunctionEventHandler(test_handler)"


class TestEventBus:
    """测试事件总线"""

    def setup_method(self):
        """测试前准备"""
        self.event_bus = EventBus()

    def test_event_bus_initialization(self):
        """测试事件总线初始化"""
        assert self.event_bus._handlers == {}
        assert self.event_bus._filters == {}
        assert self.event_bus._recursion_depth == 0
        assert self.event_bus._max_recursion_depth == 10
        assert self.event_bus._async_enabled == True

    def test_subscribe_and_publish(self):
        """测试订阅和发布事件"""
        called = []

        class TestHandler(EventHandler):
            def handle(self, event):
                called.append(event.type)

        handler = TestHandler()
        self.event_bus.subscribe(EventType.APP_INITIALIZED, handler)

        event = Event(EventType.APP_INITIALIZED)
        self.event_bus.publish(event)

        assert called == [EventType.APP_INITIALIZED]

    def test_unsubscribe(self):
        """测试取消订阅"""
        called = []

        class TestHandler(EventHandler):
            def handle(self, event):
                called.append(event.type)

        handler = TestHandler()
        self.event_bus.subscribe(EventType.APP_INITIALIZED, handler)
        self.event_bus.unsubscribe(EventType.APP_INITIALIZED, handler)

        event = Event(EventType.APP_INITIALIZED)
        self.event_bus.publish(event)

        assert called == []

    def test_publish_with_filter(self):
        """测试带过滤器的发布"""
        called = []

        class TestHandler(EventHandler):
            def handle(self, event):
                called.append(event.type)

        handler = TestHandler()
        filter = EventTypeFilter(EventType.APP_INITIALIZED)
        self.event_bus.subscribe(EventType.APP_INITIALIZED, handler, filter)

        event1 = Event(EventType.APP_INITIALIZED)
        event2 = Event(EventType.GRID_UPDATED)

        self.event_bus.publish(event1)
        self.event_bus.publish(event2)

        assert called == [EventType.APP_INITIALIZED]

    def test_recursion_depth_limit(self):
        """测试递归深度限制"""
        called = []

        class RecursiveHandler(EventHandler):
            def __init__(self, event_bus):
                self.event_bus = event_bus

            def handle(self, event):
                called.append(event.type)
                if len(called) < 15:  # 超过默认限制
                    self.event_bus.publish(Event(event.type))

        handler = RecursiveHandler(self.event_bus)
        self.event_bus.subscribe(EventType.APP_INITIALIZED, handler)

        event = Event(EventType.APP_INITIALIZED)
        self.event_bus.publish(event)

        # 应该被递归深度限制停止
        assert len(called) == 10  # 默认最大递归深度

    def test_async_publish(self):
        """测试异步发布"""
        called = []

        class AsyncTestHandler(AsyncEventHandler):
            async def handle_async(self, event):
                called.append(event.type)

        handler = AsyncTestHandler()
        self.event_bus.subscribe(EventType.APP_INITIALIZED, handler)

        event = Event(EventType.APP_INITIALIZED)

        async def run_test():
            await self.event_bus.publish_async(event)
            assert called == [EventType.APP_INITIALIZED]

        asyncio.run(run_test())

    def test_clear(self):
        """测试清除所有处理器"""
        class TestHandler(EventHandler):
            def handle(self, event):
                pass

        handler = TestHandler()
        self.event_bus.subscribe(EventType.APP_INITIALIZED, handler)

        assert self.event_bus.get_handler_count(EventType.APP_INITIALIZED) == 1

        self.event_bus.clear()

        assert self.event_bus.get_handler_count(EventType.APP_INITIALIZED) == 0

    def test_set_max_recursion_depth(self):
        """测试设置最大递归深度"""
        self.event_bus.set_max_recursion_depth(5)
        assert self.event_bus._max_recursion_depth == 5

    def test_enable_async(self):
        """测试启用/禁用异步"""
        self.event_bus.enable_async(False)
        assert not self.event_bus._async_enabled

        self.event_bus.enable_async(True)
        assert self.event_bus._async_enabled


class TestGlobalEventBus:
    """测试全局事件总线"""

    def test_global_event_bus_exists(self):
        """测试全局事件总线存在"""
        assert event_bus is not None
        assert isinstance(event_bus, EventBus)


if __name__ == "__main__":
    pytest.main([__file__])
