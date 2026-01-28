import pytest
import time
from unittest.mock import Mock
from lizi_engine.core.state import StateManager, StateChange, state_manager


class TestStateChange:
    """测试状态变更记录"""

    def test_state_change_creation(self):
        """测试状态变更创建"""
        change = StateChange("test_key", "old_value", "new_value", 1234567890.0)

        assert change.key == "test_key"
        assert change.old_value == "old_value"
        assert change.new_value == "new_value"
        assert change.timestamp == 1234567890.0

    def test_state_change_to_dict(self):
        """测试状态变更转换为字典"""
        change = StateChange("test_key", "old", "new", 1234567890.0)
        data = change.to_dict()

        assert data["key"] == "test_key"
        assert data["old_value"] == "old"
        assert data["new_value"] == "new"
        assert data["timestamp"] == 1234567890.0


class TestStateManager:
    """测试状态管理器"""

    def setup_method(self):
        """测试前准备"""
        self.manager = StateManager()

    def test_state_manager_initialization(self):
        """测试状态管理器初始化"""
        assert self.manager._state == {}
        assert self.manager._listeners == {}
        assert self.manager._change_history == []
        assert self.manager._max_history_size == 100

    def test_get_set(self):
        """测试获取和设置状态"""
        # 测试设置和获取
        self.manager.set("test_key", "test_value")
        assert self.manager.get("test_key") == "test_value"

        # 测试默认值
        assert self.manager.get("nonexistent", "default") == "default"

    def test_set_without_notify(self):
        """测试设置状态时不通知"""
        called = []

        def listener(key, old, new):
            called.append((key, old, new))

        self.manager.add_listener("test_key", listener)

        self.manager.set("test_key", "value1", notify=False)
        assert called == []

        self.manager.set("test_key", "value2", notify=True)
        assert called == [("test_key", "value1", "value2")]

    def test_update(self):
        """测试批量更新"""
        updates = {"key1": "value1", "key2": "value2"}
        self.manager.update(updates)

        assert self.manager.get("key1") == "value1"
        assert self.manager.get("key2") == "value2"

    def test_remove(self):
        """测试移除状态"""
        self.manager.set("test_key", "test_value")
        assert self.manager.remove("test_key")
        assert self.manager.get("test_key") is None

        # 移除不存在的键
        assert not self.manager.remove("nonexistent")

    def test_clear(self):
        """测试清空所有状态"""
        self.manager.set("key1", "value1")
        self.manager.set("key2", "value2")

        self.manager.clear()

        assert self.manager.get("key1") is None
        assert self.manager.get("key2") is None

    def test_get_all(self):
        """测试获取所有状态"""
        self.manager.set("key1", "value1")
        self.manager.set("key2", "value2")

        all_state = self.manager.get_all()
        assert all_state == {"key1": "value1", "key2": "value2"}

    def test_contains(self):
        """测试检查状态是否存在"""
        self.manager.set("test_key", "value")
        assert self.manager.contains("test_key")
        assert not self.manager.contains("nonexistent")

    def test_add_remove_listener(self):
        """测试添加和移除监听器"""
        called = []

        def listener(key, old, new):
            called.append((key, old, new))

        # 添加监听器
        self.manager.add_listener("test_key", listener)
        assert listener in self.manager._listeners["test_key"]

        # 触发监听器
        self.manager.set("test_key", "value")
        assert called == [("test_key", None, "value")]

        # 移除监听器
        self.manager.remove_listener("test_key", listener)
        assert listener not in self.manager._listeners["test_key"]

        # 再次设置，不应该触发
        called.clear()
        self.manager.set("test_key", "new_value")
        assert called == []

    def test_get_change_history(self):
        """测试获取变更历史"""
        self.manager.set("key1", "value1")
        self.manager.set("key2", "value2")
        self.manager.set("key1", "new_value1")

        # 获取所有历史
        history = self.manager.get_change_history()
        assert len(history) == 3

        # 获取特定键的历史
        key1_history = self.manager.get_change_history("key1")
        assert len(key1_history) == 2
        assert key1_history[0].old_value is None
        assert key1_history[0].new_value == "value1"
        assert key1_history[1].old_value == "value1"
        assert key1_history[1].new_value == "new_value1"

        # 获取限制数量的历史
        limited_history = self.manager.get_change_history(limit=1)
        assert len(limited_history) == 1

    def test_create_restore_snapshot(self):
        """测试创建和恢复快照"""
        self.manager.set("key1", "value1")
        self.manager.set("key2", "value2")

        # 创建快照
        snapshot = self.manager.create_snapshot()
        assert snapshot["state"] == {"key1": "value1", "key2": "value2"}
        assert isinstance(snapshot["timestamp"], float)

        # 修改状态
        self.manager.set("key1", "modified")

        # 恢复快照
        self.manager.restore_snapshot(snapshot)
        assert self.manager.get("key1") == "value1"
        assert self.manager.get("key2") == "value2"

    def test_context_manager(self):
        """测试上下文管理器"""
        with self.manager:
            assert self.manager._nested_level == 1

            with self.manager:
                assert self.manager._nested_level == 2

            assert self.manager._nested_level == 1

        assert self.manager._nested_level == 0

    def test_magic_methods(self):
        """测试魔术方法"""
        # __getitem__ 和 __setitem__
        self.manager["test_key"] = "test_value"
        assert self.manager["test_key"] == "test_value"

        # __delitem__
        del self.manager["test_key"]
        assert self.manager.get("test_key") is None

        # __contains__
        self.manager.set("test_key", "value")
        assert "test_key" in self.manager
        assert "nonexistent" not in self.manager

        # __len__
        assert len(self.manager) == 1

        # __iter__
        keys = list(self.manager)
        assert keys == ["test_key"]

    def test_listener_exception_handling(self):
        """测试监听器异常处理"""
        def failing_listener(key, old, new):
            raise Exception("Test exception")

        def working_listener(key, old, new):
            working_listener.called = True

        working_listener.called = False

        self.manager.add_listener("test_key", failing_listener)
        self.manager.add_listener("test_key", working_listener)

        # 即使第一个监听器失败，第二个也应该被调用
        self.manager.set("test_key", "value")

        assert working_listener.called

    def test_history_size_limit(self):
        """测试历史记录大小限制"""
        # 设置较小的历史大小
        self.manager._max_history_size = 2

        for i in range(5):
            self.manager.set(f"key{i}", f"value{i}")

        history = self.manager.get_change_history()
        assert len(history) == 2  # 应该只保留最新的2条


class TestGlobalStateManager:
    """测试全局状态管理器"""

    def test_global_state_manager_exists(self):
        """测试全局状态管理器存在"""
        assert state_manager is not None
        assert isinstance(state_manager, StateManager)


if __name__ == "__main__":
    pytest.main([__file__])
