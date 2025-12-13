"""
状态管理模块 - 提供统一的状态管理功能
支持状态变更通知和状态快照
"""
import threading
import time
import copy
from typing import Any, Dict, Callable, Optional, List, Union
from dataclasses import dataclass, asdict
from .events import Event, EventType, event_bus

@dataclass
class StateChange:
    """状态变更记录"""
    key: str
    old_value: Any
    new_value: Any
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

class StateManager:
    """状态管理器"""
    def __init__(self):
        self._state: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._listeners: Dict[str, List[Callable]] = {}
        self._change_history: List[StateChange] = []
        self._max_history_size = 100
        self._nested_level = 0

    def get(self, key: str, default: Any = None) -> Any:
        """获取状态值"""
        with self._lock:
            return self._state.get(key, default)

    def set(self, key: str, value: Any, notify: bool = True) -> None:
        """设置状态值"""
        with self._lock:
            old_value = self._state.get(key)
            self._state[key] = value

            # 记录变更历史
            change = StateChange(key, old_value, value, time.time())
            self._change_history.append(change)

            # 限制历史记录大小
            if len(self._change_history) > self._max_history_size:
                self._change_history.pop(0)

            # 如果需要通知监听器
            if notify and old_value != value:
                self._notify_listeners(key, old_value, value)

    def update(self, updates: Dict[str, Any], notify: bool = True) -> None:
        """批量更新状态"""
        with self._lock:
            for key, value in updates.items():
                self.set(key, value, notify=notify)

    def remove(self, key: str) -> bool:
        """移除状态"""
        with self._lock:
            if key in self._state:
                old_value = self._state.pop(key)

                # 记录变更历史
                change = StateChange(key, old_value, None, time.time())
                self._change_history.append(change)

                # 通知监听器
                if old_value is not None:
                    self._notify_listeners(key, old_value, None)

                return True
            return False

    def clear(self) -> None:
        """清空所有状态"""
        with self._lock:
            old_state = copy.deepcopy(self._state)
            self._state.clear()

            # 记录变更历史
            change = StateChange("*", old_state, {}, time.time())
            self._change_history.append(change)

            # 通知所有监听器
            for key in old_state:
                self._notify_listeners(key, old_state[key], None)

    def get_all(self) -> Dict[str, Any]:
        """获取所有状态"""
        with self._lock:
            return copy.deepcopy(self._state)

    def contains(self, key: str) -> bool:
        """检查状态是否存在"""
        with self._lock:
            return key in self._state

    def add_listener(self, key: str, callback: Callable[[str, Any, Any], None]) -> None:
        """添加状态变更监听器"""
        with self._lock:
            if key not in self._listeners:
                self._listeners[key] = []
            if callback not in self._listeners[key]:
                self._listeners[key].append(callback)

    def remove_listener(self, key: str, callback: Callable[[str, Any, Any], None]) -> None:
        """移除状态变更监听器"""
        with self._lock:
            if key in self._listeners and callback in self._listeners[key]:
                self._listeners[key].remove(callback)

    def clear_listeners(self) -> None:
        """清除所有监听器"""
        with self._lock:
            self._listeners.clear()

    def _notify_listeners(self, key: str, old_value: Any, new_value: Any) -> None:
        """通知状态变更监听器"""
        if key in self._listeners:
            for callback in self._listeners[key]:
                try:
                    callback(key, old_value, new_value)
                except Exception as e:
                    print(f"[状态管理] 通知监听器时出错: {e}")

    def get_change_history(self, key: Optional[str] = None, limit: Optional[int] = None) -> List[StateChange]:
        """获取变更历史"""
        with self._lock:
            history = self._change_history

            # 如果指定了key，只返回该key的变更历史
            if key is not None:
                history = [change for change in history if change.key == key]

            # 如果指定了limit，只返回最近limit条记录
            if limit is not None and limit > 0:
                history = history[-limit:]

            return copy.deepcopy(history)

    def create_snapshot(self) -> Dict[str, Any]:
        """创建状态快照"""
        with self._lock:
            return {
                "state": copy.deepcopy(self._state),
                "timestamp": time.time()
            }

    def restore_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """从快照恢复状态"""
        with self._lock:
            old_state = copy.deepcopy(self._state)
            self._state = copy.deepcopy(snapshot["state"])

            # 记录变更历史
            change = StateChange("*", old_state, self._state, time.time())
            self._change_history.append(change)

            # 通知所有监听器
            for key in old_state:
                if key in self._state:
                    if old_state[key] != self._state[key]:
                        self._notify_listeners(key, old_state[key], self._state[key])
                else:
                    self._notify_listeners(key, old_state[key], None)

            # 通知新增的键
            for key in self._state:
                if key not in old_state:
                    self._notify_listeners(key, None, self._state[key])

    def __enter__(self):
        """进入上下文管理器，增加嵌套级别"""
        with self._lock:
            self._nested_level += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文管理器，减少嵌套级别"""
        with self._lock:
            self._nested_level -= 1

    def __getitem__(self, key: str) -> Any:
        """通过索引获取状态"""
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """通过索引设置状态"""
        self.set(key, value)

    def __delitem__(self, key: str) -> None:
        """通过索引删除状态"""
        self.remove(key)

    def __contains__(self, key: str) -> bool:
        """检查键是否存在"""
        return self.contains(key)

    def __len__(self) -> int:
        """获取状态数量"""
        with self._lock:
            return len(self._state)

    def __iter__(self):
        """迭代状态键"""
        with self._lock:
            return iter(self._state.keys())

# 全局状态管理器实例
state_manager = StateManager()
