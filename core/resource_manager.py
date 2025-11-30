
"""
资源管理器 - 提供统一的资源管理功能
"""
import threading
from typing import Dict, Any, Callable, Optional
from .error_handler import error_handler, LogLevel

class ResourceManager:
    """资源管理器"""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ResourceManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._resources = {}  # 资源字典
            self._cleanup_callbacks = {}  # 清理回调字典
            self._initialized = True

    def register_resource(self, name: str, resource: Any, cleanup_func: Optional[Callable] = None):
        """注册资源"""
        with self._lock:
            self._resources[name] = resource
            if cleanup_func is not None:
                self._cleanup_callbacks[name] = cleanup_func
            error_handler.log(LogLevel.DEBUG, f"资源已注册: {name}")

    def get_resource(self, name: str) -> Optional[Any]:
        """获取资源"""
        with self._lock:
            return self._resources.get(name)

    def cleanup_resource(self, name: str) -> bool:
        """清理特定资源"""
        with self._lock:
            if name in self._resources:
                resource = self._resources.pop(name)
                success = True

                # 执行清理回调
                if name in self._cleanup_callbacks:
                    try:
                        self._cleanup_callbacks[name](resource)
                        error_handler.log(LogLevel.DEBUG, f"资源已清理: {name}")
                    except Exception as e:
                        error_handler.handle_exception(e, f"清理资源 {name}")
                        success = False

                    # 移除清理回调
                    del self._cleanup_callbacks[name]

                return success
            return False

    def cleanup_all(self) -> Dict[str, bool]:
        """清理所有资源"""
        results = {}
        with self._lock:
            # 复制资源名称列表，避免在迭代过程中修改字典
            resource_names = list(self._resources.keys())

            for name in resource_names:
                results[name] = self.cleanup_resource(name)

        return results

# 全局资源管理器实例
resource_manager = ResourceManager()
