"""
依赖注入容器 - 提供依赖注入功能，避免单例模式，降低模块间耦合
"""
from typing import Dict, Type, Any, Callable, Optional, TypeVar, Generic, get_type_hints
from inspect import isclass, isfunction, ismethod
import threading

T = TypeVar('T')

class ServiceDescriptor:
    """服务描述符，包含服务的生命周期和工厂方法"""
    def __init__(self, factory: Callable, singleton: bool = True):
        self.factory = factory
        self.singleton = singleton
        self.instance = None if singleton else None
        self._lock = threading.Lock()

    def get_instance(self, container: 'Container') -> Any:
        """获取服务实例"""
        if not self.singleton:
            return self._create_instance(container)

        with self._lock:
            if self.instance is None:
                self.instance = self._create_instance(container)
            return self.instance

    def _create_instance(self, container: 'Container') -> Any:
        """创建服务实例"""
        if isclass(self.factory):
            # 如果是类，解析构造函数参数并自动注入
            return self._create_with_injection(container, self.factory)
        else:
            # 如果是工厂函数，调用它并自动注入参数
            return self._create_with_injection(container, self.factory)

    def _create_with_injection(self, container: 'Container', factory: Callable) -> Any:
        """通过依赖注入创建实例"""
        # 获取函数或类的类型提示
        hints = get_type_hints(factory)

        # 获取参数列表
        if isclass(factory):
            # 类的构造函数
            import inspect
            params = inspect.signature(factory.__init__).parameters
        else:
            # 函数
            import inspect
            params = inspect.signature(factory).parameters

        # 准备参数字典
        kwargs = {}

        for name, param in params.items():
            if name == 'self':
                continue

            # 检查是否有类型提示
            param_type = hints.get(name)
            if param_type:
                # 尝试从容器中解析依赖
                dependency = container.resolve(param_type)
                if dependency is not None:
                    kwargs[name] = dependency
                elif param.default is inspect.Parameter.empty:
                    # 如果没有默认值且无法解析依赖，抛出异常
                    raise ValueError(f"无法解析依赖: {name} ({param_type})")
            elif param.default is not inspect.Parameter.empty:
                # 如果没有类型提示但有默认值，使用默认值
                kwargs[name] = param.default

        # 创建实例
        if isclass(factory):
            return factory(**kwargs)
        else:
            return factory(**kwargs)

class Container:
    """依赖注入容器"""
    def __init__(self):
        self._services: Dict[Type, ServiceDescriptor] = {}
        self._instances: Dict[Type, Any] = {}
        self._lock = threading.Lock()

    def register(self, service_type: Type[T], factory: Callable, singleton: bool = True) -> None:
        """注册服务"""
        with self._lock:
            self._services[service_type] = ServiceDescriptor(factory, singleton)

    def register_singleton(self, service_type: Type[T], instance: T) -> None:
        """注册单例实例"""
        with self._lock:
            self._services[service_type] = ServiceDescriptor(lambda: instance, singleton=True)
            self._instances[service_type] = instance

    def register_transient(self, service_type: Type[T], factory: Callable) -> None:
        """注册瞬态服务（每次获取都创建新实例）"""
        with self._lock:
            self._services[service_type] = ServiceDescriptor(factory, singleton=False)

    def resolve(self, service_type: Type[T]) -> Optional[T]:
        """解析服务"""
        with self._lock:
            if service_type not in self._services:
                return None

            descriptor = self._services[service_type]
            return descriptor.get_instance(self)

    def is_registered(self, service_type: Type[T]) -> bool:
        """检查服务是否已注册"""
        with self._lock:
            return service_type in self._services

    def remove(self, service_type: Type[T]) -> None:
        """移除服务"""
        with self._lock:
            if service_type in self._services:
                descriptor = self._services.pop(service_type)
                if descriptor.singleton and descriptor.instance is not None:
                    # 如果是单例且已创建实例，尝试调用其cleanup方法
                    instance = descriptor.instance
                    if hasattr(instance, 'cleanup'):
                        try:
                            instance.cleanup()
                        except Exception:
                            pass
                if service_type in self._instances:
                    self._instances.pop(service_type)

    def clear(self) -> None:
        """清除所有服务"""
        with self._lock:
            # 清理所有单例实例
            for service_type, descriptor in self._services.items():
                if descriptor.singleton and descriptor.instance is not None:
                    # 如果是单例且已创建实例，尝试调用其cleanup方法
                    instance = descriptor.instance
                    if hasattr(instance, 'cleanup'):
                        try:
                            instance.cleanup()
                        except Exception:
                            pass
                    self._instances.pop(service_type, None)

            # 清除所有服务描述符
            self._services.clear()

# 全局容器实例
container = Container()
