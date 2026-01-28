import pytest
from unittest.mock import Mock
from lizi_engine.core.container import Container, ServiceDescriptor, container


class TestServiceDescriptor:
    """测试服务描述符"""

    def test_service_descriptor_singleton(self):
        """测试单例服务描述符"""
        def factory():
            return "test_instance"

        descriptor = ServiceDescriptor(factory, singleton=True)

        # 第一次获取实例
        instance1 = descriptor.get_instance(None)
        assert instance1 == "test_instance"

        # 第二次应该返回同一个实例
        instance2 = descriptor.get_instance(None)
        assert instance1 is instance2

    def test_service_descriptor_transient(self):
        """测试瞬态服务描述符"""
        def factory():
            return object()  # Return different objects

        descriptor = ServiceDescriptor(factory, singleton=False)

        # 每次都应该返回新实例
        instance1 = descriptor.get_instance(None)
        instance2 = descriptor.get_instance(None)
        assert instance1 != instance2
        assert instance1 is not instance2

    def test_service_descriptor_with_class(self):
        """测试类工厂的服务描述符"""
        class TestClass:
            def __init__(self, value="default"):
                self.value = value

        descriptor = ServiceDescriptor(TestClass, singleton=True)
        instance = descriptor.get_instance(None)

        assert isinstance(instance, TestClass)
        assert instance.value == "default"

    def test_service_descriptor_with_dependency_injection(self):
        """测试依赖注入"""
        class Dependency:
            pass

        class Service:
            def __init__(self, dep: Dependency):
                self.dep = dep

        container = Container()
        container.register(Dependency, lambda: Dependency())

        descriptor = ServiceDescriptor(Service, singleton=True)
        instance = descriptor.get_instance(container)

        assert isinstance(instance, Service)
        assert isinstance(instance.dep, Dependency)


class TestContainer:
    """测试依赖注入容器"""

    def setup_method(self):
        """测试前准备"""
        self.container = Container()

    def test_container_initialization(self):
        """测试容器初始化"""
        assert self.container._services == {}
        assert self.container._instances == {}

    def test_register_singleton(self):
        """测试注册单例服务"""
        def factory():
            return "test_service"

        self.container.register(str, factory, singleton=True)

        assert self.container.is_registered(str)
        instance = self.container.resolve(str)
        assert instance == "test_service"

        # 再次解析应该返回同一个实例
        instance2 = self.container.resolve(str)
        assert instance is instance2

    def test_register_transient(self):
        """测试注册瞬态服务"""
        def factory():
            return object()  # Return different objects

        self.container.register(object, factory, singleton=False)

        instance1 = self.container.resolve(object)
        instance2 = self.container.resolve(object)

        assert instance1 != instance2
        assert instance1 is not instance2

    def test_register_singleton_instance(self):
        """测试注册单例实例"""
        instance = "test_instance"
        self.container.register_singleton(str, instance)

        resolved = self.container.resolve(str)
        assert resolved is instance

    def test_resolve_unregistered_service(self):
        """测试解析未注册的服务"""
        result = self.container.resolve(str)
        assert result is None

    def test_is_registered(self):
        """测试检查服务是否已注册"""
        assert not self.container.is_registered(str)

        self.container.register(str, lambda: "test")
        assert self.container.is_registered(str)

    def test_remove_service(self):
        """测试移除服务"""
        self.container.register(str, lambda: "test")
        assert self.container.is_registered(str)

        self.container.remove(str)
        assert not self.container.is_registered(str)

    def test_remove_singleton_with_cleanup(self):
        """测试移除带清理方法的单例服务"""
        class TestService:
            def __init__(self):
                self.cleaned = False

            def cleanup(self):
                self.cleaned = True

        service = TestService()
        self.container.register_singleton(TestService, service)

        self.container.remove(TestService)
        assert service.cleaned

    def test_clear_container(self):
        """测试清空容器"""
        class TestService:
            def __init__(self):
                self.cleaned = False

            def cleanup(self):
                self.cleaned = True

        service = TestService()
        self.container.register_singleton(TestService, service)
        self.container.register(str, lambda: "test")

        self.container.clear()

        assert service.cleaned
        assert not self.container.is_registered(TestService)
        assert not self.container.is_registered(str)

    def test_dependency_injection_with_type_hints(self):
        """测试带类型提示的依赖注入"""
        class Dependency:
            pass

        class Service:
            def __init__(self, dep: Dependency):
                self.dep = dep

        # Register the dependency first
        self.container.register(Dependency, Dependency)
        self.container.register(Service, Service)

        service = self.container.resolve(Service)
        assert isinstance(service, Service)
        assert isinstance(service.dep, Dependency)

    def test_dependency_injection_missing_dependency(self):
        """测试依赖注入缺少依赖"""
        class Service:
            def __init__(self, dep: str):
                self.dep = dep

        self.container.register(Service, Service)

        with pytest.raises(ValueError, match="无法解析依赖"):
            self.container.resolve(Service)

    def test_dependency_injection_with_defaults(self):
        """测试依赖注入带默认值"""
        class Service:
            def __init__(self, value: str = "default"):
                self.value = value

        self.container.register(Service, Service)
        service = self.container.resolve(Service)

        assert service.value == "default"

    def test_thread_safety(self):
        """测试线程安全"""
        import threading
        import time

        results = []

        def worker():
            for i in range(100):
                self.container.register(f"key{i}", lambda: f"value{i}")
                result = self.container.resolve(f"key{i}")
                results.append(result)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 所有操作都应该成功完成
        assert len(results) == 500


class TestGlobalContainer:
    """测试全局容器"""

    def test_global_container_exists(self):
        """测试全局容器存在"""
        assert container is not None
        assert isinstance(container, Container)


if __name__ == "__main__":
    pytest.main([__file__])
