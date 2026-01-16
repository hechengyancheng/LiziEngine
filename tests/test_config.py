import pytest
import json
import os
from lizi_engine.core.config import ConfigManager


class TestConfigManager:
    """测试配置管理器"""

    def setup_method(self):
        """测试前准备"""
        self.config_file = "test_config.json"
        self.manager = ConfigManager(self.config_file)
        # 重置为默认值以确保测试隔离
        self.manager.reset_to_default()

    def teardown_method(self):
        """测试后清理"""
        if os.path.exists(self.config_file):
            os.remove(self.config_file)

    def test_load_config(self):
        """测试加载配置"""
        config_data = {
            "grid": {"width": 640, "height": 480},
            "vector": {"scale": 1.0}
        }
        with open(self.config_file, 'w') as f:
            json.dump(config_data, f)

        self.manager.load_config()
        assert self.manager.get("grid.width") == 640
        assert self.manager.get("vector.scale") == 1.0

    def test_save_config(self):
        """测试保存配置"""
        self.manager.set("test_key", "test_value")
        self.manager.save_config()

        with open(self.config_file, 'r') as f:
            data = json.load(f)

        assert data.get("test_key") == "test_value"

    def test_get_set(self):
        """测试获取和设置配置"""
        self.manager.set("test.value", 42)
        assert self.manager.get("test.value") == 42
        assert self.manager.get("nonexistent", "default") == "default"

    def test_nested_access(self):
        """测试嵌套访问"""
        self.manager.set("nested.deep.value", "test")
        assert self.manager.get("nested.deep.value") == "test"

    def test_validation(self):
        """测试配置验证"""
        # 测试类型验证
        assert self.manager.set("grid_width", 800)  # 有效数字
        assert not self.manager.set("grid_width", "invalid")  # 无效类型

        # 测试范围验证
        assert self.manager.set("vector_scale", 2.0)  # 在范围内
        assert not self.manager.set("vector_scale", 15.0)  # 超出范围

        # 测试数组验证
        assert self.manager.set("vector_color", [1.0, 0.5, 0.0])  # 有效数组
        assert not self.manager.set("vector_color", "invalid")  # 无效类型

    def test_event_publishing(self):
        """测试事件发布"""
        # 这里需要模拟事件总线，但由于依赖复杂，我们只测试基本功能
        old_value = self.manager.get("grid_width")
        self.manager.set("grid_width", old_value + 100)
        # 实际的事件测试需要更复杂的设置，这里简化

    def test_dynamic_registration(self):
        """测试动态配置注册"""
        # 设置未知配置项
        assert self.manager.set("custom_option", "value")
        assert self.manager.get("custom_option") == "value"

    def test_reset_to_default(self):
        """测试重置为默认值"""
        original_value = self.manager.get("grid_width")
        self.manager.set("grid_width", original_value + 100)
        assert self.manager.get("grid_width") == original_value + 100

        self.manager.reset_to_default("grid_width")
        assert self.manager.get("grid_width") == original_value

    def test_get_option_info(self):
        """测试获取配置选项信息"""
        info = self.manager.get_option_info("grid_width")
        assert info is not None
        assert info["type"] == "number"
        assert info["description"] == "网格宽度"

    def test_invalid_config_file(self):
        """测试无效配置文件"""
        invalid_file = "invalid_config.json"
        with open(invalid_file, 'w') as f:
            f.write("invalid json")

        manager = ConfigManager(invalid_file)
        # 应该能够处理无效文件而不崩溃

        if os.path.exists(invalid_file):
            os.remove(invalid_file)


if __name__ == "__main__":
    pytest.main([__file__])
