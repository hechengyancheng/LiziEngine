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


if __name__ == "__main__":
    pytest.main([__file__])
