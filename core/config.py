
"""
配置管理模块 - 提供统一的配置加载和管理功能
"""
import os
import json
import threading
from typing import Dict, Any, Optional, Union
from .events import EventBus, Event, EventType
from .state import state_manager

class ConfigManager:
    """配置管理器"""
    _instance = None
    _lock = threading.Lock()  # 类级别锁，确保线程安全

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._config_file = 'config.json'
            self._settings: Dict[str, Any] = {}
            self._event_bus = EventBus()
            self._state_manager = state_manager
            self._initialized = True

            # 加载默认配置
            self._load_default_config()

            # 尝试从文件加载配置
            self.load_config()

    def _load_default_config(self) -> None:
        """加载默认配置"""
        self._settings = {
            # 窗口设置
            "window": {
                "width": 1200,
                "height": 800,
                "title": "LiziEngine"
            },

            # 向量场设置
            "vector_field": {
                "grid_size": 50,
                "default_vector_length": 0.5,
                "vector_color": [0.2, 0.6, 1.0, 1.0],
                "default_brush_size": 10,
                "vector_self_weight": 0.2,
                "vector_neighbor_weight": 0.2,
                "include_self": True,
                "enable_vector_average": True,
                "reverse_vector": False,
                "enable_vector_normalization": False
            },

            # 渲染设置
            "rendering": {
                "background_color": [0.1, 0.1, 0.1, 1.0],
                "grid_color": [0.3, 0.3, 0.3, 1.0],
                "show_grid": True,
                "cell_size": 1,
                "cam_x": 0,
                "cam_y": 0,
                "cam_zoom": 1.0,
                "update_frequency": 30.0
            },

            # 计算设置
            "compute": {
                "use_opencl_compute": True,
                "opencl_compute_threshold": 10000,
                "compute_shader_local_size_x": 32,
                "compute_shader_local_size_y": 32
            },

            # UI设置
            "ui": {
                "toolbar_type": "auto",
                "enable_debug_output": False,
                "enable_event_output": False
            },

            # 网格设置
            "grid": {
                "width": 50,
                "height": 50,
                "cell_size": 10
            }
        }

    def load_config(self, config_file: Optional[str] = None) -> None:
        """从配置文件加载设置"""
        file_path = config_file or self._config_file
        if not os.path.exists(file_path):
            print(f"[配置管理] 配置文件 {file_path} 不存在，使用默认设置")
            return

        print(f"[配置管理] 正在加载配置文件: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith('.json'):
                    # JSON格式配置文件
                    file_settings = json.load(f)

                    # 检查是否是旧的扁平格式配置文件
                    if not any(isinstance(v, dict) for v in file_settings.values()):
                        print("[配置管理] 检测到旧的扁平配置格式，正在转换...")
                        file_settings = self._convert_flat_to_nested(file_settings)
                else:
                    # 旧的键值对格式配置文件
                    file_settings = {}
                    for line in f:
                        line = line.strip()
                        # 跳过空行和注释行
                        if not line or line.startswith('#'):
                            continue

                        # 解析键值对
                        if '=' in line:
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = value.strip()
                            file_settings[key] = self._parse_value(value)

                    # 转换为嵌套格式
                    file_settings = self._convert_flat_to_nested(file_settings)

            # 更新设置
            with self._lock:
                self._settings.update(file_settings)

            # 同步到状态管理器
            self._sync_to_state()

            # 发布配置加载事件
            self._event_bus.publish(Event(
                self._get_event_type("CONFIG_LOADED", EventType.VIEW_CHANGED),
                {"file_path": file_path},
                "ConfigManager"
            ))

            print(f"[配置管理] 配置加载完成")

        except Exception as e:
            print(f"[配置管理] 加载配置文件出错: {e}")

    def _parse_value(self, value: str) -> Union[str, int, float, bool, tuple]:
        """尝试将字符串值转换为适当的数据类型"""
        # 首先移除值后面的注释（如果有）
        if '#' in value:
            value = value.split('#')[0].strip()

        # 处理带引号的字符串
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            return value[1:-1]  # 移除引号

        # 尝试解析为布尔值
        if value.lower() == 'true':
            return True
        elif value.lower() == 'false':
            return False

        # 尝试解析为整数
        try:
            return int(value)
        except ValueError:
            pass

        # 尝试解析为浮点数
        try:
            return float(value)
        except ValueError:
            pass

        # 尝试解析为元组（用于RGB颜色等）
        if ',' in value:
            try:
                return tuple(float(x.strip()) for x in value.split(','))
            except ValueError:
                pass

        # 默认返回字符串
        return value

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值，如果不存在则返回默认值

        支持嵌套键访问，例如 "window.width" 或 "vector_field.grid_size"
        """
        with self._lock:
            # 如果key包含点号，则按嵌套路径查找
            if '.' in key:
                keys = key.split('.')
                value = self._settings
                try:
                    for k in keys:
                        value = value[k]
                    return value
                except (KeyError, TypeError):
                    return default
            else:
                return self._settings.get(key, default)

    def set(self, key: str, value: Any, persist: bool = False) -> None:
        """设置配置值

        支持嵌套键设置，例如 "window.width" 或 "vector_field.grid_size"
        """
        with self._lock:
            old_value = self.get(key)

            # 如果key包含点号，则按嵌套路径设置
            if '.' in key:
                keys = key.split('.')
                target = self._settings
                for k in keys[:-1]:
                    if k not in target:
                        target[k] = {}
                    target = target[k]
                target[keys[-1]] = value
            else:
                self._settings[key] = value

            # 同步到状态管理器
            self._state_manager.set(f"config_{key}", value)

            # 如果值发生变化，发布配置变更事件
            if old_value != value:
                self._event_bus.publish(Event(
                    self._get_event_type("CONFIG_CHANGED", EventType.VIEW_CHANGED),
                    {"key": key, "old_value": old_value, "new_value": value},
                    "ConfigManager"
                ))

        # 如果需要持久化，保存到文件
        if persist:
            self.save()

    def update(self, updates: Dict[str, Any], persist: bool = False) -> None:
        """批量更新配置"""
        changed = []

        with self._lock:
            for key, value in updates.items():
                old_value = self._settings.get(key)
                self._settings[key] = value

                # 同步到状态管理器
                self._state_manager.set(f"config_{key}", value)

                if old_value != value:
                    changed.append((key, old_value, value))

        # 如果有变化，发布批量配置变更事件
        if changed:
            self._event_bus.publish(Event(
                self._get_event_type("BATCH_CONFIG_CHANGED", EventType.VIEW_CHANGED),
                {"changes": changed},
                "ConfigManager"
            ))

        # 如果需要持久化，保存到文件
        if persist:
            self.save()

    def save(self, config_file: Optional[str] = None) -> None:
        """将当前配置保存到文件"""
        file_path = config_file or self._config_file

        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)

            with open(file_path, 'w', encoding='utf-8') as f:
                if file_path.endswith('.json'):
                    # JSON格式
                    json.dump(self._settings, f, indent=2, ensure_ascii=False)
                else:
                    # 旧的键值对格式
                    f.write("# LiziEngine 配置文件\n")
                    f.write("# 此文件记录了引擎的主要参数设置\n\n")

                    for key, value in sorted(self._settings.items()):
                        # 将值转换回字符串格式
                        if isinstance(value, bool):
                            value = 'true' if value else 'false'
                        elif isinstance(value, tuple):
                            value = ','.join(str(x) for x in value)
                        else:
                            value = str(value)

                        f.write(f"{key}={value}\n")

            print(f"[配置管理] 配置已保存到: {file_path}")

            # 发布配置保存事件
            self._event_bus.publish(Event(
                self._get_event_type("CONFIG_SAVED", EventType.VIEW_CHANGED),
                {"file_path": file_path},
                "ConfigManager"
            ))
        except Exception as e:
            print(f"[配置管理] 保存配置文件出错: {e}")

    def _get_event_type(self, type_name: str, default_type):
        """安全地获取事件类型"""
        try:
            from .events import EventType
            return getattr(EventType, type_name, default_type)
        except Exception:
            return default_type

    def _sync_to_state(self) -> None:
        """将配置同步到状态管理器"""
        # 直接设置状态值，避免触发事件和递归
        with self._state_manager._lock:
            self._sync_nested_to_state(self._settings, "config")

    def _sync_nested_to_state(self, obj, prefix) -> None:
        """递归同步嵌套配置到状态管理器"""
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_prefix = f"{prefix}_{key}"
                if isinstance(value, dict):
                    self._sync_nested_to_state(value, new_prefix)
                else:
                    self._state_manager._state[new_prefix] = value
        else:
            self._state_manager._state[prefix] = obj

    def _convert_flat_to_nested(self, flat_config) -> dict:
        """将扁平配置转换为嵌套配置"""
        nested_config = {
            "window": {},
            "vector_field": {},
            "rendering": {},
            "compute": {},
            "ui": {}
        }

        # 映射旧的扁平键到新的嵌套键
        key_mapping = {
            # 窗口设置
            "window_title": ("window", "title"),
            "window_width": ("window", "width"),
            "window_height": ("window", "height"),

            # 向量场设置
            "default_brush_size": ("vector_field", "default_brush_size"),
            "default_magnitude": ("vector_field", "default_vector_length"),
            "vector_self_weight": ("vector_field", "vector_self_weight"),
            "vector_neighbor_weight": ("vector_field", "vector_neighbor_weight"),
            "include_self": ("vector_field", "include_self"),
            "enable_vector_average": ("vector_field", "enable_vector_average"),
            "reverse_vector": ("vector_field", "reverse_vector"),
            "enable_vector_normalization": ("vector_field", "enable_vector_normalization"),
            "vector_color": ("vector_field", "vector_color"),

            # 渲染设置
            "background_color": ("rendering", "background_color"),
            "grid_color": ("rendering", "grid_color"),
            "show_grid": ("rendering", "show_grid"),
            "cell_size": ("rendering", "cell_size"),
            "cam_x": ("rendering", "cam_x"),
            "cam_y": ("rendering", "cam_y"),
            "cam_zoom": ("rendering", "cam_zoom"),
            "update_frequency": ("rendering", "update_frequency"),

            # 计算设置
            "use_opencl_compute": ("compute", "use_opencl_compute"),
            "opencl_compute_threshold": ("compute", "opencl_compute_threshold"),
            "compute_shader_local_size_x": ("compute", "compute_shader_local_size_x"),
            "compute_shader_local_size_y": ("compute", "compute_shader_local_size_y"),

            # UI设置
            "enable_debug_output": ("ui", "enable_debug_output"),
            "enable_event_output": ("ui", "enable_event_output")
        }

        # 转换配置
        for old_key, value in flat_config.items():
            if old_key in key_mapping:
                section, new_key = key_mapping[old_key]
                nested_config[section][new_key] = value
            else:
                # 对于没有映射的键，尝试根据前缀进行分类
                if old_key.startswith("window"):
                    nested_config["window"][old_key[6:]] = value
                elif old_key.startswith("vector") or old_key.startswith("default"):
                    nested_config["vector_field"][old_key] = value
                elif old_key.startswith("render") or old_key.startswith("cam") or old_key.startswith("grid") or old_key.startswith("background"):
                    nested_config["rendering"][old_key] = value
                elif old_key.startswith("compute") or old_key.startswith("opencl"):
                    nested_config["compute"][old_key] = value
                elif old_key.startswith("debug") or old_key.startswith("event"):
                    nested_config["ui"][old_key] = value
                else:
                    # 默认情况下，添加到UI部分
                    nested_config["ui"][old_key] = value

        # 添加默认值，确保所有必要的键都存在
        for section, defaults in self._settings.items():
            if isinstance(defaults, dict):
                for key, default_value in defaults.items():
                    if key not in nested_config[section]:
                        nested_config[section][key] = default_value

        return nested_config

    def debug_print_all(self) -> None:
        """打印所有已加载的配置项，用于调试"""
        print("[配置管理] 所有配置项:")
        self._debug_print_nested(self._settings, 0)

    def _debug_print_nested(self, obj, indent=0) -> None:
        """递归打印嵌套配置"""
        indent_str = "  " * indent
        if isinstance(obj, dict):
            for key, value in sorted(obj.items()):
                if isinstance(value, dict):
                    print(f"{indent_str}{key}:")
                    self._debug_print_nested(value, indent + 1)
                else:
                    print(f"{indent_str}{key} = {value}")
        else:
            print(f"{indent_str}{obj}")

# 确保类级别的锁已初始化
if not hasattr(ConfigManager, '_lock') or ConfigManager._lock is None:
    ConfigManager._lock = threading.Lock()

# 全局配置管理器实例
config_manager = ConfigManager()

# 便捷函数
def get_config(key: str, default: Any = None) -> Any:
    """获取配置值的便捷函数"""
    return config_manager.get(key, default)

def set_config(key: str, value: Any, persist: bool = False) -> None:
    """设置配置值的便捷函数"""
    config_manager.set(key, value, persist)
