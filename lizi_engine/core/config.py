"""
配置管理模块 - 提供统一的配置管理功能
支持从文件加载配置和热更新
"""
import json
import os
import sys
import threading
from typing import Any, Dict, Optional, Union, List
from dataclasses import dataclass, asdict, field
from .state import StateManager, state_manager
from .events import Event, EventType, event_bus

@dataclass
class ConfigOption:
    """配置选项"""
    key: str
    value: Any
    default: Any
    description: str = ""
    type: str = "string"  # string, number, boolean, array, object
    options: List[Any] = field(default_factory=list)
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConfigOption':
        """从字典创建配置选项"""
        return cls(**data)

class ConfigManager:
    """配置管理器"""
    def __init__(self, config_file: Optional[str] = None):
        self._config_file = config_file
        self._options: Dict[str, ConfigOption] = {}
        self._lock = threading.RLock()
        self._state_manager = state_manager
        self._event_bus = event_bus

        # 初始化默认配置
        self._init_default_config()

        # 如果指定了配置文件，加载配置
        if self._config_file and os.path.exists(self._config_file):
            self.load_from_file(self._config_file)

    def _init_default_config(self) -> None:
        """初始化默认配置"""
        # 网格配置
        self.register_option("grid_width", 640, "网格宽度", type="number")
        self.register_option("grid_height", 480, "网格高度", type="number")
        self.register_option("cell_size", 1.0, "单元格大小", type="number")

        # 向量场配置
        self.register_option("vector_color", [0.2, 0.6, 1.0], "向量颜色", type="array")
        self.register_option("vector_scale", 1.0, "向量缩放", type="number", min_value=0.1, max_value=10.0)
        self.register_option("vector_self_weight", 0.2, "向量自身权重", type="number", min_value=0.0, max_value=10.0)
        self.register_option("vector_neighbor_weight", 0.2, "向量邻居权重", type="number", min_value=0.0, max_value=10.0)

        # 视图配置
        self.register_option("cam_x", 0.0, "相机X坐标", type="number")
        self.register_option("cam_y", 0.0, "相机Y坐标", type="number")
        self.register_option("cam_zoom", 1.0, "相机缩放", type="number", min_value=0.1, max_value=10.0)
        self.register_option("show_grid", True, "是否显示网格", type="boolean")
        self.register_option("grid_color", [0.3, 0.3, 0.3], "网格颜色", type="array")

        # 渲染配置
        self.register_option("background_color", [0.1, 0.1, 0.1], "背景颜色", type="array")
        self.register_option("antialiasing", True, "是否启用抗锯齿", type="boolean")
        self.register_option("line_width", 1.0, "线条宽度", type="number", min_value=0.5, max_value=5.0)

        # 事件系统配置（移除未使用的项，保持事件总线行为默认）

        # 计算配置
        self.register_option("compute_device", "cpu", "计算设备", options=["cpu", "gpu"])
        self.register_option("compute_iterations", 1, "计算迭代次数", type="number", min_value=1, max_value=100)

        # UI 配置（简化：UI 主题/缩放由前端插件自行管理）

        # 渲染配置：是否渲染向量线条（新增）
        self.register_option("render_vector_lines", True, "是否渲染向量线条", type="boolean")

    def register_option(self, key: str, default: Any, description: str = "", 
                       type: str = "string", options: List[Any] = None,
                       min_value: Optional[Union[int, float]] = None,
                       max_value: Optional[Union[int, float]] = None) -> None:
        """注册配置选项"""
        with self._lock:
            if key in self._options:
                # 如果选项已存在，更新描述和其他元数据
                option = self._options[key]
                option.description = description
                option.type = type
                if options is not None:
                    option.options = options
                option.min_value = min_value
                option.max_value = max_value
            else:
                # 创建新选项
                option = ConfigOption(
                    key=key,
                    value=default,
                    default=default,
                    description=description,
                    type=type,
                    options=options or [],
                    min_value=min_value,
                    max_value=max_value
                )
                self._options[key] = option

            # 如果状态管理器中没有该配置，则设置默认值
            if not self._state_manager.contains(key):
                self._state_manager.set(key, default, notify=False)

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        with self._lock:
            flat_key = key.replace('.', '_')
            if flat_key in self._options:
                return self._state_manager.get(flat_key, self._options[flat_key].value)
            return self._state_manager.get(flat_key, default)

    def set(self, key: str, value: Any) -> bool:
        """设置配置值"""
        with self._lock:
            flat_key = key.replace('.', '_')
            if flat_key not in self._options:
                # 动态注册未知选项
                self.register_option(flat_key, value, f"动态配置选项: {key}")

            option = self._options[flat_key]

            # 类型检查（对于动态注册的选项，跳过类型检查）
            if not self._validate_value(value, option):
                print(f"[配置管理] 配置值类型或范围不匹配: {key}")
                return False

            # 设置值
            old_value = self._state_manager.get(flat_key, option.value)
            self._state_manager.set(flat_key, value)

            # 如果值发生了变化，发布配置变更事件
            if old_value != value:
                self._event_bus.publish(Event(
                    EventType.CONFIG_CHANGED,
                    {"key": key, "old_value": old_value, "new_value": value},
                    "ConfigManager"
                ))

            # 如果配置文件已设置，保存配置
            if self._config_file:
                self.save_to_file(self._config_file)

            return True

    def _validate_value(self, value: Any, option: ConfigOption) -> bool:
        """验证配置值是否有效"""
        # 类型验证
        if option.type == "boolean":
            return isinstance(value, bool)
        elif option.type == "number":
            if not isinstance(value, (int, float)):
                return False
            # 范围验证
            if option.min_value is not None and value < option.min_value:
                return False
            if option.max_value is not None and value > option.max_value:
                return False
            return True
        elif option.type == "array":
            return isinstance(value, list)
        elif option.type == "object":
            return isinstance(value, dict)
        else:  # string
            return isinstance(value, str)

    def get_all(self) -> Dict[str, Any]:
        """获取所有配置"""
        with self._lock:
            config = {}
            for key, option in self._options.items():
                config[key] = self._state_manager.get(key, option.value)
            return config

    def _nest_dict_from_flat(self, flat: Dict[str, Any]) -> Dict[str, Any]:
        """将扁平键（使用下划线分隔）转换为嵌套字典

        例如: {'vector_color': [..], 'grid_width': 640} -> {'vector': {'color': [...]}, 'grid': {'width': 640}}
        """
        nested: Dict[str, Any] = {}
        for flat_key, value in flat.items():
            if not flat_key:
                continue  # 跳过空键
            parts = flat_key.split('_')
            cur = nested
            for i, part in enumerate(parts):
                if not part:
                    continue  # 跳过空的部分
                if i == len(parts) - 1:
                    cur[part] = value
                else:
                    if part not in cur or not isinstance(cur[part], dict):
                        cur[part] = {}
                    cur = cur[part]
        return nested

    def _flatten_dict(self, nested: Dict[str, Any], parent_key: str = '') -> Dict[str, Any]:
        """将嵌套字典转换为扁平键（使用下划线连接）"""
        items: Dict[str, Any] = {}
        for k, v in nested.items():
            new_key = f"{parent_key}_{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(self._flatten_dict(v, new_key))
            else:
                items[new_key] = v
        return items

    def reset_to_default(self, key: Optional[str] = None) -> None:
        """重置配置为默认值"""
        with self._lock:
            if key is None:
                # 重置所有配置
                for option in self._options.values():
                    self._state_manager.set(option.key, option.default)

                # 发布配置重置事件
                self._event_bus.publish(Event(
                    EventType.CONFIG_CHANGED,
                    {"action": "reset_all"},
                    "ConfigManager"
                ))
            else:
                # 重置单个配置
                if key in self._options:
                    option = self._options[key]
                    self._state_manager.set(key, option.default)

                    # 发布配置重置事件
                    self._event_bus.publish(Event(
                        EventType.CONFIG_CHANGED,
                        {"key": key, "action": "reset"},
                        "ConfigManager"
                    ))

    def load_from_file(self, file_path: str) -> bool:
        """从文件加载配置"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            # 支持嵌套或扁平的配置文件格式
            flat_config: Dict[str, Any] = {}
            if isinstance(config_data, dict):
                # 将嵌套的 dict 扁平化为使用下划线的键名
                flat_config = self._flatten_dict(config_data)

            with self._lock:
                for key, value in flat_config.items():
                    if key in self._options:
                        self.set(key, value)
                    else:
                        # 兼容：也尝试直接使用原始键（如果文件已经是扁平的）
                        if key in self._options:
                            self.set(key, value)

            return True
        except Exception as e:
            print(f"[配置管理] 加载配置文件失败: {e}")
            return False

    def save_to_file(self, file_path: Optional[str] = None) -> bool:
        """保存配置到文件"""
        try:
            config_file = file_path or self._config_file
            if not config_file:
                return False

            # 获取所有配置（包括动态注册的）
            all_config = self._state_manager.get_all()

            # 确保目录存在（仅当路径包含目录时创建）
            dir_name = os.path.dirname(config_file)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(all_config, f, indent=4)

            return True
        except Exception as e:
            print(f"[配置管理] 保存配置文件失败: {e}")
            return False

    def load_config(self) -> bool:
        """加载配置（兼容性方法）"""
        return self.load_from_file(self._config_file) if self._config_file else False

    def save_config(self) -> bool:
        """保存配置（兼容性方法）"""
        return self.save_to_file(self._config_file) if self._config_file else False

    def get_option_info(self, key: str) -> Optional[Dict[str, Any]]:
        """获取配置选项信息"""
        with self._lock:
            if key in self._options:
                return self._options[key].to_dict()
            return None

    def get_all_option_info(self) -> Dict[str, Dict[str, Any]]:
        """获取所有配置选项信息"""
        with self._lock:
            return {key: option.to_dict() for key, option in self._options.items()}

    def __getitem__(self, key: str) -> Any:
        """通过索引获取配置"""
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """通过索引设置配置"""
        self.set(key, value)

    def __contains__(self, key: str) -> bool:
        """检查配置是否存在"""
        return key in self._options

# 全局配置管理器实例
# 配置路径查找策略（打包后可动态覆盖）：
# 1. 环境变量 LIZI_CONFIG 或 LIZIENGINE_CONFIG
# 2. 当前工作目录下的 config.json
# 3. 可执行文件所在目录下的 config.json（对 PyInstaller/frozen 生效）
env_path = os.environ.get("LIZI_CONFIG") or os.environ.get("LIZIENGINE_CONFIG")
if env_path:
    config_path = env_path
else:
    cwd_path = os.path.join(os.getcwd(), "config.json")
    if os.path.exists(cwd_path):
        config_path = cwd_path
    else:
        if getattr(sys, 'frozen', False):
            base_dir = os.path.dirname(sys.executable)
        else:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        config_path = os.path.join(base_dir, "config.json")

config_manager = ConfigManager(config_path)
