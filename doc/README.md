
# LiziEngine API 文档

## 项目概述

LiziEngine 是一个现代化的向量场可视化引擎，采用模块化架构设计，提供高性能的向量场计算和渲染功能。通过模拟物理中的力场（如电磁力），LiziEngine 能够高效处理实体间的受力计算，避免传统碰撞检测的性能瓶颈。

## 核心特性

- **高性能计算**: 支持 CPU 和 GPU 加速计算，适合大规模向量场模拟
- **模块化架构**: 采用依赖注入、事件驱动和状态管理的设计模式
- **灵活的插件系统**: 支持自定义计算模式、渲染效果和用户交互
- **实时可视化**: 基于 OpenGL 的高效渲染，支持交互式操作
- **易于扩展**: 清晰的 API 接口，便于开发者添加新功能

## 快速开始

### 安装

```bash
pip install -r requirements.txt
```

### 基本使用

```python
from lizi_engine.compute.vector_field import vector_calculator

# 创建向量网格
grid = vector_calculator.create_vector_grid(width=640, height=480)

# 更新向量场
updated_grid = vector_calculator.update_grid_with_adjacent_sum(grid)

# 添加向量影响
vector_calculator.add_vector_at_position(grid, 320.0, 240.0, 1.0, 0.5)

# 获取位置向量
vx, vy = vector_calculator.fit_vector_at_position(grid, 320.0, 240.0)
```

### 错误处理

```python
try:
    grid = vector_calculator.create_vector_grid(width=640, height=480)
    if grid is None:
        raise ValueError("Failed to create vector grid")
except Exception as e:
    print(f"Error: {e}")
```

## 架构概览

LiziEngine 采用分层架构设计，主要包含以下模块：

### 核心层 (core/)

负责应用的生命周期管理和模块间协调：

- **AppCore**: 应用核心，整合所有管理器
- **Container**: 依赖注入容器
- **EventBus**: 事件系统，支持异步处理和事件过滤
- **StateManager**: 状态管理，支持变更通知和快照
- **ConfigManager**: 配置管理，支持文件加载和热更新
- **PluginManager**: 插件管理

### 计算层 (compute/)

提供向量场计算功能：

- **VectorFieldCalculator**: 主计算接口，支持 CPU/GPU 切换
- **CPUVectorFieldCalculator**: CPU 实现
- **GPUVectorFieldCalculator**: GPU 实现（基于 OpenCL）

### 渲染层 (graphics/)

处理可视化渲染：

- **VectorFieldRenderer**: 向量场渲染器
- **ShaderProgram**: 着色器程序管理

### 窗口层 (window/)

管理窗口和输入：

- **Window**: GLFW 窗口管理
- **InputHandler**: 输入事件处理

### 插件层 (plugins/)

扩展功能：

- **MarkerSystem**: 标记系统
- **UI**: 用户界面插件
- **Command**: 命令系统
- **Controller**: 控制器插件

## API 参考

### VectorFieldCalculator

向量场计算器的主要接口，支持 CPU/GPU 自动切换。

#### 属性

##### current_device: str

获取当前使用的计算设备。

**返回值:** "cpu" 或 "gpu"

#### 方法

##### create_vector_grid(width: int, height: int, default: Tuple[float, float] = (0, 0)) -> np.ndarray

创建指定大小的向量网格。

**参数:**
- `width` (int): 网格宽度
- `height` (int): 网格高度
- `default` (Tuple[float, float]): 默认向量值，默认为 (0, 0)

**返回值:** 形状为 (height, width, 2) 的 numpy 数组

**示例:**
```python
grid = vector_calculator.create_vector_grid(640, 480, (0.0, 0.0))
```

##### update_grid_with_adjacent_sum(grid: np.ndarray) -> np.ndarray

根据相邻向量更新整个网格，使用配置的权重参数。

**参数:**
- `grid` (np.ndarray): 输入网格，必须是有效的 numpy 数组

**返回值:** 更新后的网格

**异常:** TypeError 如果 grid 不是 numpy 数组

**示例:**
```python
updated_grid = vector_calculator.update_grid_with_adjacent_sum(grid)
```

##### sum_adjacent_vectors(grid: np.ndarray, x: int, y: int) -> Tuple[float, float]

计算指定位置相邻四个方向的向量之和。

**参数:**
- `grid` (np.ndarray): 输入网格
- `x` (int): X 坐标（整数）
- `y` (int): Y 坐标（整数）

**返回值:** 相邻向量之和 (sum_x, sum_y)

##### add_vector_at_position(grid: np.ndarray, x: float, y: float, vx: float, vy: float) -> None

在指定浮点坐标处添加向量，使用双线性插值分布到相邻整数坐标。

**参数:**
- `grid` (np.ndarray): 目标网格
- `x` (float): X 坐标
- `y` (float): Y 坐标
- `vx` (float): X 分量
- `vy` (float): Y 分量

##### fit_vector_at_position(grid: np.ndarray, x: float, y: float) -> Tuple[float, float]

在指定浮点坐标处使用双线性插值拟合向量值。

**参数:**
- `grid` (np.ndarray): 输入网格
- `x` (float): X 坐标
- `y` (float): Y 坐标

**返回值:** 插值后的向量 (vx, vy)

##### create_tiny_vector(grid: np.ndarray, x: float, y: float, mag: float = 1.0) -> None

在指定位置创建一个微小的向量场影响，只影响位置本身及上下左右四个邻居。

**参数:**
- `grid` (np.ndarray): 目标网格
- `x` (float): X 坐标
- `y` (float): Y 坐标
- `mag` (float): 向量大小，默认为 1.0

##### create_tiny_vectors_batch(grid: np.ndarray, positions: List[Tuple[float, float, float]]) -> None

批量创建微小向量影响，用于优化性能。

**参数:**
- `grid` (np.ndarray): 向量场网格
- `positions` (List[Tuple[float, float, float]]): 位置列表，每个元素为 (x, y, mag) 元组

##### fit_vectors_at_positions_batch(grid: np.ndarray, positions: List[Tuple[float, float]]) -> List[Tuple[float, float]]

批量拟合多个位置的向量值。

**参数:**
- `grid` (np.ndarray): 向量场网格
- `positions` (List[Tuple[float, float]]): 位置列表，每个元素为 (x, y) 元组

**返回值:** 向量列表，每个元素为 (vx, vy) 元组

##### set_device(device: str) -> bool

设置计算设备，支持运行时切换。

**参数:**
- `device` (str): "cpu" 或 "gpu"

**返回值:** 设置是否成功

**示例:**
```python
success = vector_calculator.set_device("gpu")
if not success:
    print("GPU not available, using CPU")
```

##### cleanup() -> None

清理计算资源，释放 GPU 内存等。

### ConfigManager

配置管理器，支持文件加载、热更新和类型验证。

#### 方法

##### get(key: str, default: Any = None) -> Any

获取配置值，支持点分隔的嵌套访问。

**参数:**
- `key` (str): 配置键，支持点分隔（如 "grid.width"）
- `default` (Any): 默认值

**返回值:** 配置值

**示例:**
```python
width = config_manager.get("grid_width", 640)
# 或使用嵌套访问
width = config_manager.get("grid.width", 640)
```

##### set(key: str, value: Any) -> bool

设置配置值，支持类型验证和范围检查。

**参数:**
- `key` (str): 配置键
- `value` (Any): 配置值

**返回值:** 设置是否成功

**示例:**
```python
config_manager.set("vector_scale", 1.5)
```

##### register_option(key: str, default: Any, description: str = "", type: str = "string", options: List[Any] = None, min_value: Union[int, float] = None, max_value: Union[int, float] = None) -> None

注册配置选项。

**参数:**
- `key` (str): 配置键
- `default` (Any): 默认值
- `description` (str): 描述
- `type` (str): 类型 ("string", "number", "boolean", "array", "object")
- `options` (List[Any]): 可选值列表
- `min_value` (Union[int, float]): 最小值
- `max_value` (Union[int, float]): 最大值

##### load_from_file(file_path: str) -> bool

从文件加载配置。

**参数:**
- `file_path` (str): 配置文件路径

**返回值:** 加载是否成功

##### save_to_file(file_path: Optional[str] = None) -> bool

保存配置到文件。

**参数:**
- `file_path` (Optional[str]): 文件路径，默认使用初始化路径

**返回值:** 保存是否成功

##### reset_to_default(key: Optional[str] = None) -> None

重置配置为默认值。

**参数:**
- `key` (Optional[str]): 指定键，为 None 时重置所有

##### get_all_option_info() -> Dict[str, Dict[str, Any]]

获取所有配置选项信息。

**返回值:** 配置选项信息字典

### StateManager

状态管理器，支持变更通知和快照。

#### 方法

##### get(key: str, default: Any = None) -> Any

获取状态值。

**参数:**
- `key` (str): 状态键
- `default` (Any): 默认值

**返回值:** 状态值

##### set(key: str, value: Any, notify: bool = True) -> None

设置状态值并触发变更通知。

**参数:**
- `key` (str): 状态键
- `value` (Any): 状态值
- `notify` (bool): 是否通知监听器，默认为 True

##### update(updates: Dict[str, Any], notify: bool = True) -> None

批量更新状态。

**参数:**
- `updates` (Dict[str, Any]): 更新字典
- `notify` (bool): 是否通知，默认为 True

##### remove(key: str) -> bool

移除状态。

**参数:**
- `key` (str): 状态键

**返回值:** 移除是否成功

##### clear() -> None

清空所有状态。

##### add_listener(key: str, callback: Callable[[str, Any, Any], None]) -> None

添加状态变更监听器。

**参数:**
- `key` (str): 状态键
- `callback` (Callable): 回调函数，参数为 (key, old_value, new_value)

##### remove_listener(key: str, callback: Callable[[str, Any, Any], None]) -> None

移除状态变更监听器。

##### get_change_history(key: Optional[str] = None, limit: Optional[int] = None) -> List[StateChange]

获取变更历史。

**参数:**
- `key` (Optional[str]): 指定键，为 None 时获取所有
- `limit` (Optional[int]): 限制数量

**返回值:** 变更历史列表

##### create_snapshot() -> Dict[str, Any]

创建状态快照。

**返回值:** 快照字典

##### restore_snapshot(snapshot: Dict[str, Any]) -> None

从快照恢复状态。

**参数:**
- `snapshot` (Dict[str, Any]): 快照字典

### EventBus

事件系统，支持异步处理和事件过滤。

#### 方法

##### subscribe(event_type: EventType, handler: EventHandler, filter: Optional[EventFilter] = None) -> None

订阅事件。

**参数:**
- `event_type` (EventType): 事件类型
- `handler` (EventHandler): 事件处理器
- `filter` (Optional[EventFilter]): 事件过滤器

##### unsubscribe(event_type: EventType, handler: EventHandler) -> None

取消订阅事件。

**参数:**
- `event_type` (EventType): 事件类型
- `handler` (EventHandler): 事件处理器

##### publish(event: Event) -> None

发布事件。

**参数:**
- `event` (Event): 事件对象

##### clear() -> None

清除所有事件处理器和过滤器。

##### set_max_recursion_depth(depth: int) -> None

设置最大递归深度。

**参数:**
- `depth` (int): 递归深度

##### enable_async(enabled: bool) -> None

启用或禁用异步事件处理。

**参数:**
- `enabled` (bool): 是否启用

##### get_handler_count(event_type: EventType) -> int

获取指定事件类型的处理器数量。

**参数:**
- `event_type` (EventType): 事件类型

**返回值:** 处理器数量

## 插件开发

### 创建计算插件

```python
from lizi_engine.compute.vector_field import vector_calculator
from lizi_engine.core.events import Event, EventType, event_bus

class MyComputePlugin:
    def __init__(self):
        # 订阅相关事件
        event_bus.subscribe(EventType.GRID_UPDATED, self)

    def create_custom_pattern(self, grid):
        """实现自定义向量场模式"""
        h, w = grid.shape[:2]
        for y in range(h):
            for x in range(w):
                # 创建漩涡模式
                dx = x - w // 2
                dy = y - h // 2
                dist = (dx**2 + dy**2)**0.5
                if dist > 0:
                    vx = -dy / dist * 0.1
                    vy = dx / dist * 0.1
                    grid[y, x] = (vx, vy)
        return grid

    def handle(self, event: Event):
        """处理事件"""
        if event.type == EventType.GRID_UPDATED:
            grid = event.data.get('grid')
            if grid is not None:
                self.create_custom_pattern(grid)
```

### 创建 UI 插件

```python
from plugins.ui import UIManager
from lizi_engine.input import input_handler, KeyMap
from lizi_engine.core.config import config_manager

class MyUIPlugin(UIManager):
    def register_callbacks(self, grid, **kwargs):
        super().register_callbacks(grid, **kwargs)

        def on_custom_key():
            # 切换显示模式
            current = config_manager.get("show_grid", True)
            config_manager.set("show_grid", not current)
            print(f"Grid display: {not current}")

        input_handler.register_key_callback(KeyMap.G, on_custom_key)

        def on_scale_up():
            current = config_manager.get("vector_scale", 1.0)
            config_manager.set("vector_scale", min(current + 0.1, 5.0))

        input_handler.register_key_callback(KeyMap.PLUS, on_scale_up)
```

### 事件驱动开发

```python
from lizi_engine.core.events import Event, EventType, event_bus, EventHandler

class MyEventHandler(EventHandler):
    def handle(self, event: Event):
        if event.type == EventType.VECTOR_UPDATED:
            print(f"Vector updated at {event.timestamp}")
        elif event.type == EventType.CONFIG_CHANGED:
            key = event.data.get('key')
            new_value = event.data.get('new_value')
            print(f"Config {key} changed to {new_value}")

# 注册事件处理器
handler = MyEventHandler()
event_bus.subscribe(EventType.VECTOR_UPDATED, handler)
event_bus.subscribe(EventType.CONFIG_CHANGED, handler)
```

## 配置说明

LiziEngine 使用 JSON 配置文件，支持运行时热更新和环境变量覆盖。

### 完整配置选项

```json
{
  "grid_width": 640,
  "grid_height": 480,
  "cell_size": 1.0,
  "vector_color": [0.2, 0.6, 1.0],
  "vector_scale": 1.0,
  "vector_self_weight": 0.0,
  "vector_neighbor_weight": 0.25,
  "cam_x": 0.0,
  "cam_y": 0.0,
  "cam_zoom": 1.0,
  "show_grid": true,
  "grid_color": [0.3, 0.3, 0.3],
  "background_color": [0.1, 0.1, 0.1],
  "antialiasing": true,
  "line_width": 1.0,
  "compute_device": "cpu",
  "compute_iterations": 1,
  "render_vector_lines": true,
  "target_fps": 60
}
```

### 配置类型和验证

- **数字类型**: 支持范围验证 (min_value, max_value)
- **布尔类型**: true/false
- **数组类型**: RGB颜色等
- **字符串类型**: 设备选择等，支持选项列表

### 环境变量支持

```bash
export LIZI_CONFIG=/path/to/config.json
# 或
export LIZIENGINE_CONFIG=/path/to/config.json
```

### 运行时配置更新

```python
from lizi_engine.core.config import config_manager

# 更新配置
config_manager.set("vector_scale", 2.0)
config_manager.set("compute_device", "gpu")

# 保存到文件
config_manager.save_config()
```

## 高级用法

### 批量操作优化

```python
# 批量创建向量影响
positions = [(100, 200, 1.0), (300, 150, 0.8), (500, 400, 1.2)]
vector_calculator.create_tiny_vectors_batch(grid, positions)

# 批量查询向量值
query_positions = [(100, 200), (300, 150), (500, 400)]
vectors = vector_calculator.fit_vectors_at_positions_batch(grid, query_positions)
```

### 状态管理和快照

```python
from lizi_engine.core.state import state_manager

# 创建快照
snapshot = state_manager.create_snapshot()

# 修改状态
state_manager.set("simulation_running", True)
state_manager.set("current_frame", 100)

# 恢复快照
state_manager.restore_snapshot(snapshot)
```

### 自定义事件过滤

```python
from lizi_engine.core.events import EventTypeFilter, EventSourceFilter, CompositeFilter

# 只处理特定类型的事件
type_filter = EventTypeFilter(EventType.VECTOR_UPDATED, EventType.GRID_UPDATED)

# 只处理来自特定源的事件
source_filter = EventSourceFilter("MyPlugin")

# 组合过滤器
composite_filter = CompositeFilter([type_filter, source_filter], logic="AND")

event_bus.subscribe(EventType.VECTOR_UPDATED, my_handler, composite_filter)
```

## 许可证

MIT License

