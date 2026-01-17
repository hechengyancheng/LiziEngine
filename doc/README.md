
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

# 创建径向向量场模式
vector_calculator.create_radial_pattern(grid, center=(320, 240), radius=100, magnitude=1.0)

# 更新向量场
updated_grid = vector_calculator.update_grid_with_adjacent_sum(grid)
```

## 架构概览

LiziEngine 采用分层架构设计，主要包含以下模块：

### 核心层 (core/)

负责应用的生命周期管理和模块间协调：

- **AppCore**: 应用核心，整合所有管理器
- **Container**: 依赖注入容器
- **EventBus**: 事件系统
- **StateManager**: 状态管理
- **ConfigManager**: 配置管理
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

## API 参考

### VectorFieldCalculator

向量场计算器的主要接口。

#### 方法

##### create_vector_grid(width: int, height: int, default: Tuple[float, float] = (0, 0)) -> np.ndarray

创建指定大小的向量网格。

**参数:**
- `width` (int): 网格宽度
- `height` (int): 网格高度
- `default` (Tuple[float, float]): 默认向量值

**返回值:** 形状为 (height, width, 2) 的 numpy 数组

**示例:**
```python
grid = vector_calculator.create_vector_grid(640, 480, (0.0, 0.0))
```

##### update_grid_with_adjacent_sum(grid: np.ndarray) -> np.ndarray

根据相邻向量更新整个网格。

**参数:**
- `grid` (np.ndarray): 输入网格

**返回值:** 更新后的网格

**示例:**
```python
updated_grid = vector_calculator.update_grid_with_adjacent_sum(grid)
```

##### create_radial_pattern(grid: np.ndarray, center: Tuple[float, float] = None, radius: float = None, magnitude: float = 1.0) -> np.ndarray

创建径向向量场模式（从中心向外辐射）。

**参数:**
- `grid` (np.ndarray): 目标网格
- `center` (Tuple[float, float]): 中心坐标，默认为网格中心
- `radius` (float): 影响半径，默认为网格最小边长的一半
- `magnitude` (float): 向量强度

**返回值:** 修改后的网格

##### create_tangential_pattern(grid: np.ndarray, center: Tuple[float, float] = None, radius: float = None, magnitude: float = 1.0) -> np.ndarray

创建切线向量场模式（围绕中心旋转）。

**参数:**
- `grid` (np.ndarray): 目标网格
- `center` (Tuple[float, float]): 中心坐标，默认为网格中心
- `radius` (float): 影响半径，默认为网格最小边长的一半
- `magnitude` (float): 向量强度

**返回值:** 修改后的网格

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

##### set_device(device: str) -> bool

设置计算设备。

**参数:**
- `device` (str): "cpu" 或 "gpu"

**返回值:** 设置是否成功

### ConfigManager

配置管理器。

#### 方法

##### get(key: str, default: Any = None) -> Any

获取配置值。

**参数:**
- `key` (str): 配置键，支持点分隔的嵌套访问
- `default` (Any): 默认值

**示例:**
```python
width = config_manager.get("grid.width", 640)
```

##### set(key: str, value: Any) -> None

设置配置值。

**参数:**
- `key` (str): 配置键
- `value` (Any): 配置值

### StateManager

状态管理器。

#### 方法

##### get(key: str, default: Any = None) -> Any

获取状态值。

##### set(key: str, value: Any) -> None

设置状态值并触发变更通知。

### EventBus

事件系统。

#### 方法

##### subscribe(event_type: EventType, handler: EventHandler) -> None

订阅事件。

##### publish(event: Event) -> None

发布事件。

## 插件开发

### 创建计算插件

```python
from lizi_engine.compute.vector_field import VectorFieldCalculator

class MyComputePlugin:
    def __init__(self, vector_calculator: VectorFieldCalculator):
        self.calculator = vector_calculator

    def create_custom_pattern(self, grid: np.ndarray) -> np.ndarray:
        # 实现自定义向量场模式
        h, w = grid.shape[:2]
        for y in range(h):
            for x in range(w):
                vx = np.sin(x * 0.01) * np.cos(y * 0.01)
                vy = np.cos(x * 0.01) * np.sin(y * 0.01)
                grid[y, x] = (vx, vy)
        return grid
```

### 创建 UI 插件

```python
from plugins.ui import UIManager
from lizi_engine.input import input_handler, KeyMap

class MyUIPlugin(UIManager):
    def register_callbacks(self, grid: np.ndarray, **kwargs):
        super().register_callbacks(grid, **kwargs)

        def on_custom_key():
            print("Custom action triggered!")

        input_handler.register_key_callback(KeyMap.X, on_custom_key)
```

## 配置说明

LiziEngine 使用 JSON 配置文件，支持运行时热更新。

### 示例配置

```json
{
  "grid": {
    "width": 640,
    "height": 480
  },
  "vector": {
    "scale": 1.0,
    "self": {"weight": 0.2},
    "neighbor": {"weight": 0.2}
  },
  "compute": {
    "device": "gpu",
    "iterations": 1
  },
  "render": {
    "vector": {"lines": false}
  }
}
```

## 许可证

MIT License

