
# LiziEngine

## 项目概述

LiziEngine 是一个由向量场驱动的物理引擎，它通过模拟现实中的力场（如电磁力）来驱动实体运动，从而在根本上解决传统方法在处理碰撞检测的性能瓶颈。相比与每个实体互相遍历计算，本项目仅需让它们独立计算坐标下的向量值即可处理受力问题。

## 视频演示

[【重力模拟与电荷模拟 【物理引擎开发日志#01】-哔哩哔哩】](https://b23.tv/1Bj1y19)

## 向量场计算原理

### 基本概念

向量场是一个二维网格数据结构，每个网格点包含一个二维向量 `(vx, vy)`，表示该点的方向和强度。向量场的计算涉及相邻点的相互影响，形成复杂的动态模式。

本项目坚持“大道至简”的原则，核心计算逻辑只有两个：迭代向量场与平滑向量值

### 计算方法

#### 1. 相邻向量求和
对于每个网格点，计算其自身和上下左右四个邻居向量的加权和：

```python
权重和 = 自身权重 + 4 × 邻居权重
'''
当权重和等于1时，影响范围无穷大，类似电荷力
当其小于1时，影响范围有限，类似支持力
'''
```

- **自身权重** (`vector_self_weight`): 控制自身向量对结果的影响
- **邻居权重** (`vector_neighbor_weight`): 控制邻居向量对结果的影响

#### 2. 向量场更新
迭代向量场的每个向量值，新向量=自身向量的权重积+邻居向量的权重积：

```python
# 向量场更新
vector_calculator.update_grid_with_adjacent_sum(grid)
```

#### 3. 双线性插值
在浮点坐标处添加或读取向量时，使用双线性插值进行平滑处理：

```python
# 在指定位置拟合一个向量
fit_vector_at_position(grid, x, y)

# 在指定位置添加一个向量
add_vector_at_position(grid, x, y, vx, vy)
```

### 支持的向量场模式

#### 径向模式 (Radial Pattern)
向量从中心向外辐射，形成扩散效果（仅用于测试，只支持整数坐标）：

```python
create_radial_pattern(self, grid, center, radius, magnitude)
```

#### 切线模式 (Tangential Pattern)
向量围绕中心旋转，形成漩涡效果（仅用于测试，只支持整数坐标）：

```python
create_tangential_pattern(self, grid, center, radius, magnitude)
```

#### 微小向量添加
在指定位置创建局部影响，只影响当前位置及其上下左右邻居：

```python
create_tiny_vector(self, grid, x, y, mag)
```

### CPU vs GPU 计算

- **CPU 计算**: 使用 NumPy 向量化操作，适合中小规模计算，易于调试
- **GPU 计算**: 使用 OpenCL 加速，大规模并行计算，性能显著提升，适合实时应用

## 安装和运行

### 系统要求

- Python 3.7+
- OpenGL 支持
- GPU 计算需要 OpenCL 运行时（可选）

### 安装依赖

```bash
pip install -r requirements.txt
# 或者使用 pip install lizi-engine 直接安装底层库
```

### 运行示例

```bash
# 基本使用示例 - 展示核心功能
python examples/basic_usage.py

# 重力箱示例 - 物理模拟
python examples/gravity_box.py

# 模式示例 - 各种向量场模式
python examples/patterns.py

# 输入演示 - 交互功能展示
python examples/input_demo.py
```

### 键盘控制

- **空格键**: 重新生成切线模式
- **G键**: 切换网格显示
- **C键**: 清空网格
- **U键**: 切换实时更新
- **鼠标拖拽**: 平移视图
- **滚轮**: 缩放视图

## 架构设计

### 核心模块 (core/)

- **container.py**: 依赖注入容器，避免单例模式，管理组件生命周期
- **events.py**: 事件系统，支持发布-订阅模式和异步处理
- **state.py**: 状态管理，提供统一的状态管理和变更通知
- **config.py**: 配置管理，支持文件加载和运行时热更新
- **app.py**: 应用核心，整合所有管理器

### 计算模块 (compute/)

- **vector_field.py**: 向量场计算器主接口，统一 CPU/GPU 计算
- **cpu_vector_field.py**: CPU 计算实现，使用 NumPy 向量化
- **gpu_vector_field.py**: GPU 计算实现，基于 OpenCL

### 图形和窗口模块

- **graphics/renderer.py**: OpenGL 渲染器，向量场可视化
- **window/window.py**: 窗口管理，OpenGL 窗口创建和事件处理
- **input/**: 输入处理模块，支持键盘、鼠标事件

### 插件系统 (plugins/)

- **controller.py**: 控制器插件，处理用户输入和向量场操作
- **marker_system.py**: 标记系统，支持在向量场上添加标记点
- **ui.py**: UI 管理器，处理界面交互和事件分发

## 开发指南

### 扩展计算功能

```python
from lizi_engine.compute.vector_field import VectorFieldCalculator

class CustomCalculator(VectorFieldCalculator):
    def create_custom_pattern(self, grid, params):
        # 实现自定义向量场模式
        pass

# 注册自定义计算器
custom_calc = CustomCalculator()
# 在应用中集成
```

### 添加新插件

```python
from plugins.base_plugin import BasePlugin

class MyPlugin(BasePlugin):
    def initialize(self):
        # 初始化逻辑
        pass

    def update(self, delta_time):
        # 更新逻辑
        pass

# 注册插件
app_core.plugin_manager.register_plugin(MyPlugin())
```

## 配置说明

`config.json` 中的主要配置项：

```json
{
    "grid": {
        "width": 640,
        "height": 480,
        "color": [
            0.3,
            0.3,
            0.3
        ]
    },
    "cell": {
        "size": 2.0 //格子大小
    },
    "vector": {
        "color": [
            0.2,
            0.6,
            1.0
        ],
        "scale": 1.0, //向量长度缩放
        "self": {
            "weight": 0.2 //自身权重
        },
        "neighbor": {
            "weight": 0.2 //邻居权重
        }
    },
    "cam": {
        "x": 0.0,
        "y": 0.0,
        "zoom": 1.0
    },
    "show": {
        "grid": true //是否显示网格线
    },
    "background": {
        "color": [
            0.1,
            0.1,
            0.1
        ]
    },
    "antialiasing": true, //是否开启抗锯齿
    "line": {
        "width": 1.0 //线宽
    },
    "compute": {
        "device": "gpu", //计算设备，cpu或gpu
        "iterations": 1 //迭代次数
    },
    "render": {
        "vector": {
            "lines": false //是否显示向量线
        }
    }
}
```

## 性能优化建议

1. **GPU 加速**: 大规模计算时优先使用 GPU
2. **参数调优**: 根据应用场景调整权重参数
3. **内存管理**: 及时清理不再使用的网格数据
4. **渲染优化**: 合理设置窗口大小和刷新率

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request 来改进 LiziEngine！

## 联系方式

如有问题或建议，请通过 GitHub Issues 联系项目维护者。


