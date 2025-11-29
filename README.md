# LiziEngine

![LiziEngine](https://img.shields.io/badge/LiziEngine-1.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 项目概述

LiziEngine 是一个高性能的向量场可视化和编辑工具，用于从根本上解决碰撞检查的性能消耗问题。该工具采用模块化架构，结合了现代图形计算技术和直观的用户交互界面，使用户能够轻松创建、编辑和可视化复杂的向量场。

### 主要特性

- 🎨 **直观的向量场编辑器**：通过鼠标交互直接绘制和编辑向量场
- 🚀 **高性能渲染**：支持GPU加速计算和渲染，处理大规模向量场数据
- 🛠️ **灵活的工具栏系统**：提供图形界面和键盘快捷键两种操作模式
- 📊 **实时可视化**：即时查看向量场变化，支持多种可视化效果
- 🔧 **模块化架构**：清晰的代码结构，易于扩展和维护
- ⚙️ **可配置性**：通过配置文件自定义各种参数和行为
- 🌐 **跨平台支持**：支持Windows、Linux和macOS

## 架构设计

LiziEngine 采用分层模块化架构，各模块职责明确，通过事件系统进行通信，降低耦合度。

```
LiziEngine/
├── core/          # 核心模块
├── compute/       # 计算模块
├── graphics/      # 图形渲染模块
├── ui/            # 用户界面模块
└── run.py         # 应用入口
```

### 核心模块 (core/)

- **events.py**: 事件系统，提供发布-订阅模式的事件通信机制
- **state.py**: 状态管理，提供线程安全的状态管理功能
- **config.py**: 配置管理，提供统一的配置加载和管理功能
- **app.py**: 应用核心，整合各个管理器，提供应用程序的主要功能
- **controller.py**: 应用控制器，整合各个模块，提供应用程序的主要控制逻辑
- **performance.py**: 性能监控和分析工具

### 计算模块 (compute/)

- **vector_field.py**: 向量场计算，提供向量场计算的核心功能
- **opencl_compute.py**: OpenCL计算，提供基于OpenCL的GPU并行计算功能

### 图形模块 (graphics/)

- **renderer.py**: 渲染器，提供向量场的渲染功能
- **window.py**: 窗口管理，提供OpenGL窗口的创建和管理功能

### 用户界面模块 (ui/)

- **toolbar.py**: 工具栏基类和ImGui实现
- **simple_toolbar.py**: 简单键盘工具栏实现

## 主要改进

1. **模块化设计**: 将代码按功能划分为多个模块，每个模块职责单一
2. **事件驱动**: 采用事件系统进行模块间通信，降低耦合度
3. **状态集中管理**: 统一的状态管理，避免全局变量滥用
4. **配置统一管理**: 集中的配置管理，支持动态配置
5. **资源管理**: 更好的资源生命周期管理，防止内存泄漏
6. **性能优化**: 添加性能监控模块，便于分析和优化性能瓶颈
7. **GPU加速**: 支持OpenCL GPU并行计算，大幅提升大规模向量场处理能力

## 安装与使用

### 系统要求

- Python 3.8 或更高版本
- 支持OpenGL的图形驱动
- 4GB以上RAM（推荐8GB或更多）
- 支持OpenCL的GPU（可选，用于加速计算）

### 安装依赖

基本依赖（必须安装）：
```bash
pip install -r requirements.txt
```

可选依赖（用于图形界面工具栏）：
```bash
pip install imgui[glfw]
```

注意：如果安装imgui时遇到编译错误，可以尝试以下方法：
1. 安装Visual Studio Build Tools（Windows）
2. 或者使用预编译的wheel包：`pip install --only-binary=all imgui`
3. 如果仍然失败，程序会自动使用简单的键盘快捷键工具栏

### 运行应用

```bash
python run.py
```

### 工具栏类型

LiziEngine 支持两种工具栏类型：
1. **ImGui工具栏**：图形界面，需要安装 `imgui[glfw]`
2. **简单工具栏**：基于键盘快捷键，无需额外依赖

默认情况下，程序会自动检测ImGui是否可用，并选择合适的工具栏类型。你也可以通过修改 `run.py` 中的 `create_toolbar()` 调用来指定工具栏类型：

```python
# 使用ImGui工具栏
toolbar = create_toolbar("imgui")

# 使用简单工具栏
toolbar = create_toolbar("simple")

# 自动选择（默认）
toolbar = create_toolbar("auto")
```

### 配置文件

配置文件支持JSON格式，默认配置文件为 `config.json`。您可以通过修改此文件来自定义各种参数：

```json
{
  "window": {
    "width": 1200,
    "height": 800,
    "title": "LiziEngine"
  },
  "vector_field": {
    "grid_size": 50,
    "default_vector_length": 0.5,
    "vector_color": [0.2, 0.6, 1.0, 1.0]
  },
  "rendering": {
    "background_color": [0.1, 0.1, 0.1, 1.0],
    "grid_color": [0.3, 0.3, 0.3, 1.0],
    "show_grid": true
  },
  "ui": {
    "toolbar_type": "auto"
  }
}
```

## 快速入门

1. **启动应用**：
   ```bash
   python run.py
   ```

2. **基本操作**：
   - 使用鼠标左键在网格上绘制向量
   - 使用鼠标右键拖动视图
   - 使用鼠标滚轮缩放视图

3. **工具栏操作**：
   - 使用工具栏调整画笔大小、向量大小等参数
   - 切换不同的可视化模式

4. **快捷键**：
   - `G`: 切换网格显示
   - `R`: 重置视图
   - `C`: 清空网格
   - `T`: 切换工具栏显示
   - `H`: 显示帮助信息
   - `+/-`: 增加/减少画笔大小
   - `</>`: 增加/减少向量大小
   - `V`: 切换向量方向
   - `Space`: 暂停/恢复动画
   - `S`: 保存当前向量场
   - `L`: 加载向量场文件

## 开发指南

### 添加新功能

1. 在相应的模块中添加新功能
2. 通过事件系统与其他模块通信
3. 更新配置文件（如需要）
4. 更新状态（如需要）

### 自定义渲染器

1. 继承 `VectorFieldRenderer` 类
2. 重写相应的渲染方法
3. 在应用控制器中注册新的渲染器

示例：
```python
class CustomRenderer(VectorFieldRenderer):
    def __init__(self, context):
        super().__init__(context)

    def render(self):
        # 自定义渲染逻辑
        pass

# 在控制器中注册
controller.register_renderer("custom", CustomRenderer)
```

### 自定义计算器

1. 继承 `VectorFieldCalculator` 类
2. 实现自定义的计算方法
3. 在应用控制器中注册新的计算器

示例：
```python
class CustomCalculator(VectorFieldCalculator):
    def __init__(self, context):
        super().__init__(context)

    def calculate(self, field):
        # 自定义计算逻辑
        return result

# 在控制器中注册
controller.register_calculator("custom", CustomCalculator)
```

### 事件系统

LiziEngine 使用事件系统进行模块间通信。您可以发布自定义事件或订阅现有事件：

```python
# 发布事件
events.publish("custom_event", data)

# 订阅事件
def event_handler(data):
    # 处理事件
    pass

events.subscribe("custom_event", event_handler)
```

## 性能优化

LiziEngine 提供了多种性能优化选项：

1. **GPU加速**：使用OpenCL进行并行计算
2. **LOD系统**：根据视图距离自动调整细节级别
3. **视锥剔除**：只渲染可见区域内的向量
4. **批处理渲染**：减少绘制调用次数

您可以通过配置文件调整这些参数以获得最佳性能。

## 常见问题

### Q: 如何提高渲染性能？
A: 可以尝试以下方法：
- 降低网格分辨率
- 启用GPU加速计算
- 减少向量场密度
- 调整渲染质量设置

### Q: 如何导入/导出向量场数据？
A: 使用快捷键 `S` 保存和 `L` 加载向量场文件。支持JSON和CSV格式。

### Q: 如何自定义向量场颜色？
A: 修改配置文件中的 `vector_color` 参数，或通过工具栏的颜色选择器进行调整。

## 贡献指南

欢迎为LiziEngine贡献代码！请遵循以下步骤：

1. Fork本项目
2. 创建您的功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建一个Pull Request

## 注意事项

1. 避免直接修改其他模块的内部状态
2. 使用事件系统进行模块间通信
3. 遵循单一职责原则
4. 保持配置和状态的一致性
5. 编写清晰的文档和注释

## 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件

## 致谢

感谢所有为LiziEngine做出贡献的开发者和用户！

## 联系方式

- 项目主页：[GitHub链接]
- 问题反馈：[Issues链接]
- 邮箱：[联系邮箱]
