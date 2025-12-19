# LiziEngine 底层使用手册

## 1. 简介

本手册旨在帮助开发者理解 LiziEngine 的底层架构，并指导如何编写插件和示例程序。LiziEngine 是一个用于创建、模拟和可视化向量场的强大引擎，采用模块化、事件驱动的架构设计。

## 2. 核心架构

### 2.1 依赖注入容器

LiziEngine 使用依赖注入容器管理服务实例的生命周期，实现松耦合设计。

```python
from lizi_engine.core.container import container

# 注册服务
container.register_singleton(ServiceClass, service_instance)

# 获取服务
service_instance = container.resolve(ServiceClass)
```

### 2.2 事件系统

事件系统是 LiziEngine 的核心通信机制，各模块通过事件总线进行通信。

```python
from lizi_engine.core.events import Event, EventType, event_bus, EventHandler

# 发布事件
event_bus.publish(Event(
    EventType.CUSTOM_EVENT,
    {"key": "value"},
    "Source"
))

# 订阅事件
class CustomHandler(EventHandler):
    def handle(self, event: Event) -> None:
        if event.type == EventType.CUSTOM_EVENT:
            # 处理事件
            pass

# 注册事件处理器
handler = CustomHandler()
event_bus.subscribe(EventType.CUSTOM_EVENT, handler)
```

### 2.3 状态管理

状态管理器集中管理应用状态，提供状态变更通知机制。

```python
from lizi_engine.core.state import state_manager

# 设置状态
state_manager.set("key", "value")

# 获取状态
value = state_manager.get("key", "default_value")

# 批量更新状态
state_manager.update({"key1": "value1", "key2": "value2"})
```

### 2.4 配置管理

配置管理器负责加载和管理配置文件，支持运行时配置更新。

```python
from lizi_engine.core.config import config_manager

# 获取配置
value = config_manager.get("key", "default_value")

# 设置配置
config_manager.set("key", "value")
```

## 3. 核心组件

### 3.1 应用核心 (AppCore)

应用核心是 LiziEngine 的中心组件，整合了各个管理器。

```python
from lizi_engine.core.app import AppCore
from lizi_engine.core.container import container

# 获取或创建应用核心
app_core = container.resolve(AppCore)
if app_core is None:
    app_core = AppCore()
    container.register_singleton(AppCore, app_core)

# 访问管理器
grid_manager = app_core.grid_manager
view_manager = app_core.view_manager
vector_calculator = app_core.vector_calculator
renderer = app_core.renderer
```

### 3.2 网格管理器 (GridManager)

网格管理器负责管理向量场网格数据。

```python
# 初始化网格
grid = app_core.grid_manager.init_grid(width=640, height=480)

# 更新网格中的特定点
updates = {(y, x): (vx, vy) for x, y, vx, vy in points}
app_core.grid_manager.update_grid(updates)

# 清空网格
app_core.grid_manager.clear_grid()

# 保存/加载网格
app_core.grid_manager.save_grid("path/to/file.npy")
app_core.grid_manager.load_grid("path/to/file.npy")
```

### 3.3 视图管理器 (ViewManager)

视图管理器负责相机和视图控制。

```python
# 重置视图
app_core.view_manager.reset_view(width=640, height=480)

# 获取视图参数
cam_x = app_core.state_manager.get("cam_x", 0.0)
cam_y = app_core.state_manager.get("cam_y", 0.0)
cam_zoom = app_core.state_manager.get("cam_zoom", 1.0)
```

### 3.4 向量场计算器 (VectorFieldCalculator)

向量场计算器提供向量场计算的核心功能。

```python
from lizi_engine.compute.vector_field import vector_calculator

# 创建向量网格
grid = vector_calculator.create_vector_grid(width=640, height=480)

# 创建径向模式
vector_calculator.create_radial_pattern(
    grid,
    center=(width//2, height//2),
    radius=None,
    magnitude=1.0
)

# 创建切线模式
vector_calculator.create_tangential_pattern(
    grid,
    center=(width//2, height//2),
    radius=None,
    magnitude=1.0
)

# 更新网格
vector_calculator.update_grid_with_adjacent_sum(grid, include_self=True)
```

### 3.5 渲染器 (VectorFieldRenderer)

渲染器负责向量场的可视化。

```python
from lizi_engine.graphics.renderer import VectorFieldRenderer
from lizi_engine.core.container import container

# 获取或创建渲染器
renderer = container.resolve(VectorFieldRenderer)
if renderer is None:
    renderer = VectorFieldRenderer()
    container.register_singleton(VectorFieldRenderer, renderer)

# 渲染向量场
renderer.render_vector_field(
    grid,
    cell_size=1.0,
    cam_x=0.0,
    cam_y=0.0,
    cam_zoom=1.0,
    viewport_width=800,
    viewport_height=600
)

# 渲染网格
renderer.render_grid(
    grid,
    cell_size=1.0,
    cam_x=0.0,
    cam_y=0.0,
    cam_zoom=1.0,
    viewport_width=800,
    viewport_height=600
)
```

### 3.6 窗口管理器 (Window)

窗口管理器负责创建和管理 OpenGL 窗口。

```python
from lizi_engine.window.window import Window
from lizi_engine.core.container import container

# 获取或创建窗口
window = container.resolve(Window)
if window is None:
    window = Window("My Application", 800, 600)
    container.register_singleton(Window, window)

# 初始化窗口
if not window.initialize():
    print("窗口初始化失败")
    return

# 主循环
while not window.should_close:
    # 更新窗口
    window.update()

    # 渲染内容
    window.render(grid)

# 清理资源
window.cleanup()
```

### 3.7 输入处理器 (InputHandler)

输入处理器负责处理用户输入事件。

```python
from lizi_engine.input import input_handler, KeyMap, MouseMap

# 注册键盘回调
def on_key_press():
    print("键被按下")

input_handler.register_key_callback(KeyMap.SPACE, MouseMap.PRESS, on_key_press)

# 注册鼠标回调
def on_mouse_click():
    print("鼠标被点击")

input_handler.register_mouse_callback(MouseMap.LEFT, MouseMap.PRESS, on_mouse_click)

# 检查按键状态
if input_handler.is_key_pressed(KeyMap.SPACE):
    print("空格键被按下")

# 获取鼠标位置
x, y = input_handler.get_mouse_position()
```
