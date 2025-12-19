# LiziEngine 底层使用手册 - 第二部分

## 4. 插件开发

### 4.1 插件结构

插件通常放在 `plugins` 目录下，每个插件是一个独立的 Python 模块。

```python
# plugins/my_plugin.py
"""
我的插件描述
"""
from typing import Any, Dict
from lizi_engine.core.events import Event, EventType, event_bus, EventHandler
from lizi_engine.core.container import container

class MyPlugin(EventHandler):
    """我的插件类"""
    def __init__(self):
        # 订阅事件
        event_bus.subscribe(EventType.APP_INITIALIZED, self)
        
        # 从容器获取服务
        self.app_core = container.resolve("AppCore")
        
        # 初始化插件
        self._initialized = False
    
    def initialize(self) -> None:
        """初始化插件"""
        if self._initialized:
            return
        
        # 插件初始化逻辑
        print("[我的插件] 初始化完成")
        self._initialized = True
    
    def handle(self, event: Event) -> None:
        """处理事件"""
        if event.type == EventType.APP_INITIALIZED:
            self.initialize()
    
    def cleanup(self) -> None:
        """清理插件资源"""
        print("[我的插件] 清理资源")

# 创建插件实例
my_plugin = MyPlugin()
```

### 4.2 UI 插件开发

UI 插件通常继承自 `UIManager` 类，提供用户交互功能。

```python
# plugins/my_ui_plugin.py
"""
我的UI插件描述
"""
from typing import Any, Dict, Optional, Callable
import numpy as np
from lizi_engine.input import input_handler, KeyMap, MouseMap
from plugins.ui import UIManager

class MyUIPlugin(UIManager):
    """我的UI插件类"""
    def __init__(self, app_core, window, vector_calculator):
        super().__init__(app_core, window, vector_calculator)
        
        # 初始化插件特定状态
        self.my_feature_enabled = False
    
    def register_callbacks(self, grid: np.ndarray, **kwargs) -> None:
        """注册回调函数"""
        super().register_callbacks(grid, **kwargs)
        
        def on_my_key_press():
            self.my_feature_enabled = not self.my_feature_enabled
            status = "开启" if self.my_feature_enabled else "关闭"
            print(f"[我的UI插件] 功能已{status}")
        
        # 注册自定义按键回调
        input_handler.register_key_callback(KeyMap.M, MouseMap.PRESS, on_my_key_press)
    
    def my_custom_function(self, grid: np.ndarray) -> None:
        """自定义功能"""
        if not self.my_feature_enabled:
            return
        
        # 实现自定义功能
        print("[我的UI插件] 执行自定义功能")
```

### 4.3 计算插件开发

计算插件通常扩展向量场计算功能。

```python
# plugins/my_compute_plugin.py
"""
我的计算插件描述
"""
import numpy as np
from typing import Tuple, Optional
from lizi_engine.compute.vector_field import VectorFieldCalculator
from lizi_engine.core.container import container

class MyComputePlugin:
    """我的计算插件类"""
    def __init__(self):
        # 从容器获取向量场计算器
        self.vector_calculator = container.resolve(VectorFieldCalculator)
        
        if self.vector_calculator is None:
            self.vector_calculator = VectorFieldCalculator()
            container.register_singleton(VectorFieldCalculator, self.vector_calculator)
    
    def create_custom_pattern(self, grid: np.ndarray, **kwargs) -> np.ndarray:
        """创建自定义向量场模式"""
        h, w = grid.shape[:2]
        
        # 实现自定义向量场模式
        for y in range(h):
            for x in range(w):
                # 计算向量
                vx = np.sin(x * 0.1) * np.cos(y * 0.1)
                vy = np.cos(x * 0.1) * np.sin(y * 0.1)
                
                # 设置向量
                grid[y, x, 0] = vx
                grid[y, x, 1] = vy
        
        return grid
    
    def custom_update_function(self, grid: np.ndarray, **kwargs) -> np.ndarray:
        """自定义更新函数"""
        # 实现自定义更新逻辑
        return grid

# 创建插件实例
my_compute_plugin = MyComputePlugin()
```

## 5. 示例程序开发

### 5.1 基本示例结构

```python
"""
我的示例程序
"""
import sys
import os
import numpy as np

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lizi_engine.core.container import container
from lizi_engine.core.app import AppCore
from lizi_engine.window.window import Window
from lizi_engine.compute.vector_field import vector_calculator

def main():
    """主函数"""
    print("[示例] 启动我的示例程序...")
    
    # 初始化应用核心
    app_core = container.resolve(AppCore)
    if app_core is None:
        app_core = AppCore()
        container.register_singleton(AppCore, app_core)
    
    # 初始化窗口
    window = container.resolve(Window)
    if window is None:
        window = Window("我的示例", 800, 600)
        container.register_singleton(Window, window)
    
    if not window.initialize():
        print("[示例] 窗口初始化失败")
        return
    
    # 获取网格
    grid = app_core.grid_manager.init_grid(640, 480)
    
    # 创建向量场
    vector_calculator.create_tangential_pattern(grid, magnitude=1.0)
    
    # 初始化视图
    app_core.view_manager.reset_view(grid.shape[1], grid.shape[0])
    
    # 主循环
    print("[示例] 开始主循环...")
    while not window.should_close:
        # 更新窗口
        window.update()
        
        # 渲染
        window.render(grid)
    
    # 清理资源
    print("[示例] 清理资源...")
    window.cleanup()
    app_core.shutdown()
    
    print("[示例] 示例结束")

if __name__ == "__main__":
    main()
```

### 5.2 带UI交互的示例

```python
"""
带UI交互的示例程序
"""
import sys
import os
import numpy as np

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lizi_engine.core.container import container
from lizi_engine.core.app import AppCore
from lizi_engine.window.window import Window
from lizi_engine.compute.vector_field import vector_calculator
from lizi_engine.input import input_handler, KeyMap, MouseMap

def main():
    """主函数"""
    print("[示例] 启动带UI交互的示例程序...")
    
    # 初始化应用核心
    app_core = container.resolve(AppCore)
    if app_core is None:
        app_core = AppCore()
        container.register_singleton(AppCore, app_core)
    
    # 初始化窗口
    window = container.resolve(Window)
    if window is None:
        window = Window("带UI交互的示例", 800, 600)
        container.register_singleton(Window, window)
    
    if not window.initialize():
        print("[示例] 窗口初始化失败")
        return
    
    # 获取网格
    grid = app_core.grid_manager.init_grid(640, 480)
    
    # 创建向量场
    vector_calculator.create_tangential_pattern(grid, magnitude=1.0)
    
    # 初始化视图
    app_core.view_manager.reset_view(grid.shape[1], grid.shape[0])
    
    # 定义回调函数
    def on_space_press():
        """空格键回调"""
        print("[示例] 重新生成向量场")
        vector_calculator.create_tangential_pattern(grid, magnitude=1.0)
    
    def on_r_press():
        """R键回调"""
        print("[示例] 重置视图")
        app_core.view_manager.reset_view(grid.shape[1], grid.shape[0])
    
    def on_g_press():
        """G键回调"""
        show_grid = app_core.state_manager.get("show_grid", True)
        app_core.state_manager.set("show_grid", not show_grid)
        status = "显示" if not show_grid else "隐藏"
        print(f"[示例] 网格{status}")
    
    def on_c_press():
        """C键回调"""
        print("[示例] 清空网格")
        app_core.grid_manager.clear_grid()
    
    # 注册回调函数
    input_handler.register_key_callback(KeyMap.SPACE, MouseMap.PRESS, on_space_press)
    input_handler.register_key_callback(KeyMap.R, MouseMap.PRESS, on_r_press)
    input_handler.register_key_callback(KeyMap.G, MouseMap.PRESS, on_g_press)
    input_handler.register_key_callback(KeyMap.C, MouseMap.PRESS, on_c_press)
    
    # 主循环
    print("[示例] 开始主循环...")
    print("[示例] 按空格键重新生成向量场，按R键重置视图")
    print("[示例] 按G键切换网格显示，按C键清空网格")
    
    while not window.should_close:
        # 更新窗口
        window.update()
        
        # 渲染
        window.render(grid)
    
    # 清理资源
    print("[示例] 清理资源...")
    window.cleanup()
    app_core.shutdown()
    
    print("[示例] 示例结束")

if __name__ == "__main__":
    main()
```

## 6. 最佳实践

### 6.1 代码组织

1. 将插件放在 `plugins` 目录下
2. 将示例放在 `examples` 目录下
3. 遵循 PEP 8 代码风格指南
4. 添加适当的文档字符串和注释

### 6.2 事件处理

1. 使用事件总线进行模块间通信
2. 避免直接调用其他模块的方法
3. 订阅必要的事件，及时取消订阅

### 6.3 资源管理

1. 在插件初始化时获取资源
2. 实现清理方法，释放资源
3. 使用上下文管理器管理资源

### 6.4 错误处理

1. 使用 try-except 块处理异常
2. 记录错误信息，便于调试
3. 提供合理的默认值和回退机制

## 7. 常见问题

### 7.1 如何添加新的向量场模式？

1. 在 `VectorFieldCalculator` 类中添加新方法
2. 实现向量场计算逻辑
3. 在示例程序中调用新方法

### 7.2 如何添加新的渲染效果？

1. 在 `VectorFieldRenderer` 类中添加新方法
2. 实现渲染逻辑
3. 在窗口渲染流程中调用新方法

### 7.3 如何添加新的交互功能？

1. 在 `InputHandler` 中注册新的回调函数
2. 实现交互逻辑
3. 更新状态或触发事件

## 8. 总结

LiziEngine 提供了丰富的底层 API，支持灵活的插件开发和示例程序编写。通过本手册，开发者可以更好地理解 LiziEngine 的架构，并开发出高质量的插件和示例程序。