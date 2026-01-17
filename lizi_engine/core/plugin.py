"""
插件管理器：动态加载和管理插件模块
允许通过插件管理器导入插件类和函数
"""
import importlib
import os

# 存储加载的插件
_plugins = {}

def _load_plugins():
    """加载所有插件模块"""
    # 获取插件目录路径
    plugins_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'plugins')

    # 遍历插件目录中的所有 .py 文件
    for file in os.listdir(plugins_dir):
        if file.endswith('.py') and file != '__init__.py':
            module_name = file[:-3]  # 移除 .py 扩展名
            try:
                # 动态导入插件模块
                module = importlib.import_module(f'plugins.{module_name}')
                # 将模块中的所有公开对象添加到插件字典中
                _plugins.update({name: obj for name, obj in vars(module).items()
                               if not name.startswith('_')})
            except ImportError as e:
                print(f"[警告] 无法加载插件 {module_name}: {e}")

# 加载插件
_load_plugins()

def __getattr__(name):
    """模块级别的 __getattr__，允许直接导入插件"""
    if name in _plugins:
        return _plugins[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

def get_plugin(name):
    """获取指定名称的插件"""
    return __getattr__(name)

def list_plugins():
    """列出所有可用插件"""
    return list(_plugins.keys())
