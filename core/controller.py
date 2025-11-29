
"""
应用控制器 - 整合各个模块，提供应用程序的主要控制逻辑
"""
import threading
import time
import numpy as np
try:
    import psutil
except ImportError:
    psutil = None
from typing import Dict, Any, Optional, Tuple
from .app import app_core
from .events import EventBus, Event, EventType, EventHandler, FunctionEventHandler
from .state import state_manager
from .config import config_manager

# 使用全局配置管理器实例
from core.config import config_manager as _config
from graphics.window import window_manager
from graphics.renderer import vector_field_renderer
from compute.vector_field import vector_calculator
from compute.opencl_compute import opencl_compute_manager
from ui.toolbar import create_toolbar, get_toolbar

class DebugTools:
    """调试工具类，用于帮助调试和优化程序"""
    
    @staticmethod
    def print_grid_info(grid):
        """打印网格信息"""
        if grid is None:
            print("网格为空")
            return
            
        h, w = grid.shape[:2]
        print(f"网格尺寸: {h}x{w}")
        print(f"网格总点数: {h*w}")
        
        # 计算统计信息
        vectors = grid.reshape(-1, 2)
        magnitudes = np.sqrt(vectors[:, 0]**2 + vectors[:, 1]**2)
        
        print(f"向量幅度 - 最小: {np.min(magnitudes):.3f}, 最大: {np.max(magnitudes):.3f}, 平均: {np.mean(magnitudes):.3f}")
        print(f"零向量数量: {np.sum(magnitudes == 0)}")
        
    @staticmethod
    def analyze_performance瓶颈(average_times):
        """分析性能瓶颈"""
        total_time = average_times.get('update_time', 0) + average_times.get('render_time', 0)
        
        if total_time == 0:
            print("无法分析性能瓶颈：总时间为0")
            return
            
        update_percent = (average_times.get('update_time', 0) / total_time) * 100
        render_percent = (average_times.get('render_time', 0) / total_time) * 100
        
        print(f"性能瓶颈分析:")
        print(f"更新时间占比: {update_percent:.1f}%")
        print(f"渲染时间占比: {render_percent:.1f}%")
        
        if update_percent > render_percent:
            print("主要瓶颈：计算/更新")
        else:
            print("主要瓶颈：渲染")
            
    @staticmethod
    def suggest_optimizations(grid_size, use_opencl):
        """提供优化建议"""
        print(f"优化建议:")
        
        if grid_size > 1000000:
            print("- 考虑使用更大的网格块或分块处理")
        elif grid_size > 100000:
            print("- 考虑启用GPU加速（如果可用）")
            
        if not use_opencl:
            print("- 考虑启用OpenCL以获得更好的性能")
            
        print("- 减少不必要的渲染调用")
        print("- 使用对象池减少内存分配")
        print("- 考虑使用多线程处理CPU密集型任务")


class MemoryMonitor:
    """内存监控器，用于跟踪程序的内存使用情况"""
    
    def __init__(self):
        self.enabled = True
        self.sample_interval = 1.0  # 采样间隔（秒）
        self.last_sample_time = 0
        self.memory_samples = []
        self.max_samples = 100
        
    def update(self, current_time):
        """更新内存监控"""
        if not self.enabled:
            return
            
        if current_time - self.last_sample_time >= self.sample_interval:
            try:
                if psutil is None:
                    return
                process = psutil.Process()
                memory_info = process.memory_info()
                
                sample = {
                    "time": current_time,
                    "rss": memory_info.rss,  # 物理内存使用量
                    "vms": memory_info.vms,  # 虚拟内存使用量
                    "cpu_percent": process.cpu_percent()
                }
                
                self.memory_samples.append(sample)
                
                if len(self.memory_samples) > self.max_samples:
                    self.memory_samples.pop(0)
                    
                self.last_sample_time = current_time
                
            except ImportError:
                print("[内存监控] psutil未安装，无法监控内存使用情况")
                self.enabled = False
            except Exception as e:
                print(f"[内存监控] 获取内存信息时出错: {e}")
                
    def get_memory_stats(self):
        """获取内存统计信息"""
        if not self.memory_samples:
            return None
            
        rss_values = [s["rss"] for s in self.memory_samples]
        vms_values = [s["vms"] for s in self.memory_samples]
        
        return {
            "current_rss": rss_values[-1],
            "current_vms": vms_values[-1],
            "avg_rss": sum(rss_values) / len(rss_values),
            "max_rss": max(rss_values),
            "min_rss": min(rss_values),
            "samples": len(self.memory_samples)
        }
        
    def print_memory_stats(self):
        """打印内存统计信息"""
        stats = self.get_memory_stats()
        if stats is None:
            print("没有内存数据可用")
            return
            
        print("\n=== 内存使用统计 ===")
        print(f"当前物理内存: {stats['current_rss'] / 1024 / 1024:.1f} MB")
        print(f"当前虚拟内存: {stats['current_vms'] / 1024 / 1024:.1f} MB")
        print(f"平均物理内存: {stats['avg_rss'] / 1024 / 1024:.1f} MB")
        print(f"最大物理内存: {stats['max_rss'] / 1024 / 1024:.1f} MB")
        print(f"样本数量: {stats['samples']}")
        print("=====================\n")
        
    def reset(self):
        """重置内存监控数据"""
        self.memory_samples = []
        self.last_sample_time = 0


class PerformanceMonitor:
    """性能监控器，用于跟踪和优化程序性能"""
    
    def __init__(self):
        self.metrics = {
            "frame_times": [],
            "update_times": [],
            "render_times": [],
            "opencl_times": [],
            "cpu_times": []
        }
        self.max_samples = 100  # 保留最近100个样本
        
    def record_frame_time(self, time_ms):
        """记录帧时间"""
        self.metrics["frame_times"].append(time_ms)
        if len(self.metrics["frame_times"]) > self.max_samples:
            self.metrics["frame_times"].pop(0)
            
    def record_update_time(self, time_ms):
        """记录更新时间"""
        self.metrics["update_times"].append(time_ms)
        if len(self.metrics["update_times"]) > self.max_samples:
            self.metrics["update_times"].pop(0)
            
    def record_render_time(self, time_ms):
        """记录渲染时间"""
        self.metrics["render_times"].append(time_ms)
        if len(self.metrics["render_times"]) > self.max_samples:
            self.metrics["render_times"].pop(0)
            
    def record_opencl_time(self, time_ms):
        """记录OpenCL计算时间"""
        self.metrics["opencl_times"].append(time_ms)
        if len(self.metrics["opencl_times"]) > self.max_samples:
            self.metrics["opencl_times"].pop(0)
            
    def record_cpu_time(self, time_ms):
        """记录CPU计算时间"""
        self.metrics["cpu_times"].append(time_ms)
        if len(self.metrics["cpu_times"]) > self.max_samples:
            self.metrics["cpu_times"].pop(0)
            
    def get_average_times(self):
        """获取平均时间"""
        return {
            "frame_time": np.mean(self.metrics["frame_times"]) if self.metrics["frame_times"] else 0,
            "update_time": np.mean(self.metrics["update_times"]) if self.metrics["update_times"] else 0,
            "render_time": np.mean(self.metrics["render_times"]) if self.metrics["render_times"] else 0,
            "opencl_time": np.mean(self.metrics["opencl_times"]) if self.metrics["opencl_times"] else 0,
            "cpu_time": np.mean(self.metrics["cpu_times"]) if self.metrics["cpu_times"] else 0
        }
        
    def get_fps(self):
        """获取FPS"""
        if not self.metrics["frame_times"]:
            return 0
        avg_frame_time = np.mean(self.metrics["frame_times"])
        return 1000.0 / avg_frame_time if avg_frame_time > 0 else 0
        
    def reset(self):
        """重置监控数据"""
        for key in self.metrics:
            self.metrics[key] = []
            
    def generate_report(self):
        """生成性能报告"""
        report = {
            "fps": self.get_fps(),
            "average_times": self.get_average_times(),
            "sample_counts": {key: len(values) for key, values in self.metrics.items()}
        }
        return report
        
    def print_report(self):
        """打印性能报告"""
        report = self.generate_report()
        print("\n=== 性能报告 ===")
        print(f"FPS: {report['fps']:.1f}")
        print(f"平均帧时间: {1000.0/report['fps']:.1f}ms" if report['fps'] > 0 else "平均帧时间: N/A")
        times = report['average_times']
        print(f"平均更新时间: {times['update_time']:.1f}ms")
        print(f"平均渲染时间: {times['render_time']:.1f}ms")
        print(f"样本数量: {report['sample_counts']}")
        print("================\n")


class AppController:
    """应用控制器"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AppController, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._app_core = app_core
            self._event_bus = self._app_core.event_bus
            self._state_manager = self._app_core.state_manager
            self._window_manager = window_manager
            self._vector_calculator = vector_calculator
            self._opencl_compute_manager = opencl_compute_manager
            
            # 性能监控器
            self._performance_monitor = PerformanceMonitor()
            
            # 性能监控开关
            self._enable_performance_monitoring = _config.get("enable_performance_monitoring", True)
            
            # 调试工具
            self._debug_tools = DebugTools()
            
            # 内存监控器
            self._memory_monitor = MemoryMonitor()
            
            # 调试命令处理器
            self._debug_commands = {
                "perf": self._cmd_performance,
                "grid": self._cmd_grid_info,
                "debug": self._cmd_debug_info,
                "memory": self._cmd_memory,
                "help": self._cmd_help
            }
            self._toolbar = None
            self._running = False
            self._lock = threading.Lock()
            self._initialized = True

            # 注册事件处理器
            self._register_event_handlers()

    def _register_event_handlers(self) -> None:
        """注册事件处理器"""
        # 网格更新事件
        self._event_bus.subscribe(EventType.GRID_UPDATED, FunctionEventHandler(self._on_grid_updated))
        
        # 网格更新请求事件（用于解耦window模块）
        if hasattr(EventType, 'GRID_UPDATE_REQUEST'):
            self._event_bus.subscribe(EventType.GRID_UPDATE_REQUEST, FunctionEventHandler(self._on_grid_update_request))

        # 向量更新事件
        if hasattr(EventType, 'VECTOR_UPDATED'):
            self._event_bus.subscribe(EventType.VECTOR_UPDATED, FunctionEventHandler(self._on_vector_updated))

        # 视图重置事件
        self._event_bus.subscribe(EventType.VIEW_RESET, FunctionEventHandler(self._on_view_reset))

        # 网格清空事件
        self._event_bus.subscribe(EventType.GRID_CLEARED, FunctionEventHandler(self._on_grid_cleared))

        # 网格保存事件
        self._event_bus.subscribe(EventType.GRID_SAVED, FunctionEventHandler(self._on_grid_saved))

        # 网格加载事件
        self._event_bus.subscribe(EventType.GRID_LOADED, FunctionEventHandler(self._on_grid_loaded))

        # GPU计算完成事件
        if hasattr(EventType, 'GPU_COMPUTE_COMPLETED'):
            self._event_bus.subscribe(EventType.GPU_COMPUTE_COMPLETED, FunctionEventHandler(self._on_gpu_compute_completed))
            
        # 工具栏相关事件
        if hasattr(EventType, 'TOGGLE_GRID'):
            self._event_bus.subscribe(EventType.TOGGLE_GRID, FunctionEventHandler(self._on_toggle_grid))
            
        if hasattr(EventType, 'RESET_VIEW'):
            self._event_bus.subscribe(EventType.RESET_VIEW, FunctionEventHandler(self._on_reset_view))
            
        if hasattr(EventType, 'CLEAR_GRID'):
            self._event_bus.subscribe(EventType.CLEAR_GRID, FunctionEventHandler(self._on_clear_grid))
            
        if hasattr(EventType, 'SET_BRUSH_SIZE'):
            self._event_bus.subscribe(EventType.SET_BRUSH_SIZE, FunctionEventHandler(self._on_set_brush_size))
            
        if hasattr(EventType, 'SET_MAGNITUDE'):
            self._event_bus.subscribe(EventType.SET_MAGNITUDE, FunctionEventHandler(self._on_set_magnitude))
            
        if hasattr(EventType, 'TOGGLE_REVERSE_VECTOR'):
            self._event_bus.subscribe(EventType.TOGGLE_REVERSE_VECTOR, FunctionEventHandler(self._on_toggle_reverse_vector))

    def initialize(self, title: str = "LiziEngine", width: int = 800, height: int = 600) -> bool:
        """初始化应用程序"""
        try:
            # 设置应用核心引用
            self._window_manager._app_core = self._app_core

            # 创建窗口
            window = self._window_manager.create_window(title, width, height)
            if window is None:
                return False

            # 创建工具栏
            self._toolbar = create_toolbar("auto")
            if self._toolbar is not None and hasattr(self._toolbar, "initialize"):
                self._toolbar.initialize(window)

            # 初始化网格
            grid_width = _config.get("default_grid_width", 640)
            grid_height = _config.get("default_grid_height", 480)
            self._app_core.grid_manager.init_grid(grid_width, grid_height)

            # 重置视图
            self._app_core.view_manager.reset_view(grid_width, grid_height)

            # 启用OpenCL计算
            if config_manager.get("use_opencl_compute", False):
                try:
                    self._opencl_compute_manager.init_compute(grid_width, grid_height)
                    print("[应用控制器] OpenCL计算初始化成功")
                except Exception as e:
                    print(f"[应用控制器] OpenCL计算初始化失败: {e}")
                    # 禁用OpenCL计算，回退到CPU计算
                    config_manager.set("use_opencl_compute", False)

            print("[应用控制器] 初始化完成")
            return True
        except Exception as e:
            print(f"[应用控制器] 初始化失败: {e}")
            return False

    def run(self) -> None:
        """运行应用程序主循环"""
        if not self._window_manager.get_window():
            print("[应用控制器] 窗口未初始化")
            return

        with self._lock:
            if self._running:
                return
            self._running = True

        # 初始化计算环境
        compute_env = self._initialize_compute_environment()

        # 主循环
        self._main_loop(compute_env)

        # 清理资源
        self.cleanup()

    def _initialize_compute_environment(self):
        """初始化计算环境（CPU或GPU）"""
        # 获取网格尺寸
        grid_width = self._app_core.grid_manager._grid.shape[1]
        grid_height = self._app_core.grid_manager._grid.shape[0]
        grid_size = grid_width * grid_height

        # 计算策略：根据网格大小决定使用CPU还是OpenCL计算
        # 小网格使用CPU计算（避免数据传输开销），大网格使用OpenCL计算（发挥并行优势）
        use_opencl = _config.get("use_opencl_compute", False)
        opencl_threshold = _config.get("opencl_compute_threshold", 100000)  # 默认阈值：100000个网格点
        use_opencl_computation = use_opencl and grid_size > opencl_threshold

        print(f"[应用控制器] 网格大小: {grid_width}x{grid_height}={grid_size} 点")
        print(f"[应用控制器] OpenCL计算阈值: {opencl_threshold}")
        print(f"[应用控制器] 使用OpenCL计算: {use_opencl_computation}")

        # 初始化OpenCL计算上下文（如果需要）
        opencl_ctx = None
        if use_opencl_computation:
            try:
                # 获取当前网格数据
                current_grid = self._app_core.grid_manager.grid
                opencl_ctx = self._opencl_compute_manager.init_compute(grid_width, grid_height, current_grid)
                print("[应用控制器] OpenCL计算上下文初始化成功")
            except Exception as e:
                print(f"[应用控制器] OpenCL计算初始化失败: {e}")
                use_opencl_computation = False
                print("[应用控制器] 已回退到CPU计算")

        # 更新频率控制
        update_frequency = _config.get("update_frequency", 30.0)
        update_interval = 1.0 / update_frequency

        return {
            "use_opencl_computation": use_opencl_computation,
            "opencl_ctx": opencl_ctx,
            "update_interval": update_interval,
            "last_update_time": time.time()
        }

    def _main_loop(self, compute_env):
        """主循环"""
        last_time = time.time()
        frame_count = 0
        fps = 0
        fps_update_time = last_time

        # 缓存窗口尺寸和相机参数，避免重复获取
        cached_width = None
        cached_height = None
        cached_cam_x = 0.0
        cached_cam_y = 0.0
        cached_cam_zoom = 1.0
        needs_camera_update = True
        
        # 存储OpenCL上下文，以便在网格更新时可以访问
        self._opencl_ctx = compute_env.get("opencl_ctx")

        while not self._window_manager.should_close():
            frame_start_time = time.time()

            # 处理输入和更新
            self._process_input()

            # 检查是否需要更新相机参数
            if needs_camera_update:
                cached_cam_x = self._state_manager.get("cam_x", 0.0)
                cached_cam_y = self._state_manager.get("cam_y", 0.0)
                cached_cam_zoom = self._state_manager.get("cam_zoom", 1.0)
                needs_camera_update = False

            # 更新网格
            update_start_time = time.time()
            self._update_grid(compute_env, frame_start_time)
            update_time = (time.time() - update_start_time) * 1000  # 转换为毫秒
            if self._enable_performance_monitoring:
                self._performance_monitor.record_update_time(update_time)

            # 渲染
            render_start_time = time.time()
            self._render_scene()
            render_time = (time.time() - render_start_time) * 1000  # 转换为毫秒
            if self._enable_performance_monitoring:
                self._performance_monitor.record_render_time(render_time)

            # 更新UI信息，并获取更新后的fps_update_time和frame_count
            fps_update_time, frame_count = self._update_ui_info(frame_count, frame_start_time, fps_update_time, last_time)
            
            # 记录帧时间
            if self._enable_performance_monitoring:
                frame_time = (time.time() - frame_start_time) * 1000  # 转换为毫秒
                self._performance_monitor.record_frame_time(frame_time)

            # 交换缓冲区
            self._window_manager.swap_buffers()

            # 重置网格更新标志
            grid_needs_update = self._state_manager.get("grid_updated", False)
            if grid_needs_update:
                self._state_manager.set("grid_updated", False)

                needs_camera_update = True  # 相机参数可能已改变

            last_time = frame_start_time
            frame_count += 1

    def _process_input(self):
        """处理输入事件"""
        self._window_manager.poll_events()
        
        # 处理调试快捷键
        if hasattr(self._window_manager, 'get_key_pressed'):
            # F1 - 显示帮助
            if self._window_manager.get_key_pressed(290):  # GLFW_KEY_F1
                self._cmd_help([])
            
            # F2 - 显示性能报告
            if self._window_manager.get_key_pressed(291):  # GLFW_KEY_F2
                self._cmd_performance([])
            
            # F3 - 显示网格信息
            if self._window_manager.get_key_pressed(292):  # GLFW_KEY_F3
                self._cmd_grid_info([])
            
            # F4 - 显示调试信息
            if self._window_manager.get_key_pressed(293):  # GLFW_KEY_F4
                self._cmd_debug_info([])
                
            # G键由事件系统处理，不再直接处理

    def _update_grid(self, compute_env, current_time):
        """更新网格数据"""
        # 如果启用了OpenCL计算，按固定频率执行OpenCL计算
        if (compute_env["use_opencl_computation"] and 
            compute_env["opencl_ctx"] is not None and 
            True):
            
            if current_time - compute_env["last_update_time"] >= compute_env["update_interval"]:
                try:
                    # 执行OpenCL计算
                    include_self = _config.get("include_self", False)
                    self._opencl_compute_manager.step(compute_env["opencl_ctx"], include_self)

                    # 获取计算结果
                    opencl_grid = self._opencl_compute_manager.get_grid_numpy(compute_env["opencl_ctx"])

                    # 更新网格
                    self._app_core.grid_manager._grid = opencl_grid

                    # 标记网格已更新
                    self._state_manager.set("grid_updated", True)
                    compute_env["last_update_time"] = current_time
                except Exception as e:
                    print(f"[应用控制器] OpenCL计算出错: {e}")
                    # 回退到CPU计算
                    compute_env["use_opencl_computation"] = False
                    print("[应用控制器] 已回退到CPU计算")

        # 如果网格较小或OpenCL计算失败，使用CPU计算
        elif (not compute_env["use_opencl_computation"] and 
               True):
            
            if current_time - compute_env["last_update_time"] >= compute_env["update_interval"]:
                try:
                    # 执行CPU计算 - 使用优化后的向量化版本
                    include_self = _config.get("include_self", False)
                    self._app_core.vector_calculator.update_grid_with_adjacent_sum(
                        self._app_core.grid_manager._grid,
                        include_self=include_self
                    )

                    # 标记网格已更新
                    self._state_manager.set("grid_updated", True)
                    compute_env["last_update_time"] = current_time
                except Exception as e:
                    print(f"[应用控制器] CPU计算出错: {e}")

    def _render_scene(self):
        """渲染场景"""
        # 获取窗口尺寸
        width, height = self._window_manager.get_window_size()

        # 获取相机参数
        cam_x = self._state_manager.get("cam_x", 0.0)
        cam_y = self._state_manager.get("cam_y", 0.0)
        cam_zoom = self._state_manager.get("cam_zoom", 1.0)

        # 获取网格数据
        grid = self._app_core.grid_manager.grid

        # 渲染背景
        vector_field_renderer.render_background()

        # 渲染网格
        show_grid = self._state_manager.get("show_grid", True)
        #print(f"[控制器] 渲染网格，show_grid={show_grid}")
        if show_grid:
            vector_field_renderer.render_grid(grid, 1.0, cam_x, cam_y, cam_zoom, width, height)

        # 渲染向量场
        vector_field_renderer.render_vector_field(grid, 1.0, cam_x, cam_y, cam_zoom, width, height)

        # 渲染工具栏
        if self._toolbar is not None and hasattr(self._toolbar, "render"):
            self._toolbar.render()
            
    def _format_performance_info(self, fps):
        """格式化性能信息用于显示"""
        if hasattr(self, "_enable_performance_monitoring") and self._enable_performance_monitoring:
            perf_stats = self._performance_monitor.get_average_times()
            perf_info = f" | Update: {perf_stats['update_time']:.1f}ms | Render: {perf_stats['render_time']:.1f}ms"
            fps = self._performance_monitor.get_fps()
        else:
            perf_info = ""
        
        return perf_info, fps
        
    def _update_ui_info(self, frame_count, current_time, fps_update_time, last_time):
        """更新UI信息，返回更新后的fps_update_time和frame_count"""
        # 计算FPS
        fps_update_interval = 0.1  # 将FPS更新间隔从1.0秒改为0.1秒，提高刷新频率
        if current_time - fps_update_time >= fps_update_interval:
            fps = frame_count / (current_time - fps_update_time)
            frame_count = 0
            fps_update_time = current_time

            # 更新窗口标题
            mouse_grid_x = self._state_manager.get("grid_x", 0)
            mouse_grid_y = self._state_manager.get("grid_y", 0)
            display_vec_x = self._state_manager.get("display_vec_x", 0.0)
            display_vec_y = self._state_manager.get("display_vec_y", 0.0)

            title = f"LiziEngine - FPS: {fps:.1f} - Mouse: ({mouse_grid_x}, {mouse_grid_y}) - Vec: ({display_vec_x:.2f}, {display_vec_y:.2f})"
            self._window_manager.set_window_title(title)
        
        # 返回更新后的值
        return fps_update_time, frame_count

    def _on_grid_updated(self, event: Event) -> None:
        """处理网格更新事件"""
        # 获取更新数据
        updates = event.data.get("updates", {})
        
        # 如果使用OpenCL计算，同步更新GPU缓冲区
        if _config.get("use_opencl_compute", False):
            try:
                # 获取当前OpenCL上下文
                if hasattr(self, "_opencl_ctx") and self._opencl_ctx is not None:
                    # 更新每个修改的向量到GPU缓冲区
                    for (y, x), (vx, vy) in updates.items():
                        self._opencl_compute_manager.upload_texel(self._opencl_ctx, x, y, (vx, vy))
            except Exception as e:
                print(f"[应用控制器] 同步OpenCL缓冲区失败: {e}")
        
    def _on_grid_update_request(self, event: Event) -> None:
        """处理网格更新请求事件（用于解耦window模块）"""
        # 从事件数据中获取更新信息
        updates = event.data.get("updates", {})
        
        # 执行网格更新
        if updates:
            self._app_core.grid_manager.update_grid(updates)

    def _on_vector_updated(self, event: Event) -> None:
        """处理向量更新事件"""
        grid_x = event.data.get("grid_x", 0)
        grid_y = event.data.get("grid_y", 0)
        vector = event.data.get("vector", (0.0, 0.0))

        # 更新单个向量
        updates = {(grid_y, grid_x): vector}
        self._app_core.grid_manager.update_grid(updates)

    def _on_view_reset(self, event: Event) -> None:
        """处理视图重置事件"""
        # 防止递归
        if hasattr(event, '_processed'):
            return
        event._processed = True
        
        grid = self._app_core.grid_manager.grid
        if grid is not None:
            grid_height, grid_width = grid.shape[:2]
            # 直接调用view_manager.reset_view，不触发额外事件
            self._app_core.view_manager._reset_view_internal(grid_width, grid_height)

    def _on_grid_cleared(self, event: Event) -> None:
        """处理网格清空事件"""
        # 网格已被清空，无需再次清空
        # 仅更新状态，确保UI同步
        self._state_manager.set("grid_updated", True, notify=False)

    def _on_grid_saved(self, event: Event) -> None:
        """处理网格保存事件"""
        file_path = event.data.get("file_path", "grid_save.npy")
        self._app_core.grid_manager.save_grid(file_path)

    def _on_grid_loaded(self, event: Event) -> None:
        """处理网格加载事件"""
        file_path = event.data.get("file_path", "grid_save.npy")
        self._app_core.grid_manager.load_grid(file_path)
        
    def _on_config_reloaded(self, event: Event) -> None:
        """处理配置重载事件"""
        try:
            # 重新加载配置文件
            config_file = event.data.get("config_file", "config.json")
            self._config_manager.load_config(config_file)
            
            # 重新初始化计算环境（如果需要）
            if hasattr(self, '_initialize_compute_environment'):
                compute_env = self._initialize_compute_environment()
                print(f"[应用控制器] 配置已重载，计算环境已更新")
                
            # 打印性能报告（如果启用了性能监控）
            if hasattr(self, "_enable_performance_monitoring") and self._enable_performance_monitoring:
                self._performance_monitor.print_report()
                
        except Exception as e:
            print(f"[应用控制器] 重载配置时出错: {e}")
            
    def _process_debug_command(self, command):
        """处理调试命令"""
        parts = command.strip().split()
        if not parts:
            return
            
        cmd = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        
        if cmd in self._debug_commands:
            self._debug_commands[cmd](args)
        else:
            print(f"未知命令: {cmd}. 输入 'help' 查看可用命令")
            
    def _cmd_performance(self, args):
        """性能命令"""
        if self._enable_performance_monitoring:
            self._performance_monitor.print_report()
            
            # 分析性能瓶颈
            avg_times = self._performance_monitor.get_average_times()
            self._debug_tools.analyze_performance瓶颈(avg_times)
        else:
            print("性能监控已禁用")
            
    def _cmd_grid_info(self, args):
        """网格信息命令"""
        grid = self._app_core.grid_manager.grid
        self._debug_tools.print_grid_info(grid)
        
    def _cmd_debug_info(self, args):
        """调试信息命令"""
        print("\n=== 调试信息 ===")
        print(f"性能监控: {'启用' if self._enable_performance_monitoring else '禁用'}")
        print(f"OpenCL计算: {'启用' if self._opencl_compute_manager._context else '禁用'}")
        print(f"窗口尺寸: {self._window_manager.get_window_size()}")
        print(f"相机位置: ({self._state_manager.get('cam_x', 0)}, {self._state_manager.get('cam_y', 0)})")
        print(f"相机缩放: {self._state_manager.get('cam_zoom', 1)}")
        print("================\n")
        
    def _cmd_help(self, args):
        """帮助命令"""
        print("\n=== 可用调试命令 ===")
        print("perf - 显示性能报告")
        print("grid - 显示网格信息")
        print("debug - 显示调试信息")
        print("memory - 显示内存使用情况")
        print("help - 显示此帮助信息")
        print("====================\n")
        
    def _cmd_memory(self, args):
        """内存监控命令"""
        self._memory_monitor.print_memory_stats()

    def _on_gpu_compute_completed(self, event: Event) -> None:
        """处理GPU计算完成事件"""
        # GPU计算已完成
        
    def _on_toggle_grid(self, event: Event) -> None:
        """处理切换网格显示事件"""
        show = event.data.get("show", True)
        print(f"[控制器] 收到TOGGLE_GRID事件，show={show}")
        try:
            print(f"[控制器] 准备更新state_manager中的show_grid")
            # 禁用通知，避免事件冲突
            self._state_manager.set("show_grid", show, notify=False)
            print(f"[控制器] state_manager更新完成")
            print(f"[控制器] 跳过更新config_manager中的show_grid，避免事件冲突")
            # 直接更新config_manager的内部状态，避免触发事件
            try:
                with config_manager._lock:
                    config_manager._settings["show_grid"] = show
                print(f"[控制器] config_manager内部状态更新完成")
            except Exception as e:
                print(f"[控制器] 更新config_manager内部状态时出错: {e}")
                import traceback
                traceback.print_exc()
            print(f"[控制器] 更新show_grid状态为: {show}")
        except Exception as e:
            print(f"[控制器] 更新show_grid状态时出错: {e}")
        
    def _on_reset_view(self, event: Event) -> None:
        """处理重置视图事件"""
        grid = self._app_core.grid_manager.grid
        if grid is not None:
            grid_height, grid_width = grid.shape[:2]
            self._app_core.view_manager.reset_view(grid_width, grid_height)
            
    def _on_clear_grid(self, event: Event) -> None:
        """处理清空网格事件"""
        # 防止递归
        if hasattr(event, '_processed'):
            return
        event._processed = True

        print(f"[控制器] 收到CLEAR_GRID事件")
        try:
            print(f"[控制器] 准备清空网格")
            # 直接操作网格，不触发事件
            with self._app_core.grid_manager._lock:
                if self._app_core.grid_manager._grid is not None:
                    self._app_core.grid_manager._grid.fill(0.0)
                    # 更新状态，不触发事件
                    self._state_manager.set("grid_updated", True, notify=False)

                    # 如果使用OpenCL计算，同步更新GPU缓冲区
                    if _config.get("use_opencl_compute", False):
                        try:
                            # 获取当前OpenCL上下文
                            if hasattr(self, "_opencl_ctx") and self._opencl_ctx is not None:
                                # 将清空后的网格数据上传到GPU
                                self._opencl_compute_manager.upload_grid(self._opencl_ctx, self._app_core.grid_manager._grid)
                                print(f"[控制器] OpenCL缓冲区已更新")
                        except Exception as e:
                            print(f"[控制器] 更新OpenCL缓冲区失败: {e}")

            print(f"[控制器] 网格已清空")
        except Exception as e:
            print(f"[控制器] 清空网格时出错: {e}")
        
    def _on_set_brush_size(self, event: Event) -> None:
        """处理设置画笔大小事件"""
        size = event.data.get("size", 1)
        self._state_manager.set("brush_size", size)
        
    def _on_set_magnitude(self, event: Event) -> None:
        """处理设置向量大小事件"""
        magnitude = event.data.get("magnitude", 1.0)
        self._state_manager.set("vector_magnitude", magnitude)
        
    def _on_toggle_reverse_vector(self, event: Event) -> None:
        """处理切换向量方向事件"""
        reverse = event.data.get("reverse", False)
        self._state_manager.set("reverse_vector", reverse)

    def cleanup(self) -> None:
        """清理应用程序资源"""
        with self._lock:
            if not self._running:
                return
            self._running = False

        try:
            # 清理工具栏
            if self._toolbar is not None:
                self._toolbar.cleanup()
                self._toolbar = None

            # 清理OpenCL计算资源
            if _config.get("use_opencl_compute", False):
                self._opencl_compute_manager.cleanup()

            # 清理渲染器
            vector_field_renderer.cleanup()

            # 清理窗口
            self._window_manager.cleanup()

            # 清理应用核心
            self._app_core.shutdown()

            print("[应用控制器] 资源清理完成")
        except Exception as e:
            print(f"[应用控制器] 清理资源时出错: {e}")

    # update_ui_info方法已被删除，因为_update_ui_info已经实现了相同功能

# 全局应用控制器实例
app_controller = AppController()