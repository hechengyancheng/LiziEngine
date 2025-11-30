
"""
窗口管理模块 - 提供OpenGL窗口的创建和管理功能
"""
import sys
import threading
import time
from typing import Optional, Dict, Any, Callable, Tuple
import glfw
# 尝试导入OpenGL，如果失败则提供错误信息
try:
    from OpenGL.GL import *
    print("[窗口管理] OpenGL导入成功")
except ImportError as e:
    print(f"[窗口管理] OpenGL导入失败: {e}")
    print("[窗口管理] 请确保已正确安装PyOpenGL库")
    sys.exit(1)
import numpy as np
from core.config import config_manager
from core.events import EventBus, Event, EventType
from core.state import state_manager
from .renderer import vector_field_renderer
from core.app import app_core
# 移除app_controller导入，通过事件系统解耦

class WindowManager:
    """窗口管理器"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(WindowManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            # 使用全局事件总线实例，而不是创建新的实例
            from core.events import event_bus
            self._event_bus = event_bus
            self._state_manager = state_manager
            self._window = None
            self._app_core = None
            self._lock = threading.Lock()
            self._initialized = True

    def create_window(self, title: str = "LiziEngine", width: int = 800, height: int = 600) -> Optional[int]:
        """创建OpenGL窗口"""
        # 初始化GLFW
        try:
            if not glfw.init():
                print("[窗口管理] 初始化GLFW失败")
                return None
        except Exception as e:
            print(f"[窗口管理] 初始化GLFW时发生错误: {e}")
            return None

        # 设置窗口提示
        glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)
        
        # 尝试设置OpenGL版本，从高到低尝试多个版本
        opengl_versions = [
            (3, 3, glfw.OPENGL_CORE_PROFILE),  # 首先尝试3.3核心模式
            (3, 2, glfw.OPENGL_CORE_PROFILE),  # 然后尝试3.2核心模式
            (3, 1, glfw.OPENGL_ANY_PROFILE),   # 尝试3.1
            (3, 0, glfw.OPENGL_ANY_PROFILE),   # 尝试3.0
            (2, 1, glfw.OPENGL_ANY_PROFILE),   # 最后尝试2.1
        ]

        context_created = False
        for major, minor, profile in opengl_versions:
            try:
                glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, major)
                glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, minor)
                if major >= 3 and minor >= 2:
                    glfw.window_hint(glfw.OPENGL_PROFILE, profile)

                # 尝试创建窗口
                test_window = glfw.create_window(100, 100, "Test", None, None)
                if test_window:
                    # 测试窗口创建成功，销毁它并使用这些设置
                    glfw.destroy_window(test_window)
                    print(f"[窗口管理] 成功设置OpenGL {major}.{minor} {'核心' if profile == glfw.OPENGL_CORE_PROFILE else '兼容'}模式")
                    context_created = True
                    break
            except Exception as e:
                print(f"[窗口管理] 尝试OpenGL {major}.{minor}失败: {e}")
                continue

        if not context_created:
            print("[窗口管理] 所有OpenGL版本尝试失败，使用默认设置")
            # 重置所有提示，使用系统默认设置
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 1)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 0)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_ANY_PROFILE)
        
        # 如果是MacOS，需要设置向前兼容
        if sys.platform == "darwin":
            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)

        # 创建窗口
        try:
            self._window = glfw.create_window(width, height, title, None, None)
            if not self._window:
                print("[窗口管理] 创建窗口失败")
                glfw.terminate()
                return None
            print(f"[窗口管理] 窗口创建成功: {width}x{height}")
        except KeyboardInterrupt:
            # 特别处理键盘中断，确保资源正确释放
            print("[窗口管理] 创建窗口被中断")
            glfw.terminate()
            raise  # 重新抛出中断，让上层处理
        except Exception as e:
            print(f"[窗口管理] 创建窗口时发生错误: {e}")
            print("[窗口管理] 尝试使用基本设置重新创建窗口")

            # 重置所有窗口提示到最基本设置
            glfw.default_window_hints()
            glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)

            # 尝试使用最基本设置创建窗口
            try:
                self._window = glfw.create_window(width, height, title, None, None)
                if not self._window:
                    print("[窗口管理] 使用基本设置创建窗口仍然失败")
                    glfw.terminate()
                    return None
                print(f"[窗口管理] 使用基本设置创建窗口成功: {width}x{height}")
            except KeyboardInterrupt:
                # 特别处理键盘中断，确保资源正确释放
                print("[窗口管理] 使用基本设置创建窗口被中断")
                glfw.terminate()
                raise  # 重新抛出中断，让上层处理
            except Exception as e2:
                print(f"[窗口管理] 使用基本设置创建窗口仍然失败: {e2}")
                glfw.terminate()
                return None

        # 设置窗口为当前上下文
        try:
            glfw.make_context_current(self._window)
            # 验证上下文是否成功创建
            if not glfw.get_window_attrib(self._window, glfw.CONTEXT_VERSION_MAJOR):
                print("[窗口管理] OpenGL上下文创建失败")
                glfw.destroy_window(self._window)
                self._window = None
                return None
            print("[窗口管理] OpenGL上下文创建成功")
        except Exception as e:
            print(f"[窗口管理] 设置OpenGL上下文失败: {e}")
            glfw.destroy_window(self._window)
            self._window = None
            return None

        # 检查OpenGL版本
        try:
            gl_version = glGetString(GL_VERSION).decode('utf-8')
            gl_vendor = glGetString(GL_VENDOR).decode('utf-8')
            gl_renderer = glGetString(GL_RENDERER).decode('utf-8')
            print(f"[窗口管理] OpenGL版本: {gl_version}")
            print(f"[窗口管理] OpenGL供应商: {gl_vendor}")
            print(f"[窗口管理] OpenGL渲染器: {gl_renderer}")
        except Exception as e:
            print(f"[窗口管理] 获取OpenGL信息失败: {e}")

        # 设置窗口回调
        glfw.set_window_size_callback(self._window, self._on_window_resize)
        glfw.set_key_callback(self._window, self._on_key)
        glfw.set_cursor_pos_callback(self._window, self._on_cursor_pos)
        glfw.set_mouse_button_callback(self._window, self._on_mouse_button)
        glfw.set_scroll_callback(self._window, self._on_scroll)

        # 设置窗口用户指针，用于在回调中访问WindowManager实例
        glfw.set_window_user_pointer(self._window, self)

        # 初始化渲染器
        try:
            vector_field_renderer.initialize()
            print("[窗口管理] 渲染器初始化成功")
        except Exception as e:
            print(f"[窗口管理] 渲染器初始化失败: {e}")
            print("[窗口管理] 尝试使用兼容模式继续运行")

        # 更新状态
        self._state_manager.update({
            "window_title": title,
            "window_width": width,
            "window_height": height
        })

        print(f"[窗口管理] 窗口创建成功: {width}x{height}")

        # 发布窗口创建事件
        self._event_bus.publish(Event(
            EventType.WINDOW_CREATED,
            {"title": title, "width": width, "height": height},
            "WindowManager"
        ))

        return self._window

    def _on_window_resize(self, window: int, width: int, height: int) -> None:
        """窗口大小改变回调"""
        # 检查窗口是否有效
        if window is None:
            return
            
        if height == 0:
            height = 1

        glViewport(0, 0, width, height)

        # 使用锁保护共享状态更新
        with self._lock:
            # 更新状态
            self._state_manager.update({
                "window_width": width,
                "window_height": height,
                "view_changed": True
            })

            # 发布窗口大小改变事件
            self._event_bus.publish(Event(
                EventType.WINDOW_RESIZED,
                {"width": width, "height": height},
                "WindowManager"
            ))

    def _on_key(self, window: int, key: int, scancode: int, action: int, mods: int) -> None:
        """键盘按键回调"""
        # 检查窗口是否有效
        if window is None:
            return
            
        # 使用锁保护共享状态访问和修改
        with self._lock:
            # 发布键盘事件
            self._event_bus.publish(Event(
                EventType.KEY_PRESSED,
                {"key": key, "scancode": scancode, "action": action, "mods": mods},
                "WindowManager"
            ))

            # ESC键退出
            if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
                glfw.set_window_should_close(window, True)

    def _on_cursor_pos(self, window: int, xpos: float, ypos: float) -> None:
        """鼠标位置回调"""
        # 检查窗口是否有效
        if window is None:
            return
            
        # 使用锁保护共享状态访问和修改
        with self._lock:
            # 处理中键拖动视图
            if self._state_manager.get("mouse_middle_pressed", False):
                # 获取初始状态
                start_x = self._state_manager.get("mouse_middle_start_x", 0.0)
                start_y = self._state_manager.get("mouse_middle_start_y", 0.0)
                cam_start_x = self._state_manager.get("cam_start_x", 0.0)
                cam_start_y = self._state_manager.get("cam_start_y", 0.0)
                cam_zoom = self._state_manager.get("cam_zoom", 1.0)

                # 计算鼠标移动距离并转换为世界坐标
                dx = (start_x - xpos) / cam_zoom
                dy = (start_y - ypos) / cam_zoom

                # 更新相机位置
                new_cam_x = cam_start_x + dx
                new_cam_y = cam_start_y + dy

                # 更新状态
                self._state_manager.update({
                    "cam_x": new_cam_x,
                    "cam_y": new_cam_y,
                    "view_changed": True
                })

                # 发布视图变更事件
                self._event_bus.publish(Event(
                    EventType.VIEW_CHANGED,
                    {"cam_x": new_cam_x, "cam_y": new_cam_y},
                    "WindowManager"
                ))

        # 获取窗口尺寸
        width, height = glfw.get_window_size(window)

        # 使用锁保护共享状态访问和修改
        with self._lock:
            # 获取相机参数
            cam_x = self._state_manager.get("cam_x", 0.0)
            cam_y = self._state_manager.get("cam_y", 0.0)
            cam_zoom = self._state_manager.get("cam_zoom", 1.0)

            # 计算世界坐标
            half_w = (width / 2.0) / cam_zoom
            half_h = (height / 2.0) / cam_zoom

            world_x = cam_x - half_w + (xpos / width) * (2 * half_w)
            world_y = cam_y - half_h + (ypos / height) * (2 * half_h)

            # 获取网格单元大小
            cell_size = config_manager.get("rendering.cell_size", 1.0)

            # 计算网格坐标
            grid_x = int(world_x / cell_size)
            grid_y = int(world_y / cell_size)

            # 更新状态
            self._state_manager.update({
                "mouse_x": xpos,
                "mouse_y": ypos,
                "world_x": world_x,
                "world_y": world_y,
                "grid_x": grid_x,
                "grid_y": grid_y
            })

            # 获取当前网格数据
            grid = app_core.grid_manager.grid
            if grid is not None:
                # 确保在网格范围内
                if 0 <= grid_y < grid.shape[0] and 0 <= grid_x < grid.shape[1]:
                    # 获取当前位置的向量
                    vector = grid[grid_y, grid_x]
                    self._state_manager.update({
                        "display_vec_x": float(vector[0]),
                        "display_vec_y": float(vector[1])
                    })
                else:
                    # 如果不在网格范围内，重置为0
                    self._state_manager.update({
                        "display_vec_x": 0.0,
                        "display_vec_y": 0.0
                    })
            else:
                # 如果没有网格数据，重置为0
                self._state_manager.update({
                    "display_vec_x": 0.0,
                    "display_vec_y": 0.0
                })

            # 发布鼠标移动事件
            self._event_bus.publish(Event(
                EventType.MOUSE_MOVED,
                {"x": xpos, "y": ypos, "grid_x": grid_x, "grid_y": grid_y},
                "WindowManager"
            ))

    def _on_mouse_button(self, window: int, button: int, action: int, mods: int) -> None:
        """鼠标按键回调"""
        # 检查窗口是否有效
        if window is None:
            return
            
        # 获取鼠标位置
        xpos, ypos = glfw.get_cursor_pos(window)
        
        # 获取窗口尺寸
        width, height = glfw.get_window_size(window)
        
        # 使用锁保护共享状态访问和修改
        with self._lock:
            # 获取相机参数
            cam_x = self._state_manager.get("cam_x", 0.0)
            cam_y = self._state_manager.get("cam_y", 0.0)
            cam_zoom = self._state_manager.get("cam_zoom", 1.0)
            
            # 计算世界坐标
            half_w = (width / 2.0) / cam_zoom
            half_h = (height / 2.0) / cam_zoom
            
            world_x = cam_x - half_w + (xpos / width) * (2 * half_w)
            world_y = cam_y - half_h + (ypos / height) * (2 * half_h)
            
            # 获取网格单元大小
            cell_size = config_manager.get("rendering.cell_size", 1.0)
            
            # 计算网格坐标
            grid_x = int(world_x / cell_size)
            grid_y = int(world_y / cell_size)

            # 处理中键拖动视图功能
            if button == glfw.MOUSE_BUTTON_MIDDLE:  # 中键
                if action == glfw.PRESS:  # 按下
                    # 记录鼠标中键按下状态和初始位置
                    self._state_manager.update({
                        "mouse_middle_pressed": True,
                        "mouse_middle_start_x": xpos,
                        "mouse_middle_start_y": ypos,
                        "cam_start_x": self._state_manager.get("cam_x", 0.0),
                        "cam_start_y": self._state_manager.get("cam_y", 0.0)
                    })
                elif action == glfw.RELEASE:  # 释放
                    # 清除鼠标中键按下状态
                    self._state_manager.set("mouse_middle_pressed", False)

            # 获取网格数据
            # 使用get_raw_grid方法获取原始网格数据引用，避免创建副本
            grid = app_core.grid_manager.get_raw_grid()
            if grid is not None and button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:  # 左键按下
                # 添加调试信息
                print(f"[窗口管理] 鼠标左键点击: 屏幕坐标({xpos:.1f}, {ypos:.1f}), 网格坐标({grid_x}, {grid_y})")
                
                # 检查网格坐标是否有效
                if not (0 <= grid_x < grid.shape[1] and 0 <= grid_y < grid.shape[0]):
                    print(f"[窗口管理] 鼠标点击位置超出网格范围: ({grid_x}, {grid_y})")
                    return
                # 获取当前向量大小
                magnitude = config_manager.get("vector_field.default_vector_length", 1.0)

                # 获取画笔大小
                brush_size = config_manager.get("vector_field.default_brush_size", 1)

                # 获取是否反转向量
                reverse_vector = self._state_manager.get("reverse_vector", False)
            
                # 获取向量模式（0=辐射状, 1=单一方向, 2=顺时针旋转）
                vector_mode = self._state_manager.get("vector_mode", 0)

                # 应用画笔效果
                updates = {}
                for dy_offset in range(-brush_size + 1, brush_size):
                    for dx_offset in range(-brush_size + 1, brush_size):
                        # 计算距离中心的距离
                        dist = np.sqrt(dx_offset**2 + dy_offset**2)
                        if dist < brush_size:
                            # 计算实际影响的网格位置
                            target_y = grid_y + dy_offset
                            target_x = grid_x + dx_offset

                            # 确保在网格范围内
                            if 0 <= target_y < grid.shape[0] and 0 <= target_x < grid.shape[1]:
                                # 获取当前位置的现有向量
                                current_vx = float(grid[target_y, target_x, 0])
                                current_vy = float(grid[target_y, target_x, 1])

                                # 根据距离调整向量大小的影响
                                influence = max(0.0, 1.0 - (dist / brush_size))

                                # 根据向量模式计算新向量
                                if vector_mode == 0:
                                    # 辐射状模式 - 以鼠标为中心，默认朝外
                                    # 计算从鼠标位置到目标点的向量
                                    dx = target_x - grid_x
                                    dy = target_y - grid_y

                                    # 计算距离
                                    point_dist = np.sqrt(dx**2 + dy**2)

                                    if point_dist > 0.001:  # 避免除以零
                                        # 归一化方向向量
                                        dir_x = dx / point_dist
                                        dir_y = dy / point_dist

                                        # 根据是否反转向量决定方向
                                        if reverse_vector:
                                            dir_x = -dir_x
                                            dir_y = -dir_y

                                        # 应用大小和影响因子
                                        new_vx = dir_x * influence * magnitude
                                        new_vy = dir_y * influence * magnitude
                                    else:
                                        # 如果是鼠标位置本身，创建一个默认向量
                                        if reverse_vector:
                                            new_vx = -influence * magnitude
                                        else:
                                            new_vx = influence * magnitude
                                        new_vy = 0
                                    
                                    # 添加到更新字典
                                    updates[(target_y, target_x)] = (new_vx, new_vy)
                                elif vector_mode == 1:
                                    # 单一方向模式 - 所有向量使用相同方向
                                    # 默认向右，如果反转则向左
                                    if reverse_vector:
                                        new_vx = -influence * magnitude
                                    else:
                                        new_vx = influence * magnitude
                                    new_vy = 0
                                    
                                    # 添加到更新字典
                                    updates[(target_y, target_x)] = (new_vx, new_vy)
                                else:
                                    # 顺时针旋转模式 - 向量围绕鼠标顺时针旋转
                                    # 计算从鼠标位置到目标点的向量
                                    dx = target_x - grid_x
                                    dy = target_y - grid_y

                                    # 计算距离
                                    point_dist = np.sqrt(dx**2 + dy**2)

                                    if point_dist > 0.001:  # 避免除以零
                                        # 计算切向量（顺时针旋转90度）
                                        if reverse_vector:
                                            # 反转时逆时针旋转
                                            dir_x = -dy / point_dist
                                            dir_y = dx / point_dist
                                        else:
                                            # 默认顺时针旋转
                                            dir_x = dy / point_dist
                                            dir_y = -dx / point_dist

                                        # 应用大小和影响因子
                                        new_vx = dir_x * influence * magnitude
                                        new_vy = dir_y * influence * magnitude
                                    else:
                                        # 如果是鼠标位置本身，创建一个默认向量
                                        if reverse_vector:
                                            new_vx = 0
                                            new_vy = -influence * magnitude
                                        else:
                                            new_vx = 0
                                            new_vy = influence * magnitude

                                    updates[(target_y, target_x)] = (new_vx, new_vy)
                                    print(f"[窗口管理] 更新点({target_x}, {target_y})的向量为({new_vx:.2f}, {new_vy:.2f})")

                # 使用统一接口更新网格
                if updates:
                    # 通过事件系统发布网格更新请求，而不是直接调用app_controller
                    # 这样可以避免循环导入问题
                    try:
                        event_type = EventType.GRID_UPDATE_REQUEST
                    except AttributeError:
                        print("[窗口管理] 错误: GRID_UPDATE_REQUEST 事件类型不存在，无法更新网格")
                        return
                
                    #print("[窗口管理] 发布网格更新请求")

                    # 记录向量场中心点位置（支持多个中心点）
                    centers = self._state_manager.get("vector_field_centers", [])
                    # 检查是否已存在相近的中心点，避免重复添加
                    exists = False
                    for center in centers:
                        if len(center) >= 2 and abs(center[0] - grid_x) < 5 and abs(center[1] - grid_y) < 5:
                            exists = True
                            break

                    if not exists:
                        centers.append([grid_x, grid_y])
                        self._state_manager.set("vector_field_centers", centers)

                    # 添加调试信息
                    print(f"[窗口管理] 发布网格更新请求，更新点数: {len(updates)}")
                    
                    self._event_bus.publish(Event(
                        event_type,
                        {"updates": updates},
                        "WindowManager"
                    ))
                
            # 发布鼠标按键事件
            self._event_bus.publish(Event(
                EventType.MOUSE_CLICKED,
                {"button": button, "action": action, "mods": mods, "x": xpos, "y": ypos, "grid_x": grid_x, "grid_y": grid_y},
                "WindowManager"
            ))

    def _on_scroll(self, window: int, xoffset: float, yoffset: float) -> None:
        """鼠标滚轮回调"""
        # 检查窗口是否有效
        if window is None:
            return
            
        # 使用锁保护共享状态访问和修改
        with self._lock:
            # 获取当前缩放级别
            cam_zoom = self._state_manager.get("cam_zoom", 1.0)

            # 根据滚轮方向调整缩放级别
            zoom_factor = config_manager.get("rendering.zoom_factor", 1.1)
            if yoffset > 0:
                cam_zoom *= zoom_factor
            else:
                cam_zoom /= zoom_factor

            # 限制缩放范围
            min_zoom = config_manager.get("rendering.min_zoom", 0.1)
            max_zoom = config_manager.get("rendering.max_zoom", 10.0)
            cam_zoom = max(min_zoom, min(max_zoom, cam_zoom))

            # 更新状态
            self._state_manager.update({
                "cam_zoom": cam_zoom,
                "view_changed": True
            })

            # 发布缩放事件
            self._event_bus.publish(Event(
                EventType.ZOOM_CHANGED,
                {"zoom": cam_zoom},
                "WindowManager"
            ))

    def get_window(self) -> Optional[int]:
        """获取窗口句柄"""
        return self._window

    def should_close(self) -> bool:
        """检查窗口是否应该关闭"""
        if self._window is None:
            return True
        return glfw.window_should_close(self._window)

    def swap_buffers(self) -> None:
        """交换前后缓冲区"""
        if self._window is not None:
            glfw.swap_buffers(self._window)

    def poll_events(self) -> None:
        """处理事件"""
        glfw.poll_events()

    def get_window_size(self) -> Tuple[int, int]:
        """获取窗口大小"""
        if self._window is None:
            return (800, 600)
        return glfw.get_window_size(self._window)
        
    def get_key_pressed(self, key: int) -> bool:
        """检查指定按键是否被按下"""
        if self._window is None:
            return False
        return glfw.get_key(self._window, key) == glfw.PRESS

    def get_framebuffer_size(self) -> Tuple[int, int]:
        """获取帧缓冲区大小"""
        if self._window is None:
            return (800, 600)
        return glfw.get_framebuffer_size(self._window)

    def set_window_title(self, title: str) -> None:
        """设置窗口标题"""
        if self._window is not None:
            glfw.set_window_title(self._window, title)

            # 更新状态
            self._state_manager.set("window_title", title)

    def cleanup(self) -> None:
        """清理窗口资源"""
        if self._window is not None:
            # 清理渲染器
            try:
                vector_field_renderer.cleanup()
                print("[窗口管理] 渲染器清理完成")
            except Exception as e:
                print(f"[窗口管理] 渲染器清理失败: {e}")

            # 销毁窗口
            try:
                glfw.destroy_window(self._window)
                self._window = None
                print("[窗口管理] 窗口销毁完成")
            except Exception as e:
                print(f"[窗口管理] 窗口销毁失败: {e}")
                self._window = None  # 确保即使失败也重置引用

            # 终止GLFW
            try:
                glfw.terminate()
                print("[窗口管理] GLFW终止完成")
            except Exception as e:
                print(f"[窗口管理] GLFW终止失败: {e}")

            print("[窗口管理] 窗口资源清理完成")

            # 发布窗口关闭事件
            try:
                self._event_bus.publish(Event(
                    EventType.WINDOW_CLOSED,
                    {},
                    "WindowManager"
                ))
            except Exception as e:
                print(f"[窗口管理] 发布窗口关闭事件失败: {e}")

# 全局窗口管理器实例
window_manager = WindowManager()

# 便捷函数
def create_window(title: str = "LiziEngine", width: int = 800, height: int = 600) -> Optional[int]:
    """便捷函数：创建窗口"""
    return window_manager.create_window(title, width, height)

def should_close() -> bool:
    """便捷函数：检查窗口是否应该关闭"""
    return window_manager.should_close()

def swap_buffers() -> None:
    """便捷函数：交换前后缓冲区"""
    window_manager.swap_buffers()

def poll_events() -> None:
    """便捷函数：处理事件"""
    window_manager.poll_events()
