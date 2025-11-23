"""
现代化工具栏实现，使用Tkinter提供美观的界面
"""

import numpy as np
import os
import tkinter as tk
from tkinter import ttk, colorchooser, messagebox
from OpenGL.GL import *
import glfw
from config import config
import threading
import time
from app_core import app_core, EventType, EventHandler

class ToolbarEventHandler:
    """工具栏事件处理器包装类"""
    
    def __init__(self, callback):
        self.callback = callback
        print(f"创建事件处理器: {callback.__name__}")
    
    def handle(self, event):
        """处理事件"""
        print(f"处理事件: {event.type}")
        try:
            self.callback(event)
        except Exception as e:
            print(f"调用回调函数时出错: {e}")

class ModernToolbar:
    """使用Tkinter实现的现代化工具栏"""

    def __init__(self, window):
        """初始化工具栏"""
        self.window = window

        # 工具栏状态
        self.brush_size = config.get("default_brush_size", 1)
        self.magnitude = config.get("default_magnitude", 1.0)
        self.reverse_vector = config.get("reverse_vector", False)
        self.show_grid = config.get("show_grid", True)

        # UI状态
        self.show_toolbar = True

        # 初始化GUI线程（已弃用）
        self.gui_thread = None
        self.stop_event = threading.Event()
        
        # 事件订阅将在render方法中进行
        
        # 获取应用核心的网格引用
        self.grid = app_core.grid
        
        # 从应用核心获取视图数据
        self.view_data = {
            'cam_x': app_core.state.get("cam_x"),
            'cam_y': app_core.state.get("cam_y"),
            'cam_zoom': app_core.state.get("cam_zoom"),
            'show_grid': app_core.state.get("show_grid")
        }

    def start_gui_thread(self):
        """启动GUI线程（已弃用，改为在主线程中处理）"""
        pass

    def create_tkinter_window(self):
        """创建Tkinter窗口"""
        try:
            # 创建Tkinter根窗口
            self.root = tk.Tk()
            self.root.title("LiziEngine 工具栏")
            self.root.geometry("400x500")
        except Exception as e:
            print(f"创建Tkinter窗口时出错: {e}")
            self.root = None

        # 创建控件
        # 标题
        title_label = tk.Label(self.root, text="LiziEngine 工具栏", font=("Arial", 12, "bold"))
        title_label.pack(pady=10)

        # 笔刷大小控制
        brush_frame = tk.Frame(self.root)
        brush_frame.pack(pady=5, padx=10, fill="x")
        tk.Label(brush_frame, text="笔刷大小:").pack(side="left")
        self.brush_var = tk.IntVar(value=self.brush_size)
        brush_slider = tk.Scale(brush_frame, from_=config.get("min_brush_size", 0),
                                to=config.get("max_brush_size", 15),
                                orient="horizontal", variable=self.brush_var,
                                command=self.update_brush_size)
        brush_slider.pack(side="right", fill="x", expand=True)

        # 向量模值控制
        magnitude_frame = tk.Frame(self.root)
        magnitude_frame.pack(pady=5, padx=10, fill="x")
        tk.Label(magnitude_frame, text="向量模值:").pack(side="left")
        self.magnitude_var = tk.DoubleVar(value=self.magnitude)
        magnitude_slider = tk.Scale(magnitude_frame, from_=0.1, to=2.0,
                                   resolution=0.1, orient="horizontal",
                                   variable=self.magnitude_var,
                                   command=self.update_magnitude)
        magnitude_slider.pack(side="right", fill="x", expand=True)

        # 反转向量方向
        self.reverse_var = tk.BooleanVar(value=self.reverse_vector)
        reverse_check = tk.Checkbutton(self.root, text="反转向量方向",
                                       variable=self.reverse_var,
                                       command=self.update_reverse)
        reverse_check.pack(pady=5, anchor="w", padx=10)

        # 显示网格
        self.grid_var = tk.BooleanVar(value=self.show_grid)
        grid_check = tk.Checkbutton(self.root, text="显示网格",
                                    variable=self.grid_var,
                                    command=self.update_grid)
        grid_check.pack(pady=5, anchor="w", padx=10)

        # 按钮组
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)

        clear_button = tk.Button(button_frame, text="清空网格", command=self.clear_grid)
        clear_button.pack(side="left", padx=5)

        reset_button = tk.Button(button_frame, text="重置视图", command=self.reset_view)
        reset_button.pack(side="left", padx=5)

        # 颜色选择
        color_frame = tk.Frame(self.root)
        color_frame.pack(pady=10, padx=10, fill="x")
        tk.Label(color_frame, text="界面颜色设置", font=("Arial", 10, "bold")).pack(anchor="w")

        bg_color_frame = tk.Frame(color_frame)
        bg_color_frame.pack(pady=5, fill="x")
        tk.Label(bg_color_frame, text="背景颜色:").pack(side="left")

        self.bg_color_button = tk.Button(bg_color_frame, text="选择颜色",
                                         command=self.choose_bg_color)
        self.bg_color_button.pack(side="right")

        # 菜单按钮
        menu_frame = tk.Frame(self.root)
        menu_frame.pack(pady=10)

        save_button = tk.Button(menu_frame, text="保存", command=self.save)
        save_button.pack(side="left", padx=5)

        load_button = tk.Button(menu_frame, text="加载", command=self.load)
        load_button.pack(side="left", padx=5)

        about_button = tk.Button(menu_frame, text="关于", command=self.about)
        about_button.pack(side="left", padx=5)

    def gui_thread_func(self):
        """GUI线程函数（已弃用，改为在主线程中处理）"""
        pass

    def update_brush_size(self, value):
        """更新笔刷大小"""
        self.brush_size = int(value)
        # 同步到配置文件
        config.set("default_brush_size", self.brush_size)

    def update_magnitude(self, value):
        """更新向量模值"""
        self.magnitude = float(value)
        # 同步到配置文件
        config.set("default_magnitude", self.magnitude)

    def update_reverse(self):
        """更新反转向量选项"""
        self.reverse_vector = self.reverse_var.get()
        # 同步到配置文件
        config.set("reverse_vector", self.reverse_vector)

    def update_grid(self):
        """更新显示网格选项"""
        self.show_grid = self.grid_var.get()
        # 同步到配置文件
        config.set("show_grid", self.show_grid)

    def clear_grid(self):
        """清空网格"""
        # 使用应用核心清空网格
        app_core.clear_grid()
        messagebox.showinfo("成功", "网格已清空")
    
    def _on_clear_grid(self, event):
        """处理清空网格事件"""
        print("处理清空网格事件")
        pass

    def reset_view(self):
        """重置视图"""
        # 使用应用核心重置视图
        app_core.reset_view()
        messagebox.showinfo("成功", "视图已重置到中心位置")
    
    def _on_reset_view(self, event):
        """处理重置视图事件"""
        print("处理重置视图事件")
        pass

    def choose_bg_color(self):
        """选择背景颜色"""
        color = colorchooser.askcolor(
            title="选择背景颜色",
            color=(
                int(config.get("background_color_r", 0.12) * 255),
                int(config.get("background_color_g", 0.12) * 255),
                int(config.get("background_color_b", 0.12) * 255)
            )
        )
        if color[0]:  # 如果用户选择了颜色
            r, g, b = color[0]
            config.set("background_color_r", r / 255.0)
            config.set("background_color_g", g / 255.0)
            config.set("background_color_b", b / 255.0)
            glClearColor(r / 255.0, g / 255.0, b / 255.0, 1.0)

    def save(self):
        """保存功能"""
        try:
            # 使用应用核心保存网格
            success = app_core.save_grid("grid_save.npy")
            if success:
                messagebox.showinfo("成功", "网格数据已保存到 grid_save.npy")
            else:
                messagebox.showwarning("警告", "没有可保存的网格数据")
        except Exception as e:
            messagebox.showerror("错误", f"保存失败: {str(e)}")

    def load(self):
        """加载功能"""
        try:
            # 检查文件是否存在
            if not os.path.exists("grid_save.npy"):
                messagebox.showwarning("警告", "找不到保存文件 grid_save.npy")
                return

            # 使用应用核心加载网格
            success = app_core.load_grid("grid_save.npy")
            if success:
                messagebox.showinfo("成功", "网格数据已从 grid_save.npy 加载")
            else:
                messagebox.showerror("错误", "加载网格失败")
        except Exception as e:
            messagebox.showerror("错误", f"加载失败: {str(e)}")
    
    def _on_load_grid(self, event):
        """处理加载网格事件"""
        print("处理加载网格事件")
        pass

    def about(self):
        """关于对话框"""
        messagebox.showinfo("关于", "LiziEngine - 粒子引擎")

    def process_input(self):
        """处理输入事件（Tkinter在主线程中处理）"""
        pass

    def _create_handler(self, callback):
        """创建事件处理器"""
        handler = ToolbarEventHandler(callback)
        return handler
    
    def render(self, grid):
        """渲染工具栏界面"""
        # 更新应用核心的网格引用
        if grid is not None and app_core.grid is not grid:
            app_core.set_grid(grid)
            
        # 订阅应用核心事件
        if not hasattr(self, 'events_subscribed'):
            print("订阅事件...")
            app_core.event_bus.subscribe(EventType.CLEAR_GRID, ToolbarEventHandler(self._on_clear_grid))
            app_core.event_bus.subscribe(EventType.RESET_VIEW, ToolbarEventHandler(self._on_reset_view))
            app_core.event_bus.subscribe(EventType.LOAD_GRID, ToolbarEventHandler(self._on_load_grid))
            self.events_subscribed = True
            print("事件订阅完成")

        # 从应用核心获取视图数据
        self.view_data = {
            'cam_x': app_core.state.get("cam_x"),
            'cam_y': app_core.state.get("cam_y"),
            'cam_zoom': app_core.state.get("cam_zoom"),
            'show_grid': app_core.state.get("show_grid")
        }

        # 处理Tkinter事件（在主线程中）
        try:
            # 如果Tkinter窗口尚未创建，创建它
            if not hasattr(self, 'root') or self.root is None or not self.root.winfo_exists():
                self.create_tkinter_window()

            # 处理Tkinter事件
            if self.root is not None:
                self.root.update()
        except Exception as e:
            print(f"处理Tkinter事件时出错: {e}")

        # 返回当前工具栏状态，让主程序能够获取这些值
        return self.brush_size, self.magnitude, self.reverse_vector

    def shutdown(self):
        """关闭工具栏"""
        self.stop_event.set()
        if self.gui_thread and self.gui_thread.is_alive():
            self.gui_thread.join(timeout=1.0)
