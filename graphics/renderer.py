
"""
渲染器模块 - 提供向量场的渲染功能
"""
import numpy as np
import ctypes
from typing import Optional, Dict, Any, List, Tuple
from OpenGL.GL import *
from OpenGL.GL import shaders
from core.config import config_manager
from core.events import EventBus, Event, EventType
from core.state import state_manager

class ShaderProgram:
    """着色器程序管理器"""
    def __init__(self, vertex_src: str, fragment_src: str):
        self._program = None
        self._uniform_locations = {}
        self._attribute_locations = {}
        self._vertex_src = vertex_src
        self._fragment_src = fragment_src

    def compile(self) -> None:
        """编译着色器程序"""
        try:
            # 编译顶点着色器
            vertex_shader = shaders.compileShader(self._vertex_src, GL_VERTEX_SHADER)

            # 编译片段着色器
            fragment_shader = shaders.compileShader(self._fragment_src, GL_FRAGMENT_SHADER)

            # 链接着色器程序
            self._program = shaders.compileProgram(vertex_shader, fragment_shader)

            print("[渲染器] 着色器程序编译成功")
        except Exception as e:
            print(f"[渲染器] 着色器编译错误: {e}")
            raise

    def use(self) -> None:
        """使用着色器程序"""
        if self._program is not None:
            glUseProgram(self._program)

    def get_uniform_location(self, name: str) -> int:
        """获取uniform变量位置"""
        if name not in self._uniform_locations:
            self._uniform_locations[name] = glGetUniformLocation(self._program, name)
        return self._uniform_locations[name]

    def get_attribute_location(self, name: str) -> int:
        """获取attribute变量位置"""
        if name not in self._attribute_locations:
            self._attribute_locations[name] = glGetAttribLocation(self._program, name)
        return self._attribute_locations[name]

    def set_uniform_float(self, name: str, value: float) -> None:
        """设置float类型uniform变量"""
        loc = self.get_uniform_location(name)
        if loc >= 0:
            glUniform1f(loc, value)

    def set_uniform_vec2(self, name: str, value: Tuple[float, float]) -> None:
        """设置vec2类型uniform变量"""
        loc = self.get_uniform_location(name)
        if loc >= 0:
            glUniform2f(loc, value[0], value[1])

    def set_uniform_vec3(self, name: str, value: Tuple[float, float, float]) -> None:
        """设置vec3类型uniform变量"""
        loc = self.get_uniform_location(name)
        if loc >= 0:
            glUniform3f(loc, value[0], value[1], value[2])

    def cleanup(self) -> None:
        """清理着色器程序"""
        if self._program is not None:
            glDeleteProgram(self._program)
            self._program = None
            self._uniform_locations.clear()
            self._attribute_locations.clear()

class VectorFieldRenderer:
    """向量场渲染器"""
    def __init__(self):
        self._event_bus = EventBus()
        self._state_manager = state_manager

        # 着色器源代码
        # 顶点着色器源代码
        # 功能: 将顶点位置从世界空间转换到屏幕空间, 并传递颜色数据给片段着色器
        self._vertex_shader_src = """
#version 120  // 使用OpenGL 2.1版本的GLSL

// 输入属性
attribute vec2 a_pos;  // 顶点位置坐标(x, y)
attribute vec3 a_col;  // 顶点颜色(RGB)

// 输出变量(传递给片段着色器)
varying vec3 v_col;   // 插值后的颜色

// 统一变量(对所有顶点相同的值)
uniform vec2 u_center; // 视图中心点坐标
uniform vec2 u_half;   // 视图半宽和半高

// 主函数
void main() {
    // 将顶点位置从世界坐标转换为归一化设备坐标(NDC)
    // 1. 减去视图中心, 使视图中心成为原点
    // 2. 除以视图半尺寸, 进行缩放
    vec2 ndc = (a_pos - u_center) / u_half;

    // 翻转Y轴, 因为OpenGL的NDC坐标系Y轴向上, 而我们的世界坐标系Y轴向下
    ndc.y = -ndc.y;

    // 设置最终的顶点位置, Z坐标为0, W坐标为1
    gl_Position = vec4(ndc, 0.0, 1.0);

    // 将颜色数据传递给片段着色器(会自动进行透视校正插值)
    v_col = a_col;
}
"""

        # 片段着色器源代码
        # 功能: 为每个像素设置最终颜色
        self._fragment_shader_src = """
#version 120  // 使用OpenGL 2.1版本的GLSL

// 输入变量(从顶点着色器接收, 已插值)
varying vec3 v_col;   // 插值后的颜色

// 主函数
void main() {
    // 将接收到的RGB颜色转换为RGBA, 设置alpha为1.0(不透明)
    gl_FragColor = vec4(v_col, 1.0);
}
"""

        # 着色器程序
        self._shader_program = ShaderProgram(self._vertex_shader_src, self._fragment_shader_src)

        # OpenGL 对象
        self._vao = None
        self._vbo = None
        self._grid_vao = None
        self._grid_vbo = None

        # 渲染状态
        self._initialized = False

    def initialize(self) -> None:
        """初始化渲染器"""
        if self._initialized:
            return

        try:
            # 编译着色器
            self._shader_program.compile()

            # 创建顶点数组对象和顶点缓冲对象
            self._vao = glGenVertexArrays(1)
            self._vbo = glGenBuffers(1)

            # 创建网格顶点数组对象和顶点缓冲对象
            self._grid_vao = glGenVertexArrays(1)
            self._grid_vbo = glGenBuffers(1)

            self._initialized = True
            print("[渲染器] 初始化成功")
        except Exception as e:
            print(f"[渲染器] 初始化失败: {e}")
            raise

    def render_vector_field(self, grid: np.ndarray, cell_size: float = 1.0,
                           cam_x: float = 0.0, cam_y: float = 0.0, cam_zoom: float = 1.0,
                           viewport_width: int = 800, viewport_height: int = 600) -> None:
        """
        渲染向量场到当前OpenGL上下文

        该方法将向量场数据渲染为一系列有向线段, 每个线段表示一个向量.
        使用向量化操作提高性能, 避免在Python中进行循环操作.

        参数:
            grid: 二维向量场数组, 形状为(height, width, 2), 最后一维表示向量的x和y分量
            cell_size: 网格单元的实际大小, 用于将网格坐标转换为世界坐标
            cam_x: 摄像机X位置, 用于视图变换
            cam_y: 摄像机Y位置, 用于视图变换
            cam_zoom: 摄像机缩放级别, 用于视图变换
            viewport_width: 视口宽度(像素)
            viewport_height: 视口高度(像素)

        技术细节:
            - 使用numpy向量化操作处理顶点数据, 避免Python循环
            - 只渲染非零向量, 提高渲染效率
            - 每个向量渲染为一条线段, 从起点指向终点
            - 顶点格式: (x, y, r, g, b), 每个向量需要两个顶点(起点和终点)

        性能优化:
            - 使用掩码操作过滤零向量, 减少渲染负载
            - 批量构建顶点数据, 减少OpenGL API调用次数
            - 使用单个VBO存储所有向量数据, 提高GPU访问效率
        """
        # 确保渲染器已初始化
        if not self._initialized:
            self.initialize()

        # 如果没有向量场数据, 直接返回
        if grid is None:
            return

        # 从配置管理器获取向量颜色
        vector_color = config_manager.get("vector_field.vector_color", [0.2, 0.6, 1.0])

        # 获取向量场尺寸
        h, w = grid.shape[:2]

        # 使用向量化操作准备顶点数据, 避免Python循环
        # 创建网格坐标矩阵
        y_coords, x_coords = np.mgrid[0:h, 0:w]

        # 提取向量的x和y分量
        vx = grid[:, :, 0]
        vy = grid[:, :, 1]

        # 创建非零向量掩码, 只渲染长度大于阈值的向量
        # 阈值设置为0.001, 避免渲染几乎为零的向量
        mask = (np.abs(vx) > 0.001) | (np.abs(vy) > 0.001)

        # 如果没有非零向量, 直接返回, 避免不必要的渲染操作
        if not np.any(mask):
            return

        # 提取非零向量的坐标和分量
        non_zero_x = x_coords[mask]
        non_zero_y = y_coords[mask]
        non_zero_vx = vx[mask]
        non_zero_vy = vy[mask]

        # 计算向量的起点和终点坐标
        # 起点是网格位置, 终点是起点加上向量
        start_x = non_zero_x * cell_size
        start_y = non_zero_y * cell_size
        end_x = start_x + non_zero_vx
        end_y = start_y + non_zero_vy

        # 创建顶点数组 - 每个向量需要两个点(起点和终点)
        # 每个点有5个分量 (x, y, r, g, b), 所以每个向量需要10个分量
        vertices = np.zeros(len(non_zero_x) * 2 * 5, dtype=np.float32)

        # 填充起点数据, 使用切片操作批量赋值
        vertices[0::10] = start_x  # 起点x坐标, 每10个元素取一个
        vertices[1::10] = start_y  # 起点y坐标, 每10个元素取一个
        vertices[2::10] = vector_color[0]  # R分量, 每10个元素取一个
        vertices[3::10] = vector_color[1]  # G分量, 每10个元素取一个
        vertices[4::10] = vector_color[2]  # B分量, 每10个元素取一个

        # 填充终点数据
        vertices[5::10] = end_x  # 终点x坐标
        vertices[6::10] = end_y  # 终点y坐标
        vertices[7::10] = vector_color[0]  # R
        vertices[8::10] = vector_color[1]  # G
        vertices[9::10] = vector_color[2]  # B

        # 绑定VAO和VBO
        glBindVertexArray(self._vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)

        # 上传顶点数据
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_DYNAMIC_DRAW)

        # 设置顶点属性
        pos_loc = self._shader_program.get_attribute_location("a_pos")
        col_loc = self._shader_program.get_attribute_location("a_col")

        if pos_loc >= 0:
            glEnableVertexAttribArray(pos_loc)
            glVertexAttribPointer(pos_loc, 2, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(0))

        if col_loc >= 0:
            glEnableVertexAttribArray(col_loc)
            glVertexAttribPointer(col_loc, 3, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(2 * 4))

        # 使用着色器程序
        self._shader_program.use()

        # 设置uniform变量
        half_w = (viewport_width / 2.0) / cam_zoom
        half_h = (viewport_height / 2.0) / cam_zoom
        self._shader_program.set_uniform_vec2("u_center", (cam_x, cam_y))
        self._shader_program.set_uniform_vec2("u_half", (half_w, half_h))

        # 绘制向量线
        glDrawArrays(GL_LINES, 0, len(vertices) // 5)

        # 解绑
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glUseProgram(0)

    def render_grid(self, grid: np.ndarray, cell_size: float = 1.0,
                   cam_x: float = 0.0, cam_y: float = 0.0, cam_zoom: float = 1.0,
                   viewport_width: int = 800, viewport_height: int = 600) -> None:
        """渲染网格线"""
        if not self._initialized:
            self.initialize()

        if grid is None:
            return

        # 获取配置
        grid_color = config_manager.get("rendering.grid_color", [0.3, 0.3, 0.3])
        show_grid = config_manager.get("rendering.show_grid", True)

        if not show_grid:
            return

        h, w = grid.shape[:2]
        vertices = []

        # 水平线
        for y in range(h + 1):
            py = y * cell_size
            vertices.extend([0, py, grid_color[0], grid_color[1], grid_color[2]])
            vertices.extend([w * cell_size, py, grid_color[0], grid_color[1], grid_color[2]])

        # 垂直线
        for x in range(w + 1):
            px = x * cell_size
            vertices.extend([px, 0, grid_color[0], grid_color[1], grid_color[2]])
            vertices.extend([px, h * cell_size, grid_color[0], grid_color[1], grid_color[2]])

        # 绑定VAO和VBO
        glBindVertexArray(self._grid_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._grid_vbo)

        # 上传顶点数据
        glBufferData(GL_ARRAY_BUFFER, len(vertices) * 4, np.array(vertices, dtype=np.float32), GL_STATIC_DRAW)

        # 设置顶点属性
        pos_loc = self._shader_program.get_attribute_location("a_pos")
        col_loc = self._shader_program.get_attribute_location("a_col")

        if pos_loc >= 0:
            glEnableVertexAttribArray(pos_loc)
            glVertexAttribPointer(pos_loc, 2, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(0))

        if col_loc >= 0:
            glEnableVertexAttribArray(col_loc)
            glVertexAttribPointer(col_loc, 3, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(2 * 4))

        # 使用着色器程序
        self._shader_program.use()

        # 设置uniform变量
        half_w = (viewport_width / 2.0) / cam_zoom
        half_h = (viewport_height / 2.0) / cam_zoom
        self._shader_program.set_uniform_vec2("u_center", (cam_x, cam_y))
        self._shader_program.set_uniform_vec2("u_half", (half_w, half_h))

        # 绘制网格线
        glDrawArrays(GL_LINES, 0, len(vertices) // 5)

        # 解绑
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glUseProgram(0)

    def render_vector_centers(self, centers: List[Tuple[int, int]], cell_size: float = 1.0,
                          cam_x: float = 0.0, cam_y: float = 0.0, cam_zoom: float = 1.0,
                          viewport_width: int = 800, viewport_height: int = 600) -> None:
        """渲染向量中心标记"""
        if not self._initialized:
            self.initialize()
            
        if not centers:
            return
            
        # 获取配置
        center_color = config_manager.get("vector_field.center_color", [1.0, 0.2, 0.2])
        center_size = config_manager.get("vector_field.center_size", 5)
        
        # 准备顶点数据 - 每个中心点绘制一个十字
        vertices = []
        
        for x, y in centers:
            # 转换为世界坐标
            wx = x * cell_size
            wy = y * cell_size
            
            # 水平线
            vertices.extend([wx - center_size, wy, center_color[0], center_color[1], center_color[2]])
            vertices.extend([wx + center_size, wy, center_color[0], center_color[1], center_color[2]])
            
            # 垂直线
            vertices.extend([wx, wy - center_size, center_color[0], center_color[1], center_color[2]])
            vertices.extend([wx, wy + center_size, center_color[0], center_color[1], center_color[2]])
        
        # 绑定VAO和VBO
        glBindVertexArray(self._vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        
        # 上传顶点数据
        vertices_array = np.array(vertices, dtype=np.float32)
        glBufferData(GL_ARRAY_BUFFER, vertices_array.nbytes, vertices_array, GL_DYNAMIC_DRAW)
        
        # 设置顶点属性
        pos_loc = self._shader_program.get_attribute_location("a_pos")
        col_loc = self._shader_program.get_attribute_location("a_col")
        
        if pos_loc >= 0:
            glEnableVertexAttribArray(pos_loc)
            glVertexAttribPointer(pos_loc, 2, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(0))
        
        if col_loc >= 0:
            glEnableVertexAttribArray(col_loc)
            glVertexAttribPointer(col_loc, 3, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(2 * 4))
        
        # 使用着色器程序
        self._shader_program.use()
        
        # 设置uniform变量
        half_w = (viewport_width / 2.0) / cam_zoom
        half_h = (viewport_height / 2.0) / cam_zoom
        self._shader_program.set_uniform_vec2("u_center", (cam_x, cam_y))
        self._shader_program.set_uniform_vec2("u_half", (half_w, half_h))
        
        # 绘制中心标记
        glDrawArrays(GL_LINES, 0, len(vertices) // 5)
        
        # 解绑
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glUseProgram(0)
    
    def render_background(self) -> None:
        """渲染背景"""
        # 获取配置
        bg_color = config_manager.get("rendering.background_color", [0.1, 0.1, 0.1])

        # 设置清屏颜色
        glClearColor(bg_color[0], bg_color[1], bg_color[2], 1.0)

        # 清除颜色和深度缓冲
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    def cleanup(self) -> None:
        """清理渲染器资源"""
        if not self._initialized:
            return

        try:
            # 删除着色器程序
            if hasattr(self, '_shader_program') and self._shader_program is not None:
                self._shader_program.cleanup()
                self._shader_program = None

            # 删除VAO和VBO
            gl_resources = [
                ('_vao', glDeleteVertexArrays),
                ('_vbo', glDeleteBuffers),
                ('_grid_vao', glDeleteVertexArrays),
                ('_grid_vbo', glDeleteBuffers)
            ]

            for attr_name, delete_func in gl_resources:
                if hasattr(self, attr_name):
                    resource_id = getattr(self, attr_name)
                    if resource_id is not None:
                        delete_func(1, [resource_id])
                        setattr(self, attr_name, None)

            self._initialized = False
            print("[渲染器] 资源清理完成")
        except Exception as e:
            print(f"[渲染器] 清理资源时出错: {e}")
            # 确保即使出错也标记为未初始化，避免重复尝试清理
            self._initialized = False
            # 重新抛出异常，以便上层可以处理
            raise

# 全局向量场渲染器实例
vector_field_renderer = VectorFieldRenderer()

# 便捷函数
def render_vector_field(grid: np.ndarray, cell_size: float = 1.0,
                       cam_x: float = 0.0, cam_y: float = 0.0, cam_zoom: float = 1.0,
                       viewport_width: int = 800, viewport_height: int = 600) -> None:
    """便捷函数：渲染向量场"""
    vector_field_renderer.render_vector_field(grid, cell_size, cam_x, cam_y, cam_zoom, viewport_width, viewport_height)

def render_grid(grid: np.ndarray, cell_size: float = 1.0,
               cam_x: float = 0.0, cam_y: float = 0.0, cam_zoom: float = 1.0,
               viewport_width: int = 800, viewport_height: int = 600) -> None:
    """便捷函数：渲染网格"""
    vector_field_renderer.render_grid(grid, cell_size, cam_x, cam_y, cam_zoom, viewport_width, viewport_height)

def render_background() -> None:
    """便捷函数：渲染背景"""
    vector_field_renderer.render_background()

def render_vector_centers(centers: List[Tuple[int, int]], cell_size: float = 1.0,
                         cam_x: float = 0.0, cam_y: float = 0.0, cam_zoom: float = 1.0,
                         viewport_width: int = 800, viewport_height: int = 600) -> None:
    """便捷函数：渲染向量中心标记"""
    vector_field_renderer.render_vector_centers(centers, cell_size, cam_x, cam_y, cam_zoom, viewport_width, viewport_height)
