"""
渲染器模块 - 提供向量场的渲染功能
支持OpenGL渲染和着色器程序
"""
import numpy as np
import ctypes
from typing import Optional, Dict, Any, List, Tuple
from OpenGL.GL import *
from OpenGL.GL import shaders
from ..core.config import config_manager
from ..core.events import Event, EventType, event_bus, EventHandler
from ..core.state import state_manager

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
            # 检查 OpenGL 上下文是否存在
            if callable(glGetString) and callable(glDeleteProgram):
                try:
                    # 尝试获取 OpenGL 版本，如果成功则说明上下文存在
                    glGetString(GL_VERSION)
                    glDeleteProgram(self._program)
                except:
                    # 如果 OpenGL 上下文不存在，忽略错误
                    pass
            self._program = None
            self._uniform_locations.clear()
            self._attribute_locations.clear()

class VectorFieldRenderer(EventHandler):
    """向量场渲染器"""
    def __init__(self):
        self._event_bus = event_bus
        self._state_manager = state_manager
        self._config_manager = config_manager

        # 着色器源代码
        self._vertex_shader_src = """
        #version 120
        attribute vec2 a_pos;
        attribute vec3 a_col;
        varying vec3 v_col;
        uniform vec2 u_center;
        uniform vec2 u_half;
        void main() {
            vec2 ndc = (a_pos - u_center) / u_half;
            ndc.y = -ndc.y;
            gl_Position = vec4(ndc, 0.0, 1.0);
            v_col = a_col;
        }
        """
        self._fragment_shader_src = """
        #version 120
        varying vec3 v_col;
        void main() {
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

        # 订阅事件
        self._event_bus.subscribe(EventType.APP_INITIALIZED, self)

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
        """渲染向量场"""
        if not self._initialized:
            self.initialize()

        if grid is None:
            return

        # 获取配置
        vector_color = self._config_manager.get("vector_color", [0.2, 0.6, 1.0])
        vector_scale = self._config_manager.get("vector_scale", 1.0)
        line_width = self._config_manager.get("line_width", 1.0)
        render_lines = self._config_manager.get("render_vector_lines", True)

        # 如果关闭渲染向量线条，直接返回
        if not render_lines:
            return

        # 设置线条宽度
        glLineWidth(line_width)

        # 准备顶点数据
        h, w = grid.shape[:2]

        # 使用向量化操作准备顶点数据，避免循环
        # 创建网格坐标
        y_coords, x_coords = np.mgrid[0:h, 0:w]

        # 获取向量分量
        vx = grid[:, :, 0]
        vy = grid[:, :, 1]

        # 创建非零向量掩码
        mask = (np.abs(vx) > 0.001) | (np.abs(vy) > 0.001)

        # 如果没有非零向量，直接返回
        if not np.any(mask):
            return

        # 提取非零向量的坐标和分量
        non_zero_x = x_coords[mask]
        non_zero_y = y_coords[mask]
        non_zero_vx = vx[mask] * vector_scale
        non_zero_vy = vy[mask] * vector_scale

        # 计算起点和终点坐标
        start_x = non_zero_x * cell_size
        start_y = non_zero_y * cell_size
        end_x = start_x + non_zero_vx
        end_y = start_y + non_zero_vy

        # 创建顶点数组 - 每个向量需要两个点（起点和终点）
        # 每个点有5个分量 (x, y, r, g, b)
        vertices = np.zeros(len(non_zero_x) * 2 * 5, dtype=np.float32)

        # 填充起点数据
        vertices[0::10] = start_x  # 起点x坐标
        vertices[1::10] = start_y  # 起点y坐标
        vertices[2::10] = vector_color[0]  # R
        vertices[3::10] = vector_color[1]  # G
        vertices[4::10] = vector_color[2]  # B

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
        grid_color = self._config_manager.get("grid_color", [0.3, 0.3, 0.3])
        # 统一从配置管理器读取 show_grid，确保配置文件和运行时配置一致
        show_grid = self._config_manager.get("show_grid" , True)

        if not show_grid :
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

    def render_background(self) -> None:
        """渲染背景"""
        # 获取配置
        bg_color = self._config_manager.get("background_color", [0.1, 0.1, 0.1])

        # 设置清屏颜色
        glClearColor(bg_color[0], bg_color[1], bg_color[2], 1.0)

        # 清除颜色和深度缓冲
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    def render_markers(self, cell_size: float = 1.0,
                       cam_x: float = 0.0, cam_y: float = 0.0, cam_zoom: float = 1.0,
                       viewport_width: int = 800, viewport_height: int = 600) -> None:
        """渲染在 state_manager 中注册的标记（点）"""
        if not self._initialized:
            self.initialize()

        markers = self._state_manager.get("markers", [])
        if not markers:
            return

        # 统一颜色和点大小
        marker_color = self._config_manager.get("marker_color", [1.0, 0.2, 0.2])
        point_size = int(self._config_manager.get("marker_size", 8))

        # 构建顶点数组，每个点 (x, y, r, g, b)
        verts = np.zeros(len(markers) * 5, dtype=np.float32)
        for i, m in enumerate(markers):
            try:
                gx = float(m.get("x", 0.0))
                gy = float(m.get("y", 0.0))
            except Exception:
                gx = 0.0
                gy = 0.0

            wx = gx * cell_size
            wy = gy * cell_size

            verts[i * 5 + 0] = wx
            verts[i * 5 + 1] = wy
            verts[i * 5 + 2] = marker_color[0]
            verts[i * 5 + 3] = marker_color[1]
            verts[i * 5 + 4] = marker_color[2]

        # 绑定VAO和VBO（可复用已有 _vao/_vbo）
        glBindVertexArray(self._vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)

        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_DYNAMIC_DRAW)

        pos_loc = self._shader_program.get_attribute_location("a_pos")
        col_loc = self._shader_program.get_attribute_location("a_col")

        if pos_loc >= 0:
            glEnableVertexAttribArray(pos_loc)
            glVertexAttribPointer(pos_loc, 2, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(0))

        if col_loc >= 0:
            glEnableVertexAttribArray(col_loc)
            glVertexAttribPointer(col_loc, 3, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(2 * 4))

        # 使用着色器并设置uniform
        self._shader_program.use()
        half_w = (viewport_width / 2.0) / cam_zoom
        half_h = (viewport_height / 2.0) / cam_zoom
        self._shader_program.set_uniform_vec2("u_center", (cam_x, cam_y))
        self._shader_program.set_uniform_vec2("u_half", (half_w, half_h))

        # 点大小
        glPointSize(point_size)

        # 绘制点
        glDrawArrays(GL_POINTS, 0, len(markers))

        # 清理绑定
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glUseProgram(0)

    def _is_opengl_context_valid(self) -> bool:
        """检查OpenGL上下文是否有效"""
        try:
            glGetString(GL_VERSION)
            return True
        except:
            return False

    def _safe_delete_buffer(self, buffer_id: Optional[int], delete_func) -> None:
        """安全删除OpenGL缓冲对象"""
        if buffer_id is not None and self._is_opengl_context_valid():
            try:
                delete_func(1, [buffer_id])
            except Exception as e:
                print(f"[渲染器] 删除缓冲对象失败: {e}")

    def cleanup(self) -> None:
        """清理渲染器资源"""
        if not self._initialized:
            return

        try:
            # 删除着色器程序
            self._shader_program.cleanup()

            # 删除VAO和VBO
            self._safe_delete_buffer(self._vao, glDeleteVertexArrays)
            self._safe_delete_buffer(self._vbo, glDeleteBuffers)
            self._safe_delete_buffer(self._grid_vao, glDeleteVertexArrays)
            self._safe_delete_buffer(self._grid_vbo, glDeleteBuffers)

            # 清理引用
            self._vao = None
            self._vbo = None
            self._grid_vao = None
            self._grid_vbo = None

            self._initialized = False
            print("[渲染器] 资源清理完成")
        except Exception as e:
            print(f"[渲染器] 清理资源时出错: {e}")

    def handle(self, event: Event) -> None:
        """处理事件"""
        if event.type == EventType.APP_INITIALIZED:
            # 处理应用初始化事件
            pass

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
