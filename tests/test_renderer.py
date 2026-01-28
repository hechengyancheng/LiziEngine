import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from lizi_engine.graphics.renderer import VectorFieldRenderer, ShaderProgram, vector_field_renderer
from lizi_engine.core.events import EventType, event_bus
from lizi_engine.core.state import state_manager

# Mock OpenGL constants and functions
GL_LINES = 1
GL_POINTS = 0
GL_COLOR_BUFFER_BIT = 16384
GL_DEPTH_BUFFER_BIT = 256


class TestShaderProgram:
    """测试着色器程序"""

    def setup_method(self):
        """测试前准备"""
        self.vertex_src = """
        #version 120
        attribute vec2 a_pos;
        void main() {
            gl_Position = vec4(a_pos, 0.0, 1.0);
        }
        """
        self.fragment_src = """
        #version 120
        void main() {
            gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0);
        }
        """

    @patch('OpenGL.GL.shaders.compileShader')
    @patch('OpenGL.GL.shaders.compileProgram')
    def test_compile_success(self, mock_compile_program, mock_compile_shader):
        """测试着色器编译成功"""
        mock_compile_program.return_value = 1
        mock_compile_shader.return_value = 2

        shader = ShaderProgram(self.vertex_src, self.fragment_src)
        shader.compile()

        assert shader._program == 1
        mock_compile_shader.assert_called()
        mock_compile_program.assert_called_once_with(2, 2)

    @patch('OpenGL.GL.shaders.compileShader')
    def test_compile_failure(self, mock_compile_shader):
        """测试着色器编译失败"""
        mock_compile_shader.side_effect = Exception("Compile error")

        shader = ShaderProgram(self.vertex_src, self.fragment_src)

        with pytest.raises(Exception):
            shader.compile()

    @patch('lizi_engine.graphics.renderer.glUseProgram')
    def test_use_program(self, mock_use_program):
        """测试使用着色器程序"""
        shader = ShaderProgram(self.vertex_src, self.fragment_src)
        shader._program = 1

        shader.use()

        mock_use_program.assert_called_once_with(1)

    @patch('lizi_engine.graphics.renderer.glGetUniformLocation')
    def test_get_uniform_location(self, mock_get_uniform):
        """测试获取uniform位置"""
        mock_get_uniform.return_value = 5

        shader = ShaderProgram(self.vertex_src, self.fragment_src)
        shader._program = 1

        loc = shader.get_uniform_location("test_uniform")

        assert loc == 5
        mock_get_uniform.assert_called_once_with(1, "test_uniform")

        # 测试缓存
        loc2 = shader.get_uniform_location("test_uniform")
        assert loc2 == 5
        # 应该只调用一次
        mock_get_uniform.assert_called_once()

    @patch('lizi_engine.graphics.renderer.glGetAttribLocation')
    def test_get_attribute_location(self, mock_get_attrib):
        """测试获取attribute位置"""
        mock_get_attrib.return_value = 3

        shader = ShaderProgram(self.vertex_src, self.fragment_src)
        shader._program = 1

        loc = shader.get_attribute_location("test_attr")

        assert loc == 3
        mock_get_attrib.assert_called_once_with(1, "test_attr")

    @patch('lizi_engine.graphics.renderer.glUniform1f')
    @patch('lizi_engine.graphics.renderer.glGetUniformLocation')
    def test_set_uniform_float(self, mock_get_uniform, mock_uniform1f):
        """测试设置float uniform"""
        mock_get_uniform.return_value = 5

        shader = ShaderProgram(self.vertex_src, self.fragment_src)
        shader._program = 1

        shader.set_uniform_float("test_float", 3.14)

        mock_uniform1f.assert_called_once_with(5, 3.14)

    @patch('lizi_engine.graphics.renderer.glUniform2f')
    @patch('lizi_engine.graphics.renderer.glGetUniformLocation')
    def test_set_uniform_vec2(self, mock_get_uniform, mock_uniform2f):
        """测试设置vec2 uniform"""
        mock_get_uniform.return_value = 5

        shader = ShaderProgram(self.vertex_src, self.fragment_src)
        shader._program = 1

        shader.set_uniform_vec2("test_vec2", (1.0, 2.0))

        mock_uniform2f.assert_called_once_with(5, 1.0, 2.0)

    @patch('lizi_engine.graphics.renderer.glUniform3f')
    @patch('lizi_engine.graphics.renderer.glGetUniformLocation')
    def test_set_uniform_vec3(self, mock_get_uniform, mock_uniform3f):
        """测试设置vec3 uniform"""
        mock_get_uniform.return_value = 5

        shader = ShaderProgram(self.vertex_src, self.fragment_src)
        shader._program = 1

        shader.set_uniform_vec3("test_vec3", (1.0, 2.0, 3.0))

        mock_uniform3f.assert_called_once_with(5, 1.0, 2.0, 3.0)

    @patch('lizi_engine.graphics.renderer.glGetString')
    @patch('lizi_engine.graphics.renderer.glDeleteProgram')
    def test_cleanup_with_context(self, mock_delete_program, mock_get_string):
        """测试在OpenGL上下文存在时清理"""
        mock_get_string.return_value = b"OpenGL 3.3"

        shader = ShaderProgram(self.vertex_src, self.fragment_src)
        shader._program = 1

        shader.cleanup()

        mock_delete_program.assert_called_once_with(1)
        assert shader._program is None

    @patch('lizi_engine.graphics.renderer.glGetString')
    def test_cleanup_without_context(self, mock_get_string):
        """测试在OpenGL上下文不存在时清理"""
        mock_get_string.side_effect = Exception("No context")

        shader = ShaderProgram(self.vertex_src, self.fragment_src)
        shader._program = 1

        shader.cleanup()

        assert shader._program is None


class TestVectorFieldRenderer:
    """测试向量场渲染器"""

    def setup_method(self):
        """测试前准备"""
        self.renderer = VectorFieldRenderer()

    @patch('lizi_engine.graphics.renderer.glGenVertexArrays')
    @patch('lizi_engine.graphics.renderer.glGenBuffers')
    @patch('lizi_engine.graphics.renderer.ShaderProgram.compile')
    def test_initialize_success(self, mock_compile, mock_gen_buffers, mock_gen_vertex_arrays):
        """测试渲染器初始化成功"""
        mock_gen_vertex_arrays.return_value = 1
        mock_gen_buffers.return_value = 2

        self.renderer.initialize()

        assert self.renderer._initialized
        assert self.renderer._vao == 1
        assert self.renderer._vbo == 2
        mock_compile.assert_called_once()

    @patch('lizi_engine.graphics.renderer.ShaderProgram.compile')
    def test_initialize_failure(self, mock_compile):
        """测试渲染器初始化失败"""
        mock_compile.side_effect = Exception("Compile failed")

        with pytest.raises(Exception):
            self.renderer.initialize()

        assert not self.renderer._initialized

    @patch('lizi_engine.graphics.renderer.glLineWidth')
    @patch('lizi_engine.graphics.renderer.glUseProgram')
    @patch('lizi_engine.graphics.renderer.glBindVertexArray')
    @patch('lizi_engine.graphics.renderer.glBindBuffer')
    @patch('lizi_engine.graphics.renderer.glBufferData')
    @patch('lizi_engine.graphics.renderer.glEnableVertexAttribArray')
    @patch('lizi_engine.graphics.renderer.glVertexAttribPointer')
    @patch('lizi_engine.graphics.renderer.glDrawArrays')
    @patch('lizi_engine.graphics.renderer.ShaderProgram.get_attribute_location')
    @patch('lizi_engine.graphics.renderer.ShaderProgram.set_uniform_vec2')
    def test_render_vector_field_empty_grid(self, mock_set_uniform, mock_get_attr,
                                           mock_draw_arrays, mock_vertex_attrib, mock_enable_attr,
                                           mock_buffer_data, mock_bind_buffer, mock_bind_vao,
                                           mock_use_program, mock_line_width):
        """测试渲染空向量场"""
        grid = np.zeros((10, 10, 2), dtype=np.float32)

        self.renderer._initialized = True
        self.renderer.render_vector_field(grid)

        # 应该不会绘制任何东西，因为所有向量都是零
        mock_draw_arrays.assert_not_called()

    @patch('lizi_engine.graphics.renderer.glLineWidth')
    @patch('lizi_engine.graphics.renderer.glUseProgram')
    @patch('lizi_engine.graphics.renderer.glBindVertexArray')
    @patch('lizi_engine.graphics.renderer.glBindBuffer')
    @patch('lizi_engine.graphics.renderer.glBufferData')
    @patch('lizi_engine.graphics.renderer.glEnableVertexAttribArray')
    @patch('lizi_engine.graphics.renderer.glVertexAttribPointer')
    @patch('lizi_engine.graphics.renderer.glDrawArrays')
    @patch('lizi_engine.graphics.renderer.ShaderProgram.get_attribute_location')
    @patch('lizi_engine.graphics.renderer.ShaderProgram.set_uniform_vec2')
    def test_render_vector_field_with_vectors(self, mock_set_uniform, mock_get_attr,
                                             mock_draw_arrays, mock_vertex_attrib, mock_enable_attr,
                                             mock_buffer_data, mock_bind_buffer, mock_bind_vao,
                                             mock_use_program, mock_line_width):
        """测试渲染包含向量的向量场"""
        grid = np.zeros((3, 3, 2), dtype=np.float32)
        grid[1, 1] = (1.0, 1.0)  # 一个非零向量

        mock_get_attr.return_value = 0

        self.renderer._initialized = True
        self.renderer.render_vector_field(grid)

        # 应该绘制向量线
        mock_draw_arrays.assert_called_once_with(GL_LINES, 0, 2)  # 2个点（起点和终点）

    @patch('lizi_engine.graphics.renderer.glClearColor')
    @patch('lizi_engine.graphics.renderer.glClear')
    def test_render_background(self, mock_clear, mock_clear_color):
        """测试渲染背景"""
        self.renderer.render_background()

        mock_clear_color.assert_called_once()
        mock_clear.assert_called_once_with(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    @patch('lizi_engine.graphics.renderer.glPointSize')
    @patch('lizi_engine.graphics.renderer.glUseProgram')
    @patch('lizi_engine.graphics.renderer.glBindVertexArray')
    @patch('lizi_engine.graphics.renderer.glBindBuffer')
    @patch('lizi_engine.graphics.renderer.glBufferData')
    @patch('lizi_engine.graphics.renderer.glEnableVertexAttribArray')
    @patch('lizi_engine.graphics.renderer.glVertexAttribPointer')
    @patch('lizi_engine.graphics.renderer.glDrawArrays')
    @patch('lizi_engine.graphics.renderer.ShaderProgram.get_attribute_location')
    @patch('lizi_engine.graphics.renderer.ShaderProgram.set_uniform_vec2')
    def test_render_markers(self, mock_set_uniform, mock_get_attr,
                           mock_draw_arrays, mock_vertex_attrib, mock_enable_attr,
                           mock_buffer_data, mock_bind_buffer, mock_bind_vao,
                           mock_use_program, mock_point_size):
        """测试渲染标记"""
        # 设置标记数据
        markers = [{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}]
        self.renderer._state_manager.set("markers", markers)

        mock_get_attr.return_value = 0

        self.renderer._initialized = True
        self.renderer.render_markers()

        # 应该绘制点
        mock_draw_arrays.assert_called_once_with(GL_POINTS, 0, 2)
        mock_point_size.assert_called_once()

    @patch('lizi_engine.graphics.renderer.glDeleteVertexArrays')
    @patch('lizi_engine.graphics.renderer.glDeleteBuffers')
    @patch('lizi_engine.graphics.renderer.glGetString')
    def test_cleanup(self, mock_get_string, mock_delete_buffers, mock_delete_vertex_arrays):
        """测试清理资源"""
        mock_get_string.return_value = b"OpenGL 3.3"

        self.renderer._initialized = True
        self.renderer._vao = 1
        self.renderer._vbo = 2
        self.renderer._grid_vao = 3
        self.renderer._grid_vbo = 4

        self.renderer.cleanup()

        assert not self.renderer._initialized
        assert self.renderer._vao is None
        assert self.renderer._vbo is None
        assert self.renderer._grid_vao is None
        assert self.renderer._grid_vbo is None

    def test_handle_app_initialized(self):
        """测试处理应用初始化事件"""
        event = Mock()
        event.type = EventType.APP_INITIALIZED

        # 不应该抛出异常
        self.renderer.handle(event)


class TestGlobalRenderer:
    """测试全局渲染器"""

    def test_global_renderer_exists(self):
        """测试全局渲染器存在"""
        assert vector_field_renderer is not None
        assert isinstance(vector_field_renderer, VectorFieldRenderer)


if __name__ == "__main__":
    pytest.main([__file__])
