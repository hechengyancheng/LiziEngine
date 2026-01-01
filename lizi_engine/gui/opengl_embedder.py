"""
OpenGL嵌入器模块 - 将OpenGL渲染嵌入到Dear PyGui中
"""
import glfw
import numpy as np
from OpenGL.GL import *
from typing import Optional, Tuple
import dearpygui.dearpygui as dpg

class OpenGLEmbedder:
    """OpenGL嵌入器 - 在Dear PyGui中嵌入OpenGL渲染"""

    def __init__(self, width: int = 800, height: int = 600):
        self._width = width
        self._height = height

        # OpenGL资源
        self._fbo = None  # 帧缓冲对象
        self._texture = None  # 纹理对象
        self._rbo = None  # 渲染缓冲对象

        # Dear PyGui资源
        self._dpg_texture = None
        self._dpg_image = None

        # GLFW窗口（隐藏的）
        self._glfw_window = None

        self._initialized = False

    def initialize(self) -> bool:
        """初始化OpenGL嵌入"""
        try:
            # 创建隐藏的GLFW窗口用于OpenGL上下文
            if not glfw.init():
                print("[OpenGL嵌入器] GLFW初始化失败")
                return False

            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)  # 隐藏窗口
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

            self._glfw_window = glfw.create_window(1, 1, "OpenGL Context", None, None)
            if not self._glfw_window:
                print("[OpenGL嵌入器] 创建GLFW窗口失败")
                glfw.terminate()
                return False

            glfw.make_context_current(self._glfw_window)

            # 初始化OpenGL
            self._init_opengl()

            # 创建帧缓冲和纹理
            self._create_framebuffer()

            # 创建Dear PyGui纹理
            self._create_dpg_texture()

            self._initialized = True
            print("[OpenGL嵌入器] 初始化成功")
            return True

        except Exception as e:
            print(f"[OpenGL嵌入器] 初始化失败: {e}")
            self._cleanup()
            return False

    def _init_opengl(self):
        """初始化OpenGL设置"""
        # 启用深度测试
        glEnable(GL_DEPTH_TEST)

        # 设置视口
        glViewport(0, 0, self._width, self._height)

        # 设置清除颜色
        glClearColor(0.1, 0.1, 0.1, 1.0)

        # 启用抗锯齿
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

    def _create_framebuffer(self):
        """创建帧缓冲对象"""
        # 生成帧缓冲对象
        self._fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo)

        # 创建纹理附件
        self._texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self._texture)

        # 设置纹理参数
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self._width, self._height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        # 将纹理附加到帧缓冲
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self._texture, 0)

        # 创建渲染缓冲对象用于深度和模板
        self._rbo = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self._rbo)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, self._width, self._height)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, self._rbo)

        # 检查帧缓冲完整性
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            print("[OpenGL嵌入器] 帧缓冲不完整")
            raise RuntimeError("Framebuffer is not complete")

        # 解绑帧缓冲
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def _create_dpg_texture(self):
        """创建Dear PyGui纹理"""
        # 创建空的纹理数据
        texture_data = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        texture_data[:, :, :] = [25, 25, 25]  # 深灰色背景

        # 添加纹理到Dear PyGui
        with dpg.texture_registry():
            self._dpg_texture = dpg.add_raw_texture(
                width=self._width,
                height=self._height,
                default_value=texture_data.flatten(),
                format=dpg.mvFormat_FloatRgb
            )

    def begin_render(self):
        """开始渲染到帧缓冲"""
        if not self._initialized:
            return

        # 绑定帧缓冲
        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo)

        # 设置视口
        glViewport(0, 0, self._width, self._height)

        # 清除缓冲
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    def end_render(self):
        """结束渲染并更新Dear PyGui纹理"""
        if not self._initialized:
            return

        # 解绑帧缓冲
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        # 读取纹理数据
        glBindTexture(GL_TEXTURE_2D, self._texture)
        texture_data = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE)

        # 转换为numpy数组
        texture_array = np.frombuffer(texture_data, dtype=np.uint8).reshape((self._height, self._width, 3))

        # 垂直翻转（OpenGL坐标系与Dear PyGui不同）
        texture_array = np.flipud(texture_array)

        # 归一化到0-1范围
        texture_float = texture_array.astype(np.float32) / 255.0

        # 更新Dear PyGui纹理
        dpg.set_value(self._dpg_texture, texture_float.flatten())

        # 解绑纹理
        glBindTexture(GL_TEXTURE_2D, 0)

    def get_dpg_texture(self) -> Optional[int]:
        """获取Dear PyGui纹理ID"""
        return self._dpg_texture

    def get_dpg_image(self) -> Optional[int]:
        """获取或创建Dear PyGui图像"""
        if self._dpg_image is None and self._dpg_texture is not None:
            self._dpg_image = dpg.add_image(self._dpg_texture, width=self._width, height=self._height)
        return self._dpg_image

    def resize(self, width: int, height: int):
        """调整渲染区域大小"""
        if width == self._width and height == self._height:
            return

        self._width = width
        self._height = height

        if not self._initialized:
            return

        # 重新创建帧缓冲
        self._cleanup_framebuffer()
        self._create_framebuffer()

        # 重新创建Dear PyGui纹理
        self._cleanup_dpg_texture()
        self._create_dpg_texture()

    def _cleanup_framebuffer(self):
        """清理帧缓冲资源"""
        if self._fbo is not None:
            glDeleteFramebuffers(1, [self._fbo])
            self._fbo = None

        if self._texture is not None:
            glDeleteTextures(1, [self._texture])
            self._texture = None

        if self._rbo is not None:
            glDeleteRenderbuffers(1, [self._rbo])
            self._rbo = None

    def _cleanup_dpg_texture(self):
        """清理Dear PyGui纹理"""
        if self._dpg_texture is not None:
            # 删除旧的纹理和图像
            if self._dpg_image is not None:
                dpg.delete_item(self._dpg_image)
                self._dpg_image = None
            dpg.delete_item(self._dpg_texture)
            self._dpg_texture = None

    def _cleanup(self):
        """清理所有资源"""
        self._cleanup_framebuffer()
        self._cleanup_dpg_texture()

        if self._glfw_window is not None:
            glfw.destroy_window(self._glfw_window)
            self._glfw_window = None

        glfw.terminate()

    def cleanup(self):
        """清理资源"""
        if not self._initialized:
            return

        self._cleanup()
        self._initialized = False
        print("[OpenGL嵌入器] 资源清理完成")

    def __del__(self):
        """析构函数"""
        self.cleanup()
