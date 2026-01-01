"""
GUI模块 - 提供图形用户界面功能
包含GUI管理器和OpenGL嵌入器
"""

from .gui_manager import gui_manager
from .opengl_embedder import OpenGLEmbedder

__all__ = ['gui_manager', 'OpenGLEmbedder']
