
"""
错误处理和日志记录模块 - 提供统一的错误处理和日志记录功能
"""
import logging
import traceback
import time
from typing import Optional, Any, Dict
from enum import Enum

class LogLevel(Enum):
    """日志级别枚举"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class ErrorHandler:
    """统一的错误处理器"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ErrorHandler, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized') or not self._initialized:
            self._setup_logger()
            self._error_counts = {}
            self._initialized = True

    def _setup_logger(self):
        """设置日志记录器"""
        # 创建日志记录器
        self._logger = logging.getLogger('LiziEngine')
        self._logger.setLevel(logging.DEBUG)

        # 创建文件处理器
        file_handler = logging.FileHandler('liziengine.log', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)

        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 添加处理器
        self._logger.addHandler(file_handler)
        self._logger.addHandler(console_handler)

    def log(self, level: LogLevel, message: str, exception: Optional[Exception] = None):
        """记录日志"""
        log_level = getattr(logging, level.value)
        self._logger.log(log_level, message)

        # 如果有异常，记录堆栈跟踪
        if exception is not None:
            self._logger.error(f"异常详情: {traceback.format_exc()}")

            # 统计错误类型
            error_type = type(exception).__name__
            if error_type not in self._error_counts:
                self._error_counts[error_type] = 0
            self._error_counts[error_type] += 1

    def get_error_statistics(self) -> Dict[str, int]:
        """获取错误统计信息"""
        return self._error_counts.copy()

    def reset_error_statistics(self):
        """重置错误统计"""
        self._error_counts = {}

    def handle_exception(self, exception: Exception, context: str = ""):
        """处理异常"""
        error_msg = f"在 {context} 中发生异常: {str(exception)}"
        self.log(LogLevel.ERROR, error_msg, exception)

        # 返回处理后的异常信息
        return {
            "type": type(exception).__name__,
            "message": str(exception),
            "context": context,
            "timestamp": time.time()
        }

    def safe_execute(self, func, *args, **kwargs):
        """安全执行函数，捕获并记录异常"""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            context = f"函数 {func.__name__}"
            self.handle_exception(e, context)
            return None

# 全局错误处理器实例
error_handler = ErrorHandler()
