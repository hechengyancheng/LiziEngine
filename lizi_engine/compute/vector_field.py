"""
向量场计算模块 - 提供向量场计算的核心功能
支持CPU和GPU加速计算
"""
import numpy as np
from typing import Tuple, Union, List, Optional, Any
from ..core.config import config_manager
from ..core.events import Event, EventType, event_bus, EventHandler
from ..core.state import state_manager
from .cpu_vector_field import CPUVectorFieldCalculator
from .gpu_vector_field import GPUVectorFieldCalculator

class VectorFieldCalculator(EventHandler):
    """向量场计算器，支持CPU和GPU计算"""
    def __init__(self):
        self._event_bus = event_bus
        self._state_manager = state_manager
        self._config_manager = config_manager

        # 初始化计算器
        self._cpu_calculator = CPUVectorFieldCalculator()
        self._gpu_calculator = None

        # 尝试初始化GPU计算器
        try:
            self._gpu_calculator = GPUVectorFieldCalculator()
            print("[向量场计算] GPU计算器初始化成功")
        except Exception as e:
            print(f"[向量场计算] GPU计算器初始化失败: {e}")

        # 当前使用的计算设备
        self._current_device = self._config_manager.get("compute_device", "cpu")

        # 订阅事件
        self._event_bus.subscribe(EventType.APP_INITIALIZED, self)
        # 监听配置变更以便在运行时响应 compute_device 的修改
        self._event_bus.subscribe(EventType.CONFIG_CHANGED, self)

    @property
    def current_device(self) -> str:
        """获取当前计算设备"""
        return self._current_device

    def set_device(self, device: str) -> bool:
        """设置计算设备"""
        if device not in ["cpu", "gpu"]:
            return False

        if device == "gpu" and self._gpu_calculator is None:
            print("[向量场计算] GPU计算器不可用")
            return False

        self._current_device = device
        self._config_manager.set("compute_device", device)

        # 发布设备变更事件
        self._event_bus.publish(Event(
            EventType.APP_INITIALIZED,
            {"device": device},
            "VectorFieldCalculator"
        ))

        return True

    def sum_adjacent_vectors(self, grid: np.ndarray, x: int, y: int) -> Tuple[float, float]:
        """
        读取目标 (x,y) 的上下左右四个相邻格子的向量并相加（越界安全）。
        返回 (sum_x, sum_y) 的 tuple。
        """
        # 从配置管理器读取权重参数
        self_weight = self._config_manager.get("vector_self_weight", 0.2)
        neighbor_weight = self._config_manager.get("vector_neighbor_weight", 0.2)

        if grid is None:
            return (0.0, 0.0)

        if not isinstance(grid, np.ndarray):
            raise TypeError("grid 必须是 numpy.ndarray 类型")

        # 根据当前设备选择计算器
        calculator = self._gpu_calculator if self._current_device == "gpu" and self._gpu_calculator else self._cpu_calculator

        return calculator.sum_adjacent_vectors(
            grid, x, y, self_weight, neighbor_weight
        )

    def update_grid_with_adjacent_sum(self, grid: np.ndarray) -> np.ndarray:
        """
        使用NumPy的向量化操作高效计算相邻向量之和，替换原有的双重循环实现。
        返回修改后的 grid。
        """
        if grid is None or not isinstance(grid, np.ndarray):
            return grid

        # 根据当前设备选择计算器
        calculator = self._gpu_calculator if self._current_device == "gpu" and self._gpu_calculator else self._cpu_calculator

        return calculator.update_grid_with_adjacent_sum(grid)

    def create_vector_grid(self, width: int = 640, height: int = 480, default: Tuple[float, float] = (0, 0)) -> np.ndarray:
        """创建一个 height x width 的二维向量网格"""
        # 根据当前设备选择计算器
        calculator = self._gpu_calculator if self._current_device == "gpu" and self._gpu_calculator else self._cpu_calculator

        return calculator.create_vector_grid(width, height, default)

    def create_radial_pattern(self, grid: np.ndarray, center: Tuple[float, float] = None,
                            radius: float = None, magnitude: float = 1.0) -> np.ndarray:
        """在网格上创建径向向量模式"""
        # 根据当前设备选择计算器
        calculator = self._gpu_calculator if self._current_device == "gpu" and self._gpu_calculator else self._cpu_calculator

        result = calculator.create_radial_pattern(grid, center, radius, magnitude)

        return result

    def create_tangential_pattern(self, grid: np.ndarray, center: Tuple[float, float] = None,
                               radius: float = None, magnitude: float = 1.0) -> np.ndarray:
        """在网格上创建切线向量模式（旋转）"""
        # 根据当前设备选择计算器
        calculator = self._gpu_calculator if self._current_device == "gpu" and self._gpu_calculator else self._cpu_calculator

        result = calculator.create_tangential_pattern(grid, center, radius, magnitude)

        return result

    def handle(self, event: Event) -> None:
        """处理事件"""
        if event.type == EventType.APP_INITIALIZED:
            if "device" in event.data:
                self.set_device(event.data["device"])
        elif event.type == EventType.CONFIG_CHANGED:
            # 响应配置变更事件，及时更新当前计算设备而不进入死循环
            data = event.data or {}
            key = data.get("key")
            if key == "compute_device":
                new_device = data.get("new_value")
                if new_device and new_device != self._current_device:
                    if new_device == "gpu" and self._gpu_calculator is None:
                        print("[向量场计算] GPU计算器不可用 (来自配置变更)")
                    else:
                        self._current_device = new_device
                        print(f"[向量场计算] 计算设备已切换为: {new_device} (来自配置变更)")

    def cleanup(self) -> None:
        """清理资源"""
        if self._gpu_calculator:
            self._gpu_calculator.cleanup()

# 全局向量场计算器实例
vector_calculator = VectorFieldCalculator()

# 便捷函数
def sum_adjacent_vectors(grid: np.ndarray, x: int, y: int) -> Tuple[float, float]:
    """便捷函数：计算相邻向量之和"""
    return vector_calculator.sum_adjacent_vectors(grid, x, y)

def update_grid_with_adjacent_sum(grid: np.ndarray) -> np.ndarray:
    """便捷函数：更新整个网格"""
    return vector_calculator.update_grid_with_adjacent_sum(grid)

def create_vector_grid(width: int = 640, height: int = 480, default: Tuple[float, float] = (0, 0)) -> np.ndarray:
    """便捷函数：创建向量网格"""
    return vector_calculator.create_vector_grid(width, height, default)
