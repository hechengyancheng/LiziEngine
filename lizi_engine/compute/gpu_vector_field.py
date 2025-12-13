"""
GPU向量场计算模块 - 提供基于GPU的向量场计算功能
使用OpenCL实现高性能计算
"""
import numpy as np
import pyopencl as cl
from typing import Tuple, Union, List, Optional, Any
from ..core.config import config_manager
from ..core.events import Event, EventType, event_bus

class GPUVectorFieldCalculator:
    """GPU向量场计算器，使用OpenCL实现"""
    def __init__(self):
        self._event_bus = event_bus
        self._config_manager = config_manager
        self._ctx = None
        self._queue = None
        self._programs = {}
        self._initialized = False

        # 初始化OpenCL
        self._init_opencl()

    def _init_opencl(self) -> None:
        """初始化OpenCL环境"""
        try:
            # 获取平台和设备
            platforms = cl.get_platforms()
            if not platforms:
                raise RuntimeError("未找到OpenCL平台")

            # 选择第一个平台
            platform = platforms[0]
            devices = platform.get_devices(cl.device_type.GPU)

            if not devices:
                # 如果没有GPU设备，尝试使用CPU设备
                devices = platform.get_devices(cl.device_type.CPU)
                if not devices:
                    raise RuntimeError("未找到可用的OpenCL设备")

            # 创建上下文和命令队列
            self._ctx = cl.Context(devices)
            self._queue = cl.CommandQueue(self._ctx)

            # 编译OpenCL程序
            self._compile_programs()

            self._initialized = True
            print("[GPU计算] OpenCL初始化成功")

            # 发布GPU计算器初始化事件
            self._event_bus.publish(Event(
                EventType.GPU_COMPUTE_STARTED,
                {"device": "gpu", "status": "initialized"},
                "GPUVectorFieldCalculator"
            ))
        except Exception as e:
            print(f"[GPU计算] OpenCL初始化失败: {e}")
            self._initialized = False

            # 发布GPU计算器初始化失败事件
            self._event_bus.publish(Event(
                EventType.GPU_COMPUTE_ERROR,
                {"device": "gpu", "status": "failed", "error": str(e)},
                "GPUVectorFieldCalculator"
            ))

    def _compile_programs(self) -> None:
        """编译OpenCL程序"""
        # 相邻向量求和程序
        sum_adjacent_kernel = """
        __kernel void sum_adjacent_vectors(
            __global const float2* grid,
            __global float2* result,
            const int width,
            const int height,
            const int x,
            const int y,
            const float self_weight,
            const float neighbor_weight,
            const int include_self,
            const int enable_average
        ) {
            const int idx = get_global_id(0);
            const int idy = get_global_id(1);

            if (idx >= width || idy >= height) {
                return;
            }

            float2 sum = (float2)(0.0f, 0.0f);
            int count = 0;

            // 检查自身
            if (include_self && x == idx && y == idy) {
                sum += grid[idy * width + idx] * self_weight;
                count++;
            }

            // 检查四个邻居
            int2 neighbors[4] = {(int2)(0, -1), (int2)(0, 1), (int2)(-1, 0), (int2)(1, 0)};

            for (int i = 0; i < 4; i++) {
                int nx = idx + neighbors[i].x;
                int ny = idy + neighbors[i].y;

                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    sum += grid[ny * width + nx] * neighbor_weight;
                    count++;
                }
            }

            // 如果启用平均值，则除以有效向量数量
            if (enable_average && count > 0) {
                sum /= (float)count;
            }

            result[idy * width + idx] = sum;
        }
        """

        # 更新网格程序
        update_grid_kernel = """
        __kernel void update_grid_with_adjacent_sum(
            __global float2* grid,
            const int width,
            const int height,
            const float self_weight,
            const float neighbor_weight,
            const int include_self,
            const int enable_average,
            const int enable_normalization
        ) {
            const int idx = get_global_id(0);
            const int idy = get_global_id(1);

            if (idx >= width || idy >= height) {
                return;
            }

            float2 sum = (float2)(0.0f, 0.0f);
            int neighbor_count = 4;

            // 检查四个邻居
            int2 neighbors[4] = {(int2)(0, -1), (int2)(0, 1), (int2)(-1, 0), (int2)(1, 0)};

            for (int i = 0; i < 4; i++) {
                int nx = idx + neighbors[i].x;
                int ny = idy + neighbors[i].y;

                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    sum += grid[ny * width + nx] * neighbor_weight;
                } else {
                    neighbor_count--;
                }
            }

            // 如果包含自身，添加自身贡献
            if (include_self) {
                sum += grid[idy * width + idx] * self_weight;
                neighbor_count++;
            }

            // 如果需要平均值，计算有效邻居数量并归一化
            if (enable_average) {
                if (neighbor_count > 0) {
                    sum /= (float)neighbor_count;
                }
            }
            // 如果需要权重归一化
            else if (enable_normalization) {
                float weight_sum = neighbor_count * neighbor_weight;
                if (include_self) {
                    weight_sum += self_weight;
                }

                // 确保权重总和大于0，避免除以0
                weight_sum = fmax(weight_sum, 0.1f);

                sum /= weight_sum;
            }

            grid[idy * width + idx] = sum;
        }
        """

        # 创建径向模式程序
        radial_pattern_kernel = """
        __kernel void create_radial_pattern(
            __global float2* grid,
            const int width,
            const int height,
            const float center_x,
            const float center_y,
            const float radius,
            const float magnitude
        ) {
            const int idx = get_global_id(0);
            const int idy = get_global_id(1);

            if (idx >= width || idy >= height) {
                return;
            }

            // 计算到中心的距离
            float dx = (float)idx - center_x;
            float dy = (float)idy - center_y;
            float dist = sqrt(dx * dx + dy * dy);

            // 只处理在半径内且不在中心的点
            if (dist >= radius || dist <= 0.0f) {
                return;
            }

            // 计算径向角度
            float angle = atan2(dy, dx);

            // 计算向量大小（从中心向外递减）
            float vec_magnitude = magnitude * (1.0f - (dist / radius));

            // 计算向量分量
            float vx = vec_magnitude * cos(angle);
            float vy = vec_magnitude * sin(angle);

            // 应用到网格
            grid[idy * width + idx] = (float2)(vx, vy);
        }
        """

        # 创建切线模式程序
        tangential_pattern_kernel = """
        __kernel void create_tangential_pattern(
            __global float2* grid,
            const int width,
            const int height,
            const float center_x,
            const float center_y,
            const float radius,
            const float magnitude
        ) {
            const int idx = get_global_id(0);
            const int idy = get_global_id(1);

            if (idx >= width || idy >= height) {
                return;
            }

            // 计算到中心的距离
            float dx = (float)idx - center_x;
            float dy = (float)idy - center_y;
            float dist = sqrt(dx * dx + dy * dy);

            // 只处理在半径内且不在中心的点
            if (dist >= radius || dist <= 0.0f) {
                return;
            }

            // 计算切线角度（径向角度+90度）
            float angle = atan2(dy, dx) + M_PI_2;

            // 计算向量大小（从中心向外递减）
            float vec_magnitude = magnitude * (1.0f - (dist / radius));

            // 计算向量分量
            float vx = vec_magnitude * cos(angle);
            float vy = vec_magnitude * sin(angle);

            // 应用到网格
            grid[idy * width + idx] = (float2)(vx, vy);
        }
        """

        # 编译程序
        try:
            self._programs['sum_adjacent_vectors'] = cl.Program(self._ctx, sum_adjacent_kernel).build()
            self._programs['update_grid_with_adjacent_sum'] = cl.Program(self._ctx, update_grid_kernel).build()
            self._programs['create_radial_pattern'] = cl.Program(self._ctx, radial_pattern_kernel).build()
            self._programs['create_tangential_pattern'] = cl.Program(self._ctx, tangential_pattern_kernel).build()
        except Exception as e:
            print(f"[GPU计算] OpenCL程序编译失败: {e}")
            raise

    def sum_adjacent_vectors(self, grid: np.ndarray, x: int, y: int, include_self: bool = None,
                           self_weight: float = 1.0, neighbor_weight: float = 0.1, 
                           enable_average: bool = False) -> Tuple[float, float]:
        """使用GPU计算相邻向量之和"""
        if not self._initialized:
            raise RuntimeError("GPU计算器未初始化")

        if grid is None:
            return (0.0, 0.0)

        if not isinstance(grid, np.ndarray):
            raise TypeError("grid 必须是 numpy.ndarray 类型")

        h, w = grid.shape[:2]

        # 创建输出缓冲区
        result = np.zeros((h, w, 2), dtype=np.float32)

        # 创建OpenCL缓冲区
        grid_buf = cl.Buffer(self._ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=grid)
        result_buf = cl.Buffer(self._ctx, cl.mem_flags.WRITE_ONLY, result.nbytes)

        # 设置参数
        include_self_int = 1 if include_self else 0
        enable_average_int = 1 if enable_average else 0

        # 执行内核
        self._programs['sum_adjacent_vectors'].sum_adjacent_vectors(
            self._queue, (w, h), None,
            grid_buf, result_buf,
            np.int32(w), np.int32(h), np.int32(x), np.int32(y),
            np.float32(self_weight), np.float32(neighbor_weight),
            np.int32(include_self_int), np.int32(enable_average_int)
        )

        # 读取结果
        cl.enqueue_copy(self._queue, result, result_buf)

        # 返回指定位置的向量
        return (result[y, x, 0], result[y, x, 1])

    def update_grid_with_adjacent_sum(self, grid: np.ndarray, include_self: bool = None) -> np.ndarray:
        """使用GPU更新网格"""
        if not self._initialized:
            raise RuntimeError("GPU计算器未初始化")

        if grid is None or not isinstance(grid, np.ndarray):
            return grid

        h, w = grid.shape[:2]

        # 获取配置参数
        neighbor_weight = self._config_manager.get("vector_neighbor_weight", 0.1)
        self_weight = self._config_manager.get("vector_self_weight", 1.0)
        enable_average = self._config_manager.get("enable_vector_average", False)
        enable_normalization = self._config_manager.get("enable_vector_normalization", True)

        # 如果未指定include_self，则使用配置管理器中的默认值
        if include_self is None:
            include_self = self._config_manager.get("include_self", False)

        # 创建OpenCL缓冲区
        grid_buf = cl.Buffer(self._ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=grid)

        # 设置参数
        include_self_int = 1 if include_self else 0
        enable_average_int = 1 if enable_average else 0
        enable_normalization_int = 1 if enable_normalization else 0

        # 执行内核
        self._programs['update_grid_with_adjacent_sum'].update_grid_with_adjacent_sum(
            self._queue, (w, h), None,
            grid_buf, np.int32(w), np.int32(h),
            np.float32(self_weight), np.float32(neighbor_weight),
            np.int32(include_self_int), np.int32(enable_average_int),
            np.int32(enable_normalization_int)
        )

        # 读取结果
        cl.enqueue_copy(self._queue, grid, grid_buf)

        return grid

    def create_vector_grid(self, width: int = 640, height: int = 480, default: Tuple[float, float] = (0, 0)) -> np.ndarray:
        """创建一个 height x width 的二维向量网格"""
        # GPU版本与CPU版本相同，因为初始化网格不需要GPU计算
        grid = np.zeros((height, width, 2), dtype=np.float32)
        if default != (0, 0):
            grid[:, :, 0] = default[0]
            grid[:, :, 1] = default[1]
        return grid

    def create_radial_pattern(self, grid: np.ndarray, center: Tuple[float, float] = None,
                            radius: float = None, magnitude: float = 1.0) -> np.ndarray:
        """使用GPU在网格上创建径向向量模式"""
        if not self._initialized:
            raise RuntimeError("GPU计算器未初始化")

        if grid is None or not isinstance(grid, np.ndarray):
            raise TypeError("grid 必须是 numpy.ndarray 类型")

        h, w = grid.shape[:2]

        # 如果未指定中心，则使用网格中心
        if center is None:
            center = (w // 2, h // 2)

        # 如果未指定半径，则使用网格尺寸的1/4
        if radius is None:
            radius = min(w, h) // 4

        cx, cy = center

        # 创建OpenCL缓冲区
        grid_buf = cl.Buffer(self._ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=grid)

        # 执行内核
        self._programs['create_radial_pattern'].create_radial_pattern(
            self._queue, (w, h), None,
            grid_buf, np.int32(w), np.int32(h),
            np.float32(cx), np.float32(cy),
            np.float32(radius), np.float32(magnitude)
        )

        # 读取结果
        cl.enqueue_copy(self._queue, grid, grid_buf)

        return grid

    def create_tangential_pattern(self, grid: np.ndarray, center: Tuple[float, float] = None,
                               radius: float = None, magnitude: float = 1.0) -> np.ndarray:
        """使用GPU在网格上创建切线向量模式（旋转）"""
        if not self._initialized:
            raise RuntimeError("GPU计算器未初始化")

        if grid is None or not isinstance(grid, np.ndarray):
            raise TypeError("grid 必须是 numpy.ndarray 类型")

        h, w = grid.shape[:2]

        # 如果未指定中心，则使用网格中心
        if center is None:
            center = (w // 2, h // 2)

        # 如果未指定半径，则使用网格尺寸的1/4
        if radius is None:
            radius = min(w, h) // 4

        cx, cy = center

        # 创建OpenCL缓冲区
        grid_buf = cl.Buffer(self._ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=grid)

        # 执行内核
        self._programs['create_tangential_pattern'].create_tangential_pattern(
            self._queue, (w, h), None,
            grid_buf, np.int32(w), np.int32(h),
            np.float32(cx), np.float32(cy),
            np.float32(radius), np.float32(magnitude)
        )

        # 读取结果
        cl.enqueue_copy(self._queue, grid, grid_buf)

        return grid

    def cleanup(self) -> None:
        """清理资源"""
        if self._ctx:
            del self._ctx
        if self._queue:
            del self._queue
        self._programs.clear()
        self._initialized = False
