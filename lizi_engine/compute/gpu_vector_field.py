"""
GPU向量场计算模块 - 提供基于GPU的向量场计算功能
使用OpenCL实现高性能计算
"""
import numpy as np
import pyopencl as cl
from typing import Tuple, Union, List, Optional, Any
from ..core.state import state_manager
from ..core.events import Event, EventType, event_bus

class GPUVectorFieldCalculator:
    """GPU向量场计算器，使用OpenCL实现"""
    def __init__(self):
        self._event_bus = event_bus
        self._state_manager = state_manager
        self._ctx = None
        self._queue = None
        self._programs = {}
        self._kernels = {}
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
            const float neighbor_weight
        ) {
            const int idx = get_global_id(0);
            const int idy = get_global_id(1);

            if (idx >= width || idy >= height) {
                return;
            }

            float2 sum = (float2)(0.0f, 0.0f);

            // 检查自身
            if (x == idx && y == idy) {
                sum += grid[idy * width + idx] * self_weight;
            }

            // 检查四个邻居
            int2 neighbors[4] = {(int2)(0, -1), (int2)(0, 1), (int2)(-1, 0), (int2)(1, 0)};

            for (int i = 0; i < 4; i++) {
                int nx = idx + neighbors[i].x;
                int ny = idy + neighbors[i].y;

                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    sum += grid[ny * width + nx] * neighbor_weight;
                }
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
            const float neighbor_weight
        ) {
            const int idx = get_global_id(0);
            const int idy = get_global_id(1);

            if (idx >= width || idy >= height) {
                return;
            }

            float2 sum = (float2)(0.0f, 0.0f);

            // 检查四个邻居
            int2 neighbors[4] = {(int2)(0, -1), (int2)(0, 1), (int2)(-1, 0), (int2)(1, 0)};

            for (int i = 0; i < 4; i++) {
                int nx = idx + neighbors[i].x;
                int ny = idy + neighbors[i].y;

                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    sum += grid[ny * width + nx] * neighbor_weight;
                }
            }

            // 总是添加自身贡献
            sum += grid[idy * width + idx] * self_weight;

            grid[idy * width + idx] = sum;
        }
        """

        # 批量创建微小向量程序
        create_tiny_vectors_batch_kernel = """
        // 原子加法函数（用于float类型）
        inline void atomicAddFloat(__global float* addr, float val) {
            union {
                unsigned int u32;
                float f32;
            } next, expected, current;
            current.f32 = *addr;
            do {
                expected.f32 = current.f32;
                next.f32 = expected.f32 + val;
                current.u32 = atomic_cmpxchg((volatile __global unsigned int*)addr, expected.u32, next.u32);
            } while (current.u32 != expected.u32);
        }

        __kernel void create_tiny_vectors_batch_kernel(
            __global float* temp_grid,
            __global const float* positions,
            const int width,
            const int height,
            const int num_positions
        ) {
            const int pos_idx = get_global_id(0);

            if (pos_idx >= num_positions) {
                return;
            }

            // 获取当前位置（从展平数组中读取）
            int pos_base_idx = pos_idx * 3;
            float x = clamp(positions[pos_base_idx], 0.0f, (float)(width - 1));
            float y = clamp(positions[pos_base_idx + 1], 0.0f, (float)(height - 1));
            float mag = positions[pos_base_idx + 2];

            // 处理四个邻居：上下左右
            int2 neighbors[4] = {(int2)(0, -1), (int2)(0, 1), (int2)(-1, 0), (int2)(1, 0)};

            for (int i = 0; i < 4; i++) {
                float nx = x + (float)neighbors[i].x;
                float ny = y + (float)neighbors[i].y;

                // 确保邻居坐标在范围内
                nx = clamp(nx, 0.0f, (float)(width - 1));
                ny = clamp(ny, 0.0f, (float)(height - 1));

                // 计算向量分量（方向 * 幅度）
                float vx = (float)neighbors[i].x * mag;
                float vy = (float)neighbors[i].y * mag;

                // 双线性插值：计算四个最近的整数坐标
                int x0 = (int)floor(nx);
                int x1 = min(x0 + 1, width - 1);
                int y0 = (int)floor(ny);
                int y1 = min(y0 + 1, height - 1);

                // 计算插值权重
                float wx = nx - (float)x0;
                float wy = ny - (float)y0;

                // 双线性插值的逆：将向量按权重分布到四个角
                float w00 = (1.0f - wx) * (1.0f - wy);
                float w01 = wx * (1.0f - wy);
                float w10 = (1.0f - wx) * wy;
                float w11 = wx * wy;

                // 计算网格索引（直接写入共享临时缓冲区）
                int base_idx00 = (y0 * width + x0) * 2;
                int base_idx01 = (y0 * width + x1) * 2;
                int base_idx10 = (y1 * width + x0) * 2;
                int base_idx11 = (y1 * width + x1) * 2;

                // 使用原子操作累加到临时缓冲区
                atomicAddFloat(&temp_grid[base_idx00], w00 * vx);
                atomicAddFloat(&temp_grid[base_idx00 + 1], w00 * vy);
                atomicAddFloat(&temp_grid[base_idx01], w01 * vx);
                atomicAddFloat(&temp_grid[base_idx01 + 1], w01 * vy);
                atomicAddFloat(&temp_grid[base_idx10], w10 * vx);
                atomicAddFloat(&temp_grid[base_idx10 + 1], w10 * vy);
                atomicAddFloat(&temp_grid[base_idx11], w11 * vx);
                atomicAddFloat(&temp_grid[base_idx11 + 1], w11 * vy);
            }
        }
        """

        # 批量拟合向量程序
        fit_vectors_at_positions_batch_kernel = """
        __kernel void fit_vectors_at_positions_batch_kernel(
            __global const float2* grid,
            __global const float* positions,
            __global float2* results,
            const int width,
            const int height,
            const int num_positions
        ) {
            const int pos_idx = get_global_id(0);

            if (pos_idx >= num_positions) {
                return;
            }

            // 获取当前位置
            int pos_base_idx = pos_idx * 2;
            float x = clamp(positions[pos_base_idx], 0.0f, (float)(width - 1));
            float y = clamp(positions[pos_base_idx + 1], 0.0f, (float)(height - 1));

            // 计算四个最近的整数坐标
            int x0 = (int)floor(x);
            int x1 = min(x0 + 1, width - 1);
            int y0 = (int)floor(y);
            int y1 = min(y0 + 1, height - 1);

            // 获取四个角的向量值
            float2 v00 = grid[y0 * width + x0];
            float2 v01 = grid[y0 * width + x1];
            float2 v10 = grid[y1 * width + x0];
            float2 v11 = grid[y1 * width + x1];

            // 计算插值权重
            float wx = x - (float)x0;
            float wy = y - (float)y0;

            // 双线性插值
            float vx = (1.0f - wx) * (1.0f - wy) * v00.x + wx * (1.0f - wy) * v01.x +
                      (1.0f - wx) * wy * v10.x + wx * wy * v11.x;
            float vy = (1.0f - wx) * (1.0f - wy) * v00.y + wx * (1.0f - wy) * v01.y +
                      (1.0f - wx) * wy * v10.y + wx * wy * v11.y;

            // 存储结果
            results[pos_idx] = (float2)(vx, vy);
        }
        """

        # 编译程序
        try:
            self._programs['sum_adjacent_vectors'] = cl.Program(self._ctx, sum_adjacent_kernel).build()
            self._programs['update_grid_with_adjacent_sum'] = cl.Program(self._ctx, update_grid_kernel).build()
            self._programs['create_tiny_vectors_batch'] = cl.Program(self._ctx, create_tiny_vectors_batch_kernel).build()
            self._programs['fit_vectors_at_positions_batch'] = cl.Program(self._ctx, fit_vectors_at_positions_batch_kernel).build()

            # 预先创建并存储内核实例，避免重复检索
            self._kernels['sum_adjacent_vectors'] = cl.Kernel(self._programs['sum_adjacent_vectors'], 'sum_adjacent_vectors')
            self._kernels['update_grid_with_adjacent_sum'] = cl.Kernel(self._programs['update_grid_with_adjacent_sum'], 'update_grid_with_adjacent_sum')
            self._kernels['create_tiny_vectors_batch'] = cl.Kernel(self._programs['create_tiny_vectors_batch'], 'create_tiny_vectors_batch_kernel')
            self._kernels['fit_vectors_at_positions_batch'] = cl.Kernel(self._programs['fit_vectors_at_positions_batch'], 'fit_vectors_at_positions_batch_kernel')
        except Exception as e:
            print(f"[GPU计算] OpenCL程序编译失败: {e}")
            raise

    def sum_adjacent_vectors(self, grid: np.ndarray, x: int, y: int,
                           self_weight: float = 1.0, neighbor_weight: float = 0.1) -> Tuple[float, float]:
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

        # 执行内核
        self._kernels['sum_adjacent_vectors'].set_args(
            grid_buf, result_buf,
            np.int32(w), np.int32(h), np.int32(x), np.int32(y),
            np.float32(self_weight), np.float32(neighbor_weight)
        )
        cl.enqueue_nd_range_kernel(self._queue, self._kernels['sum_adjacent_vectors'], (w, h), None)

        # 读取结果
        cl.enqueue_copy(self._queue, result, result_buf)

        # 返回指定位置的向量
        return (result[y, x, 0], result[y, x, 1])

    def update_grid_with_adjacent_sum(self, grid: np.ndarray) -> np.ndarray:
        """使用GPU更新网格"""
        if not self._initialized:
            raise RuntimeError("GPU计算器未初始化")

        if grid is None or not isinstance(grid, np.ndarray):
            return grid

        h, w = grid.shape[:2]

        # 获取配置参数
        neighbor_weight = self._state_manager.get("vector_neighbor_weight", 0.1)
        self_weight = self._state_manager.get("vector_self_weight", 1.0)

        # 创建OpenCL缓冲区
        grid_buf = cl.Buffer(self._ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=grid)

        # 执行内核
        self._kernels['update_grid_with_adjacent_sum'].set_args(
            grid_buf, np.int32(w), np.int32(h),
            np.float32(self_weight), np.float32(neighbor_weight)
        )
        cl.enqueue_nd_range_kernel(self._queue, self._kernels['update_grid_with_adjacent_sum'], (w, h), None)

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

    def create_tiny_vector(self, grid: np.ndarray, x: float, y: float, mag: float = 1.0) -> None:
        """在指定位置创建一个微小的向量场影响,只影响位置本身及上下左右四个邻居"""
        if not self._initialized:
            return

        if not hasattr(grid, "ndim"):
            return

        h, w = grid.shape[0], grid.shape[1]

        # 确保坐标在有效范围内
        x = max(0.0, min(w - 1.0, float(x)))
        y = max(0.0, min(h - 1.0, float(y)))

        # 只影响当前位置及其上下左右邻居，使用浮点坐标
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if abs(dx) + abs(dy) == 1:  # 上下左右邻居
                    self.add_vector_at_position(grid, x + dx, y + dy, dx * mag, dy * mag)

    def create_tiny_vectors_batch(self, grid: np.ndarray, positions: List[Tuple[float, float, float]]) -> None:
        """批量创建微小向量影响，使用GPU并行处理

        Args:
            grid: 向量场网格
            positions: 位置列表，每个元素为 (x, y, mag) 元组
        """
        if not self._initialized:
            raise RuntimeError("GPU计算器未初始化")

        if not hasattr(grid, "ndim") or not positions:
            return

        h, w = grid.shape[0], grid.shape[1]
        num_positions = len(positions)

        # 将位置列表转换为numpy数组，确保是连续的float32数组
        positions_array = np.array(positions, dtype=np.float32).flatten()

        # 创建临时网格用于累积结果（大小为网格大小，使用原子操作累加）
        temp_grid_flat = np.zeros((h * w * 2,), dtype=np.float32)

        # 创建OpenCL缓冲区
        temp_grid_buf = cl.Buffer(self._ctx, cl.mem_flags.WRITE_ONLY, temp_grid_flat.nbytes)
        positions_buf = cl.Buffer(self._ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=positions_array)

        # 执行内核
        self._kernels['create_tiny_vectors_batch'].set_args(
            temp_grid_buf, positions_buf,
            np.int32(w), np.int32(h), np.int32(num_positions)
        )
        cl.enqueue_nd_range_kernel(self._queue, self._kernels['create_tiny_vectors_batch'], (num_positions,), None)

        # 读取临时网格结果
        cl.enqueue_copy(self._queue, temp_grid_flat, temp_grid_buf)

        # 将累加结果添加到主网格
        temp_grid = temp_grid_flat.reshape((h, w, 2))
        grid += temp_grid

    def add_vector_at_position(self, grid: np.ndarray, x: float, y: float, vx: float, vy: float) -> None:
        """在浮点坐标处添加向量，使用双线性插值的逆方法，将向量分布到四个最近的整数坐标"""
        if not self._initialized:
            return

        if not hasattr(grid, "ndim") or grid.ndim < 3 or grid.shape[2] < 2:
            return

        h, w = grid.shape[0], grid.shape[1]

        # 确保坐标在有效范围内
        x = max(0.0, min(w - 1.0, float(x)))
        y = max(0.0, min(h - 1.0, float(y)))

        # 计算四个最近的整数坐标
        x0 = int(np.floor(x))
        x1 = min(x0 + 1, w - 1)
        y0 = int(np.floor(y))
        y1 = min(y0 + 1, h - 1)

        # 计算插值权重
        wx = x - x0
        wy = y - y0

        # 双线性插值的逆：将向量按权重分布到四个角
        w00 = (1 - wx) * (1 - wy)
        w01 = wx * (1 - wy)
        w10 = (1 - wx) * wy
        w11 = wx * wy

        try:
            grid[y0, x0, 0] += w00 * vx
            grid[y0, x0, 1] += w00 * vy
            grid[y0, x1, 0] += w01 * vx
            grid[y0, x1, 1] += w01 * vy
            grid[y1, x0, 0] += w10 * vx
            grid[y1, x0, 1] += w10 * vy
            grid[y1, x1, 0] += w11 * vx
            grid[y1, x1, 1] += w11 * vy
        except Exception:
            pass

    def fit_vector_at_position(self, grid: np.ndarray, x: float, y: float) -> Tuple[float, float]:
        """在浮点坐标处拟合向量值，使用双线性插值"""
        if not self._initialized:
            raise RuntimeError("GPU计算器未初始化")

        if not hasattr(grid, "ndim") or grid.ndim < 3 or grid.shape[2] < 2:
            return (0.0, 0.0)

        h, w = grid.shape[0], grid.shape[1]

        # 确保坐标在有效范围内
        x = max(0.0, min(w - 1.0, float(x)))
        y = max(0.0, min(h - 1.0, float(y)))

        # 计算四个最近的整数坐标
        x0 = int(np.floor(x))
        x1 = min(x0 + 1, w - 1)
        y0 = int(np.floor(y))
        y1 = min(y0 + 1, h - 1)

        # 获取四个角的向量值
        v00 = (grid[y0, x0, 0], grid[y0, x0, 1])
        v01 = (grid[y0, x1, 0], grid[y0, x1, 1])
        v10 = (grid[y1, x0, 0], grid[y1, x0, 1])
        v11 = (grid[y1, x1, 0], grid[y1, x1, 1])

        # 计算插值权重
        wx = x - x0
        wy = y - y0

        # 双线性插值
        vx = (1 - wx) * (1 - wy) * v00[0] + wx * (1 - wy) * v01[0] + (1 - wx) * wy * v10[0] + wx * wy * v11[0]
        vy = (1 - wx) * (1 - wy) * v00[1] + wx * (1 - wy) * v01[1] + (1 - wx) * wy * v10[1] + wx * wy * v11[1]

        return (vx, vy)

    def fit_vectors_at_positions_batch(self, grid: np.ndarray, positions: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """批量拟合多个位置的向量值，使用GPU并行处理

        Args:
            grid: 向量场网格
            positions: 位置列表，每个元素为 (x, y) 元组

        Returns:
            向量列表，每个元素为 (vx, vy) 元组
        """
        if not self._initialized:
            raise RuntimeError("GPU计算器未初始化")

        if not hasattr(grid, "ndim") or grid.ndim < 3 or grid.shape[2] < 2 or not positions:
            return [(0.0, 0.0)] * len(positions)

        h, w = grid.shape[0], grid.shape[1]
        num_positions = len(positions)

        # 将位置列表转换为numpy数组，确保是连续的float32数组
        positions_array = np.array(positions, dtype=np.float32).flatten()

        # 创建结果数组
        results = np.zeros((num_positions, 2), dtype=np.float32)

        # 创建OpenCL缓冲区
        grid_buf = cl.Buffer(self._ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=grid)
        positions_buf = cl.Buffer(self._ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=positions_array)
        results_buf = cl.Buffer(self._ctx, cl.mem_flags.WRITE_ONLY, results.nbytes)

        # 执行内核
        self._kernels['fit_vectors_at_positions_batch'].set_args(
            grid_buf, positions_buf, results_buf,
            np.int32(w), np.int32(h), np.int32(num_positions)
        )
        cl.enqueue_nd_range_kernel(self._queue, self._kernels['fit_vectors_at_positions_batch'], (num_positions,), None)

        # 读取结果
        cl.enqueue_copy(self._queue, results, results_buf)

        # 返回结果列表
        return [(results[i, 0], results[i, 1]) for i in range(num_positions)]

    def cleanup(self) -> None:
        """清理资源"""
        if self._ctx:
            del self._ctx
        if self._queue:
            del self._queue
        self._programs.clear()
        self._kernels.clear()
        self._initialized = False
