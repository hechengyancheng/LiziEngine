
"""
OpenCL计算模块 - 提供基于OpenCL的向量场计算功能
"""
import numpy as np
from typing import Optional, Dict, Any, Tuple
import pyopencl as cl
from core.config import config_manager
from core.events import EventBus, Event, EventType
from core.state import state_manager

class OpenCLComputeManager:
    """OpenCL计算管理器"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OpenCLComputeManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._event_bus = EventBus()
            self._state_manager = state_manager
            self._context = None
            self._queue = None
            self._program = None
            self._kernel = None  # 添加内核实例缓存
            self._initialized = True

    def get_compute_kernel_src(self) -> str:
        """从配置管理器获取OpenCL计算内核源代码，动态设置权重参数"""
        self_weight = config_manager.get("vector_field.vector_self_weight", 1.0)
        neighbor_weight = config_manager.get("vector_field.vector_neighbor_weight", 0.1)
        enable_average = config_manager.get("vector_field.enable_vector_average", False)
        enable_normalization = config_manager.get("vector_field.enable_vector_normalization", False)

        kernel_src = """
__kernel void vector_field_compute(
    __global const float2* input,
    __global float2* output,
    const int width,
    const int height,
    const int include_self,
    const int enable_average,
    const int enable_normalization,
    const float self_weight,
    const float neighbor_weight
) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) {
        return;
    }

    int idx = y * width + x;
    float2 sum = (float2)(0.0f, 0.0f);
    int count = 0;

    // 处理自身向量
    if (include_self != 0) {
        sum += input[idx] * self_weight;
        count++;
    }

    // 处理邻居向量
    // 上
    if (y > 0) {
        sum += input[(y-1) * width + x] * neighbor_weight;
        count++;
    }
    // 下
    if (y < height - 1) {
        sum += input[(y+1) * width + x] * neighbor_weight;
        count++;
    }
    // 左
    if (x > 0) {
        sum += input[y * width + (x-1)] * neighbor_weight;
        count++;
    }
    // 右
    if (x < width - 1) {
        sum += input[y * width + (x+1)] * neighbor_weight;
        count++;
    }

    // 根据配置决定是否进行归一化
    // 使用权重归一化而不是简单计数归一化，以保持向量场的稳定性
    if (enable_normalization != 0 && count > 0) {
        // 计算有效权重总和
        float weight_sum = 0.0f;
        if (include_self != 0) {
            weight_sum += self_weight;
        }
        
        // 每个有效邻居贡献neighbor_weight
        // 注意：count已经包含了自身，但自身权重已经单独计算过，所以这里需要减去1
        weight_sum += neighbor_weight * (float)(count - (include_self != 0 ? 1 : 0));
        
        // 使用权重总和进行归一化
        if (weight_sum > 0.0f) {
            sum /= weight_sum;
        }
    }

    output[idx] = sum;
}
"""
        return kernel_src

    def init_compute(self, width: int = 640, height: int = 480, init_grid: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        初始化OpenCL计算上下文
        init_grid: 可选 numpy (h,w,2) 初始向量场。
        返回 context dict，供后续 step / get_grid_numpy 使用。
        """
        try:
            # 获取OpenCL平台和设备
            platforms = cl.get_platforms()
            if not platforms:
                raise RuntimeError("没有找到可用的OpenCL平台")

            # 尝试使用GPU，如果没有则使用CPU
            device_type = cl.device_type.GPU
            devices = None
            for platform in platforms:
                try:
                    devices = platform.get_devices(device_type=device_type)
                    if devices:
                        break
                except:
                    continue

            # 如果没有GPU设备，尝试使用CPU
            if not devices:
                device_type = cl.device_type.CPU
                for platform in platforms:
                    try:
                        devices = platform.get_devices(device_type=device_type)
                        if devices:
                            break
                    except:
                        continue

            if not devices:
                raise RuntimeError("没有找到可用的OpenCL设备")

            device = devices[0]  # 使用第一个可用设备
            print(f"[OpenCL计算] 使用设备: {device.name} ({'GPU' if device_type == cl.device_type.GPU else 'CPU'})")

            # 创建上下文和命令队列
            self._context = cl.Context([device])
            self._queue = cl.CommandQueue(self._context)

            # 编译内核
            kernel_src = self.get_compute_kernel_src()
            self._program = cl.Program(self._context, kernel_src).build()
            # 创建并缓存内核实例
            self._kernel = cl.Kernel(self._program, "vector_field_compute")

            # 创建缓冲区
            input_buffer = cl.Buffer(self._context, cl.mem_flags.READ_WRITE, size=width*height*2*4)  # float2
            output_buffer = cl.Buffer(self._context, cl.mem_flags.READ_WRITE, size=width*height*2*4)  # float2

            ctx = {
                "width": width,
                "height": height,
                "input_buffer": input_buffer,
                "output_buffer": output_buffer,
                "ping": True
            }

            # 如果有初始数据，上传到GPU
            if init_grid is not None:
                h, w = init_grid.shape[:2]
                flat_grid = np.asarray(init_grid, dtype=np.float32).reshape(-1)
                cl.enqueue_copy(self._queue, input_buffer, flat_grid)
                # 确保数据上传完成
                self._queue.finish()
            else:
                # 如果没有初始数据，初始化为零
                zeros = np.zeros(width*height*2, dtype=np.float32)
                cl.enqueue_copy(self._queue, input_buffer, zeros)
                # 确保数据上传完成
                self._queue.finish()

            print("[OpenCL计算] OpenCL上下文创建成功")

            # 发布OpenCL计算初始化事件
            self._event_bus.publish(Event(
                EventType.GPU_COMPUTE_STARTED,
                {"width": width, "height": height},
                "OpenCLComputeManager"
            ))

            return ctx
        except Exception as e:
            # 发布OpenCL计算错误事件
            self._event_bus.publish(Event(
                EventType.GPU_COMPUTE_ERROR,
                {"error": str(e)},
                "OpenCLComputeManager"
            ))
            raise

    def step(self, ctx: Dict[str, Any], include_self: Optional[bool] = None) -> Dict[str, Any]:
        """
        在 GPU 上执行一次邻域求和（OpenCL内核），并在 ctx 中 ping-pong 交换输入/输出缓冲区。
        """
        if ctx is None:
            print("[OpenCL计算] 警告: OpenCL上下文为空，跳过计算")
            return ctx

        try:
            width = ctx["width"]
            height = ctx["height"]

            # 确定输入和输出缓冲区
            input_buffer = ctx["input_buffer"] if ctx["ping"] else ctx["output_buffer"]
            output_buffer = ctx["output_buffer"] if ctx["ping"] else ctx["input_buffer"]

            # 从配置获取参数
            self_weight = config_manager.get("vector_field.vector_self_weight", 1.0)
            neighbor_weight = config_manager.get("vector_field.vector_neighbor_weight", 0.1)
            enable_average = config_manager.get("vector_field.enable_vector_average", False)
            enable_normalization = config_manager.get("vector_field.enable_vector_normalization", False)
            
            # 如果没有显式传入include_self参数，则从配置中读取
            if include_self is None:
                include_self = config_manager.get("vector_field.include_self", True)
                
            #print(f"[OpenCL计算] 计算参数 - include_self: {include_self}, self_weight: {self_weight}, neighbor_weight: {neighbor_weight}, enable_average: {enable_average}, enable_normalization: {enable_normalization}")

            # 设置内核参数并执行
            self._kernel.set_args(
                input_buffer,
                output_buffer,
                np.int32(width),
                np.int32(height),
                np.int32(1 if include_self else 0),
                np.int32(1 if enable_average else 0),
                np.int32(1 if enable_normalization else 0),
                np.float32(self_weight),
                np.float32(neighbor_weight)
            )

            # 执行内核
            # 确保宽度和高度大于0
            if width <= 0 or height <= 0:
                print(f"[OpenCL计算] 错误: 无效的网格尺寸 {width}x{height}")
                return ctx

            global_size = (width, height)
            cl.enqueue_nd_range_kernel(self._queue, self._kernel, global_size, None)

            # 确保内核执行完成
            self._queue.finish()

            # 等待完成
            self._queue.finish()

            # 交换ping-pong标志
            ctx["ping"] = not ctx["ping"]

            # 发布OpenCL计算完成事件
            self._event_bus.publish(Event(
                EventType.GPU_COMPUTE_COMPLETED,
                {"include_self": include_self},
                "OpenCLComputeManager"
            ))

            return ctx
        except Exception as e:
            # 发布OpenCL计算错误事件
            self._event_bus.publish(Event(
                EventType.GPU_COMPUTE_ERROR,
                {"error": str(e)},
                "OpenCLComputeManager"
            ))
            raise

    def get_grid_numpy(self, ctx: Dict[str, Any]) -> np.ndarray:
        """
        从当前"输入"缓冲区读取回 numpy 数组，返回形状 (h,w,2) 的 float32 数组。
        注意：这会把缓冲区数据从 GPU 下载到 CPU。
        """
        if ctx is None:
            return np.array([])

        try:
            width = ctx["width"]
            height = ctx["height"]

            # 确定输入缓冲区
            input_buffer = ctx["input_buffer"] if ctx["ping"] else ctx["output_buffer"]

            # 创建numpy数组并从GPU复制数据
            result = np.empty((height, width, 2), dtype=np.float32)
            flat_result = result.reshape(-1)
            cl.enqueue_copy(self._queue, flat_result, input_buffer)

            # 确保数据传输完成
            self._queue.finish()

            return result
        except Exception as e:
            print(f"[OpenCL计算] 获取网格数据失败: {e}")
            return np.array([])

    def upload_texel(self, ctx: Dict[str, Any], x: int, y: int, vec2: Tuple[float, float]) -> None:
        """
        将单个格子的向量上传到当前输入缓冲区。
        x,y: texel 坐标（整型）
        vec2: 可迭代 (vx, vy)
        """
        if ctx is None:
            return

        try:
            width = ctx["width"]
            height = ctx["height"]

            if x < 0 or x >= width or y < 0 or y >= height:
                return

            # 确定输入缓冲区
            input_buffer = ctx["input_buffer"] if ctx["ping"] else ctx["output_buffer"]

            # 创建临时缓冲区并更新单个元素
            idx = y * width + x
            vec = np.array([float(vec2[0]), float(vec2[1])], dtype=np.float32)
            cl.enqueue_copy(self._queue, input_buffer, vec, dst_offset=idx*2*4)

            # 确保写入完成
            self._queue.finish()
        except Exception as e:
            print(f"[OpenCL计算] 上传单个向量失败: {e}")

    def upload_grid(self, ctx: Dict[str, Any], grid: np.ndarray) -> None:
        """
        将整个 (h,w,2) numpy grid 上传到当前输入缓冲区（覆盖）。
        """
        if ctx is None or grid is None:
            return

        try:
            h, w = grid.shape[:2]
            if h != ctx["height"] or w != ctx["width"]:
                print(f"[OpenCL计算] 网格尺寸不匹配，期望: {ctx['height']}x{ctx['width']}, 实际: {h}x{w}")
                return

            # 确定输入缓冲区
            input_buffer = ctx["input_buffer"] if ctx["ping"] else ctx["output_buffer"]

            # 上传数据
            flat_grid = np.asarray(grid, dtype=np.float32).reshape(-1)
            cl.enqueue_copy(self._queue, input_buffer, flat_grid)

            # 确保写入完成
            self._queue.finish()
        except Exception as e:
            print(f"[OpenCL计算] 上传网格数据失败: {e}")

    def cleanup(self) -> None:
        """清理OpenCL资源"""
        if self._context is not None:
            try:
                # OpenCL资源会随着上下文的销毁自动清理
                self._context = None
                self._queue = None
                self._program = None
                self._kernel = None  # 清理内核实例

                # 发布OpenCL计算清理事件
                self._event_bus.publish(Event(
                    self._get_event_type("GPU_COMPLETED", EventType.APP_SHUTDOWN),
                    {},
                    "OpenCLComputeManager"
                ))
            except Exception as e:
                print(f"[OpenCL计算] 清理资源时出错: {e}")
            
    def _get_event_type(self, type_name: str, default_type):
        """安全地获取事件类型"""
        try:
            from core.events import EventType
            return getattr(EventType, type_name, default_type)
        except Exception:
            return default_type

# 全局OpenCL计算管理器实例
opencl_compute_manager = OpenCLComputeManager()

# 便捷函数
def init_compute(width: int = 640, height: int = 480, init_grid: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """便捷函数：初始化OpenCL计算"""
    return opencl_compute_manager.init_compute(width, height, init_grid)

def opencl_step(ctx: Dict[str, Any], include_self: Optional[bool] = None) -> Dict[str, Any]:
    """便捷函数：执行OpenCL计算步骤"""
    return opencl_compute_manager.step(ctx, include_self)

def opencl_get_grid_numpy(ctx: Dict[str, Any]) -> np.ndarray:
    """便捷函数：从OpenCL获取网格数据"""
    return opencl_compute_manager.get_grid_numpy(ctx)
