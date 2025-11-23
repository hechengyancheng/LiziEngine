import numpy as np
from OpenGL.GL import *
from config import config

def create_vector_grid(width=640, height=480, default=(0, 0)):
    """返回一个 height x width 的二维列表，每个元素为一个二位向量（tuple）。"""
    return [[default for _ in range(width)] for _ in range(height)]

# 改为使用 numpy 原生数组作为默认全局网格，dtype 为 float32 以节省内存并兼顾性能
GRID_640x480 = np.zeros((480, 640, 2), dtype=np.float32)

def sum_adjacent_vectors(grid, x, y, include_self=None):
    """
    读取目标 (x,y) 的上下左右四个相邻格子的向量并相加（越界安全）。
    返回 (sum_x, sum_y) 的 tuple。
    支持 list-of-tuples 和 numpy.ndarray。
    """
    # 从配置文件读取权重参数和平均值开关
    self_weight = config.get("vector_self_weight", 1.0)
    neighbor_weight = config.get("vector_neighbor_weight", 0.1)
    enable_average = config.get("enable_vector_average", False)
    
    # 如果未指定include_self，则使用配置文件中的默认值
    if include_self is None:
        include_self = config.get("include_self", False)
    if grid is None:
        return (0.0, 0.0)

    if isinstance(grid, np.ndarray):
        h, w = grid.shape[:2]
    else:
        if not grid or not grid[0]:
            return (0.0, 0.0)
        h = len(grid)
        w = len(grid[0])

    sum_x = 0.0
    sum_y = 0.0
    count = 0

    if include_self and 0 <= x < w and 0 <= y < h:
        if isinstance(grid, np.ndarray):
            vx, vy = float(grid[y, x, 0]), float(grid[y, x, 1])
        else:
            vx, vy = grid[y][x]
        sum_x += vx * self_weight  # 使用配置文件中的自身权重
        sum_y += vy * self_weight
        count += 1

    neighbors = ((0, -1), (0, 1), (-1, 0), (1, 0))
    for dx, dy in neighbors:
        nx = x + dx
        ny = y + dy
        if 0 <= nx < w and 0 <= ny < h:
            if isinstance(grid, np.ndarray):
                vx, vy = float(grid[ny, nx, 0]), float(grid[ny, nx, 1])
            else:
                vx, vy = grid[ny][nx]
            sum_x += vx * neighbor_weight  # 使用配置文件中的邻居权重
            sum_y += vy * neighbor_weight
            count += 1

    # 如果启用平均值功能，则除以有效向量数量
    if enable_average and count > 0:
        sum_x /= count
        sum_y /= count

    return (sum_x, sum_y)

def update_grid_with_adjacent_sum(grid, include_self=None):
    """
    遍历整个网格，使用 sum_adjacent_vectors 计算新值并一次性替换。
    返回修改后的 grid。
    支持 numpy.ndarray 和 list-of-tuples。
    """
    # 如果未指定include_self，则使用配置文件中的默认值
    if include_self is None:
        include_self = config.get("include_self", False)
    if not grid or not grid[0]:
        return grid

    # 如果是 numpy 数组，使用向量化操作
    if isinstance(grid, np.ndarray):
        h, w = grid.shape[:2]
        new_grid = np.zeros_like(grid)
        for y in range(h):
            for x in range(w):
                new_grid[y, x] = sum_adjacent_vectors(grid, x, y, include_self=include_self)
        grid[:] = new_grid
    else:
        # list-of-tuples 实现
        h = len(grid)
        w = len(grid[0])
        new_grid = [[(0, 0) for _ in range(w)] for _ in range(h)]
        for y in range(h):
            for x in range(w):
                new_grid[y][x] = sum_adjacent_vectors(grid, x, y, include_self=include_self)
        grid[:] = new_grid
    
    return grid



# ---------- GPU compute shader 实现（OpenGL 4.3+） ----------
def get_compute_shader_src():
    """从配置文件获取GPU计算着色器源代码，动态设置工作组和权重参数"""
    local_size_x = config.get("compute_shader_local_size_x", 16)
    local_size_y = config.get("compute_shader_local_size_y", 16)
    self_weight = config.get("vector_self_weight", 1.0)
    neighbor_weight = config.get("vector_neighbor_weight", 0.1)
    enable_average = config.get("enable_vector_average", False)
    
    shader_src = """
#version 430
layout(local_size_x = LOCAL_SIZE_X, local_size_y = LOCAL_SIZE_Y) in;
layout(binding = 0, rgba32f) uniform image2D img_in;
layout(binding = 1, rgba32f) uniform image2D img_out;
uniform int u_include_self;
uniform int u_enable_average;
void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 dims = imageSize(img_in);
    if(coord.x < 0 || coord.y < 0 || coord.x >= dims.x || coord.y >= dims.y) return;
    vec2 sum = vec2(0.0);
    int count = 0;
    if(u_include_self != 0) {
        sum += imageLoad(img_in, coord).xy * SELF_WEIGHT;  // 保持自身向量不变
        count++;
    }
    ivec2 n;
    n = coord + ivec2(0, -1); if(n.y >= 0) {
        sum += imageLoad(img_in, n).xy * NEIGHBOR_WEIGHT;  // 添加邻居向量的影响
        count++;
    }
    n = coord + ivec2(0,  1); if(n.y < dims.y) {
        sum += imageLoad(img_in, n).xy * NEIGHBOR_WEIGHT;
        count++;
    }
    n = coord + ivec2(-1, 0); if(n.x >= 0) {
        sum += imageLoad(img_in, n).xy * NEIGHBOR_WEIGHT;
        count++;
    }
    n = coord + ivec2( 1, 0); if(n.x < dims.x) {
        sum += imageLoad(img_in, n).xy * NEIGHBOR_WEIGHT;
        count++;
    }
    
    // 如果启用平均值功能，则除以有效向量数量
    if(u_enable_average != 0 && count > 0) {
        sum /= float(count);
    }

    imageStore(img_out, coord, vec4(sum, 0.0, 0.0));
}
"""
    # 使用字符串替换方法替换变量，避免格式化问题
    return shader_src.replace("LOCAL_SIZE_X", str(local_size_x)).replace("LOCAL_SIZE_Y", str(local_size_y)).replace("SELF_WEIGHT", str(self_weight)).replace("NEIGHBOR_WEIGHT", str(neighbor_weight))

# 默认计算着色器源代码（向后兼容）
COMPUTE_SHADER_SRC = get_compute_shader_src()



def _compile_compute_shader(src):
    sh = glCreateShader(GL_COMPUTE_SHADER)
    glShaderSource(sh, src)
    glCompileShader(sh)
    ok = glGetShaderiv(sh, GL_COMPILE_STATUS)
    if not ok:
        msg = glGetShaderInfoLog(sh).decode('utf-8')
        glDeleteShader(sh)
        raise RuntimeError("Compute shader compile error: " + msg)
    prog = glCreateProgram()
    glAttachShader(prog, sh)
    glLinkProgram(prog)
    ok = glGetProgramiv(prog, GL_LINK_STATUS)
    if not ok:
        msg = glGetProgramInfoLog(prog).decode('utf-8')
        glDeleteProgram(prog)
        glDeleteShader(sh)
        raise RuntimeError("Compute program link error: " + msg)
    glDeleteShader(sh)
    return prog

def _create_texture_float(w, h, data=None):
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    # 内部格式为 RGBA32F，外部使用 RGBA,GL_FLOAT；data shape must be (h,w,4) or None
    if data is None:
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGBA, GL_FLOAT, None)
    else:
        arr = np.asarray(data, dtype=np.float32)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGBA, GL_FLOAT, arr)
    glBindTexture(GL_TEXTURE_2D, 0)
    return tex

def gpu_init_compute(width=640, height=480, init_grid=None):
    """
    在当前 GL 上下文中创建 compute program 和两个浮点纹理（ping-pong）。
    init_grid: 可选 numpy (h,w,2) 初始向量场。
    返回 context dict，供后续 gpu_step / gpu_get_grid_numpy 使用。
    """
    # 使用动态生成的计算着色器源代码
    prog = _compile_compute_shader(get_compute_shader_src())
    tex_a = _create_texture_float(width, height, None)
    tex_b = _create_texture_float(width, height, None)
    ctx = {
        "prog": prog,
        "tex_a": tex_a,
        "tex_b": tex_b,
        "w": width,
        "h": height,
        "ping": True
    }
    if init_grid is not None:
        # 上传初值：将 (h,w,2) -> (h,w,4) RGBA
        h, w = init_grid.shape[:2]
        rgba = np.zeros((h, w, 4), dtype=np.float32)
        rgba[:, :, 0:2] = np.asarray(init_grid, dtype=np.float32)
        # 写入 tex_a
        glBindTexture(GL_TEXTURE_2D, ctx["tex_a"])
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_FLOAT, rgba)
        glBindTexture(GL_TEXTURE_2D, 0)
    return ctx

def gpu_step(ctx, include_self=False):
    """
    在 GPU 上执行一次邻域求和（compute shader），并在 ctx 中 ping-pong 交换输入/输出纹理。
    """
    prog = ctx["prog"]
    tex_in = ctx["tex_a"] if ctx["ping"] else ctx["tex_b"]
    tex_out = ctx["tex_b"] if ctx["ping"] else ctx["tex_a"]
    glUseProgram(prog)
    loc = glGetUniformLocation(prog, b"u_include_self")
    glUniform1i(loc, 1 if include_self else 0)

    # 传递平均值计算参数
    enable_average = config.get("enable_vector_average", False)
    loc = glGetUniformLocation(prog, b"u_enable_average")
    glUniform1i(loc, 1 if enable_average else 0)
    # 绑定 image 单元
    glBindImageTexture(0, tex_in, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F)
    glBindImageTexture(1, tex_out, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F)
    # 使用配置文件中的工作组大小参数
    local_size_x = config.get("compute_shader_local_size_x", 16)
    local_size_y = config.get("compute_shader_local_size_y", 16)
    groups_x = (ctx["w"] + local_size_x - 1) // local_size_x
    groups_y = (ctx["h"] + local_size_y - 1) // local_size_y
    glDispatchCompute(groups_x, groups_y, 1)
    # 确保写回完成
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)
    glUseProgram(0)
    ctx["ping"] = not ctx["ping"]
    return ctx

def gpu_get_grid_numpy(ctx):
    """
    从当前“输入”纹理读取回 numpy 数组，返回形状 (h,w,2) 的 float32 数组（只取 RG 分量）。
    注意：这会把纹理数据从 GPU 下载到 CPU。
    """
    tex = ctx["tex_a"] if ctx["ping"] else ctx["tex_b"]  # 注意：gpu_step 已交换 ping
    w = ctx["w"]; h = ctx["h"]
    glBindTexture(GL_TEXTURE_2D, tex)
    # glGetTexImage 返回 bytes; PyOpenGL 可以直接返回 numpy array
    data = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT)
    glBindTexture(GL_TEXTURE_2D, 0)
    arr = np.frombuffer(data, dtype=np.float32)
    arr = arr.reshape((h, w, 4))
    return arr[:, :, 0:2].copy()

def gpu_upload_texel(ctx, x, y, vec2):
    """
    将单个格子的向量上传到当前输入纹理（在下一次 gpu_step 中被读取）。
    x,y: texel 坐标（整型）
    vec2: 可迭代 (vx, vy)
    """
    if ctx is None:
        return
    tex = ctx["tex_a"] if ctx.get("ping", True) else ctx["tex_b"]
    rgba = np.array([float(vec2[0]), float(vec2[1]), 0.0, 0.0], dtype=np.float32)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexSubImage2D(GL_TEXTURE_2D, 0, int(x), int(y), 1, 1, GL_RGBA, GL_FLOAT, rgba)
    glBindTexture(GL_TEXTURE_2D, 0)

def gpu_upload_grid(ctx, grid):
    """
    将整个 (h,w,2) numpy grid 上传到当前输入纹理（覆盖）。
    """
    if ctx is None or grid is None:
        return
    h, w = grid.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    rgba[:, :, 0:2] = np.asarray(grid, dtype=np.float32)
    tex = ctx["tex_a"] if ctx.get("ping", True) else ctx["tex_b"]
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_FLOAT, rgba)
    glBindTexture(GL_TEXTURE_2D, 0)

def update_grid_with_adjacent_sum_numpy(grid, include_self=False):
    """
    专门针对numpy数组优化的向量场更新函数
    """
    if not isinstance(grid, np.ndarray) or grid.size == 0:
        return grid
        
    h, w = grid.shape[:2]
    new_grid = np.zeros_like(grid)
    
    # 使用numpy的向量化操作提高性能
    for y in range(h):
        for x in range(w):
            new_grid[y, x] = sum_adjacent_vectors(grid, x, y, include_self=include_self)
    
    grid[:] = new_grid
    return grid