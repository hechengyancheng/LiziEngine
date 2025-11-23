import threading
import time
import numpy as np
import glfw
from OpenGL.GL import *
import ctypes
import LiziLib
from config import config
from app_core import app_core, EventType

# 导入工具栏模块
try:
    from toolbar import ModernToolbar
    HAS_IMGUI = True
    print("成功导入工具栏")
except ImportError as e:
    print(f"警告: 无法导入工具栏，将使用默认工具栏。错误信息: {e}")
    HAS_IMGUI = False

VERTEX_SHADER_SRC = """
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

FRAGMENT_SHADER_SRC = """
#version 120
varying vec3 v_col;
void main() {
    gl_FragColor = vec4(v_col, 1.0);
}
"""

def compile_shader(src, shader_type):
    sh = glCreateShader(shader_type)
    glShaderSource(sh, src)
    glCompileShader(sh)
    ok = glGetShaderiv(sh, GL_COMPILE_STATUS)
    if not ok:
        msg = glGetShaderInfoLog(sh).decode('utf-8')
        glDeleteShader(sh)
        raise RuntimeError(f"Shader compile error: {msg}")
    return sh

def create_program(vs_src, fs_src):
    vs = compile_shader(vs_src, GL_VERTEX_SHADER)
    fs = compile_shader(fs_src, GL_FRAGMENT_SHADER)
    prog = glCreateProgram()
    glAttachShader(prog, vs)
    glAttachShader(prog, fs)
    glLinkProgram(prog)
    ok = glGetProgramiv(prog, GL_LINK_STATUS)
    if not ok:
        msg = glGetProgramInfoLog(prog).decode('utf-8')
        glDeleteProgram(prog)
        raise RuntimeError(f"Program link error: {msg}")
    glDeleteShader(vs)
    glDeleteShader(fs)
    return prog

def run_opengl_view(grid, cell_size=4, show_grid=True, toolbar_draw_func=None, grid_updated_flag=None):
    """运行 OpenGL 视图，带可选的工具栏回调函数"""
    # 初始化应用核心
    app_core.set_grid(grid)

    # 如果提供了grid_updated_flag，则使用它
    if grid_updated_flag is not None:
        # 创建一个全局引用，以便在主循环中使用
        global grid_updated_ref
        grid_updated_ref = grid_updated_flag
    
    # 初始化 GLFW
    if not glfw.init():
        print("glfw init failed"); return

    # 检查是否使用工具栏
    toolbar_enabled = config.get("toolbar_enabled", True)
    use_toolbar = HAS_IMGUI and toolbar_draw_func is None and toolbar_enabled
    h, w = grid.shape[:2]

    # 如果网格全零，填充示例场以便观察
    # if np.all(grid == 0):
    #     for yi in range(h):
    #         for xi in range(w):
    #             ang = (xi / max(1, w - 1)) * 2.0 * math.pi
    #             grid[yi, xi] = (math.cos(ang), math.sin(ang))

    # 从配置文件读取窗口尺寸限制
    max_win_w = config.get("window_max_width", 1200)
    max_win_h = config.get("window_max_height", 800)
    min_win_w = config.get("window_min_width", 200)
    min_win_h = config.get("window_min_height", 200)
    win_w = min(max(min_win_w, w * cell_size), max_win_w)
    win_h = min(max(min_win_h, h * cell_size), max_win_h)

    # 打印所有配置项用于调试
    config.debug_print_all()
    
    window_title = config.get("window_title", "LiziEngine - OpenGL")
    print(f"使用窗口标题: {window_title}")
    # 从配置文件读取抗锯齿设置
    enable_msaa = config.get("enable_msaa", True)
    msaa_samples = config.get("msaa_samples", 4)
    
    # 设置窗口提示以支持多重采样抗锯齿
    if enable_msaa:
        glfw.window_hint(glfw.SAMPLES, msaa_samples)
        
    try:
        window = glfw.create_window(win_w, win_h, window_title, None, None)
        if not window:
            glfw.terminate(); print("glfw window failed"); return
        glfw.make_context_current(window)
    except Exception as e:
        print(f"创建OpenGL窗口时出错: {e}")
        glfw.terminate()
        return

    # 初始化工具栏（如果可用）
    toolbar = None
    if use_toolbar:
        toolbar = ModernToolbar(window)
    
    # 启用多重采样抗锯齿
    if enable_msaa:
        glEnable(GL_MULTISAMPLE)
        
    # 从配置文件读取垂直同步设置
    vsync_enabled = config.get("enable_vsync", True)
    glfw.swap_interval(1 if vsync_enabled else 0)

    ver = glGetString(GL_VERSION)
    if ver is not None:
        print("GL_VERSION:", ver.decode() if isinstance(ver, bytes) else ver)

    try:
        program = create_program(VERTEX_SHADER_SRC, FRAGMENT_SHADER_SRC)
    except Exception as e:
        print("着色器创建失败，确保驱动支持 OpenGL 3.3+. 错误：", e)
        glfw.terminate()
        return

    # 从配置文件读取GPU计算设置
    use_gpu_compute = config.get("use_gpu_compute", True)
    
    # 初始化 GPU compute 资源（在 GL 上下文中）
    gpu_ctx = None
    if use_gpu_compute:
        print("[INFO] 正在尝试初始化GPU计算资源...")
        try:
            gpu_ctx = LiziLib.gpu_init_compute(w, h, init_grid=app_core.grid)
            print("[INFO] GPU计算资源初始化成功")
        except Exception as e:
            print(f"[WARNING] GPU compute 初始化失败，退回 CPU（numpy）实现：{e}")
            gpu_ctx = None
            use_gpu_compute = False
            print("[INFO] 已切换到CPU计算模式")
    else:
        print("[INFO] GPU计算已禁用，使用CPU计算模式")

    # VAO / VBO：预分配 VBO 容量（允许在可见区达到全局上限时仍能容纳）
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    # 从配置文件读取最大顶点数设置
    max_vertices = config.get("max_vertices", h * w * 2)
    max_bytes = max_vertices * 5 * ctypes.sizeof(ctypes.c_float)  # 每顶点 5 float (x,y,r,g,b)
    # 预分配一次大缓冲，之后用 BufferSubData 更新实际内容
    glBufferData(GL_ARRAY_BUFFER, max_bytes, None, GL_DYNAMIC_DRAW)

    a_pos_loc = glGetAttribLocation(program, b"a_pos")
    a_col_loc = glGetAttribLocation(program, b"a_col")
    stride = (2 + 3) * ctypes.sizeof(ctypes.c_float)
    glEnableVertexAttribArray(a_pos_loc)
    glVertexAttribPointer(a_pos_loc, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(a_col_loc)
    glVertexAttribPointer(a_col_loc, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(2 * ctypes.sizeof(ctypes.c_float)))
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)
    # UI (工具栏) VAO/VBO，由外部函数负责
    ui_vao = glGenVertexArrays(1)
    glBindVertexArray(ui_vao)
    ui_vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, ui_vbo)
    # 预分配空间：最多 60 顶点 (阴影 + 背景 + 边框 + 7 个按钮，每个用 6 顶点)，每顶点 5 float
    ui_max_bytes = 60 * 5 * ctypes.sizeof(ctypes.c_float)
    glBufferData(GL_ARRAY_BUFFER, ui_max_bytes, None, GL_DYNAMIC_DRAW)
    # 与主顶点布局相同
    glEnableVertexAttribArray(a_pos_loc)
    glVertexAttribPointer(a_pos_loc, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(a_col_loc)
    glVertexAttribPointer(a_col_loc, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(2 * ctypes.sizeof(ctypes.c_float)))
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)
    toolbar_enabled = True if toolbar_draw_func else False

    u_center_loc = glGetUniformLocation(program, b"u_center")
    u_half_loc = glGetUniformLocation(program, b"u_half")
    # 从配置文件读取FPS更新间隔设置
    last_frame_time = time.time()
    smooth_fps = 0.0
    last_title_update = 0.0
    title_update_interval = config.get("fps_update_interval", 0.25)
    # 从配置文件读取工具栏设置
    toolbar_px_w = config.get("toolbar_width", 320)
    toolbar_px_h = config.get("toolbar_height", 40)
    toolbar_padding = config.get("toolbar_padding", 10)

    # 从应用核心获取相机设置
    cam_zoom = app_core.state.get("cam_zoom", 1.0)
    cam_x = app_core.state.get("cam_x", (w * cell_size) / 2.0)
    cam_y = app_core.state.get("cam_y", (h * cell_size) / 2.0)
    panning = False
    pan_start = (0,0)
    cam_start = (cam_x, cam_y)

    # 更新应用核心的视图状态
    app_core.state.update({
        "cam_x": cam_x,
        "cam_y": cam_y,
        "cam_zoom": cam_zoom,
        "show_grid": show_grid
    })

    # 创建一个字典来存储视图状态，供工具栏函数使用
    view_state = {
        'cam_x': cam_x,
        'cam_y': cam_y,
        'cam_zoom': cam_zoom,
        'show_grid': show_grid
    }

    # 将视图状态设置为窗口用户指针
    view_state_ptr = ctypes.py_object(view_state)
    glfw.set_window_user_pointer(window, view_state_ptr)

    grid_lock = threading.Lock()
    stop_event = threading.Event()
    grid_updated_event = threading.Event()

    # 鼠标交互状态：左键绘制（dragging），右/中键平移（panning 已存在）
    dragging = False
    drag_button = None
    # 记录上一次绘制的格子坐标，避免重复设置
    last_drawn_cell = None
    # 标记是否是新笔触的开始
    new_stroke = True
    # 从配置文件读取笔刷和向量默认设置
    brush_size = config.get("default_brush_size", 1)
    magnitude = config.get("default_magnitude", 1.0)
    reverse_vector = config.get("reverse_vector", False)  # 是否反转向量方向

    # 创建可变对象，使工具栏能够修改这些值
    class MutableState:
        def __init__(self):
            self.brush_size = brush_size
            self.magnitude = magnitude
            self.reverse_vector = reverse_vector

    mutable_state = MutableState()

    def background_updater():
        # 从配置文件读取更新频率设置
        hz = config.get("update_frequency", 30.0)  # 使用配置中的更新频率
        interval = 1.0 / hz
        print(f"[DEBUG] 后台更新线程启动，频率: {hz}Hz，间隔: {interval:.3f}秒")
        update_count = 0
        while not stop_event.is_set():
            t0 = time.time()
            # 后台线程只执行 CPU 路径；如果启用了 GPU，则在主线程执行 GPU step
            if not use_gpu_compute:
                try:
                    with grid_lock:
                        LiziLib.update_grid_with_adjacent_sum_numpy(grid, include_self=False)
                    grid_updated_event.set()
                    update_count += 1
                    if update_count % 30 == 0:  # 每隔30次更新打印一次日志
                        print(f"[DEBUG] 后台更新线程已执行{update_count}次网格更新")
                except Exception as e:
                    print(f"[ERROR] 后台更新线程发生错误: {e}")
                    # 尝试恢复
                    time.sleep(interval * 2)
                    continue
            # 否则什么也不做（避免在无上下文线程里调用 OpenGL）
            dt = time.time() - t0
            sleep_time = max(0.0, interval - dt)
            time.sleep(sleep_time)
    th = threading.Thread(target=background_updater, name="updater")
    th.start()

    # 回调：缩放/拖动（保持原逻辑）
    def scroll_cb(window, xoffset, yoffset):
        nonlocal cam_zoom, cam_x, cam_y
        mx, my = glfw.get_cursor_pos(window)
        sx, sy = glfw.get_window_size(window)
        world_before_x = cam_x + (mx - sx/2.0) / cam_zoom
        world_before_y = cam_y + (my - sy/2.0) / cam_zoom
        # 从配置文件读取缩放因子和范围设置
        zoom_factor = config.get("zoom_factor", 1.15)
        min_zoom = config.get("min_zoom", 0.05)
        max_zoom = config.get("max_zoom", 64.0)
        factor = zoom_factor if yoffset > 0 else (1.0 / zoom_factor)
        cam_zoom = max(min_zoom, min(cam_zoom * factor, max_zoom))
        cam_x = world_before_x - (mx - sx/2.0) / cam_zoom
        cam_y = world_before_y - (my - sy/2.0) / cam_zoom

        # 更新应用核心和view_state字典中的相机参数
        app_core.state.update({
            "cam_x": cam_x,
            "cam_y": cam_y,
            "cam_zoom": cam_zoom
        })
        
        view_ptr = glfw.get_window_user_pointer(window)
        if view_ptr:
            view_data = view_ptr.value
            view_data['cam_x'] = cam_x
            view_data['cam_y'] = cam_y
            view_data['cam_zoom'] = cam_zoom

    # 提取向量绘制逻辑为单独的函数
    def draw_vectors_at_position(world_x, world_y, check_duplicate=False):
        """在指定世界坐标位置绘制向量

        Args:
            world_x: 世界坐标x
            world_y: 世界坐标y
            check_duplicate: 是否检查重复绘制（用于拖拽时避免重复设置同一格子）
        """
        nonlocal last_drawn_cell
        # 获取中心格子
        ix = int(world_x // cell_size)
        iy = int(world_y // cell_size)

        # 如果需要检查重复，且与上次绘制的格子相同，则跳过
        if check_duplicate:
            current_cell = (ix, iy)
            if last_drawn_cell is not None and current_cell == last_drawn_cell:
                return False  # 表示没有进行绘制

        # 计算中心格子的坐标
        center_x = ix * cell_size + cell_size / 2.0
        center_y = iy * cell_size + cell_size / 2.0

        # 计算从格子中心到鼠标位置的向量
        vx = world_x - center_x
        vy = world_y - center_y
        vec_mag = np.sqrt(vx*vx + vy*vy)

        # 归一化并应用模值
        if vec_mag > 0:
            vx = vx / vec_mag * mutable_state.magnitude
            vy = vy / vec_mag * mutable_state.magnitude
        else:
            # 如果鼠标在格子中心，使用默认方向
            vx = mutable_state.magnitude
            vy = 0

        # 如果需要反转方向
        if mutable_state.reverse_vector:
            vx = -vx
            vy = -vy

        # 根据笔刷大小设置格子向量
        for dy in range(-mutable_state.brush_size, mutable_state.brush_size + 1):
            for dx in range(-mutable_state.brush_size, mutable_state.brush_size + 1):
                nx, ny = ix + dx, iy + dy
                if 0 <= nx < w and 0 <= ny < h:
                    # 计算到中心点的距离，如果超出笔刷范围则跳过
                    if dx*dx + dy*dy <= mutable_state.brush_size*mutable_state.brush_size:
                        # 设置向量
                        with grid_lock:
                            grid[ny, nx] = (vx, vy)
                            # 使用app_core的update_grid方法更新单个格子
                            app_core.update_grid({(ny, nx): (vx, vy)})

                        # 同步到 GPU
                        if use_gpu_compute and gpu_ctx is not None:
                            LiziLib.gpu_upload_texel(gpu_ctx, nx, ny, (vx, vy))

        # 更新上次绘制的格子坐标
        if check_duplicate:
            last_drawn_cell = (ix, iy)

        return True  # 表示进行了绘制

    def mouse_button_cb(window, button, action, mods):
        nonlocal panning, pan_start, cam_start, dragging, drag_button, brush_size, magnitude, reverse_vector, last_drawn_cell, new_stroke

        # 如果有工具栏回调函数，则由它处理工具栏点击
        if toolbar_enabled and toolbar_draw_func is not None and button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
            mx, my = glfw.get_cursor_pos(window)
            sx, sy = glfw.get_window_size(window)
            # 将屏幕坐标转换为世界坐标
            world_x = cam_x + (mx - sx/2.0) / cam_zoom
            world_y = cam_y + (my - sy/2.0) / cam_zoom
            
            # 检查点击是否在工具栏区域内（从配置文件读取）
            toolbar_px_w = config.get("toolbar_width", 320)
            toolbar_px_h = config.get("toolbar_height", 40)
            toolbar_padding = config.get("toolbar_padding", 10)
            half_w = (sx/2.0) / cam_zoom
            half_h = (sy/2.0) / cam_zoom
            world_left = cam_x - half_w
            world_top = cam_y - half_h
            toolbar_x0 = world_left + (toolbar_padding) / cam_zoom
            toolbar_y0 = world_top + (toolbar_padding) / cam_zoom
            toolbar_x1 = toolbar_x0 + (toolbar_px_w) / cam_zoom
            toolbar_y1 = toolbar_y0 + (toolbar_px_h) / cam_zoom
            
            # 只有在工具栏区域内才调用工具栏回调
            if toolbar_x0 <= world_x <= toolbar_x1 and toolbar_y0 <= world_y <= toolbar_y1:
                result = toolbar_draw_func(
                    window, cam_x, cam_y, cam_zoom, sx, sy, ui_vao, ui_vbo,
                    a_pos_loc, a_col_loc, stride, toolbar_enabled,
                    brush_size, magnitude, reverse_vector,  # 传递当前值
                    mouse_x=world_x, mouse_y=world_y, click=True, grid=grid
                )
                if result:
                    # 如果回调函数返回了新的brush_size, magnitude, reverse_vector，则更新它们
                    brush_size, magnitude, reverse_vector = result
                # 不管是否处理了点击，只要在工具栏区域内就不再继续处理
                return

        # 平移（中键或右键）
        if button in (glfw.MOUSE_BUTTON_MIDDLE, glfw.MOUSE_BUTTON_RIGHT):
            if action == glfw.PRESS:
                panning = True
                pan_start = glfw.get_cursor_pos(window)
                cam_start = (cam_x, cam_y)
            else:
                panning = False
            return

        # 左键用于设置格子向量（单击 & 拖拽）
        if button == glfw.MOUSE_BUTTON_LEFT:
            mx, my = glfw.get_cursor_pos(window)
            sx, sy = glfw.get_window_size(window)
            if action == glfw.PRESS:
                dragging = True
                drag_button = glfw.MOUSE_BUTTON_LEFT
                # 重置上一次绘制的格子坐标
                last_drawn_cell = None
                # 添加一个标志，表示这是新的绘制开始
                new_stroke = True

                # 计算 world 坐标
                world_x = cam_x + (mx - sx/2.0) / cam_zoom
                world_y = cam_y + (my - sy/2.0) / cam_zoom

                # 使用提取的函数绘制向量
                draw_vectors_at_position(world_x, world_y, check_duplicate=False)

            else:
                # release
                if button == glfw.MOUSE_BUTTON_LEFT:
                    dragging = False
                    drag_button = None
                    last_drawn_cell = None  # 重置上一次绘制的格子坐标

    def cursor_cb(window, xpos, ypos):
        nonlocal cam_x, cam_y, last_drawn_cell, new_stroke
        # 平移
        if panning:
            dx = xpos - pan_start[0]
            dy = ypos - pan_start[1]
            cam_x = cam_start[0] - dx / cam_zoom
            cam_y = cam_start[1] - dy / cam_zoom

            # 更新view_state字典中的相机参数
            view_ptr = glfw.get_window_user_pointer(window)
            if view_ptr:
                view_data = view_ptr.value
                view_data['cam_x'] = cam_x
                view_data['cam_y'] = cam_y
            return
        
        # 拖拽绘制（左键）
        if dragging and drag_button == glfw.MOUSE_BUTTON_LEFT:
            mx, my = xpos, ypos
            sx, sy = glfw.get_window_size(window)
            world_x = cam_x + (mx - sx/2.0) / cam_zoom
            world_y = cam_y + (my - sy/2.0) / cam_zoom

            # 使用提取的函数绘制向量，并检查重复
            if draw_vectors_at_position(world_x, world_y, check_duplicate=True):
                # 更新上次绘制的格子坐标（已在函数内部处理）
                pass

    glfw.set_scroll_callback(window, scroll_cb)
    glfw.set_mouse_button_callback(window, mouse_button_cb)
    glfw.set_cursor_pos_callback(window, cursor_cb)

    # 从配置文件读取背景颜色设置
    bg_r = config.get("background_color_r", 0.12)
    bg_g = config.get("background_color_g", 0.12)
    bg_b = config.get("background_color_b", 0.12)
    glClearColor(bg_r, bg_g, bg_b, 1.0)
    glEnable(GL_PROGRAM_POINT_SIZE)
    
    # 从配置文件读取线宽设置
    line_width = config.get("line_width", 1.0)
    glLineWidth(line_width)
    
    # 从配置文件读取线条平滑设置
    enable_line_smooth = config.get("enable_line_smooth", True)
    if enable_line_smooth:
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
    
    # 从配置文件读取点平滑设置
    enable_point_smooth = config.get("enable_point_smooth", True)
    if enable_point_smooth:
        glEnable(GL_POINT_SMOOTH)
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)

    # 缓存：上一次视图/窗口参数，用于判断是否需要重建顶点数据
    last_cam = (cam_x, cam_y, cam_zoom)
    last_size = glfw.get_window_size(window)
    verts_count = 0  # 当前有效顶点数

    try:
        while not glfw.window_should_close(window):
            glfw.poll_events()
            # 计算 FPS（平滑）
            now_frame = time.time()
            dt_frame = max(1e-6, now_frame - last_frame_time)
            this_fps = 1.0 / dt_frame
            smooth_fps = smooth_fps * 0.90 + this_fps * 0.10 if smooth_fps != 0.0 else this_fps
            last_frame_time = now_frame

            # 每隔一段时间更新窗口标题以显示精确 FPS 数值
            if now_frame - last_title_update >= title_update_interval:
                window_title_base = config.get("window_title", "LiziEngine - OpenGL")
                glfw.set_window_title(window, f"{window_title_base} - FPS: {smooth_fps:.1f}")
                last_title_update = now_frame

            # 在窗口上显示当前鼠标所在的格子的坐标和向量         
            # 获取窗口尺寸和鼠标位置
            sx, sy = glfw.get_window_size(window)
            mouse_pos = glfw.get_cursor_pos(window)
            if mouse_pos is not None:
                # 将鼠标屏幕坐标转换为世界坐标
                world_x = cam_x + (mouse_pos[0] - sx/2.0) / cam_zoom
                world_y = cam_y + (mouse_pos[1] - sy/2.0) / cam_zoom

                # 计算对应的网格位置
                mouse_grid_x = int(world_x // cell_size)
                mouse_grid_y = int(world_y // cell_size)

                # 确保索引在有效范围内
                mouse_grid_x = max(0, min(w-1, mouse_grid_x))
                mouse_grid_y = max(0, min(h-1, mouse_grid_y))

                # 安全访问网格数据
                mouse_vec = grid[mouse_grid_y][mouse_grid_x]
                window_title_base = config.get("window_title", "LiziEngine - OpenGL")
                glfw.set_window_title(window, f"{window_title_base} - FPS: {smooth_fps:.1f} - Mouse: ({mouse_grid_x}, {mouse_grid_y}) - Vec: ({mouse_vec[0]:.2f}, {mouse_vec[1]:.2f})")


            # 如果启用了 GPU compute，在主线程（有 GL 上下文）按固定频率执行 GPU step 并拷回结果
            if use_gpu_compute and gpu_ctx is not None:
                # 控制频率（与之前后台线程 hz 保持一致）
                if not hasattr(run_opengl_view, "_gpu_last_time"):
                    run_opengl_view._gpu_last_time = 0.0
                gpu_interval = 1.0 / config.get("update_frequency", 30.0)  # 使用配置中的更新频率
                now = time.time()
                if now - run_opengl_view._gpu_last_time >= gpu_interval:
                    try:
                        #print(f"[DEBUG] 执行GPU计算，间隔: {gpu_interval:.3f}秒")
                        LiziLib.gpu_step(gpu_ctx, include_self=False)
                        res = LiziLib.gpu_get_grid_numpy(gpu_ctx)  # (h,w,2)
                        with grid_lock:
                            app_core.set_grid(res.astype(grid.dtype))
                            grid_updated_event.set()  # 确保设置更新标志
                        #print(f"[DEBUG] GPU计算完成并更新网格")
                    except Exception as e:
                        # 若 GPU 在运行时出现错误，回退到 CPU 路径
                        print(f"[ERROR] GPU step 发生错误，切换到 CPU 实现：{e}")
                        use_gpu_compute = False
                        print(f"[DEBUG] 已禁用GPU计算，使用CPU路径")
                    run_opengl_view._gpu_last_time = now

            sx, sy = glfw.get_window_size(window)
            # UI 在每帧都更新（位置随相机/尺寸/缩放变化）


            # 判断是否需要重建顶点缓冲：网格更新、相机变化或窗口尺寸变化
            cam_changed = (abs(cam_x - last_cam[0]) > 1e-6 or abs(cam_y - last_cam[1]) > 1e-6 or abs(cam_zoom - last_cam[2]) > 1e-6)
            size_changed = (sx != last_size[0] or sy != last_size[1])
            
            # 初始化 need_rebuild 变量
            need_rebuild = False
            
            # 检查工具栏设置的标志
            view_ptr = glfw.get_window_user_pointer(window)
            force_rebuild = False
            if view_ptr:
                view_data = view_ptr.value
                # 检查网格更新标志
                if view_data.get('grid_updated', False):
                    grid_updated_event.set()
                    view_data['grid_updated'] = False
                    
                    # 强制设置网格更新标志，确保重建
                    force_rebuild = True
                
                # 检查视图变化标志
                if view_data.get('view_changed', False):
                    cam_changed = True
                    view_data['view_changed'] = False
                    
                    # 更新相机参数
                    cam_x = view_data.get('cam_x', cam_x)
                    cam_y = view_data.get('cam_y', cam_y)
                    cam_zoom = view_data.get('cam_zoom', cam_zoom)
                    last_cam = (cam_x, cam_y, cam_zoom)
            
            # 检查全局网格更新标志
            if 'grid_updated_ref' in globals() and grid_updated_ref:
                grid_updated_event.set()
                grid_updated_ref = False
                force_rebuild = True

            # 如果网格更新事件被设置或强制重建标志被设置，强制重建
            if grid_updated_event.is_set() or force_rebuild:
                need_rebuild = True
                #print(f"[DEBUG] 网格重建被触发 - grid_updated_event: {grid_updated_event.is_set()}, force_rebuild: {force_rebuild}")

            half_w = (sx/2.0) / cam_zoom
            half_h = (sy/2.0) / cam_zoom
            # 计算可见格子索引范围
            left = cam_x - half_w
            right = cam_x + half_w
            top = cam_y - half_h
            bottom = cam_y + half_h
            x0 = max(0, int(left // cell_size) - 1)
            x1 = min(w, int(right // cell_size) + 2)
            y0 = max(0, int(top // cell_size) - 1)
            y1 = min(h, int(bottom // cell_size) + 2)

            if need_rebuild:
                #print(f"[DEBUG] 开始重建网格顶点数据，可见区域: x0={x0}, x1={x1}, y0={y0}, y1={y1}")
                with grid_lock:
                    # 从应用核心获取网格数据
                    sub = app_core.grid[y0:y1, x0:x1].astype(np.float32, copy=False)
                if sub.size == 0:
                    verts_count = 0
                    print("[DEBUG] 可见区域内没有网格数据")
                else:
                    # 用模长判断非空向量，避免 float 比较误判
                    mag = np.hypot(sub[:, :, 0], sub[:, :, 1])
                    mask = mag > 0.05  # 使用更合理的阈值，确保有意义的向量能被渲染
                    if np.any(mask):    # 至少有一个非空向量
                       # 构建顶点数据，颜色根据角度变化，长度随向量大小变化
                        ys, xs = np.nonzero(mask) #  获取mask中非零元素的坐标
                        vals = sub[ys, xs]  # (n,2) #  根据坐标获取sub数组中对应位置的值，形状为(n,2)
                        abs_xs = xs + x0 #  计算绝对x坐标，加上偏移量x0
                        abs_ys = ys + y0 #  计算绝对y坐标，加上偏移量y0
                        cx = (abs_xs.astype(np.float32) + 0.5) * cell_size #  计算中心点x坐标，将坐标转换为浮点型并加上0.5的偏移，再乘以cell_size
                        cy = (abs_ys.astype(np.float32) + 0.5) * cell_size #  计算中心点y坐标，将坐标转换为浮点型并加上0.5的偏移，再乘以cell_size
                        ang = np.arctan2(vals[:, 1], vals[:, 0]).astype(np.float32) #  计算角度，使用arctan2函数计算vals中第二列与第一列的比值，并转换为浮点型

                        # 计算向量模长
                        magnitudes = np.sqrt(vals[:, 0]**2 + vals[:, 1]**2).astype(np.float32)
                        
                        ex = cx + np.cos(ang) * magnitudes
                        ey = cy + np.sin(ang) * magnitudes

                        # 颜色向量化
                        r = 0.5 * (1.0 + np.cos(ang))
                        g = 0.5 * (1.0 + np.cos(ang - 2.0 * np.pi / 3.0))
                        b = 0.5 * (1.0 + np.cos(ang + 2.0 * np.pi / 3.0))
                        n = cx.shape[0]
                        # 从配置文件读取向量渲染平滑参数
                        enable_vector_smoothing = config.get("enable_vector_smoothing", True)
                        vector_smooth_factor = config.get("vector_smooth_factor", 0.85)
                        
                        # 应用平滑处理，减少高频变化
                        if enable_vector_smoothing:
                            magnitudes = magnitudes * vector_smooth_factor
                            ex = cx + np.cos(ang) * magnitudes
                            ey = cy + np.sin(ang) * magnitudes
                        else:
                            ex = cx + np.cos(ang) * magnitudes
                            ey = cy + np.sin(ang) * magnitudes
                        
                        # 构建顶点数组 (n*2, 5)
                        verts = np.empty((n * 2, 5), dtype=np.float32)
                        verts[0::2, 0] = cx; verts[0::2, 1] = cy
                        verts[1::2, 0] = ex; verts[1::2, 1] = ey
                        verts[0::2, 2] = r; verts[0::2, 3] = g; verts[0::2, 4] = b
                        verts[1::2, 2] = r; verts[1::2, 3] = g; verts[1::2, 4] = b
                        verts_flat = verts.ravel()
                        verts_nbytes = verts_flat.nbytes
                        # 更新 VBO（使用 SubData，保留预分配空间）
                        glBindBuffer(GL_ARRAY_BUFFER, vbo)
                        glBufferSubData(GL_ARRAY_BUFFER, 0, verts_nbytes, verts_flat)
                        glBindBuffer(GL_ARRAY_BUFFER, 0)
                        verts_count = n * 2
                    else:
                        verts_count = 0

                grid_updated_event.clear()
                #print(f"[DEBUG] 网格重建完成，顶点数: {verts_count}")
                last_cam = (cam_x, cam_y, cam_zoom)
                last_size = (sx, sy)

            # 渲染
            glViewport(0, 0, sx, sy)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glUseProgram(program)
            glUniform2f(u_center_loc, cam_x, cam_y)
            glUniform2f(u_half_loc, half_w, half_h)

            # 获取网格显示状态
            view_ptr = glfw.get_window_user_pointer(window)
            show_grid_current = show_grid  # 默认值
            if view_ptr:
                view_data = view_ptr.value
                show_grid_current = view_data.get('show_grid', show_grid)

            # 只有在启用网格显示时才绘制网格边缘
            if show_grid_current:
                # 设置边缘颜色为白色
                edge_color = (1.0, 1.0, 1.0)
                # 计算网格边缘坐标
                left_edge = 0
                right_edge = w * cell_size
                top_edge = 0
                bottom_edge = h * cell_size

                # 创建边缘顶点数据 (8个顶点，4条线)
                edge_verts = np.array([
                    # 左边缘
                    left_edge, top_edge, *edge_color,
                    left_edge, bottom_edge, *edge_color,
                    # 右边缘
                    right_edge, top_edge, *edge_color,
                    right_edge, bottom_edge, *edge_color,
                    # 上边缘
                    left_edge, top_edge, *edge_color,
                    right_edge, top_edge, *edge_color,
                    # 下边缘
                    left_edge, bottom_edge, *edge_color,
                    right_edge, bottom_edge, *edge_color
                ], dtype=np.float32)

                # 上传边缘顶点数据
                glBindBuffer(GL_ARRAY_BUFFER, vbo)
                glBufferSubData(GL_ARRAY_BUFFER, 0, edge_verts.nbytes, edge_verts)
                glBindBuffer(GL_ARRAY_BUFFER, 0)

                # 绘制网格边缘
                glBindVertexArray(vao)
                glBindBuffer(GL_ARRAY_BUFFER, vbo)
                glEnableVertexAttribArray(a_pos_loc)
                glVertexAttribPointer(a_pos_loc, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
                glEnableVertexAttribArray(a_col_loc)
                glVertexAttribPointer(a_col_loc, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(2 * ctypes.sizeof(ctypes.c_float)))
                glDrawArrays(GL_LINES, 0, 8)
                glBindVertexArray(0)
                glBindBuffer(GL_ARRAY_BUFFER, 0)

            # 绘制向量
            if verts_count > 0:
                glBindVertexArray(vao)
                glBindBuffer(GL_ARRAY_BUFFER, vbo)
                # 确保 attrib 指针正确（某些驱动需要显式）
                glEnableVertexAttribArray(a_pos_loc)
                glVertexAttribPointer(a_pos_loc, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
                glEnableVertexAttribArray(a_col_loc)
                glVertexAttribPointer(a_col_loc, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(2 * ctypes.sizeof(ctypes.c_float)))
                glDrawArrays(GL_LINES, 0, verts_count)
                glBindVertexArray(0)
                glBindBuffer(GL_ARRAY_BUFFER, 0)

            # 绘制工具栏（放在对象之后，保证可见）
            if use_toolbar and toolbar:
                # 使用工具栏
                toolbar.process_input()
                result = toolbar.render(grid)
                if result:
                    # 如果回调函数返回了新的brush_size, magnitude, reverse_vector，则更新它们
                    new_brush_size, new_magnitude, new_reverse_vector = result
                    # 更新mutable_state中的值
                    mutable_state.brush_size = new_brush_size
                    mutable_state.magnitude = new_magnitude
                    mutable_state.reverse_vector = new_reverse_vector

                # 同步show_grid状态
                view_ptr = glfw.get_window_user_pointer(window)
                if view_ptr:
                    view_data = view_ptr.value
                    view_data['show_grid'] = toolbar.show_grid
            elif toolbar_enabled and toolbar_draw_func is not None:
                # 使用原始工具栏
                result = toolbar_draw_func(
                    window, cam_x, cam_y, cam_zoom, sx, sy, ui_vao, ui_vbo,
                    a_pos_loc, a_col_loc, stride, toolbar_enabled,
                    brush_size, magnitude, reverse_vector, grid=grid
                )
                if result:
                    # 如果回调函数返回了新的brush_size, magnitude, reverse_vector，则更新它们
                    brush_size, magnitude, reverse_vector = result
            glUseProgram(0)

            glfw.swap_buffers(window)

    finally:
        stop_event.set()
        th.join(timeout=2.0)
        try:
            if vbo:
                glDeleteBuffers(1, [vbo])
            if ui_vbo:
                glDeleteBuffers(1, [ui_vbo])
            if vao:
                glDeleteVertexArrays(1, [vao])
            if ui_vao:
                glDeleteVertexArrays(1, [ui_vao])
            if program:
                glDeleteProgram(program)
            # 关闭工具栏（如果可用）
            if toolbar:
                toolbar.shutdown()
        except Exception:
            pass
        glfw.terminate()