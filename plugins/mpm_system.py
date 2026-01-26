"""
MPM系统插件，实现基于Material Point Method的物质点模拟
"""

import numpy as np
from typing import List, Dict, Any, Tuple

class MPMParticle:
    """MPM物质点类"""

    def __init__(self, x: float, y: float, mass: float = 1.0, vx: float = 0.0, vy: float = 0.0,
                 rho0: float = 1.0):
        self.x = x  # 位置
        self.y = y
        self.vx = vx  # 速度
        self.vy = vy
        self.mass = mass  # 质量
        self.rho0 = rho0  # 初始密度
        self.volume0 = mass / rho0  # 参考体积
        self.volume = self.volume0  # 当前体积
        self.stress = np.zeros((2, 2))  # 应力张量
        self.deformation = np.eye(2)  # 变形梯度
        self.elastic_energy = 0.0  # 弹性能量
        self.C = np.zeros((2, 2))  # Affine velocity matrix for MLS-MPM

    def update_position(self, dt: float):
        """更新位置"""
        self.x += self.vx * dt
        self.y += self.vy * dt

    def update_velocity(self, dt: float, gravity: float = 0.0):
        """更新速度（添加重力）"""
        self.vy += gravity * dt

class MPMSystem:
    """MPM系统，管理物质点和网格交互"""

    def __init__(self, app_core, grid_size: Tuple[int, int] = (64, 64),
                 dt: float = 0.01, gravity: float = 0.01,
                 young_modulus: float = 1000.0, poisson_ratio: float = 0.3,
                 damping: float = 0.95, restitution: float = 0.5,
                 friction: float = 0.2):
        self.app_core = app_core
        self.grid_width, self.grid_height = grid_size
        self.particles: List[MPMParticle] = []
        self.grid_mass = np.zeros((self.grid_height, self.grid_width))  # 网格质量
        self.grid_velocity = np.zeros((self.grid_height, self.grid_width, 2))  # 网格速度
        self.grid_force = np.zeros((self.grid_height, self.grid_width, 2))  # 网格力
        self.grid_B = np.zeros((self.grid_height, self.grid_width, 2, 2))  # Affine momentum matrix for MLS-MPM

        # MPM参数
        self.dt = dt  # 时间步长
        self.gravity = gravity  # 重力
        self.young_modulus = young_modulus  # 杨氏模量
        self.poisson_ratio = poisson_ratio  # 泊松比
        self.damping = damping  # 阻尼
        self.restitution = restitution  # 恢复系数（弹性碰撞）
        self.friction = friction  # 摩擦系数

        # Clamp Poisson ratio to valid range to prevent division by zero in lambda calculation
        self.poisson_ratio = max(0.0, min(0.499, self.poisson_ratio))

        # 计算Lame参数
        self.mu = self.young_modulus / (2 * (1 + self.poisson_ratio))  # 剪切模量
        self.lambda_ = self.young_modulus * self.poisson_ratio / ((1 + self.poisson_ratio) * (1 - 2 * self.poisson_ratio))  # 第一Lame参数

        # CFL参数
        self.cfl_factor = 0.1  # CFL因子
        self.min_dt = 1e-6  # 最小时间步长
        self.max_dt = 0.1  # 最大时间步长

    def quadratic_bspline_weight(self, r: float):
        """二次B样条权重函数"""
        abs_r = abs(r)
        if abs_r <= 0.5:
            return 0.75 - abs_r**2
        elif abs_r <= 1.5:
            return 0.5 * (1.5 - abs_r)**2
        else:
            return 0.0

    def quadratic_bspline_weight_derivative(self, r: float):
        """二次B样条权重函数的导数"""
        if r >= -0.5 and r <= 0.5:
            return -2 * r
        elif r > 0.5 and r <= 1.5:
            return - (1.5 - r)
        elif r < -0.5 and r >= -1.5:
            return (1.5 + r)
        else:
            return 0.0

    def get_interpolation_weights(self, x: float, y: float):
        """获取标准MPM二次B样条插值权重，使用3x3网格"""
        grid_x = int(x)
        grid_y = int(y)
        weights = []
        offsets = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                nx = grid_x + dx
                ny = grid_y + dy
                if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                    # 计算相对于网格节点的相对位置
                    rx = x - nx
                    ry = y - ny
                    # 二次B样条权重（分离x和y方向）
                    wx = self.quadratic_bspline_weight(rx)
                    wy = self.quadratic_bspline_weight(ry)
                    w = wx * wy
                    weights.append(w)
                    offsets.append((dx, dy))
        return weights, offsets, grid_x, grid_y

    def compute_mls_interpolation(self, particle: MPMParticle):
        """计算MLS插值，得到粒子速度和仿射矩阵C"""
        weights, offsets, grid_x, grid_y = self.get_interpolation_weights(particle.x, particle.y)

        # 初始化矩阵A和B
        A = np.zeros((2, 2))
        B = np.zeros((2, 2))

        for i, (dx, dy) in enumerate(offsets):
            nx, ny = grid_x + dx, grid_y + dy
            weight = weights[i]

            # 相对位置向量
            dx_vec = np.array([particle.x - nx, particle.y - ny])

            # 网格速度
            v_grid = np.array([self.grid_velocity[ny, nx, 0], self.grid_velocity[ny, nx, 1]])

            # 累加矩阵A和B
            A += weight * np.outer(dx_vec, dx_vec)
            B += weight * np.outer(dx_vec, v_grid)

        # 计算仿射矩阵C
        if np.linalg.det(A) > 1e-6:  # 检查矩阵是否可逆
            C = np.linalg.inv(A) @ B
        else:
            C = np.zeros((2, 2))

        # 计算粒子速度
        v_particle = np.zeros(2)
        for i, (dx, dy) in enumerate(offsets):
            nx, ny = grid_x + dx, grid_y + dy
            weight = weights[i]
            v_grid = np.array([self.grid_velocity[ny, nx, 0], self.grid_velocity[ny, nx, 1]])
            dx_vec = np.array([particle.x - nx, particle.y - ny])
            v_particle += weight * (v_grid + C @ dx_vec)

        return v_particle, C

    def add_particle(self, x: float, y: float, mass: float = 1.0, vx: float = 0.0, vy: float = 0.0):
        """添加物质点"""
        particle = MPMParticle(x, y, mass, vx, vy)
        self.particles.append(particle)

    def clear_particles(self):
        """清除所有物质点"""
        self.particles = []

    def compute_stress(self, particle: MPMParticle):
        """计算粒子应力（Neo-Hookean弹性模型）"""
        F = particle.deformation

        # 检查变形梯度是否有效
        if np.any(np.isnan(F)) or np.any(np.isinf(F)):
            F = np.eye(2)
            particle.deformation = F

        # 限制变形梯度的范围以防止数值不稳定
        F_norm = np.linalg.norm(F)
        if F_norm > 10.0:  # 限制最大变形
            F = F * (10.0 / F_norm)
            particle.deformation = F

        J = np.linalg.det(F)
        if not np.isfinite(J) or J <= 0:
            J = 1.0  # 重置为无变形状态

        # 更新体积
        particle.volume = particle.volume0 * J

        # Neo-Hookean应力
        try:
            b = F @ F.T  # 左Cauchy-Green张量
            if not np.all(np.isfinite(b)):
                b = np.eye(2)

            dev_b = b - (np.trace(b) / 2) * np.eye(2)  # 偏应力部分

            # Compute stress terms separately to avoid invalid value warnings
            term1 = (self.mu / J) * dev_b
            term2 = (self.lambda_ * (J - 1) / J) * np.eye(2)

            # Check for invalid values before addition
            if not np.all(np.isfinite(term1)) or not np.all(np.isfinite(term2)):
                sigma = np.zeros((2, 2))
            else:
                sigma = term1 + term2

            # 检查应力是否有效
            if not np.all(np.isfinite(sigma)):
                sigma = np.zeros((2, 2))

            particle.stress = sigma
        except:
            # 如果计算失败，重置应力
            particle.stress = np.zeros((2, 2))

    def update_deformation_gradient(self, particle: MPMParticle):
        """更新变形梯度，使用MLS-MPM方式（基于仿射矩阵C）"""
        # 检查粒子位置是否有效
        if not (np.isfinite(particle.x) and np.isfinite(particle.y)):
            return  # 跳过无效粒子

        # MLS-MPM变形梯度更新：F_new = (I + dt * C) @ F_old
        dF = np.eye(2) + self.dt * particle.C

        # 检查dF是否有效
        if not np.all(np.isfinite(dF)):
            dF = np.eye(2)

        # 更新变形梯度
        new_F = dF @ particle.deformation

        # 检查新变形梯度是否有效
        if np.all(np.isfinite(new_F)) and np.linalg.norm(new_F) < 100.0:
            particle.deformation = new_F
        else:
            # 重置为单位矩阵
            particle.deformation = np.eye(2)

    def particles_to_grid(self):
        """P2G: 将粒子信息映射到网格，包括变形梯度更新、应力计算和力映射"""
        # 重置网格
        self.grid_mass.fill(0)
        self.grid_velocity.fill(0)
        self.grid_force.fill(0)
        self.grid_B.fill(0)

        # 首先映射质量和动量（MLS-MPM方式）
        for particle in self.particles:
            # 跳过无效粒子
            if not (np.isfinite(particle.x) and np.isfinite(particle.y) and
                    np.isfinite(particle.vx) and np.isfinite(particle.vy)):
                continue

            # 使用安全的插值权重计算
            weights, offsets, grid_x, grid_y = self.get_interpolation_weights(particle.x, particle.y)

            for weight, (dx, dy) in zip(weights, offsets):
                nx, ny = grid_x + dx, grid_y + dy
                # 相对位置向量
                dx_vec = np.array([particle.x - nx, particle.y - ny])
                # 仿射动量
                affine_momentum = particle.mass * (np.array([particle.vx, particle.vy]) + particle.C @ dx_vec)

                # 质量映射
                self.grid_mass[ny, nx] += particle.mass * weight
                # 动量映射（包括仿射部分）
                self.grid_velocity[ny, nx, 0] += affine_momentum[0] * weight
                self.grid_velocity[ny, nx, 1] += affine_momentum[1] * weight
                # 仿射矩阵B映射
                self.grid_B[ny, nx] += particle.mass * particle.C * weight

        # 归一化速度
        mask = self.grid_mass > 0
        self.grid_velocity[mask, 0] /= self.grid_mass[mask]
        self.grid_velocity[mask, 1] /= self.grid_mass[mask]

        # 更新变形梯度
        for particle in self.particles:
            self.update_deformation_gradient(particle)

        # 计算应力
        for particle in self.particles:
            self.compute_stress(particle)

        # 映射力到网格
        for particle in self.particles:
            weights, offsets, grid_x, grid_y = self.get_interpolation_weights(particle.x, particle.y)
            h = 1.0  # 平滑长度

            # 计算权重梯度
            dw_dx = []
            dw_dy = []
            for (dx, dy), weight in zip(offsets, weights):
                nx = grid_x + dx
                ny = grid_y + dy
                px = particle.x - nx
                py = particle.y - ny
                dw_dx.append(-2 * px / h**2 * weight)
                dw_dy.append(-2 * py / h**2 * weight)

            # 应力散度
            stress_div_x = particle.stress[0, 0] * np.array(dw_dx) + particle.stress[0, 1] * np.array(dw_dy)
            stress_div_y = particle.stress[1, 0] * np.array(dw_dx) + particle.stress[1, 1] * np.array(dw_dy)

            for i, (dx, dy) in enumerate(offsets):
                nx, ny = grid_x + dx, grid_y + dy
                # 力映射（负应力散度）
                self.grid_force[ny, nx, 0] -= particle.volume * stress_div_x[i]
                self.grid_force[ny, nx, 1] -= particle.volume * stress_div_y[i]

        # 添加重力（作为力）
        mask = self.grid_mass > 0
        self.grid_force[mask, 1] += self.gravity * self.grid_mass[mask]

        # 更新网格速度（应用力）
        self.grid_velocity[mask, 0] += self.dt * self.grid_force[mask, 0] / self.grid_mass[mask]
        self.grid_velocity[mask, 1] += self.dt * self.grid_force[mask, 1] / self.grid_mass[mask]

    def grid_to_particles(self):
        """G2P: 将网格信息映射回粒子（MLS-MPM方式）"""
        for particle in self.particles:
            # 使用MLS插值计算粒子速度和仿射矩阵C
            v_particle, C_particle = self.compute_mls_interpolation(particle)

            # 更新粒子速度和仿射矩阵
            particle.vx = v_particle[0]
            particle.vy = v_particle[1]
            particle.C = C_particle

    def update_particles(self):
        """更新所有粒子，包含改进的边界处理和数值稳定性"""
        for particle in self.particles:
            # 位置更新（速度已在G2P中更新）
            particle.update_position(self.dt)

            # 严格的边界条件：防止粒子逃逸
            margin = 0.01  # 边界裕度

            # 左边界 (x接近0)
            if particle.x < margin:
                particle.x = margin
                if particle.vx < 0:
                    particle.vx = -particle.vx * self.restitution
                particle.vy *= (1 - self.friction)

            # 右边界 (x接近grid_width)
            elif particle.x > self.grid_width - margin:
                particle.x = self.grid_width - margin
                if particle.vx > 0:
                    particle.vx = -particle.vx * self.restitution
                particle.vy *= (1 - self.friction)

            # 下边界 (y接近0)
            if particle.y < margin:
                particle.y = margin
                if particle.vy < 0:
                    particle.vy = -particle.vy * self.restitution
                particle.vx *= (1 - self.friction)

            # 上边界 (y接近grid_height)
            elif particle.y > self.grid_height - margin:
                particle.y = self.grid_height - margin
                if particle.vy > 0:
                    particle.vy = -particle.vy * self.restitution
                particle.vx *= (1 - self.friction)

            # 数值稳定性：限制速度范围
            max_speed = 10.0
            particle.vx = max(-max_speed, min(max_speed, particle.vx))
            particle.vy = max(-max_speed, min(max_speed, particle.vy))

            # 通用阻尼
            particle.vx *= self.damping
            particle.vy *= self.damping

            # 确保位置在合理范围内
            particle.x = max(margin, min(self.grid_width - margin, particle.x))
            particle.y = max(margin, min(self.grid_height - margin, particle.y))

    def update_vector_field(self, grid: np.ndarray):
        """将MPM结果转换为LiziEngine向量场"""
        grid.fill(0.0)

        # 将网格速度转换为向量场
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                if self.grid_mass[y, x] > 0:
                    vx = self.grid_velocity[y, x, 0]
                    vy = self.grid_velocity[y, x, 1]
                    grid[y, x, 0] = vx * 10  # 放大显示
                    grid[y, x, 1] = vy * 10

    def check_cfl_condition(self):
        """检查CFL条件并调整时间步长"""
        if not self.particles:
            return

        # 计算最大速度
        max_velocity = 0.0
        for particle in self.particles:
            velocity = np.sqrt(particle.vx**2 + particle.vy**2)
            max_velocity = max(max_velocity, velocity)

        # 计算声速（基于杨氏模量和密度）
        sound_speed = np.sqrt(self.young_modulus / 1.0)  # 假设密度为1.0

        # CFL条件：dt <= dx / (max_velocity + sound_speed)
        dx = 1.0  # 网格间距
        cfl_dt = self.cfl_factor * dx / (max_velocity + sound_speed + 1e-6)

        # 限制时间步长范围
        self.dt = max(self.min_dt, min(cfl_dt, self.max_dt))

    def step(self, grid: np.ndarray):
        """执行一步MPM模拟"""
        if not self.particles:
            return

        # 检查CFL条件并调整时间步长
        self.check_cfl_condition()

        # P2G (包括变形梯度更新、应力计算、力和速度更新)
        self.particles_to_grid()

        # G2P (将网格信息映射回粒子)
        self.grid_to_particles()

        # 更新粒子位置和速度
        self.update_particles()

        # 更新向量场用于显示
        self.update_vector_field(grid)
