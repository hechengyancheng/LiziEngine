"""
MPM系统插件，实现基于Material Point Method的物质点模拟
"""

import numpy as np
from typing import List, Dict, Any, Tuple

class MPMParticle:
    """MPM物质点类"""

    def __init__(self, x: float, y: float, mass: float = 1.0, vx: float = 0.0, vy: float = 0.0):
        self.x = x  # 位置
        self.y = y
        self.vx = vx  # 速度
        self.vy = vy
        self.mass = mass  # 质量
        self.volume = 1.0  # 体积
        self.stress = np.zeros((2, 2))  # 应力张量
        self.deformation = np.eye(2)  # 变形梯度
        self.elastic_energy = 0.0  # 弹性能量

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
                 dt: float = 0.1, gravity: float = 0.01,
                 young_modulus: float = 1000.0, poisson_ratio: float = 0.3,
                 damping: float = 0.99):
        self.app_core = app_core
        self.grid_width, self.grid_height = grid_size
        self.particles: List[MPMParticle] = []
        self.grid_mass = np.zeros((self.grid_height, self.grid_width))  # 网格质量
        self.grid_velocity = np.zeros((self.grid_height, self.grid_width, 2))  # 网格速度
        self.grid_force = np.zeros((self.grid_height, self.grid_width, 2))  # 网格力

        # MPM参数
        self.dt = dt  # 时间步长
        self.gravity = gravity  # 重力
        self.young_modulus = young_modulus  # 杨氏模量
        self.poisson_ratio = poisson_ratio  # 泊松比
        self.damping = damping  # 阻尼

        # 计算Lame参数
        self.mu = self.young_modulus / (2 * (1 + self.poisson_ratio))  # 剪切模量
        self.lambda_ = self.young_modulus * self.poisson_ratio / ((1 + self.poisson_ratio) * (1 - 2 * self.poisson_ratio))  # 第一Lame参数

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

        # Neo-Hookean应力
        try:
            b = F @ F.T  # 左Cauchy-Green张量
            if not np.all(np.isfinite(b)):
                b = np.eye(2)

            dev_b = b - (np.trace(b) / 2) * np.eye(2)  # 偏应力部分

            sigma = (self.mu / J) * dev_b + (self.lambda_ * (J - 1) / J) * np.eye(2)

            # 检查应力是否有效
            if not np.all(np.isfinite(sigma)):
                sigma = np.zeros((2, 2))

            particle.stress = sigma
        except:
            # 如果计算失败，重置应力
            particle.stress = np.zeros((2, 2))

    def update_deformation_gradient(self, particle: MPMParticle):
        """更新变形梯度"""
        # 检查粒子位置是否有效
        if not (np.isfinite(particle.x) and np.isfinite(particle.y)):
            return  # 跳过无效粒子

        grid_x = int(particle.x)
        grid_y = int(particle.y)

        if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
            fx = particle.x - grid_x
            fy = particle.y - grid_y

            # 权重梯度
            dw_dx = np.array([-(1 - fy), (1 - fy), -fy, fy])
            dw_dy = np.array([-(1 - fx), -fx, (1 - fx), fx])

            offsets = [(0, 0), (1, 0), (0, 1), (1, 1)]

            # 速度梯度
            grad_vx = np.zeros(2)
            grad_vy = np.zeros(2)

            for i, (dx, dy) in enumerate(offsets):
                nx, ny = grid_x + dx, grid_y + dy
                if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                    grad_vx[0] += self.grid_velocity[ny, nx, 0] * dw_dx[i]
                    grad_vx[1] += self.grid_velocity[ny, nx, 0] * dw_dy[i]
                    grad_vy[0] += self.grid_velocity[ny, nx, 1] * dw_dx[i]
                    grad_vy[1] += self.grid_velocity[ny, nx, 1] * dw_dy[i]

            # 变形梯度增量
            dF = np.eye(2) + self.dt * np.array([[grad_vx[0], grad_vx[1]],
                                                [grad_vy[0], grad_vy[1]]])

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

        # 首先映射质量和动量
        for particle in self.particles:
            # 跳过无效粒子
            if not (np.isfinite(particle.x) and np.isfinite(particle.y) and
                    np.isfinite(particle.vx) and np.isfinite(particle.vy)):
                continue

            # 计算粒子所在的网格单元
            grid_x = int(particle.x)
            grid_y = int(particle.y)

            # 确保在网格范围内
            if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
                # 计算权重（简单线性插值）
                fx = particle.x - grid_x
                fy = particle.y - grid_y

                # 四个相邻网格点的权重
                weights = [
                    (1 - fx) * (1 - fy),  # (0,0)
                    fx * (1 - fy),        # (1,0)
                    (1 - fx) * fy,        # (0,1)
                    fx * fy              # (1,1)
                ]

                offsets = [(0, 0), (1, 0), (0, 1), (1, 1)]

                for weight, (dx, dy) in zip(weights, offsets):
                    nx, ny = grid_x + dx, grid_y + dy
                    if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                        # 质量映射
                        self.grid_mass[ny, nx] += particle.mass * weight
                        # 动量映射
                        self.grid_velocity[ny, nx, 0] += particle.mass * particle.vx * weight
                        self.grid_velocity[ny, nx, 1] += particle.mass * particle.vy * weight

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
            grid_x = int(particle.x)
            grid_y = int(particle.y)

            if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
                fx = particle.x - grid_x
                fy = particle.y - grid_y

                # 权重梯度
                dw_dx = np.array([-(1 - fy), (1 - fy), -fy, fy])
                dw_dy = np.array([-(1 - fx), -fx, (1 - fx), fx])

                offsets = [(0, 0), (1, 0), (0, 1), (1, 1)]

                # 应力散度
                stress_div_x = particle.stress[0, 0] * dw_dx + particle.stress[0, 1] * dw_dy
                stress_div_y = particle.stress[1, 0] * dw_dx + particle.stress[1, 1] * dw_dy

                for i, (dx, dy) in enumerate(offsets):
                    nx, ny = grid_x + dx, grid_y + dy
                    if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                        # 力映射（负应力散度）
                        self.grid_force[ny, nx, 0] -= particle.volume * stress_div_x[i]
                        self.grid_force[ny, nx, 1] -= particle.volume * stress_div_y[i]

        # 更新网格速度（应用力）
        mask = self.grid_mass > 0
        self.grid_velocity[mask, 0] += self.dt * self.grid_force[mask, 0] / self.grid_mass[mask]
        self.grid_velocity[mask, 1] += self.dt * self.grid_force[mask, 1] / self.grid_mass[mask]

        # 添加重力
        self.grid_velocity[mask, 1] += self.gravity * self.dt

    def grid_to_particles(self):
        """G2P: 将网格信息映射回粒子"""
        for particle in self.particles:
            grid_x = int(particle.x)
            grid_y = int(particle.y)

            if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
                fx = particle.x - grid_x
                fy = particle.y - grid_y

                weights = [
                    (1 - fx) * (1 - fy),
                    fx * (1 - fy),
                    (1 - fx) * fy,
                    fx * fy
                ]

                offsets = [(0, 0), (1, 0), (0, 1), (1, 1)]

                # 插值网格速度
                vx_new = 0.0
                vy_new = 0.0
                total_weight = 0.0

                for weight, (dx, dy) in zip(weights, offsets):
                    nx, ny = grid_x + dx, grid_y + dy
                    if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                        vx_new += self.grid_velocity[ny, nx, 0] * weight
                        vy_new += self.grid_velocity[ny, nx, 1] * weight
                        total_weight += weight

                if total_weight > 0:
                    particle.vx = vx_new / total_weight
                    particle.vy = vy_new / total_weight

    def update_particles(self):
        """更新所有粒子"""
        for particle in self.particles:
            # 位置更新（速度已在G2P中更新）
            particle.update_position(self.dt)

            # 边界条件
            if particle.x < 0:
                particle.x = 0
                particle.vx = -particle.vx * 0.5
            elif particle.x >= self.grid_width:
                particle.x = self.grid_width - 1
                particle.vx = -particle.vx * 0.5

            if particle.y < 0:
                particle.y = 0
                particle.vy = -particle.vy * 0.5
            elif particle.y >= self.grid_height:
                particle.y = self.grid_height - 1
                particle.vy = -particle.vy * 0.5

            # 阻尼
            particle.vx *= self.damping
            particle.vy *= self.damping

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

    def step(self, grid: np.ndarray):
        """执行一步MPM模拟"""
        if not self.particles:
            return

        # P2G (包括变形梯度更新、应力计算、力和速度更新)
        self.particles_to_grid()

        # G2P (将网格信息映射回粒子)
        self.grid_to_particles()

        # 更新粒子位置和速度
        self.update_particles()

        # 更新向量场用于显示
        self.update_vector_field(grid)
