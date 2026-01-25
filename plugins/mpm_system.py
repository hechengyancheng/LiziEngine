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

    def __init__(self, app_core, grid_size: Tuple[int, int] = (64, 64)):
        self.app_core = app_core
        self.grid_width, self.grid_height = grid_size
        self.particles: List[MPMParticle] = []
        self.grid_mass = np.zeros((self.grid_height, self.grid_width))  # 网格质量
        self.grid_velocity = np.zeros((self.grid_height, self.grid_width, 2))  # 网格速度
        self.grid_force = np.zeros((self.grid_height, self.grid_width, 2))  # 网格力

        # MPM参数
        self.dt = 0.1  # 时间步长
        self.gravity = 0.01  # 重力
        self.young_modulus = 1000.0  # 杨氏模量
        self.poisson_ratio = 0.3  # 泊松比
        self.damping = 0.99  # 阻尼

    def add_particle(self, x: float, y: float, mass: float = 1.0, vx: float = 0.0, vy: float = 0.0):
        """添加物质点"""
        particle = MPMParticle(x, y, mass, vx, vy)
        self.particles.append(particle)

    def clear_particles(self):
        """清除所有物质点"""
        self.particles = []

    def particles_to_grid(self):
        """P2G: 将粒子信息映射到网格"""
        # 重置网格
        self.grid_mass.fill(0)
        self.grid_velocity.fill(0)
        self.grid_force.fill(0)

        for particle in self.particles:
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
            particle.update_velocity(self.dt, self.gravity)
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

        # P2G
        self.particles_to_grid()

        # G2P
        self.grid_to_particles()

        # 更新粒子
        self.update_particles()

        # 更新向量场用于显示
        self.update_vector_field(grid)