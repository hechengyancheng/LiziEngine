import pytest
import numpy as np
from plugins.mpm_system import MPMSystem, MPMParticle


class MockAppCore:
    """Mock app core for testing"""
    pass


class TestMPMSystem:
    """测试MPM系统"""

    def setup_method(self):
        """测试前准备"""
        self.app_core = MockAppCore()
        self.mpm_system = MPMSystem(self.app_core, grid_size=(32, 32))

    def test_initialization(self):
        """测试初始化"""
        assert self.mpm_system.grid_width == 32
        assert self.mpm_system.grid_height == 32
        assert len(self.mpm_system.particles) == 0
        assert self.mpm_system.dt == 0.01
        assert self.mpm_system.gravity == 0.01
        assert self.mpm_system.young_modulus == 1000.0
        assert self.mpm_system.poisson_ratio == 0.3

    def test_add_particle(self):
        """测试添加粒子"""
        self.mpm_system.add_particle(10.0, 20.0, mass=2.0, vx=1.0, vy=-1.0)

        assert len(self.mpm_system.particles) == 1
        particle = self.mpm_system.particles[0]
        assert particle.x == 10.0
        assert particle.y == 20.0
        assert particle.mass == 2.0
        assert particle.vx == 1.0
        assert particle.vy == -1.0

    def test_clear_particles(self):
        """测试清除粒子"""
        self.mpm_system.add_particle(10.0, 20.0)
        self.mpm_system.add_particle(15.0, 25.0)

        assert len(self.mpm_system.particles) == 2

        self.mpm_system.clear_particles()
        assert len(self.mpm_system.particles) == 0

    def test_quadratic_bspline_weight(self):
        """测试二次B样条权重函数"""
        # 测试r=0
        assert abs(self.mpm_system.quadratic_bspline_weight(0.0) - 0.75) < 1e-6

        # 测试r=0.5
        assert abs(self.mpm_system.quadratic_bspline_weight(0.5) - 0.5) < 1e-6

        # 测试r=1.0
        assert abs(self.mpm_system.quadratic_bspline_weight(1.0) - 0.125) < 1e-6

        # 测试r=1.5
        assert self.mpm_system.quadratic_bspline_weight(1.5) == 0.0

        # 测试r=2.0
        assert self.mpm_system.quadratic_bspline_weight(2.0) == 0.0

    def test_quadratic_bspline_weight_derivative(self):
        """测试二次B样条权重函数导数"""
        # 测试r=0
        assert abs(self.mpm_system.quadratic_bspline_weight_derivative(0.0)) < 1e-6

        # 测试r=0.25
        assert abs(self.mpm_system.quadratic_bspline_weight_derivative(0.25) - (-0.5)) < 1e-6

        # 测试r=1.0
        assert abs(self.mpm_system.quadratic_bspline_weight_derivative(1.0) - (-0.5)) < 1e-6

    def test_get_interpolation_weights(self):
        """测试插值权重获取"""
        weights, offsets, grid_x, grid_y = self.mpm_system.get_interpolation_weights(10.5, 20.3)

        assert grid_x == 10
        assert grid_y == 20
        assert len(weights) == len(offsets)
        assert len(weights) <= 9  # 最多3x3网格

        # 权重和应该接近1（B样条性质）
        total_weight = sum(weights)
        assert abs(total_weight - 1.0) < 0.1  # 允许一些误差

    def test_compute_stress(self):
        """测试应力计算"""
        particle = MPMParticle(10.0, 20.0)
        particle.deformation = np.array([[1.1, 0.0], [0.0, 0.9]])  # 简单变形

        self.mpm_system.compute_stress(particle)

        assert particle.stress.shape == (2, 2)
        assert np.all(np.isfinite(particle.stress))

    def test_particles_to_grid(self):
        """测试粒子到网格映射"""
        self.mpm_system.add_particle(10.5, 20.3, mass=1.0, vx=1.0, vy=0.5)

        # 重置网格
        self.mpm_system.grid_mass.fill(0)
        self.mpm_system.grid_velocity.fill(0)

        self.mpm_system.particles_to_grid()

        # 检查是否有质量被映射
        total_mass = np.sum(self.mpm_system.grid_mass)
        assert total_mass > 0

        # 检查速度是否被映射
        has_velocity = np.any(self.mpm_system.grid_velocity != 0)
        assert has_velocity

    def test_grid_to_particles(self):
        """测试网格到粒子映射"""
        self.mpm_system.add_particle(10.5, 20.3)

        # 设置一些网格速度
        self.mpm_system.grid_velocity[20, 10] = [1.0, 0.5]
        self.mpm_system.grid_mass[20, 10] = 1.0

        self.mpm_system.grid_to_particles()

        particle = self.mpm_system.particles[0]
        # 粒子速度应该被更新
        assert particle.vx != 0.0 or particle.vy != 0.0

    def test_update_particles(self):
        """测试粒子更新"""
        self.mpm_system.add_particle(10.0, 20.0, vx=1.0, vy=0.5)

        original_x = self.mpm_system.particles[0].x
        original_y = self.mpm_system.particles[0].y

        self.mpm_system.update_particles()

        # 位置应该更新
        assert self.mpm_system.particles[0].x != original_x
        assert self.mpm_system.particles[0].y != original_y

    def test_boundary_conditions(self):
        """测试边界条件"""
        # 测试左边界
        self.mpm_system.add_particle(0.005, 16.0, vx=-1.0, vy=0.0)
        self.mpm_system.update_particles()

        particle = self.mpm_system.particles[0]
        assert particle.x >= 0.01  # 应该被推回边界内
        assert particle.vx >= 0  # 速度应该反转

        # 清除粒子
        self.mpm_system.clear_particles()

        # 测试右边界
        self.mpm_system.add_particle(31.995, 16.0, vx=1.0, vy=0.0)
        self.mpm_system.update_particles()

        particle = self.mpm_system.particles[0]
        assert particle.x <= 31.99  # 应该被推回边界内
        assert particle.vx <= 0  # 速度应该反转

    def test_step(self):
        """测试完整模拟步骤"""
        self.mpm_system.add_particle(16.0, 16.0)

        grid = np.zeros((32, 32, 2))

        # 执行一步
        self.mpm_system.step(grid)

        # 检查向量场是否被更新
        has_vectors = np.any(grid != 0.0)
        assert has_vectors

    def test_cfl_condition(self):
        """测试CFL条件检查"""
        self.mpm_system.add_particle(16.0, 16.0, vx=5.0, vy=0.0)

        original_dt = self.mpm_system.dt
        self.mpm_system.check_cfl_condition()

        # dt应该在合理范围内
        assert self.mpm_system.min_dt <= self.mpm_system.dt <= self.mpm_system.max_dt


if __name__ == "__main__":
    pytest.main([__file__])
