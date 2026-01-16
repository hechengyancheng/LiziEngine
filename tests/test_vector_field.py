import pytest
import numpy as np
from lizi_engine.compute.vector_field import VectorFieldCalculator


class TestVectorFieldCalculator:
    """测试向量场计算器"""

    def setup_method(self):
        """测试前准备"""
        self.calculator = VectorFieldCalculator()

    def test_create_vector_grid(self):
        """测试创建向量网格"""
        grid = self.calculator.create_vector_grid(10, 20, (1.0, 2.0))

        assert grid.shape == (20, 10, 2)  # height, width, 2
        assert np.all(grid == (1.0, 2.0))

    def test_sum_adjacent_vectors(self):
        """测试相邻向量求和"""
        grid = self.calculator.create_vector_grid(5, 5, (0.0, 0.0))
        # 设置一些测试向量
        grid[1, 1] = (1.0, 0.0)  # 中心
        grid[0, 1] = (0.0, 1.0)  # 上
        grid[2, 1] = (0.0, -1.0)  # 下
        grid[1, 0] = (-1.0, 0.0)  # 左
        grid[1, 2] = (2.0, 0.0)  # 右

        vx, vy = self.calculator.sum_adjacent_vectors(grid, 1, 1)

        # 预期结果：自身权重 * 自身 + 邻居权重 * (上+下+左+右)
        # 使用配置中的权重：self_weight=0.0, neighbor_weight=0.25
        expected_vx = 0.0 * 1.0 + 0.25 * (0.0 + 0.0 + (-1.0) + 2.0)
        expected_vy = 0.0 * 0.0 + 0.25 * (1.0 + (-1.0) + 0.0 + 0.0)

        assert abs(vx - expected_vx) < 1e-6
        assert abs(vy - expected_vy) < 1e-6

    def test_update_grid_with_adjacent_sum(self):
        """测试更新网格"""
        grid = self.calculator.create_vector_grid(3, 3, (0.0, 0.0))
        grid[1, 1] = (1.0, 1.0)

        original_grid = grid.copy()
        updated_grid = self.calculator.update_grid_with_adjacent_sum(grid)

        assert updated_grid.shape == original_grid.shape
        # 中心值应该不同（因为有邻居影响）
        assert not np.array_equal(updated_grid[1, 1], original_grid[1, 1])

    def test_create_radial_pattern(self):
        """测试创建径向模式"""
        grid = self.calculator.create_vector_grid(10, 10, (0.0, 0.0))
        center = (5.0, 5.0)
        radius = 3.0

        result = self.calculator.create_radial_pattern(grid, center, radius, 1.0)

        assert result.shape == grid.shape
        # 检查是否有向量被创建（至少有一些点不为零）
        has_vectors = np.any(result != 0.0)
        assert has_vectors, "径向模式应该创建一些向量"

    def test_create_tangential_pattern(self):
        """测试创建切线模式"""
        grid = self.calculator.create_vector_grid(10, 10, (0.0, 0.0))
        center = (5.0, 5.0)
        radius = 3.0

        result = self.calculator.create_tangential_pattern(grid, center, radius, 1.0)

        assert result.shape == grid.shape
        # 中心点应该有向量
        center_vx, center_vy = result[5, 5]
        assert center_vx != 0.0 or center_vy != 0.0

    def test_add_vector_at_position(self):
        """测试在位置添加向量"""
        grid = self.calculator.create_vector_grid(5, 5, (0.0, 0.0))

        self.calculator.add_vector_at_position(grid, 2.5, 2.5, 1.0, 1.0)

        # 检查周围的点是否受到影响
        affected = False
        for y in range(2, 4):
            for x in range(2, 4):
                if grid[y, x, 0] != 0.0 or grid[y, x, 1] != 0.0:
                    affected = True
                    break

        assert affected

    def test_fit_vector_at_position(self):
        """测试在位置拟合向量"""
        grid = self.calculator.create_vector_grid(5, 5, (0.0, 0.0))
        grid[2, 2] = (1.0, 1.0)
        grid[2, 3] = (1.0, 0.0)
        grid[3, 2] = (0.0, 1.0)
        grid[3, 3] = (0.0, 0.0)

        vx, vy = self.calculator.fit_vector_at_position(grid, 2.5, 2.5)

        assert isinstance(vx, float)
        assert isinstance(vy, float)


if __name__ == "__main__":
    pytest.main([__file__])
