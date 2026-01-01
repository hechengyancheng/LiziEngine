
# LiziEngine

![Licence](https://img.shields.io/badge/Licence-MIT-96d35f?style=flat)
![Version](https://img.shields.io/badge/Version-0.1.0Alpha-000000?style=flat)
![Language](https://img.shields.io/badge/Language-Python-f5ec00?style=flat)

## 目录

- [项目概述](#项目概述)
- [视频演示](#视频演示)
- [安装和运行](#安装和运行)
  - [系统要求](#系统要求)
  - [下载方式](#下载方式)
  - [安装依赖](#安装依赖)
  - [运行示例](#运行示例)
  - [键盘控制](#键盘控制)
- [向量场计算原理](#向量场计算原理)
  - [基本概念](#基本概念)
  - [计算方法](#计算方法)
  - [支持的向量场模式](#支持的向量场模式)
  - [CPU vs GPU 计算](#cpu-vs-gpu-计算)
- [架构设计](#架构设计)
  - [底层引擎 (lizi_engine/)](#底层引擎-lizi_engine)
  - [插件系统 (plugins/)](#插件系统-plugins)
- [配置说明](#配置说明)
- [更新日志](#更新日志)
- [常见问题](#常见问题)
- [许可证](#许可证)
- [联系方式](#联系方式)
- [贡献](#贡献)

## 项目概述

LiziEngine 是一个由向量场驱动的物理引擎，它通过模拟现实中的力场（如电磁力）来驱动实体运动，从而在根本上解决传统方法在处理碰撞检测的性能瓶颈。相比与每个实体互相遍历计算，本项目仅需让它们独立计算坐标下的向量值即可处理受力问题。

## 视频演示

[重力模拟与电荷模拟 【物理引擎开发日志#01】-哔哩哔哩](https://b23.tv/1Bj1y19)

[A Physics Engine Without Collision Detection -Youtube](https://youtube.com/shorts/vhOnfVl3CO8?si=xlLsO2X8CiU1n6YD)

## 安装和运行

### 系统要求

- Python 3.7+
- OpenGL 支持
- GPU 计算需要 OpenCL 运行时（可选）

### 下载方式

直接下载源码并解压，发行版里是打包的底层库。

### 安装依赖

```bash
pip install -r requirements.txt
# 或者使用 pip install lizi-engine 直接安装底层库
```

主要依赖包：
- numpy>=1.21.0 (数值计算)
- glfw>=2.5.0 (窗口管理)
- PyOpenGL>=3.1.6 (OpenGL 渲染)
- pyopencl>=2021.1.0 (GPU 计算，可选)
- pybind11>=2.6.0 (C++ 绑定)

### 运行示例

```bash
# 基本使用示例 - 展示核心功能
python examples/basic_usage.py

# 重力箱示例 - 物理模拟
python examples/gravity_box.py

# 模式示例 - 各种向量场模式
python examples/patterns.py

# 输入演示 - 交互功能展示
python examples/input_demo.py
```

### 键盘控制

- **空格键**: 重新生成切线模式
- **G键**: 切换网格显示
- **C键**: 清空网格
- **U键**: 切换实时更新
- **鼠标拖拽**: 平移视图
- **滚轮**: 缩放视图

## 向量场计算原理

### 基本概念

向量场是一个二维网格数据结构，每个网格点包含一个二维向量 `(vx, vy)`，表示该点的方向和强度。向量场的计算涉及相邻点的相互影响，形成复杂的动态模式。

### 计算方法

#### 1. 相邻向量求和
对于每个网格点，计算其自身和上下左右四个邻居向量的加权和：

```python
权重和 = 自身权重 + 4 × 邻居权重
'''
当权重和等于1时，影响范围无穷大，类似电荷力
当其小于1时，影响范围有限，类似支持力
'''
```

- **自身权重** (`vector_self_weight`): 控制自身向量对结果的影响
- **邻居权重** (`vector_neighbor_weight`): 控制邻居向量对结果的影响

#### 2. 向量场更新
迭代向量场的每个向量值，新向量=自身向量的权重积+邻居向量的权重积：

```python
# 向量场更新
vector_calculator.update_grid_with_adjacent_sum(grid)
```

#### 3. 双线性插值
在浮点坐标处添加或读取向量时，使用双线性插值进行平滑处理：

```python
# 在指定位置拟合一个向量
fit_vector_at_position(grid, x, y)

# 在指定位置添加一个向量
add_vector_at_position(grid, x, y, vx, vy)
```

### 支持的向量场模式

#### 微小向量添加
在指定位置创建局部影响，只影响当前位置及其上下左右邻居：

```python
create_tiny_vector(self, grid, x, y, mag)
```
#### 径向模式 (Radial Pattern)
向量从中心向外辐射，形成扩散效果（仅用于测试，只支持整数坐标）：

```python
create_radial_pattern(self, grid, center, radius, magnitude)
```

#### 切线模式 (Tangential Pattern)
向量围绕中心旋转，形成漩涡效果（仅用于测试，只支持整数坐标）：

```python
create_tangential_pattern(self, grid, center, radius, magnitude)
```

### CPU vs GPU 计算

- **CPU 计算**: 使用 NumPy 向量化操作，适合中小规模计算，易于调试
- **GPU 计算**: 使用 OpenCL 加速，大规模并行计算，性能显著提升，适合实时应用

## 架构设计

### 底层引擎 (lizi_engine/)

- **core**: 核心模块
- **compute**: 计算模块，由主接口统一cpu与gpu计算方式
- **graphics**: 渲染器模块
- **window**: 窗口管理器模块
- **input**: 输入管理器模块

### 插件系统 (plugins/)

- **controller.py**: 控制器插件，处理用户输入和向量场操作
- **marker_system.py**: 标记系统，支持在向量场上添加标记点
- **ui.py**: UI 管理器，处理界面交互和事件分发

## 配置说明

`config.json` 中的主要配置项：

```json
{
    "grid": {
        "width": 640,
        "height": 480,
        "color": [
            0.3,
            0.3,
            0.3
        ]
    },
    "cell": {
        "size": 2.0 //格子大小
    },
    "vector": {
        "color": [
            0.2,
            0.6,
            1.0
        ],
        "scale": 1.0, //向量长度缩放
        "self": {
            "weight": 0.2 //自身权重
        },
        "neighbor": {
            "weight": 0.2 //邻居权重
        }
    },
    "cam": {
        "x": 0.0,
        "y": 0.0,
        "zoom": 1.0
    },
    "show": {
        "grid": true //是否显示网格线
    },
    "background": {
        "color": [
            0.1,
            0.1,
            0.1
        ]
    },
    "antialiasing": true, //是否开启抗锯齿
    "line": {
        "width": 1.0 //线宽
    },
    "compute": {
        "device": "gpu", //计算设备，cpu或gpu
        "iterations": 1 //迭代次数
    },
    "render": {
        "vector": {
            "lines": false //是否显示向量线
        }
    }
}
```

## 更新日志

### v0.1.0 Alpha (2026-01-01)
- 初始发布版本
- 支持向量场驱动的物理模拟
- 实现 CPU 和 GPU 计算模式
- 提供基本插件系统
- 包含多个示例程序

## 常见问题

### Q: 如何切换 CPU 和 GPU 计算模式？
A: 在 `config.json` 中设置 `"compute": {"device": "cpu"}` 或 `"gpu"`。

### Q: 运行示例时出现 OpenGL 错误怎么办？
A: 确保您的系统支持 OpenGL，并安装了合适的图形驱动。

### Q: 如何添加自定义向量场模式？
A: 参考 `lizi_engine/compute/vector_field.py` 中的现有模式，实现新的计算函数。

### Q: 项目支持哪些操作系统？
A: 目前主要在 Windows 上测试，其他操作系统可能需要调整依赖。

## 许可证

[MIT License](LICENSE)

## 联系方式

2273902027@qq.com

## 贡献

本项目处于demo阶段，可能存在一些未发现的问题。
我们需要你的支持与协助！
如果你感兴趣，请务必点一个star让我们被更多人看到！

### 如何贡献

1. **报告问题**: 如果您发现bug或有功能建议，请在GitHub Issues中提交。
2. **提交代码**: Fork本仓库，创建功能分支，提交PR。
3. **文档改进**: 帮助完善文档或添加示例。
4. **测试**: 在不同平台上测试并报告兼容性问题。

### 开发环境设置

```bash
git clone https://github.com/hechengyancheng/LiziEngine.git
cd LiziEngine
pip install -r requirements.txt
```

### 代码规范

- 使用PEP 8风格
- 添加必要的注释和文档字符串
- 提交前进行测试

### API 文档

详细的API文档请参考 `doc/` 目录下的文档：
- [READEME.md](doc/README.md)





