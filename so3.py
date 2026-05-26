import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np

import numpy as np

def hat(omega):
    """向量 → 反对称矩阵 (so(3) 的标准形式)"""
    wx, wy, wz = omega
    return np.array([
        [0,   -wz,  wy],
        [wz,   0,  -wx],
        [-wy,  wx,  0]
    ])

def vee(Omega):
    """反对称矩阵 → 向量 (hat 的逆)"""
    return np.array([Omega[2,1], Omega[0,2], Omega[1,0]])

def exp_SO3(omega):
    """指数映射: so(3) → SO(3) (Rodrigues公式)"""
    theta = np.linalg.norm(omega)
    if theta < 1e-8:
        # 小角度：用泰勒展开避免数值问题
        return np.eye(3) + hat(omega)
    
    K = hat(omega / theta)  # 单位轴的反对称矩阵
    return np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*K @ K

def log_SO3(R):
    """对数映射: SO(3) → so(3)"""
    # 第1步：算角度 θ
    cos_theta = (np.trace(R) - 1) / 2
    cos_theta = np.clip(cos_theta, -1, 1)  # 数值安全
    theta = np.arccos(cos_theta)
    
    # 第2步：处理特殊情况
    if theta < 1e-8:
        # 几乎不旋转，返回零向量
        return np.zeros(3)
    
    if abs(theta - np.pi) < 1e-8:
        # 旋转180度，sin(θ) ≈ 0，需要特殊处理
        # 此时 R + I = 2 n̂ n̂^T，从对角元提取
        diag = np.diag(R)
        n = np.sqrt((diag + 1) / 2)
        # 还需要确定符号，略
        return theta * n
    
    # 第3步：一般情况
    Omega = (theta / (2*np.sin(theta))) * (R - R.T)
    return vee(Omega)


def slerp_SO3(R1, R2, t):
    """从 R1 到 R2 的球面线性插值"""
    # 第1步：算"差异旋转"
    delta_R = R1.T @ R2  # 等价于 R1⁻¹ R2
    
    # 第2步：取对数，进入李代数
    omega = log_SO3(delta_R)
    
    # 第3步：在李代数上线性插值
    omega_t = t * omega
    
    # 第4步：指数映射回李群
    delta_R_t = exp_SO3(omega_t)
    
    # 第5步：从 R1 出发应用这个插值后的旋转
    return R1 @ delta_R_t

# ===== 测试 =====
# 绕z轴转90度
# R = np.array([
#     [0, -1, 0],
#     [1,  0, 0],
#     [0,  0, 1]
# ])
# omega = log_SO3(R)
# print(f"log(R) = {omega}")  # 应该是 [0, 0, π/2] ≈ [0, 0, 1.5708]

# # 验证：exp(log(R)) == R
# R_recovered = exp_SO3(omega)
# print(f"误差: {np.linalg.norm(R - R_recovered)}")  # 应该接近0

def draw_frame(ax, R, origin=np.zeros(3), scale=1.0, alpha=1.0):
    """画一个坐标系（红x绿y蓝z）"""
    colors = ['r', 'g', 'b']
    for i in range(3):
        axis = R[:, i] * scale
        ax.quiver(*origin, *axis, color=colors[i], alpha=alpha)

# 起始和终止旋转
# R1 = np.eye(3)  # 不旋转
R1 = np.array([
    [0, -1, 0],
    [1,  0, 0],
    [0,  0, 1]
])
R2 = exp_SO3(np.array([np.pi/2, np.pi/2, 0]))  # 复杂旋转

# 准备动画
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

def update(frame):
    ax.clear()
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    
    t = frame / 100
    R_t = slerp_SO3(R1, R2, t)
    
    # 画当前旋转
    draw_frame(ax, R_t, scale=1.0)
    
    # 画起点和终点（淡色）
    draw_frame(ax, R1, scale=0.8, alpha=0.2)
    draw_frame(ax, R2, scale=0.8, alpha=0.2)
    
    ax.set_title(f't = {t:.2f}')

ani = FuncAnimation(fig, update, frames=101, interval=50)
plt.show()