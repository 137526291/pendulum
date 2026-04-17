"""
2D 倒立摆 (Inverted Pendulum) 仿真程序
========================================

物理模型:
  - 小车 (cart) 在水平轨道上运动, 质量 M
  - 摆杆 (pendulum) 铰接在小车顶部, 质量 m, 长度 L (质心到铰接点距离 l = L/2)
  - θ = 0 表示摆杆竖直向上 (不稳定平衡点)

动力学方程 (拉格朗日推导):
  设广义坐标 q = [x, θ]

  拉格朗日量 L = T - V
    T = ½(M+m)ẋ² + m·l·ẋ·θ̇·cos(θ) + ½·m·l²·θ̇²
    V = m·g·l·cos(θ)

  对 x 的 Euler-Lagrange 方程:
    (M+m)·ẍ + m·l·θ̈·cos(θ) - m·l·θ̇²·sin(θ) = F - b·ẋ

  对 θ 的 Euler-Lagrange 方程:
    m·l²·θ̈ + m·l·ẍ·cos(θ) - m·g·l·sin(θ) = 0

  解联立方程得加速度:
    Δ = (M+m)·m·l² - (m·l·cos(θ))²

    ẍ  = [ m·l²·(F - b·ẋ + m·l·θ̇²·sin(θ)) + (m·l·cos(θ))·(m·g·l·sin(θ)) ] / Δ
    θ̈  = [ (M+m)·(m·g·l·sin(θ)) - m·l·cos(θ)·(F - b·ẋ + m·l·θ̇²·sin(θ)) ] / Δ

状态空间方程 (在 θ≈0 线性化):
  状态向量 X = [x, ẋ, θ, θ̇]ᵀ
  cos(θ)≈1, sin(θ)≈θ, θ̇²≈0

        ┌                        ┐       ┌              ┐
        │ 0  1  0  0             │       │     0        │
  Ẋ =   │ 0 -b/M  -mg/M  0       │ X +   │   1/M        │ F
        │ 0  0  0  1             │       │     0        │
        │ 0  b/(Ml) (M+m)g/(Ml) 0│       │ -1/(Ml)      │
        └                        ┘       └              ┘

  其中简化假设 Δ₀ = M·m·l² (θ=0 时)

操作说明:
  左/右方向键  → 施加水平力 F
  空格键       → 重置仿真
  1 / 2       → 切换求解方法 (1=差分法 Euler, 2=状态空间 LQR)
  ESC / 关闭窗口 → 退出
"""

import numpy as np
import pygame
import sys
from scipy import linalg

# ============================================================
#  物理参数 (SI 单位)
# ============================================================
M = 1.0        # 小车质量 (kg)
m = 0.3        # 摆杆质量 (kg)
L = 1.0        # 摆杆全长 (m)
l = L / 2      # 摆杆质心到铰接点距离 (m)
g = 9.81       # 重力加速度 (m/s²)
b = 0.1        # 小车摩擦阻尼系数 (N·s/m)
F_mag = 30.0   # 键盘施加的力幅值 (N)

# ============================================================
#  仿真参数
# ============================================================
dt = 0.005              # 积分步长 (s)
SUB_STEPS = 4           # 每帧的积分子步数
FPS = 60                # 渲染帧率

# ============================================================
#  显示参数
# ============================================================
SCREEN_W, SCREEN_H = 1000, 600
SCALE = 150             # 物理米 → 像素比例
CART_W, CART_H = 80, 40 # 小车绘制尺寸 (px)
TRACK_Y = SCREEN_H * 0.65  # 轨道 y 坐标 (px)

# 颜色
WHITE      = (255, 255, 255)
BG_COLOR   = (30,  30,  40)
CART_COLOR  = (70, 130, 210)
POLE_COLOR  = (230, 180, 50)
PIVOT_COLOR = (255, 100, 80)
TRACK_COLOR = (80,  80,  100)
TEXT_COLOR  = (200, 200, 200)
FORCE_COLOR = (100, 255, 120)


# ============================================================
#  非线性动力学: 计算加速度
# ============================================================
def compute_accelerations(state: np.ndarray, F_ext: float) -> tuple[float, float]:
    """
    根据当前状态和外力，求解非线性运动方程，返回 (ẍ, θ̈).

    state = [x, x_dot, theta, theta_dot]
    F_ext = 施加在小车上的水平外力 (N)

    推导见文件顶部的动力学方程。
    """
    _, x_dot, theta, theta_dot = state

    cos_th = np.cos(theta)
    sin_th = np.sin(theta)

    # 分母 (惯性矩阵行列式)
    #   Δ = (M+m)·m·l² - (m·l·cosθ)²
    delta = (M + m) * m * l**2 - (m * l * cos_th)**2

    # 合力项 (含摩擦和离心力)
    F_eff = F_ext - b * x_dot + m * l * theta_dot**2 * sin_th

    # ẍ = [ m·l²·F_eff + m²·l²·g·sinθ·cosθ ] / Δ
    x_ddot = (m * l**2 * F_eff + m**2 * l**2 * g * sin_th * cos_th) / delta

    # θ̈ = [ (M+m)·m·g·l·sinθ - m·l·cosθ·F_eff ] / Δ
    theta_ddot = ((M + m) * m * g * l * sin_th - m * l * cos_th * F_eff) / delta

    return x_ddot, theta_ddot


# ============================================================
#  方法 1: 显式欧拉差分法 (Euler forward)
# ============================================================
def step_euler(state: np.ndarray, F_ext: float, dt_step: float) -> np.ndarray:
    """
    用一阶显式欧拉法对非线性 ODE 做一步积分.

    Xₙ₊₁ = Xₙ + dt · f(Xₙ, u)
    """
    x, x_dot, theta, theta_dot = state
    x_ddot, theta_ddot = compute_accelerations(state, F_ext)

    x_new        = x         + x_dot      * dt_step
    x_dot_new    = x_dot     + x_ddot     * dt_step
    theta_new    = theta     + theta_dot   * dt_step
    theta_dot_new = theta_dot + theta_ddot * dt_step

    return np.array([x_new, x_dot_new, theta_new, theta_dot_new])


# ============================================================
#  方法 1b: 四阶 Runge-Kutta (更精确的差分法)
# ============================================================
def state_derivative(state: np.ndarray, F_ext: float) -> np.ndarray:
    """状态导数向量 f(X, u) = [ẋ, ẍ, θ̇, θ̈]"""
    x_ddot, theta_ddot = compute_accelerations(state, F_ext)
    return np.array([state[1], x_ddot, state[3], theta_ddot])


def step_rk4(state: np.ndarray, F_ext: float, dt_step: float) -> np.ndarray:
    """
    经典四阶 Runge-Kutta 积分.
    比 Euler 法精度高很多, 但计算量为 4 倍。
    """
    k1 = state_derivative(state, F_ext)
    k2 = state_derivative(state + 0.5 * dt_step * k1, F_ext)
    k3 = state_derivative(state + 0.5 * dt_step * k2, F_ext)
    k4 = state_derivative(state + dt_step * k3, F_ext)
    return state + (dt_step / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# ============================================================
#  方法 2: 状态空间线性化 + LQR 控制器
# ============================================================
def build_state_space() -> tuple[np.ndarray, np.ndarray]:
    """
    在 θ=0 (竖直向上) 处线性化, 构造状态空间矩阵 A, B.

    状态 X = [x, ẋ, θ, θ̇]ᵀ,  输入 u = F

    线性化后:
      cosθ≈1, sinθ≈θ, θ̇²·sinθ≈0

    解联立方程:
      (M+m)ẍ + mlθ̈ = F - bẋ          ... (1)
      ml²θ̈ + mlẍ = mglθ              ... (2)

    由 (2): ẍ = lθ̈ - gθ  (代入后求解)

    最终:
      Δ₀ = Ml  (简化后的等效惯量)

      ẍ  = (-b·ẋ + m·g·θ + F) / M      (近似, M >> m·cos²θ 项)
      θ̈  = (-b·ẋ/(Ml) + (M+m)gθ/(Ml) - F/(Ml))   (精确线性化)
    """
    denom = M * l

    A = np.array([
        [0, 1,               0,              0],
        [0, -b/M,            m*g/M,          0],
        [0, 0,               0,              1],
        [0, b/denom,  (M+m)*g/denom,         0],
    ])

    B = np.array([
        [0],
        [1/M],
        [0],
        [-1/denom],
    ])

    return A, B


def compute_lqr_gain(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    求解连续时间 LQR 问题:

    最小化 J = ∫(XᵀQX + uᵀRu) dt

    通过求解代数 Riccati 方程:
      AᵀP + PA - PBR⁻¹BᵀP + Q = 0

    最优增益:
      K = R⁻¹BᵀP
    """
    # 代价权重矩阵 — 对位置/角度偏差惩罚较大
    Q = np.diag([1.0, 1.0, 50.0, 10.0])   # [x, ẋ, θ, θ̇] 权重
    R = np.array([[0.01]])                  # 控制力代价 (小 → 允许较大力)

    P = linalg.solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    return K.flatten()  # shape (4,)


# 预计算 LQR 增益
A_ss, B_ss = build_state_space()
K_lqr = compute_lqr_gain(A_ss, B_ss)


# ============================================================
#  仿真状态
# ============================================================
def initial_state() -> np.ndarray:
    """返回初始状态: 小车居中, 摆杆略偏 (便于观察控制效果)"""
    return np.array([0.0, 0.0, 0.15, 0.0])  # [x, ẋ, θ, θ̇]


# ============================================================
#  Pygame 渲染
# ============================================================
def world_to_screen(wx: float, wy: float) -> tuple[int, int]:
    """世界坐标 (m) → 屏幕像素坐标"""
    sx = int(SCREEN_W / 2 + wx * SCALE)
    sy = int(TRACK_Y - wy * SCALE)
    return sx, sy


def draw_scene(screen: pygame.Surface, state: np.ndarray,
               F_ext: float, method_name: str, font: pygame.font.Font):
    """绘制整个场景: 轨道、小车、摆杆、力箭头、信息文字"""
    screen.fill(BG_COLOR)

    x, x_dot, theta, theta_dot = state

    # --- 轨道 ---
    pygame.draw.line(screen, TRACK_COLOR,
                     (0, int(TRACK_Y) + CART_H // 2),
                     (SCREEN_W, int(TRACK_Y) + CART_H // 2), 3)

    # --- 小车 ---
    cart_sx, cart_sy = world_to_screen(x, 0)
    cart_rect = pygame.Rect(0, 0, CART_W, CART_H)
    cart_rect.center = (cart_sx, cart_sy)
    pygame.draw.rect(screen, CART_COLOR, cart_rect, border_radius=6)

    # 车轮
    wheel_r = 8
    pygame.draw.circle(screen, (50, 50, 60),
                       (cart_rect.left + 15, cart_rect.bottom), wheel_r)
    pygame.draw.circle(screen, (50, 50, 60),
                       (cart_rect.right - 15, cart_rect.bottom), wheel_r)

    # --- 摆杆 ---
    pivot = world_to_screen(x, 0)
    pole_end_x = x + L * np.sin(theta)
    pole_end_y = L * np.cos(theta)
    pole_end = world_to_screen(pole_end_x, pole_end_y)

    pygame.draw.line(screen, POLE_COLOR, pivot, pole_end, 6)
    pygame.draw.circle(screen, PIVOT_COLOR, pivot, 7)

    # 摆杆末端小球
    pygame.draw.circle(screen, (255, 220, 100), pole_end, 10)

    # --- 外力箭头 ---
    if abs(F_ext) > 0.1:
        arrow_len = int(np.sign(F_ext) * 40)
        arrow_start = (cart_sx - arrow_len, cart_sy)
        arrow_end   = (cart_sx, cart_sy)
        pygame.draw.line(screen, FORCE_COLOR, arrow_start, arrow_end, 4)
        # 箭头头部
        tip_dir = int(np.sign(F_ext))
        pygame.draw.polygon(screen, FORCE_COLOR, [
            (cart_sx + tip_dir * 5, cart_sy),
            (cart_sx - tip_dir * 8, cart_sy - 6),
            (cart_sx - tip_dir * 8, cart_sy + 6),
        ])

    # --- 信息面板 ---
    info_lines = [
        f"Method: {method_name}",
        f"x = {x:+.3f} m    dx/dt = {x_dot:+.3f} m/s",
        f"theta = {np.degrees(theta):+.2f} deg   d(theta)/dt = {np.degrees(theta_dot):+.2f} deg/s",
        f"F = {F_ext:+.1f} N",
        "",
        "L/R: force | Space: reset | 1: diff | 2: SS(LQR) | ESC: quit",
    ]
    for i, line in enumerate(info_lines):
        surf = font.render(line, True, TEXT_COLOR)
        screen.blit(surf, (15, 10 + i * 24))


def _load_chinese_font(size: int) -> pygame.font.Font:
    """
    按优先级尝试加载支持中文的字体文件.
    macOS 系统字体路径可能因版本不同而变化, 逐个尝试。
    """
    import os
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return pygame.font.Font(path, size)
            except Exception:
                continue
    # 最终回退: 用 SysFont 尝试几个名字
    for name in ["STHeiti", "SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei"]:
        f = pygame.font.SysFont(name, size)
        if f:
            return f
    return pygame.font.SysFont(None, size)


# ============================================================
#  主循环
# ============================================================
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("2D Inverted Pendulum - Euler/RK4 & State-Space LQR")
    clock = pygame.time.Clock()

    font = _load_chinese_font(18)

    state = initial_state()
    F_ext = 0.0

    # 求解方法: "euler" / "ss"
    method = "euler"
    method_labels = {"euler": "什么RK4 Nonlinear (Manual)", "ss": "State-Space LQR (Auto)"}

    running = True
    while running:
        # ---- 事件处理 ----
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    state = initial_state()
                elif event.key == pygame.K_1:
                    method = "euler"
                elif event.key == pygame.K_2:
                    method = "ss"

        # ---- 键盘持续检测 (左右方向键) ----
        keys = pygame.key.get_pressed()
        manual_force = 0.0
        if keys[pygame.K_LEFT]:
            manual_force -= F_mag
        if keys[pygame.K_RIGHT]:
            manual_force += F_mag

        # ---- 物理积分 (多子步提高稳定性) ----
        for _ in range(SUB_STEPS):
            if method == "euler":
                # 差分法: 人工施力, 用 RK4 积分非线性方程
                F_ext = manual_force
                state = step_rk4(state, F_ext, dt)
            else:
                # 状态空间: LQR 自动控制 + 手动叠加
                # u = -K·X (负反馈) + 手动力
                F_lqr = -K_lqr @ state
                F_ext = float(F_lqr) + manual_force
                # 仍用非线性方程积分 (控制器是线性的, 被控对象是非线性的)
                state = step_rk4(state, F_ext, dt)

        # 限制小车位置避免跑出屏幕
        x_limit = (SCREEN_W / 2) / SCALE * 0.9
        if abs(state[0]) > x_limit:
            state[0] = np.clip(state[0], -x_limit, x_limit)
            state[1] = 0.0

        # ---- 渲染 ----
        draw_scene(screen, state, F_ext, method_labels[method], font)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
