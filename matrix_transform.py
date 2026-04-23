"""
Interactive 2D Matrix Transformation Visualizer
================================================

A teaching tool for linear algebra that visualizes:

  1. det(A)  — The determinant as the signed area of the transformed unit square.
               |det(A)| = area scale factor; sign = orientation (flip or not).

  2. Eigenvalues & Eigenvectors — Real eigenvalues shown as special directions
               that only get scaled (not rotated) by A.
               A v = lambda v  =>  the eigenvector v stays on its span.

  3. A x     — Matrix-vector multiplication shown as mapping an input vector x
               to its image Ax, with both drawn on the same coordinate system.

  4. Grid deformation — The original square grid is drawn alongside the
               transformed grid to show how A warps space (shear, rotation,
               scaling, reflection).

Controls:
  Matrix editing:
    Q/W  — adjust a11 (top-left entry)     E/R — adjust a12 (top-right)
    A/S  — adjust a21 (bottom-left)         D/F — adjust a22 (bottom-right)
    (lowercase = -0.1, uppercase/right key = +0.1)

  Input vector x:
    Click and drag in the viewport to set the input vector x interactively.

  Presets (number keys):
    1 — Identity          I = [[1,0],[0,1]]
    2 — Rotation 45deg
    3 — Shear             [[1,1],[0,1]]
    4 — Reflection (y)    [[-1,0],[0,1]]
    5 — Scaling           [[2,0],[0,0.5]]
    6 — Singular          [[1,2],[2,4]]  (det=0)
    7 — Projection        [[1,0],[0,0]]

  Other:
    SPACE — reset to identity
    ESC   — quit

    可视化内容：

视觉元素	含义
灰色网格	原始坐标网格（单位矩阵 I）
蓝色网格	经过矩阵 A 变换后的网格，直观展示空间如何被拉伸/旋转/剪切
绿色小方块	原始单位正方形 [0,1]×[0,1]
红/蓝色平行四边形	变换后的单位正方形，面积 = |det(A)|；红色=方向保持，蓝色=方向翻转
黄/青色线 + 箭头	特征向量及其像 Av = λv，只在实特征值时显示
绿色箭头 x	你设定的输入向量
粉色箭头 Ax	矩阵作用后的输出向量
操作方式：

鼠标拖拽 — 实时设置输入向量 x，观察 Ax 如何跟随变化
Q/W, E/R, A/S, D/F — 逐步调整矩阵四个元素（每次 ±0.1），实时看到网格变形
数字键 1-7 — 7 个经典预设矩阵（单位阵、旋转、剪切、反射、缩放、奇异矩阵、投影）
左上角信息面板 — 实时显示矩阵值、det(A)、特征值、特征方程 λ²-tr·λ+det=0、向量模长
教学时可以让学生：

按 6 加载奇异矩阵，观察 det=0 时平行四边形退化为线段（降维）
按 2 加载旋转矩阵，观察特征值变为复数（没有实特征向量）
手动调整矩阵，观察 det 符号翻转时平行四边形颜色从红变蓝（方向反转）
"""

import numpy as np
import pygame
import sys
import os

# ============================================================
#  Display constants
# ============================================================
SCREEN_W, SCREEN_H = 1100, 800
ORIGIN_X, ORIGIN_Y = SCREEN_W // 2, SCREEN_H // 2
SCALE = 80  # pixels per unit

# ============================================================
#  Colors
# ============================================================
BG           = (18, 18, 28)
GRID_ORIG    = (45, 45, 65)       # original grid (faint)
GRID_TRANS   = (40, 90, 140)      # transformed grid
AXIS_COLOR   = (80, 80, 110)
UNIT_SQ      = (60, 180, 120, 90) # original unit square (with alpha)
DET_FILL     = (220, 80, 80, 70)  # transformed unit square fill
DET_OUTLINE  = (240, 100, 100)
EIGEN_COLORS = [(255, 200, 60), (100, 220, 255)]  # two eigenvectors
VEC_X_COLOR  = (120, 220, 120)    # input vector x
VEC_AX_COLOR = (255, 120, 180)    # output vector Ax
TEXT_COLOR   = (210, 210, 210)
LABEL_BG     = (30, 30, 45, 200)
HELP_COLOR   = (140, 140, 160)
ENTRY_HI     = (255, 255, 100)    # highlighted matrix entry


# ============================================================
#  Helper: load a font that supports basic math/ASCII well
# ============================================================
def load_font(size: int) -> pygame.font.Font:
    candidates = [
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/SFMono-Regular.otf",
        "/System/Library/Fonts/Monaco.dfont",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return pygame.font.Font(p, size)
            except Exception:
                continue
    return pygame.font.SysFont("monospace", size)


# ============================================================
#  Coordinate transforms
# ============================================================
def world_to_screen(wx: float, wy: float) -> tuple[int, int]:
    return int(ORIGIN_X + wx * SCALE), int(ORIGIN_Y - wy * SCALE)


def screen_to_world(sx: int, sy: int) -> tuple[float, float]:
    return (sx - ORIGIN_X) / SCALE, (ORIGIN_Y - sy) / SCALE


# ============================================================
#  Drawing primitives
# ============================================================
def draw_arrow(surface: pygame.Surface, color, start_w, end_w,
               width: int = 3, head_size: int = 10):
    """Draw an arrow from world-coord start to end."""
    s = world_to_screen(*start_w)
    e = world_to_screen(*end_w)
    dx, dy = e[0] - s[0], e[1] - s[1]
    length = (dx*dx + dy*dy) ** 0.5
    if length < 2:
        return
    pygame.draw.line(surface, color, s, e, width)
    # arrowhead
    ux, uy = dx / length, dy / length
    px, py = -uy, ux  # perpendicular
    tip = e
    left  = (int(tip[0] - head_size * ux + head_size * 0.4 * px),
             int(tip[1] - head_size * uy + head_size * 0.4 * py))
    right = (int(tip[0] - head_size * ux - head_size * 0.4 * px),
             int(tip[1] - head_size * uy - head_size * 0.4 * py))
    pygame.draw.polygon(surface, color, [tip, left, right])


def draw_grid(surface: pygame.Surface, A: np.ndarray, color, alpha: int = 255):
    """Draw a transformed grid: maps integer grid lines through matrix A."""
    extent = 6
    for i in range(-extent, extent + 1):
        pts_h, pts_v = [], []
        for t_val in np.linspace(-extent, extent, 80):
            # horizontal line at y=i
            ph = A @ np.array([t_val, i])
            pts_h.append(world_to_screen(ph[0], ph[1]))
            # vertical line at x=i
            pv = A @ np.array([i, t_val])
            pts_v.append(world_to_screen(pv[0], pv[1]))
        if len(pts_h) > 1:
            pygame.draw.lines(surface, color, False, pts_h, 1)
        if len(pts_v) > 1:
            pygame.draw.lines(surface, color, False, pts_v, 1)


def draw_parallelogram(surface: pygame.Surface, A: np.ndarray):
    """
    Draw the image of the unit square [0,1]x[0,1] under A.
    Area of this parallelogram = |det(A)|.
    """
    corners_world = [np.array([0, 0]), np.array([1, 0]),
                     np.array([1, 1]), np.array([0, 1])]
    corners_transformed = [A @ c for c in corners_world]
    pts = [world_to_screen(c[0], c[1]) for c in corners_transformed]

    # semi-transparent fill
    overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
    det_val = np.linalg.det(A)
    fill_color = (220, 80, 80, 60) if det_val >= 0 else (80, 80, 220, 60)
    pygame.draw.polygon(overlay, fill_color, pts)
    surface.blit(overlay, (0, 0))

    # outline
    outline_color = DET_OUTLINE if det_val >= 0 else (100, 100, 240)
    pygame.draw.polygon(surface, outline_color, pts, 2)

    # original unit square (faint green)
    orig_pts = [world_to_screen(c[0], c[1]) for c in corners_world]
    overlay2 = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
    pygame.draw.polygon(overlay2, UNIT_SQ, orig_pts)
    surface.blit(overlay2, (0, 0))
    pygame.draw.polygon(surface, (60, 180, 120), orig_pts, 1)


def draw_eigenvectors(surface: pygame.Surface, A: np.ndarray, font: pygame.font.Font):
    """
    Compute and draw eigenvectors for real eigenvalues.

    For A v = lambda v, the eigenvector v is drawn as a line through the origin
    (its span), and both v and A v are shown to demonstrate scaling.
    """
    try:
        eigenvalues, eigenvectors = np.linalg.eig(A)
    except np.linalg.LinAlgError:
        return

    for i in range(2):
        lam = eigenvalues[i]
        if np.iscomplex(lam):
            continue
        lam = float(np.real(lam))
        v = np.real(eigenvectors[:, i])
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-10:
            continue
        v = v / v_norm  # normalize

        color = EIGEN_COLORS[i]

        # draw the span (dashed-style: a long line through origin)
        far = 8
        p1 = (-far * v[0], -far * v[1])
        p2 = (far * v[0], far * v[1])
        s1, s2 = world_to_screen(*p1), world_to_screen(*p2)
        dash_color = (color[0]//2, color[1]//2, color[2]//2)
        pygame.draw.line(surface, dash_color, s1, s2, 1)

        # draw eigenvector v (unit length)
        draw_arrow(surface, color, (0, 0), (v[0], v[1]), width=3, head_size=9)

        # draw A v = lambda * v
        av = A @ v
        draw_arrow(surface, color, (0, 0), (av[0], av[1]), width=2, head_size=8)

        # label
        label_pos = world_to_screen(v[0] * 1.3, v[1] * 1.3)
        txt = font.render(f"lam={lam:+.2f}", True, color)
        surface.blit(txt, (label_pos[0] + 5, label_pos[1] - 10))


# ============================================================
#  Info panel rendering
# ============================================================
def draw_info_panel(surface: pygame.Surface, A: np.ndarray,
                    x_vec: np.ndarray, font: pygame.font.Font,
                    font_sm: pygame.font.Font):
    """Draw the matrix, det, eigenvalues, and vector info."""
    det_val = np.linalg.det(A)

    try:
        eigenvalues = np.linalg.eigvals(A)
    except np.linalg.LinAlgError:
        eigenvalues = np.array([float('nan'), float('nan')])

    ax_vec = A @ x_vec
    trace_val = np.trace(A)

    # background panel
    panel = pygame.Surface((320, 370), pygame.SRCALPHA)
    panel.fill((20, 20, 35, 210))
    surface.blit(panel, (10, 10))

    y = 18
    gap = 22

    def put(text, color=TEXT_COLOR, x_off=20, f=font):
        nonlocal y
        s = f.render(text, True, color)
        surface.blit(s, (x_off, y))
        y += gap

    put("  MATRIX  A", (180, 180, 220))
    put(f"  [{A[0,0]:+6.2f}  {A[0,1]:+6.2f} ]")
    put(f"  [{A[1,0]:+6.2f}  {A[1,1]:+6.2f} ]")
    y += 4

    # Determinant with color coding
    det_color = (255, 100, 100) if det_val < 0 else (100, 255, 130) if det_val > 0.01 else (200, 200, 80)
    put(f"  det(A) = {det_val:+.4f}", det_color)
    if abs(det_val) < 1e-8:
        put("    => SINGULAR (area=0)", (255, 200, 80))
    else:
        put(f"    area scale = |det| = {abs(det_val):.4f}")
        orient = "preserved" if det_val > 0 else "FLIPPED"
        put(f"    orientation: {orient}", det_color)
    y += 4

    put(f"  tr(A) = {trace_val:+.4f}")
    y += 2

    # Eigenvalues
    put("  EIGENVALUES", (180, 180, 220))
    for i, lam in enumerate(eigenvalues):
        c = EIGEN_COLORS[i]
        if np.iscomplex(lam):
            put(f"    lam{i+1} = {lam.real:+.3f} {lam.imag:+.3f}i  (complex)", c)
        else:
            put(f"    lam{i+1} = {np.real(lam):+.4f}", c)
    # characteristic equation: lam^2 - tr*lam + det = 0
    put(f"    lam^2 - ({trace_val:+.2f})lam + ({det_val:+.2f}) = 0", HELP_COLOR, f=font_sm)
    y += 4

    # Vectors
    put("  VECTORS", (180, 180, 220))
    put(f"    x  = ({x_vec[0]:+.2f}, {x_vec[1]:+.2f})", VEC_X_COLOR)
    put(f"    Ax = ({ax_vec[0]:+.2f}, {ax_vec[1]:+.2f})", VEC_AX_COLOR)
    put(f"    |x|={np.linalg.norm(x_vec):.2f}  |Ax|={np.linalg.norm(ax_vec):.2f}")


def draw_controls_help(surface: pygame.Surface, font: pygame.font.Font):
    """Draw keyboard controls reference at bottom."""
    lines = [
        "Q/W: a11   E/R: a12   A/S: a21   D/F: a22   (-/+0.1)",
        "Drag mouse: set x | 1-7: presets | SPACE: reset | ESC: quit",
    ]
    for i, line in enumerate(lines):
        s = font.render(line, True, HELP_COLOR)
        surface.blit(s, (15, SCREEN_H - 50 + i * 22))


# ============================================================
#  Preset matrices
# ============================================================
PRESETS = {
    pygame.K_1: ("Identity",        np.eye(2)),
    pygame.K_2: ("Rotation 45deg",  np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)],
                                               [np.sin(np.pi/4),  np.cos(np.pi/4)]])),
    pygame.K_3: ("Shear",           np.array([[1, 1], [0, 1]], dtype=float)),
    pygame.K_4: ("Reflect-y",       np.array([[-1, 0], [0, 1]], dtype=float)),
    pygame.K_5: ("Scale 2x, 0.5y",  np.array([[2, 0], [0, 0.5]])),
    pygame.K_6: ("Singular",        np.array([[1, 2], [2, 4]], dtype=float)),
    pygame.K_7: ("Projection x",    np.array([[1, 0], [0, 0]], dtype=float)),
}

# Matrix entry adjustment key mapping: (key_minus, key_plus) -> (row, col)
ENTRY_KEYS = {
    (pygame.K_q, pygame.K_w): (0, 0),
    (pygame.K_e, pygame.K_r): (0, 1),
    (pygame.K_a, pygame.K_s): (1, 0),
    (pygame.K_d, pygame.K_f): (1, 1),
}


# ============================================================
#  Main loop
# ============================================================
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Linear Algebra: det(A), Eigenvalues, and Ax")
    clock = pygame.time.Clock()

    font    = load_font(16)
    font_sm = load_font(13)

    A = np.eye(2, dtype=float)
    x_vec = np.array([1.0, 0.5])
    dragging = False

    running = True
    while running:
        # ---- Events ----
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    A = np.eye(2, dtype=float)
                elif event.key in PRESETS:
                    _, A = PRESETS[event.key]
                    A = A.copy()
                else:
                    for (k_minus, k_plus), (r, c) in ENTRY_KEYS.items():
                        if event.key == k_minus:
                            A[r, c] -= 0.1
                        elif event.key == k_plus:
                            A[r, c] += 0.1

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    dragging = True
                    wx, wy = screen_to_world(*event.pos)
                    x_vec = np.array([wx, wy])

            elif event.type == pygame.MOUSEBUTTONUP:
                dragging = False

            elif event.type == pygame.MOUSEMOTION:
                if dragging:
                    wx, wy = screen_to_world(*event.pos)
                    x_vec = np.array([wx, wy])

        # ---- Draw ----
        screen.fill(BG)

        # Axes
        pygame.draw.line(screen, AXIS_COLOR, (0, ORIGIN_Y), (SCREEN_W, ORIGIN_Y), 1)
        pygame.draw.line(screen, AXIS_COLOR, (ORIGIN_X, 0), (ORIGIN_X, SCREEN_H), 1)

        # Axis tick marks and labels
        for i in range(-7, 8):
            if i == 0:
                continue
            sx, sy = world_to_screen(i, 0)
            pygame.draw.line(screen, AXIS_COLOR, (sx, sy - 4), (sx, sy + 4), 1)
            lbl = font_sm.render(str(i), True, (70, 70, 90))
            screen.blit(lbl, (sx - 4, sy + 6))
            sx2, sy2 = world_to_screen(0, i)
            pygame.draw.line(screen, AXIS_COLOR, (sx2 - 4, sy2), (sx2 + 4, sy2), 1)
            lbl2 = font_sm.render(str(i), True, (70, 70, 90))
            screen.blit(lbl2, (sx2 + 8, sy2 - 6))

        # Original grid (identity)
        draw_grid(screen, np.eye(2), GRID_ORIG)

        # Transformed grid
        draw_grid(screen, A, GRID_TRANS)

        # Unit square -> parallelogram (det visualization)
        draw_parallelogram(screen, A)

        # Eigenvectors
        draw_eigenvectors(screen, A, font)

        # Input vector x
        draw_arrow(screen, VEC_X_COLOR, (0, 0), (x_vec[0], x_vec[1]),
                   width=3, head_size=11)
        lbl_x = font.render("x", True, VEC_X_COLOR)
        sx, sy = world_to_screen(x_vec[0], x_vec[1])
        screen.blit(lbl_x, (sx + 8, sy - 18))

        # Output vector Ax
        ax_vec = A @ x_vec
        draw_arrow(screen, VEC_AX_COLOR, (0, 0), (ax_vec[0], ax_vec[1]),
                   width=3, head_size=11)
        lbl_ax = font.render("Ax", True, VEC_AX_COLOR)
        sx2, sy2 = world_to_screen(ax_vec[0], ax_vec[1])
        screen.blit(lbl_ax, (sx2 + 8, sy2 - 18))

        # Info panel
        draw_info_panel(screen, A, x_vec, font, font_sm)

        # Controls help
        draw_controls_help(screen, font_sm)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
