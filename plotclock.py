"""
PlotClock Simulation — Two-Servo Planar Linkage
================================================

A PlotClock uses two servo motors mounted at fixed base positions to drive
a two-link mechanism that positions a pen tip (end effector) in 2D space.

Mechanism:
  - Two servos are located at fixed base points: Left servo at (-d/2, 0),
    Right servo at (+d/2, 0).
  - Each servo drives an upper arm of length L1.
  - The upper arms connect to forearms of length L2.
  - The forearms meet at the pen tip (end effector point P).

Kinematics:
  Forward: Given servo angles (theta1, theta2) -> compute pen position (x, y)
  Inverse: Given target (x, y) -> compute required servo angles (theta1, theta2)

Controls:
  Mouse drag   — move the target point for inverse kinematics
  LEFT/RIGHT   — adjust left servo angle (forward kinematics mode)
  UP/DOWN      — adjust right servo angle (forward kinematics mode)
  TAB          — toggle between Forward / Inverse kinematics mode
  T            — toggle pen trace (draws the path)
  C            — clear trace
  1-5          — preset patterns (circle, square, sine, figure-8, text "Hi")
  SPACE        — pause/resume pattern animation
  ESC          — quit
"""

import numpy as np
import pygame
import sys
import os
import math

# ============================================================
#  Mechanism parameters (in mm, rendered scaled to pixels)
# ============================================================
SERVO_DISTANCE = 50.0     # distance between two servo pivot centers
L1 = 35.0                 # upper arm length (servo to elbow)
L2 = 50.0                 # forearm length (elbow to pen tip)

# ============================================================
#  Display constants
# ============================================================
SCREEN_W, SCREEN_H = 1100, 800
ORIGIN_X, ORIGIN_Y = SCREEN_W // 2, int(SCREEN_H * 0.7)
SCALE = 5.0  # pixels per mm

BG = (18, 18, 28)
GRID_COLOR = (35, 35, 50)
AXIS_COLOR = (60, 60, 85)
ARM_COLOR_L = (80, 200, 255)
ARM_COLOR_R = (255, 160, 80)
JOINT_COLOR = (240, 240, 240)
PEN_COLOR = (255, 80, 120)
TARGET_COLOR = (120, 255, 120)
TRACE_COLOR = (255, 200, 60)
TEXT_COLOR = (210, 210, 210)
HELP_COLOR = (130, 130, 150)
SERVO_COLOR = (180, 80, 80)
REACHABLE_COLOR = (40, 60, 40, 60)


# ============================================================
#  Font loader
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
#  Coordinate transforms (mechanism mm <-> screen pixels)
# ============================================================
def mech_to_screen(mx: float, my: float) -> tuple[int, int]:
    return int(ORIGIN_X + mx * SCALE), int(ORIGIN_Y - my * SCALE)


def screen_to_mech(sx: int, sy: int) -> tuple[float, float]:
    return (sx - ORIGIN_X) / SCALE, (ORIGIN_Y - sy) / SCALE


# ============================================================
#  Forward Kinematics
# ============================================================
def forward_kinematics(theta1: float, theta2: float):
    """
    Given servo angles theta1 (left) and theta2 (right), compute the pen tip.

    theta1: angle of left upper arm from horizontal (radians)
    theta2: angle of right upper arm from horizontal (radians)

    Returns: (pen_x, pen_y, elbow_L, elbow_R) or None if unreachable.
    """
    half_d = SERVO_DISTANCE / 2.0

    # Servo pivot positions
    S1 = np.array([-half_d, 0.0])
    S2 = np.array([half_d, 0.0])

    # Elbow positions (end of upper arms)
    E1 = S1 + L1 * np.array([math.cos(theta1), math.sin(theta1)])
    E2 = S2 + L2 * np.array([math.cos(theta2), math.sin(theta2)])

    # The pen tip P must satisfy: |P - E1| = L2 and |P - E2| = L1
    # This is a two-circle intersection problem
    pen = circle_intersection(E1, L2, E2, L1)
    if pen is None:
        return None

    # Choose the solution with larger y (pen above mechanism)
    p1, p2 = pen
    P = p1 if p1[1] >= p2[1] else p2

    return P[0], P[1], E1, E2


def circle_intersection(c1: np.ndarray, r1: float, c2: np.ndarray, r2: float):
    """
    Find intersection points of two circles.
    Circle 1: center c1, radius r1
    Circle 2: center c2, radius r2
    Returns (point1, point2) or None if no intersection.
    """
    d_vec = c2 - c1
    d = np.linalg.norm(d_vec)

    if d > r1 + r2 or d < abs(r1 - r2) or d < 1e-10:
        return None

    a = (r1**2 - r2**2 + d**2) / (2 * d)
    h_sq = r1**2 - a**2
    if h_sq < 0:
        return None
    h = math.sqrt(h_sq)

    # Point along the line between centers
    unit = d_vec / d
    mid = c1 + a * unit

    # Perpendicular direction
    perp = np.array([-unit[1], unit[0]])

    p1 = mid + h * perp
    p2 = mid - h * perp
    return p1, p2


# ============================================================
#  Segment intersection (overlap detection)
# ============================================================
def _ccw(A, B, C):
    """True if A, B, C are in counter-clockwise order (strict)."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def segments_intersect(p1, p2, p3, p4):
    """
    Test if segment (p1-p2) intersects segment (p3-p4) in their interiors.
    Uses the standard CCW orientation test.
    Shared endpoints are NOT considered as intersections.
    """
    # Skip if segments share an endpoint
    eps = 1e-8
    for a in (p1, p2):
        for b in (p3, p4):
            if abs(a[0] - b[0]) < eps and abs(a[1] - b[1]) < eps:
                return False

    d1 = _ccw(p3, p4, p1)
    d2 = _ccw(p3, p4, p2)
    d3 = _ccw(p1, p2, p3)
    d4 = _ccw(p1, p2, p4)

    if d1 != d2 and d3 != d4:
        return True
    return False


def check_no_overlap(S1, E1, S2, E2, P):
    """
    Check that no arm segments cross each other.
    Segments:
      A: S1 -> E1 (left upper arm)
      B: S2 -> E2 (right upper arm)
      C: E1 -> P  (left forearm)
      D: E2 -> P  (right forearm)

    Tests 3 critical pairs: A-B, A-D, B-C.
    Returns True if NO overlap (configuration is valid).
    """
    # A vs B: left upper arm vs right upper arm
    if segments_intersect(S1, E1, S2, E2):
        return False
    # A vs D: left upper arm vs right forearm
    if segments_intersect(S1, E1, E2, P):
        return False
    # B vs C: right upper arm vs left forearm
    if segments_intersect(S2, E2, E1, P):
        return False
    return True


# ============================================================
#  Inverse Kinematics
# ============================================================
def _angle_distance(a, b):
    """Shortest angular distance between two angles (handles wraparound)."""
    d = (a - b) % (2 * math.pi)
    if d > math.pi:
        d -= 2 * math.pi
    return abs(d)


# Maximum angular change per IK solve (prevents violent jumps)
MAX_ANGLE_STEP = math.radians(15)


def _clamp_angle_step(new_angle, prev_angle, max_step=MAX_ANGLE_STEP):
    """Limit the angular change from prev to new, respecting wraparound."""
    diff = (new_angle - prev_angle + math.pi) % (2 * math.pi) - math.pi
    if abs(diff) <= max_step:
        return new_angle
    clamped_diff = max_step if diff > 0 else -max_step
    return prev_angle + clamped_diff


def inverse_kinematics(target_x: float, target_y: float,
                       prev_theta1=None, prev_theta2=None):
    """
    Given target pen position (x, y), compute servo angles (theta1, theta2).

    Uses geometric approach with overlap rejection:
      1. Compute both elbow candidates for each servo (2 x 2 = 4 combinations).
      2. Reject any combination where arm segments intersect (overlap).
      3. Among valid solutions, pick the one closest to previous angles
         (continuity preference). If no prev angles, prefer elbow-up.
      4. Apply angular rate limiting.

    Returns: (theta1, theta2, elbow_L, elbow_R) or None if unreachable/all overlap.
    """
    half_d = SERVO_DISTANCE / 2.0
    P = np.array([target_x, target_y])

    S1 = np.array([-half_d, 0.0])
    S2 = np.array([half_d, 0.0])

    # Find elbow E1 candidates: intersection of circle(S1, L1) and circle(P, L2)
    e1_result = circle_intersection(S1, L1, P, L2)
    if e1_result is None:
        return None
    e1_candidates = list(e1_result)

    # Find elbow E2 candidates: intersection of circle(S2, L2) and circle(P, L1)
    e2_result = circle_intersection(S2, L2, P, L1)
    if e2_result is None:
        return None
    e2_candidates = list(e2_result)

    # Build all 4 combinations, compute angles, check overlap
    valid_solutions = []
    for e1 in e1_candidates:
        t1 = math.atan2(e1[1] - S1[1], e1[0] - S1[0])
        for e2 in e2_candidates:
            t2 = math.atan2(e2[1] - S2[1], e2[0] - S2[0])
            # Check no arm segments overlap
            if not check_no_overlap(S1, e1, S2, e2, P):
                continue
            valid_solutions.append((t1, t2, e1, e2))

    if not valid_solutions:
        return None

    # Pick the best solution: closest to previous angles (continuity)
    if prev_theta1 is not None and prev_theta2 is not None:
        def angle_cost(sol):
            return _angle_distance(sol[0], prev_theta1) + _angle_distance(sol[1], prev_theta2)
        valid_solutions.sort(key=angle_cost)
    else:
        # Default: prefer both elbows up (larger sum of y)
        valid_solutions.sort(key=lambda sol: -(sol[2][1] + sol[3][1]))

    theta1, theta2, E1, E2 = valid_solutions[0]

    # Apply angular rate limiting to prevent abrupt jumps
    if prev_theta1 is not None:
        theta1 = _clamp_angle_step(theta1, prev_theta1)
    if prev_theta2 is not None:
        theta2 = _clamp_angle_step(theta2, prev_theta2)

    # Recompute elbow positions from clamped angles for consistency
    if prev_theta1 is not None or prev_theta2 is not None:
        E1 = S1 + L1 * np.array([math.cos(theta1), math.sin(theta1)])
        E2 = S2 + L2 * np.array([math.cos(theta2), math.sin(theta2)])

    return theta1, theta2, E1, E2


# ============================================================
#  Pattern generation
# ============================================================
def generate_circle(cx=0.0, cy=60.0, radius=20.0, n_points=120):
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    return [(cx + radius * math.cos(a), cy + radius * math.sin(a)) for a in t]


def generate_square(cx=0.0, cy=60.0, size=30.0, n_points=120):
    half = size / 2
    corners = [
        (cx - half, cy - half), (cx + half, cy - half),
        (cx + half, cy + half), (cx - half, cy + half),
    ]
    pts = []
    per_side = n_points // 4
    for i in range(4):
        x0, y0 = corners[i]
        x1, y1 = corners[(i + 1) % 4]
        for j in range(per_side):
            frac = j / per_side
            pts.append((x0 + frac * (x1 - x0), y0 + frac * (y1 - y0)))
    return pts


def generate_sine(cx=0.0, cy=60.0, amp=15.0, width=50.0, n_points=120):
    pts = []
    for i in range(n_points):
        frac = i / n_points
        x = cx - width / 2 + frac * width
        y = cy + amp * math.sin(frac * 4 * math.pi)
        pts.append((x, y))
    return pts


def generate_figure8(cx=0.0, cy=60.0, size=20.0, n_points=120):
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    return [(cx + size * math.sin(a), cy + size * math.sin(a) * math.cos(a)) for a in t]


def generate_text_hi(cx=-15.0, cy=60.0, scale=0.8):
    """Generate stroke points for 'Hi' text."""
    pts = []
    # H
    hx = cx
    for i in range(10):
        pts.append((hx, cy - 15 * scale + i * 3 * scale))
    for i in range(8):
        pts.append((hx + i * 2 * scale, cy))
    for i in range(10):
        pts.append((hx + 14 * scale, cy - 15 * scale + i * 3 * scale))
    # i
    ix = cx + 22 * scale
    for i in range(8):
        pts.append((ix, cy - 12 * scale + i * 3 * scale))
    pts.append((ix, cy + 15 * scale))  # dot
    return pts


PATTERNS = {
    pygame.K_1: ("Circle", generate_circle),
    pygame.K_2: ("Square", generate_square),
    pygame.K_3: ("Sine wave", generate_sine),
    pygame.K_4: ("Figure-8", generate_figure8),
    pygame.K_5: ("Text 'Hi'", generate_text_hi),
}


# ============================================================
#  Drawing helpers
# ============================================================
def draw_arm_segment(surface, p1, p2, color, width=4):
    s1 = mech_to_screen(*p1)
    s2 = mech_to_screen(*p2)
    pygame.draw.line(surface, color, s1, s2, width)


def draw_joint(surface, pos, radius=5, color=JOINT_COLOR):
    sp = mech_to_screen(*pos)
    pygame.draw.circle(surface, color, sp, radius)
    pygame.draw.circle(surface, (50, 50, 70), sp, radius, 1)


def draw_servo(surface, pos, angle, color=SERVO_COLOR):
    sp = mech_to_screen(*pos)
    pygame.draw.rect(surface, color, (sp[0] - 12, sp[1] - 8, 24, 16), border_radius=3)
    pygame.draw.rect(surface, (100, 40, 40), (sp[0] - 12, sp[1] - 8, 24, 16), 1,
                     border_radius=3)
    # angle indicator
    end_x = sp[0] + int(10 * math.cos(-angle))
    end_y = sp[1] + int(10 * math.sin(-angle))
    pygame.draw.line(surface, (255, 255, 255), sp, (end_x, end_y), 2)


def draw_grid(surface):
    for x in range(-120, 121, 10):
        sx = int(ORIGIN_X + x * SCALE)
        pygame.draw.line(surface, GRID_COLOR, (sx, 0), (sx, SCREEN_H), 1)
    for y in range(-40, 121, 10):
        sy = int(ORIGIN_Y - y * SCALE)
        pygame.draw.line(surface, GRID_COLOR, (0, sy), (SCREEN_W, sy), 1)


def draw_workspace_boundary(surface):
    """Draw approximate reachable workspace as a shaded region."""
    overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
    half_d = SERVO_DISTANCE / 2.0
    n = 200
    for angle in np.linspace(0, math.pi, n):
        # Approximate max reach from each servo
        max_r = L1 + L2
        x_l = -half_d + max_r * math.cos(angle)
        y_l = max_r * math.sin(angle)
        sx, sy = mech_to_screen(x_l, y_l)
        pygame.draw.circle(overlay, (40, 80, 40, 15), (sx, sy), 3)

        x_r = half_d + max_r * math.cos(angle)
        y_r = max_r * math.sin(angle)
        sx2, sy2 = mech_to_screen(x_r, y_r)
        pygame.draw.circle(overlay, (40, 80, 40, 15), (sx2, sy2), 3)
    surface.blit(overlay, (0, 0))


# ============================================================
#  Info panel
# ============================================================
def draw_info_panel(surface, font, font_sm, mode, theta1, theta2,
                    pen_pos, target_pos, pattern_name):
    panel = pygame.Surface((340, 320), pygame.SRCALPHA)
    panel.fill((20, 20, 35, 210))
    surface.blit(panel, (10, 10))

    y = 18
    gap = 22

    def put(text, color=TEXT_COLOR, x_off=20, f=font):
        nonlocal y
        s = f.render(text, True, color)
        surface.blit(s, (x_off, y))
        y += gap

    mode_str = "INVERSE KINEMATICS" if mode == "inverse" else "FORWARD KINEMATICS"
    mode_color = TARGET_COLOR if mode == "inverse" else ARM_COLOR_L
    put(f"  Mode: {mode_str}", mode_color)
    y += 4

    put(f"  Servo L (theta1): {math.degrees(theta1):+7.2f} deg", ARM_COLOR_L)
    put(f"  Servo R (theta2): {math.degrees(theta2):+7.2f} deg", ARM_COLOR_R)
    y += 4

    if pen_pos is not None:
        put(f"  Pen X: {pen_pos[0]:+7.2f} mm")
        put(f"  Pen Y: {pen_pos[1]:+7.2f} mm")
    else:
        put("  Pen: UNREACHABLE", (255, 80, 80))
    y += 4

    if mode == "inverse":
        put(f"  Target X: {target_pos[0]:+7.2f} mm", TARGET_COLOR)
        put(f"  Target Y: {target_pos[1]:+7.2f} mm", TARGET_COLOR)
    y += 4

    put(f"  L1 (upper arm):  {L1:.1f} mm", HELP_COLOR, f=font_sm)
    put(f"  L2 (forearm):    {L2:.1f} mm", HELP_COLOR, f=font_sm)
    put(f"  Servo dist:      {SERVO_DISTANCE:.1f} mm", HELP_COLOR, f=font_sm)

    if pattern_name:
        put(f"  Pattern: {pattern_name}", TRACE_COLOR)


def draw_controls_help(surface, font):
    lines = [
        "TAB: toggle FK/IK | Mouse: set target (IK mode)",
        "LEFT/RIGHT: theta1 | UP/DOWN: theta2 (FK mode)",
        "1-5: patterns | T: trace | C: clear | SPACE: pause | ESC: quit",
    ]
    for i, line in enumerate(lines):
        s = font.render(line, True, HELP_COLOR)
        surface.blit(s, (15, SCREEN_H - 64 + i * 22))


# ============================================================
#  Main loop
# ============================================================
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("PlotClock Simulation — 2-Servo Linkage (FK & IK)")
    clock = pygame.time.Clock()

    font = load_font(16)
    font_sm = load_font(13)

    # State
    mode = "inverse"  # "forward" or "inverse"
    theta1 = math.radians(70)
    theta2 = math.radians(110)
    target = np.array([0.0, 65.0])
    pen_pos = None
    elbow_L = None
    elbow_R = None

    trace_on = True
    trace_points = []
    pattern_pts = []
    pattern_idx = 0
    pattern_name = ""
    pattern_playing = False

    dragging = False

    running = True
    while running:
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_TAB:
                    mode = "forward" if mode == "inverse" else "inverse"

                elif event.key == pygame.K_t:
                    trace_on = not trace_on

                elif event.key == pygame.K_c:
                    trace_points.clear()

                elif event.key == pygame.K_SPACE:
                    if pattern_pts:
                        pattern_playing = not pattern_playing
                    else:
                        pattern_playing = False

                elif event.key in PATTERNS:
                    pname, pfunc = PATTERNS[event.key]
                    pattern_name = pname
                    pattern_pts = pfunc()
                    pattern_idx = 0
                    pattern_playing = True
                    mode = "inverse"
                    trace_points.clear()

                elif mode == "forward":
                    step = math.radians(2)
                    if event.key == pygame.K_LEFT:
                        theta1 -= step
                    elif event.key == pygame.K_RIGHT:
                        theta1 += step
                    elif event.key == pygame.K_DOWN:
                        theta2 -= step
                    elif event.key == pygame.K_UP:
                        theta2 += step

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1 and mode == "inverse":
                    dragging = True
                    mx, my = screen_to_mech(*event.pos)
                    target = np.array([mx, my])

            elif event.type == pygame.MOUSEBUTTONUP:
                dragging = False

            elif event.type == pygame.MOUSEMOTION:
                if dragging and mode == "inverse":
                    mx, my = screen_to_mech(*event.pos)
                    target = np.array([mx, my])

        # Pattern animation
        if pattern_playing and pattern_pts:
            target = np.array(pattern_pts[pattern_idx])
            pattern_idx = (pattern_idx + 1) % len(pattern_pts)
            mode = "inverse"

        # Compute kinematics
        if mode == "inverse":
            result = inverse_kinematics(target[0], target[1], theta1, theta2)
            if result is not None:
                theta1, theta2, elbow_L, elbow_R = result
                pen_pos = (target[0], target[1])
                if trace_on:
                    trace_points.append(pen_pos)
            else:
                pen_pos = None
                elbow_L = None
                elbow_R = None
        else:
            result = forward_kinematics(theta1, theta2)
            if result is not None:
                px, py, elbow_L, elbow_R = result
                pen_pos = (px, py)
                target = np.array([px, py])
                if trace_on:
                    trace_points.append(pen_pos)
            else:
                pen_pos = None
                elbow_L = None
                elbow_R = None

        # ---- Rendering ----
        screen.fill(BG)
        draw_grid(screen)

        # Axes
        pygame.draw.line(screen, AXIS_COLOR, (0, ORIGIN_Y), (SCREEN_W, ORIGIN_Y), 1)
        pygame.draw.line(screen, AXIS_COLOR, (ORIGIN_X, 0), (ORIGIN_X, SCREEN_H), 1)

        # Workspace boundary hint
        draw_workspace_boundary(screen)

        half_d = SERVO_DISTANCE / 2.0
        S1 = (-half_d, 0.0)
        S2 = (half_d, 0.0)

        # Draw trace
        if len(trace_points) > 1:
            screen_pts = [mech_to_screen(*p) for p in trace_points[-2000:]]
            pygame.draw.lines(screen, TRACE_COLOR, False, screen_pts, 2)

        # Draw mechanism
        if elbow_L is not None and elbow_R is not None and pen_pos is not None:
            # Upper arms
            draw_arm_segment(screen, S1, (elbow_L[0], elbow_L[1]), ARM_COLOR_L, 5)
            draw_arm_segment(screen, S2, (elbow_R[0], elbow_R[1]), ARM_COLOR_R, 5)

            # Forearms
            draw_arm_segment(screen, (elbow_L[0], elbow_L[1]), pen_pos, ARM_COLOR_L, 3)
            draw_arm_segment(screen, (elbow_R[0], elbow_R[1]), pen_pos, ARM_COLOR_R, 3)

            # Joints
            draw_joint(screen, (elbow_L[0], elbow_L[1]), 5)
            draw_joint(screen, (elbow_R[0], elbow_R[1]), 5)

            # Pen tip
            pen_sp = mech_to_screen(*pen_pos)
            pygame.draw.circle(screen, PEN_COLOR, pen_sp, 7)
            pygame.draw.circle(screen, (255, 255, 255), pen_sp, 7, 1)

        # Servos
        draw_servo(screen, S1, theta1, SERVO_COLOR)
        draw_servo(screen, S2, theta2, (80, 80, 180))

        # Base platform
        sp1 = mech_to_screen(*S1)
        sp2 = mech_to_screen(*S2)
        pygame.draw.line(screen, (100, 100, 100), sp1, sp2, 3)

        # Target crosshair (IK mode)
        if mode == "inverse":
            ts = mech_to_screen(target[0], target[1])
            pygame.draw.line(screen, TARGET_COLOR, (ts[0] - 8, ts[1]), (ts[0] + 8, ts[1]), 1)
            pygame.draw.line(screen, TARGET_COLOR, (ts[0], ts[1] - 8), (ts[0], ts[1] + 8), 1)
            pygame.draw.circle(screen, TARGET_COLOR, ts, 10, 1)

        # Info panel
        draw_info_panel(screen, font, font_sm, mode, theta1, theta2,
                        pen_pos, target, pattern_name)
        draw_controls_help(screen, font_sm)

        # Angle arcs for visual clarity
        arc_r = 20
        if theta1 is not None:
            sp = mech_to_screen(*S1)
            rect = pygame.Rect(sp[0] - arc_r, sp[1] - arc_r, arc_r * 2, arc_r * 2)
            start_angle = 0
            end_angle = theta1
            if end_angle > start_angle:
                pygame.draw.arc(screen, ARM_COLOR_L, rect, -end_angle, -start_angle, 2)

        if theta2 is not None:
            sp = mech_to_screen(*S2)
            rect = pygame.Rect(sp[0] - arc_r, sp[1] - arc_r, arc_r * 2, arc_r * 2)
            start_angle = 0
            end_angle = theta2
            if end_angle > start_angle:
                pygame.draw.arc(screen, ARM_COLOR_R, rect, -end_angle, -start_angle, 2)

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
