"""
Cylinder + Cruciform Fins on Ball Joint
========================================

A cylinder (∅5 cm × 20 cm, 80 g) is pinned at its top to a universal
ball joint.  Wind blows upward from below.  Four cruciform fins (35×25 mm)
at the bottom can deflect ±30° to steer the tilt.

DOF:  θ_x  (lean toward +x)  and  θ_y  (lean toward +y)

Physics:
  ① gravity restoring  — m g L_cg sin θ  (toward vertical)
  ② wind destabilising — F_drag L_cp sin θ  (pushes bottom away)
  ③ fin torque         — F_fin(δ) L_fin  (user/PID controlled)
  ④ angular damping    — C ω

At v_wind < ~17 m/s gravity wins (passively stable).
Above that the pendulum diverges and needs active fin control.

Controls
  A / D        δ_x  (tilt left / right)
  W / S        δ_y  (tilt forward / back)
  UP / DOWN    wind speed ± 0.5 m/s
  1            PID ON
  2            PID OFF (manual)
  R            reset
  SPACE        pause
  ESC          quit
"""

import numpy as np
import pygame
import sys
import os
import math

# ============================================================
#  Physical constants
# ============================================================
RHO = 1.225
G   = 9.81

# ── geometry ───
ROCKET_R = 0.025    # radius [m]
ROCKET_H = 0.200    # height [m]
ROCKET_M = 0.180    # mass   [kg]
ROCKET_A = math.pi * ROCKET_R ** 2

CD_AXIAL = 1.10

L_CG  = ROCKET_H / 2                       # pivot → CG
L_CP  = ROCKET_H                            # pivot → centre of pressure
L_FIN = ROCKET_H - 0.035 / 2               # pivot → fin mid-chord

# ── fins ───
FIN_L   = 0.035
FIN_W   = 0.025
FIN_A   = FIN_L * FIN_W
FIN_AR  = FIN_W / FIN_L
CL_ALPHA = 2 * math.pi * FIN_AR / (2 + math.sqrt(4 + FIN_AR ** 2))
CD_N     = 2.0
MAX_DEF  = math.radians(30)

# ── inertia about pivot (parallel-axis) ───
I_CG  = ROCKET_M * (3 * ROCKET_R ** 2 + ROCKET_H ** 2) / 12
I_PIV = I_CG + ROCKET_M * L_CG ** 2

# ── damping ───
C_DAMP = 1.8e-3   # [N·m·s/rad]

# ── critical wind speed ───
V_CRIT = math.sqrt(ROCKET_M * G * L_CG / (L_CP * 0.5 * RHO * CD_AXIAL * ROCKET_A))

# ============================================================
#  Display
# ============================================================
SCR_W, SCR_H = 1100, 820
SCALE = 1900   # px / m

BG        = (12, 12, 22)
CYL_CLR   = (230, 150, 50)
CYL_EDGE  = (180, 110, 30)
FIN_POS   = (80, 200, 255)
FIN_NEG   = (255, 90, 90)
FIN_ZERO  = (160, 160, 175)
PIVOT_CLR = (220, 220, 220)
WIND_CLR  = (60, 160, 230)
ARC_CLR   = (255, 220, 80)
TEXT_CLR  = (210, 210, 210)
DIM_CLR   = (110, 110, 130)
GREEN     = (80, 220, 100)
RED       = (255, 80, 80)
CYAN      = (80, 220, 255)
PANEL_BG  = (20, 20, 38)

FPS    = 60
DT_PHY = 0.0005
STEPS  = 28


def load_font(size):
    for p in ["/System/Library/Fonts/Menlo.ttc",
              "/System/Library/Fonts/SFMono-Regular.otf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"]:
        if os.path.exists(p):
            try:
                return pygame.font.Font(p, size)
            except Exception:
                continue
    return pygame.font.SysFont("monospace", size)


def fin_CN(delta):
    """Signed normal-force coefficient for a low-AR flat-plate fin."""
    s = math.sin(delta)
    c = math.cos(delta)
    mag = CL_ALPHA * abs(s) * c + CD_N * s * s
    return math.copysign(mag, delta)


# ============================================================
#  Simulation
# ============================================================
class Sim:
    def __init__(self):
        pygame.init()
        self.scr   = pygame.display.set_mode((SCR_W, SCR_H))
        pygame.display.set_caption("Cylinder + Fins on Ball Joint")
        self.clock = pygame.time.Clock()
        self.font    = load_font(14)
        self.font_sm = load_font(12)
        self.font_lg = load_font(18)

        # state [θ_x, θ_y, ω_x, ω_y]
        self.st = np.zeros(4)
        self.dx = 0.0   # fin δ_x  (controls θ_x via ±y-axis fin pair)
        self.dy = 0.0   # fin δ_y  (controls θ_y via ±x-axis fin pair)
        self.v_wind = 12.0
        self.pid_on = False
        self.paused = False
        self.time   = 0.0

        self.pid_x = [4.0, 0.3, 0.6]   # [Kp, Ki, Kd]
        self.pid_y = [4.0, 0.3, 0.6]
        self._ix = 0.0; self._ex = 0.0
        self._iy = 0.0; self._ey = 0.0

    def reset(self):
        self.st[:] = 0
        self.dx = self.dy = 0.0
        self.time = 0.0
        self._ix = self._iy = self._ex = self._ey = 0.0

    # ── physics ──────────────────────────────────────────────
    def _deriv(self, s, dx, dy, vw):
        tx, ty, wx, wy = s
        q = 0.5 * RHO * vw * vw
        Fd = q * CD_AXIAL * ROCKET_A

        # gravity restoring − wind destabilising
        tau_x = (-ROCKET_M * G * L_CG + Fd * L_CP) * math.sin(tx)
        tau_y = (-ROCKET_M * G * L_CG + Fd * L_CP) * math.sin(ty)

        # fin torques (two fins per pair)
        # positive δ deflects trailing edge toward +x → aero reaction pushes bottom toward −x
        qf = q  # fins see the freestream
        tau_x -= 2 * qf * fin_CN(dx) * FIN_A * L_FIN
        tau_y -= 2 * qf * fin_CN(dy) * FIN_A * L_FIN

        # damping
        tau_x -= C_DAMP * wx
        tau_y -= C_DAMP * wy

        return np.array([wx, wy, tau_x / I_PIV, tau_y / I_PIV])

    def _rk4(self):
        s, dx, dy, vw, dt = self.st, self.dx, self.dy, self.v_wind, DT_PHY
        k1 = self._deriv(s,          dx, dy, vw)
        k2 = self._deriv(s+0.5*dt*k1, dx, dy, vw)
        k3 = self._deriv(s+0.5*dt*k2, dx, dy, vw)
        k4 = self._deriv(s+dt*k3,     dx, dy, vw)
        self.st = s + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        for i in (0, 1):
            self.st[i] = np.clip(self.st[i], -math.pi/2, math.pi/2)

    def _pid_step(self, dt):
        tx, ty = self.st[0], self.st[1]
        wx, wy = self.st[2], self.st[3]
        kp, ki, kd = self.pid_x
        self._ix += tx * dt
        self._ix = np.clip(self._ix, -0.5, 0.5)
        d = (tx - self._ex) / dt if dt > 0 else 0
        self._ex = tx
        self.dx = float(np.clip(kp*tx + ki*self._ix + kd*d, -MAX_DEF, MAX_DEF))

        kp, ki, kd = self.pid_y
        self._iy += ty * dt
        self._iy = np.clip(self._iy, -0.5, 0.5)
        d = (ty - self._ey) / dt if dt > 0 else 0
        self._ey = ty
        self.dy = float(np.clip(kp*ty + ki*self._iy + kd*d, -MAX_DEF, MAX_DEF))

    def update(self):
        if self.paused:
            return
        for _ in range(STEPS):
            if self.pid_on:
                self._pid_step(DT_PHY)
            self._rk4()
            self.time += DT_PHY

    # ── coordinate helpers ───────────────────────────────────
    @staticmethod
    def _body2scr(bx, bz, theta, cx, cy):
        """Rotate body-frame (bx,bz) by theta about pivot, return screen."""
        x =  bx * math.cos(theta) - bz * math.sin(theta)
        z =  bx * math.sin(theta) + bz * math.cos(theta)
        return int(cx + x * SCALE), int(cy - z * SCALE)

    # ── drawing ──────────────────────────────────────────────
    def _draw_view(self, cx, cy, theta, delta, label, axis_label):
        """Draw one side-view panel (front or side)."""
        scr = self.scr
        tx, ty = self.st[0], self.st[1]

        # background panel
        pw, ph = 420, 660
        panel = pygame.Surface((pw, ph), pygame.SRCALPHA)
        panel.fill((18, 20, 32, 220))
        scr.blit(panel, (cx - pw//2, cy - 40))

        # label
        lbl = self.font_lg.render(label, True, TEXT_CLR)
        scr.blit(lbl, (cx - lbl.get_width()//2, cy - 34))

        py = cy + 20   # pivot screen-y

        # equilibrium line (dashed)
        for yy in range(py, py + int(ROCKET_H * SCALE) + 40, 8):
            if (yy // 8) % 2 == 0:
                pygame.draw.line(scr, (40, 40, 55), (cx, yy), (cx, yy + 4))

        # wind arrows
        n_arrows = 7
        aw = int(2 * ROCKET_R * SCALE) + 80
        bot_y = py + int(ROCKET_H * SCALE) + 20
        for i in range(n_arrows):
            ax = cx - aw//2 + int(aw * (i + 0.5) / n_arrows)
            phase = (self.time * 50 + i * 5) % 20
            ay = int(bot_y + 35 - phase * 2)
            leng = max(5, int(self.v_wind / 30 * 18))
            c = (50, int(120 + self.v_wind * 3), min(255, 180 + int(self.v_wind * 2)))
            pygame.draw.line(scr, c, (ax, ay + leng), (ax, ay), 2)
            pygame.draw.line(scr, c, (ax, ay), (ax - 3, ay + 4), 1)
            pygame.draw.line(scr, c, (ax, ay), (ax + 3, ay + 4), 1)

        b2s = lambda bx, bz: self._body2scr(bx, bz, theta, cx, py)

        # cylinder body
        R, H = ROCKET_R, ROCKET_H
        body = [b2s(-R, 0), b2s(R, 0), b2s(R, -H), b2s(-R, -H)]
        pygame.draw.polygon(scr, CYL_CLR, body)
        pygame.draw.polygon(scr, CYL_EDGE, body, 2)

        # CG marker
        cg = b2s(0, -L_CG)
        pygame.draw.circle(scr, (255, 255, 255), cg, 4)
        pygame.draw.circle(scr, CYL_EDGE, cg, 4, 1)

        # fins (the pair visible in THIS view controls the OTHER axis)
        fl = int(FIN_L * SCALE)
        fw = int(FIN_W * SCALE)
        fin_base_z = -H + FIN_L / 2

        # left fin
        lf = [b2s(-R, fin_base_z - FIN_L/2),
              b2s(-R, fin_base_z + FIN_L/2),
              b2s(-R - FIN_W, fin_base_z + FIN_L/2),
              b2s(-R - FIN_W, fin_base_z - FIN_L/2)]
        pygame.draw.polygon(scr, FIN_ZERO, lf)
        pygame.draw.polygon(scr, DIM_CLR, lf, 1)
        # right fin
        rf = [b2s(R, fin_base_z - FIN_L/2),
              b2s(R, fin_base_z + FIN_L/2),
              b2s(R + FIN_W, fin_base_z + FIN_L/2),
              b2s(R + FIN_W, fin_base_z - FIN_L/2)]
        pygame.draw.polygon(scr, FIN_ZERO, rf)
        pygame.draw.polygon(scr, DIM_CLR, rf, 1)

        # edge-on active fins (the pair that controls THIS axis)
        # draw as angled line at the bottom centre
        fin_half = FIN_W * 0.7
        fin_cz = -H + FIN_L / 2
        p1 = b2s(-fin_half * math.sin(delta), fin_cz - fin_half * math.cos(delta))
        p2 = b2s( fin_half * math.sin(delta), fin_cz + fin_half * math.cos(delta))
        clr = FIN_POS if delta > 0.01 else (FIN_NEG if delta < -0.01 else FIN_ZERO)
        pygame.draw.line(scr, clr, p1, p2, 3)

        # pivot joint
        pygame.draw.circle(scr, (50, 50, 70), (cx, py), 12)
        pygame.draw.circle(scr, PIVOT_CLR, (cx, py), 8)
        pygame.draw.circle(scr, (100, 100, 110), (cx, py), 8, 1)

        # angle arc
        if abs(theta) > 0.005:
            arc_r = 60
            start = math.pi / 2
            end   = math.pi / 2 - theta
            a1, a2 = min(start, end), max(start, end)
            rect = pygame.Rect(cx - arc_r, py - arc_r, arc_r * 2, arc_r * 2)
            pygame.draw.arc(scr, ARC_CLR, rect, a1, a2, 2)
            deg = math.degrees(theta)
            t = self.font_sm.render(f"{deg:+.1f}°", True, ARC_CLR)
            scr.blit(t, (cx + arc_r + 4, py - t.get_height()//2))

        # axis label
        al = self.font_sm.render(axis_label, True, DIM_CLR)
        scr.blit(al, (cx - al.get_width()//2, cy + ph - 60))

    def _draw_top_view(self):
        """Overhead cross-section showing all 4 fins."""
        scr = self.scr
        ox, oy = 960, 170
        r_tube = 80

        panel = pygame.Surface((200, 200), pygame.SRCALPHA)
        panel.fill((PANEL_BG[0], PANEL_BG[1], PANEL_BG[2], 210))
        scr.blit(panel, (ox - 100, oy - 100))

        # cylinder cross-section
        cr = int(ROCKET_R / 0.05 * r_tube * 0.5)
        pygame.draw.circle(scr, CYL_CLR, (ox, oy), cr)
        pygame.draw.circle(scr, CYL_EDGE, (ox, oy), cr, 2)

        # tilt indicator (dot showing where bottom points)
        tx, ty = self.st[0], self.st[1]
        tilt_scale = r_tube * 0.9 / (math.pi / 4)
        dx_px = int(math.sin(tx) * tilt_scale)
        dy_px = int(-math.sin(ty) * tilt_scale)
        pygame.draw.circle(scr, ARC_CLR, (ox + dx_px, oy + dy_px), 5)
        pygame.draw.line(scr, (60, 60, 80), (ox, oy), (ox + dx_px, oy + dy_px), 1)

        # four fins
        flen = int(FIN_W / 0.05 * r_tube * 0.5)

        def draw_fin(angle, delta, lbl):
            ca, sa = math.cos(angle), math.sin(angle)
            x1 = ox + int(cr * ca)
            y1 = oy - int(cr * sa)
            x2 = ox + int((cr + flen) * ca)
            y2 = oy - int((cr + flen) * sa)
            clr = FIN_POS if delta > 0.02 else (FIN_NEG if delta < -0.02 else FIN_ZERO)
            pygame.draw.line(scr, clr, (x1, y1), (x2, y2), 4)
            t = self.font_sm.render(lbl, True, DIM_CLR)
            scr.blit(t, (x2 + int(6 * ca) - t.get_width()//2,
                         y2 - int(6 * sa) - t.get_height()//2))

        draw_fin(0,           self.dx, "+x")
        draw_fin(math.pi,    -self.dx, "−x")
        draw_fin(math.pi/2,   self.dy, "+y")
        draw_fin(-math.pi/2, -self.dy, "−y")

        lbl = self.font_sm.render("Top (bottom cross-sec.)", True, DIM_CLR)
        scr.blit(lbl, (ox - lbl.get_width()//2, oy + r_tube + 14))

    def _draw_info(self):
        scr = self.scr
        ox, oy = 870, 320
        panel = pygame.Surface((220, 460), pygame.SRCALPHA)
        panel.fill((PANEL_BG[0], PANEL_BG[1], PANEL_BG[2], 210))
        scr.blit(panel, (ox - 5, oy - 5))

        y = oy
        gap = 19
        def put(txt, clr=TEXT_CLR, f=None):
            nonlocal y
            s = (f or self.font).render(txt, True, clr)
            scr.blit(s, (ox, y)); y += gap
        def sec(t):
            nonlocal y; y += 4; put(t, CYAN, self.font_lg); y += 2

        sec("STATE")
        put(f" θ_x  {math.degrees(self.st[0]):+7.2f}°")
        put(f" θ_y  {math.degrees(self.st[1]):+7.2f}°")
        put(f" ω_x  {math.degrees(self.st[2]):+7.1f} °/s")
        put(f" ω_y  {math.degrees(self.st[3]):+7.1f} °/s")

        sec("CONTROL")
        mode = "PID" if self.pid_on else "MANUAL"
        put(f" Mode   {mode}", GREEN if self.pid_on else RED)
        put(f" δ_x   {math.degrees(self.dx):+6.1f}°",
            FIN_POS if self.dx > 0.01 else (FIN_NEG if self.dx < -0.01 else TEXT_CLR))
        put(f" δ_y   {math.degrees(self.dy):+6.1f}°",
            FIN_POS if self.dy > 0.01 else (FIN_NEG if self.dy < -0.01 else TEXT_CLR))

        sec("WIND")
        put(f" v     {self.v_wind:5.1f} m/s")
        put(f" v_crit {V_CRIT:5.1f} m/s", DIM_CLR)
        q = 0.5 * RHO * self.v_wind ** 2
        Fd = q * CD_AXIAL * ROCKET_A
        ratio = Fd * L_CP / (ROCKET_M * G * L_CG)
        bar = "STABLE" if ratio < 1 else "UNSTABLE"
        clr = GREEN if ratio < 1 else RED
        put(f" F·L/mgL = {ratio:.2f}  {bar}", clr)

        sec("PARAMS")
        put(f" m={ROCKET_M*1e3:.0f}g  ∅{ROCKET_R*200:.0f}cm  "
            f"H={ROCKET_H*100:.0f}cm", DIM_CLR)
        put(f" I_piv  {I_PIV:.2e} kg·m²", DIM_CLR)
        put(f" CL_α   {CL_ALPHA:.2f} /rad", DIM_CLR)
        put(f" L_cg={L_CG*100:.0f}  L_fin={L_FIN*100:.1f} cm", DIM_CLR)
        put(f" t = {self.time:.2f} s", DIM_CLR)

    def _draw_help(self):
        scr = self.scr
        lines = [
            "A/D: δ_x (tilt L/R) | W/S: δ_y (tilt F/B) | "
            "UP/DN: wind | 1:PID on | 2:PID off | R:reset | SPACE:pause | ESC:quit"
        ]
        for i, ln in enumerate(lines):
            t = self.font_sm.render(ln, True, DIM_CLR)
            scr.blit(t, (12, SCR_H - 26 + i * 16))

    def render(self):
        self.scr.fill(BG)

        title = self.font_lg.render(
            "Cylinder + Cruciform Fins — Ball Joint  (2-DOF pendulum in wind)",
            True, TEXT_CLR)
        self.scr.blit(title, (SCR_W//2 - title.get_width()//2, 8))

        self._draw_view(220,  40, self.st[0], self.dx,
                         "Front View  (x-z)", "← x →")
        self._draw_view(640,  40, self.st[1], self.dy,
                         "Side View  (y-z)", "← y →")
        self._draw_top_view()
        self._draw_info()
        self._draw_help()

        if self.paused:
            ps = self.font_lg.render("▌▌ PAUSED", True, ARC_CLR)
            self.scr.blit(ps, (SCR_W//2 - ps.get_width()//2, 42))

        pygame.display.flip()

    # ── input ────────────────────────────────────────────────
    def handle(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return False
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE: return False
                if ev.key == pygame.K_SPACE: self.paused = not self.paused
                if ev.key == pygame.K_r:     self.reset()
                if ev.key == pygame.K_1:     self.pid_on = True
                if ev.key == pygame.K_2:
                    self.pid_on = False
                    self.dx = self.dy = 0.0
        keys = pygame.key.get_pressed()
        step_v = 0.3
        if keys[pygame.K_UP]:   self.v_wind = min(40, self.v_wind + step_v)
        if keys[pygame.K_DOWN]: self.v_wind = max(0,  self.v_wind - step_v)

        if not self.pid_on:
            rate = math.radians(3.0)  # per-frame slew toward target
            if keys[pygame.K_d]:
                tgt_x = MAX_DEF
            elif keys[pygame.K_a]:
                tgt_x = -MAX_DEF
            else:
                tgt_x = 0.0
            if keys[pygame.K_w]:
                tgt_y = MAX_DEF
            elif keys[pygame.K_s]:
                tgt_y = -MAX_DEF
            else:
                tgt_y = 0.0
            # slew toward target (smooth but no oscillation at limits)
            diff_x = tgt_x - self.dx
            self.dx += max(-rate, min(rate, diff_x))
            diff_y = tgt_y - self.dy
            self.dy += max(-rate, min(rate, diff_y))
        return True

    def run(self):
        running = True
        while running:
            running = self.handle()
            self.update()
            self.render()
            self.clock.tick(FPS)
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    Sim().run()
