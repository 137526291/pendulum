"""
Wind Tunnel Rocket Simulation — EDF Vertical Hover
===================================================

Simulates a cylindrical rocket model hovering inside a vertical acrylic
tube (acting as a small wind tunnel), suspended by airflow from a 5-inch
EDF mounted at the base.

Geometry:
  Tube   — inner diameter 10 cm, height 60 cm
  Rocket — diameter 5 cm, height 20 cm, mass ~80 g (flat-bottom cylinder)
  Fins   — 4 cruciform, 35×25×2 mm, at the rocket base

Physics:
  - Axial drag (v² law) balanced against gravity for hover
  - Bernoulli-effect wall restoring force (lateral centering)
  - Wind speed gradient along tube height (passive z-stability)
  - Aerodynamic destabilizing moment (CP below CG)
  - 4 cruciform control fins: pitch / yaw / roll
  - Cascaded PID: position → attitude → fin deflection
  - 6-DOF rigid body dynamics (small-angle regime, RK4 integration)

Controls:
  UP / DOWN    — EDF power ±2 %
  H            — toggle height-hold PID (auto EDF)
  P            — random perturbation (impulse + tilt)
  1            — PID ON  (auto stabilisation)
  2            — PID OFF (free response / manual)
  W / S        — manual pitch fins   (PID OFF only)
  A / D        — manual yaw fins
  Q / E        — manual roll fins
  F            — toggle force-vector overlay
  R            — reset to hover equilibrium
  SPACE        — pause / resume
  ESC          — quit
"""

import numpy as np
import pygame
import sys
import os
import math

# ============================================================
#  Physical constants
# ============================================================
RHO = 1.225       # air density  [kg/m³]
G   = 9.81        # gravity      [m/s²]

# ============================================================
#  Tube geometry
# ============================================================
TUBE_R = 0.050    # inner radius [m]  (∅10 cm)
TUBE_H = 0.600    # height       [m]  (60 cm)

# ============================================================
#  Rocket geometry & mass
# ============================================================
ROCKET_R = 0.025   # body radius  [m]  (∅5 cm)
ROCKET_H = 0.200   # body height  [m]  (20 cm)
ROCKET_M = 0.080   # total mass   [kg] (80 g)
ROCKET_A = math.pi * ROCKET_R ** 2   # frontal area [m²]
GAP      = TUBE_R - ROCKET_R         # max lateral clearance [m]

# ============================================================
#  Aerodynamic coefficients
# ============================================================
CD_AXIAL   = 1.10   # flat-bottom cylinder axial drag
CD_LATERAL = 1.20   # crossflow drag
C_STATIC   = 0.03   # net aerodynamic static instability
K_LOSS     = 0.15   # wind-speed loss gradient along tube height

# ── Bernoulli eccentric-annulus model ──
# Inviscid theory: inner cylinder displaced by ε = r/GAP in a
# pipe with axial flow experiences a force TOWARD the nearest
# wall (destabilising):
#   f_x = π ρ v_ann² R_rocket ε / (1 − ε²)^(3/2)   [N/m]
# At Re ≈ 80 000 the flow is turbulent; viscous/wake effects
# drastically reduce the inviscid suction and add a separate
# restoring component (boundary-layer pressure recovery).
C_BERN_TURB = 0.02   # turbulence reduction on inviscid Bernoulli (Re≈80k)
C_VISC_REST = 1.50   # viscous / wake restoring coefficient
C_SQUEEZE   = 0.30   # squeeze-film lubrication near wall (∝ ε³/(1−ε)²)
V_RATIO = TUBE_R ** 2 / (TUBE_R ** 2 - ROCKET_R ** 2)  # ≈ 1.333

# ============================================================
#  Fin parameters  (4 × cruciform)
# ============================================================
FIN_L     = 0.035          # chord along rocket axis  [m]
FIN_W     = 0.025          # radial span              [m]
FIN_A     = FIN_L * FIN_W  # single-fin planform area [m²]
FIN_AR    = FIN_W / FIN_L  # aspect ratio ≈ 0.71
FIN_ARM   = ROCKET_H / 2 - FIN_L / 2   # moment arm from CG [m]
MAX_FIN   = math.radians(25)           # max deflection [rad]

# Lift-curve slope corrected for low AR  (Helmbold / Jones):
#   CL_α = 2π AR / (2 + √(4 + AR²))
# AR = 0.71 → CL_α ≈ 1.09 /rad  (vs 6.28 for infinite AR)
CL_ALPHA_INF = 2 * math.pi
CL_ALPHA = CL_ALPHA_INF * FIN_AR / (2 + math.sqrt(4 + FIN_AR ** 2))

# Flat-plate normal-force drag coefficient (for large δ)
CD_N_FIN = 2.0

# ============================================================
#  Moments of inertia  (solid cylinder approximation)
# ============================================================
I_ROLL  = 0.5 * ROCKET_M * ROCKET_R ** 2
I_PITCH = ROCKET_M * (3 * ROCKET_R ** 2 + ROCKET_H ** 2) / 12
I_YAW   = I_PITCH

# ============================================================
#  Damping
# ============================================================
C_ANG_DAMP = 1.2e-3  # angular damping  [N·m·s/rad]
C_LAT_DAMP = 0.80    # lateral aero drag scale
C_LAT_LIN  = 0.15    # linear lateral damping  [N·s/m]

# ============================================================
#  Derived hover quantities
# ============================================================
V_HOVER = math.sqrt(2 * ROCKET_M * G / (RHO * CD_AXIAL * ROCKET_A))
V_MAX   = V_HOVER * 1.6
Z_TARGET = TUBE_H * 0.50   # default target hover height

# Wind speed needed so that hover occurs at Z_TARGET
V_WIND_INIT = V_HOVER / (1 - K_LOSS * Z_TARGET / TUBE_H)

# ============================================================
#  Display constants
# ============================================================
SCR_W, SCR_H = 1300, 860

SIDE_CX    = 460                        # tube centre-x on screen
SIDE_SCALE = 1100                       # px / m
SIDE_BOT_Y = 800                        # tube bottom y-pixel

TUBE_PIX_H = int(TUBE_H * SIDE_SCALE)
TUBE_PIX_W = int(2 * TUBE_R * SIDE_SCALE)
TUBE_TOP_Y = SIDE_BOT_Y - TUBE_PIX_H
TUBE_LX    = SIDE_CX - TUBE_PIX_W // 2
TUBE_RX    = SIDE_CX + TUBE_PIX_W // 2

TOP_CX, TOP_CY = 1080, 175
TOP_R     = 110
TOP_SCALE = TOP_R / TUBE_R

# Physics stepping
PHYS_DT       = 0.0008
STEPS_PER_FRAME = 21
FPS            = 60

# ── colours ──────────────────────────────────────────────────
BG           = (12, 12, 22)
TUBE_CLR     = (45, 95, 155)
TUBE_INNER   = (16, 22, 36)
ROCKET_CLR   = (230, 150, 50)
ROCKET_EDGE  = (180, 110, 30)
FIN_CLR      = (175, 185, 205)
AIR_CLR      = (70, 170, 235)
F_GRAV_CLR   = (255, 80, 80)
F_DRAG_CLR   = (80, 255, 120)
F_WALL_CLR   = (255, 255, 80)
F_FIN_CLR    = (200, 100, 255)
TEXT_CLR     = (210, 210, 210)
DIM_CLR      = (110, 110, 130)
PANEL_BG     = (20, 20, 38)
GREEN        = (80, 220, 100)
RED          = (255, 80, 80)
YELLOW       = (255, 220, 80)
CYAN         = (80, 220, 255)
EDF_CLR      = (60, 60, 90)


# ============================================================
#  Helpers
# ============================================================
def load_font(size: int) -> pygame.font.Font:
    for p in [
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/SFMono-Regular.otf",
        "/System/Library/Fonts/Monaco.dfont",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ]:
        if os.path.exists(p):
            try:
                return pygame.font.Font(p, size)
            except Exception:
                continue
    return pygame.font.SysFont("monospace", size)


def phys_to_side(x: float, z: float):
    """Physical (x, z) → side-view screen pixel."""
    return int(SIDE_CX + x * SIDE_SCALE), int(SIDE_BOT_Y - z * SIDE_SCALE)


def phys_to_top(x: float, y: float):
    """Physical (x, y) → top-view screen pixel."""
    return int(TOP_CX + x * TOP_SCALE), int(TOP_CY - y * TOP_SCALE)


# ============================================================
#  PID controller
# ============================================================
class PID:
    __slots__ = ("kp", "ki", "kd", "lim", "_i", "_ep")

    def __init__(self, kp, ki, kd, lim=None):
        self.kp, self.ki, self.kd, self.lim = kp, ki, kd, lim
        self._i = 0.0
        self._ep = 0.0

    def step(self, err, dt):
        self._i += err * dt
        if self.lim:
            cap = self.lim / max(self.ki, 1e-9)
            self._i = np.clip(self._i, -cap, cap)
        d = (err - self._ep) / dt if dt > 1e-9 else 0.0
        self._ep = err
        out = self.kp * err + self.ki * self._i + self.kd * d
        if self.lim:
            out = float(np.clip(out, -self.lim, self.lim))
        return out

    def reset(self):
        self._i = 0.0
        self._ep = 0.0


# ============================================================
#  Air particle  (visual only)
# ============================================================
class AirParticle:
    __slots__ = ("x", "z", "sm", "sz", "al")

    def __init__(self):
        self.x  = np.random.uniform(-TUBE_R * 0.92, TUBE_R * 0.92)
        self.z  = np.random.uniform(0, TUBE_H)
        self.sm = np.random.uniform(0.65, 1.35)
        self.sz = np.random.choice([1, 2, 2, 3])
        self.al = np.random.randint(40, 110)

    def _reset_bottom(self):
        self.x  = np.random.uniform(-TUBE_R * 0.92, TUBE_R * 0.92)
        self.z  = 0.0
        self.sm = np.random.uniform(0.65, 1.35)

    def update(self, v_wind, dt, rx, rz):
        vz = v_wind * self.sm
        vx = 0.0
        rb = rz - ROCKET_H / 2 - 0.008
        rt = rz + ROCKET_H / 2 + 0.008
        if rb < self.z < rt:
            dx = self.x - rx
            if abs(dx) < ROCKET_R * 1.4:
                g = max(abs(dx), 0.002) / (ROCKET_R * 1.4)
                push = 0.35 * v_wind * (1 - g)
                vx = push if dx >= 0 else -push
                vz *= 1.0 + 0.4 * (1 - g)
        self.x += vx * dt
        self.z += vz * dt
        self.x = np.clip(self.x, -TUBE_R * 0.97, TUBE_R * 0.97)
        if self.z > TUBE_H or self.z < 0:
            self._reset_bottom()


# ============================================================
#  Simulation class
# ============================================================
class WindTunnelSim:
    # state indices
    X, Y, Z    = 0, 1, 2
    VX, VY, VZ = 3, 4, 5
    ROLL, PITCH, YAW = 6, 7, 8
    WX, WY, WZ = 9, 10, 11

    def __init__(self):
        pygame.init()
        self.scr = pygame.display.set_mode((SCR_W, SCR_H))
        pygame.display.set_caption("Wind Tunnel Rocket — EDF Hover Sim")
        self.clock = pygame.time.Clock()
        self.font    = load_font(14)
        self.font_sm = load_font(12)
        self.font_lg = load_font(16)
        self.font_tl = load_font(20)

        self.state = np.zeros(12)
        self.fins  = np.zeros(3)      # [δ_pitch, δ_yaw, δ_roll]
        self.v_wind = V_WIND_INIT
        self.edf_pct = self.v_wind / V_MAX * 100

        # inner-loop PIDs  (attitude → fin)
        # gains raised to compensate for low-AR CL_α = 1.09
        self.pid_pitch = PID(6.0, 1.2, 0.25, lim=MAX_FIN)
        self.pid_yaw   = PID(6.0, 1.2, 0.25, lim=MAX_FIN)
        self.pid_roll  = PID(3.0, 0.5, 0.15, lim=MAX_FIN)
        # outer-loop PIDs  (position → attitude cmd)
        # kept very gentle: Bernoulli wall effect is primary centering;
        # large attitude commands from pos-PID cause parasitic lateral
        # fin force that fights the wall centering.
        self.pid_x = PID(0.5, 0.05, 0.3, lim=math.radians(3))
        self.pid_y = PID(0.5, 0.05, 0.3, lim=math.radians(3))
        # height hold  (z error → Δv_wind)
        self.pid_z = PID(18.0, 3.0, 9.0, lim=V_MAX * 0.35)

        self.pid_on     = True
        self.height_hold = True
        self.paused      = False
        self.show_forces = True
        self.z_target    = Z_TARGET
        self.sim_time    = 0.0

        # display force cache
        self.dsp_Fg   = 0.0
        self.dsp_Fd   = 0.0
        self.dsp_Fb   = 0.0   # Bernoulli toward wall
        self.dsp_Fv   = 0.0   # viscous toward centre
        self.dsp_Fw   = np.zeros(2)
        self.dsp_Ffin = np.zeros(2)

        self.particles = [AirParticle() for _ in range(80)]
        self.hist_z = []
        self.hist_t = []

        self.reset()

    # ── reset ────────────────────────────────────────────────
    def reset(self):
        self.state[:] = 0.0
        self.state[self.Z] = Z_TARGET
        self.fins[:] = 0.0
        self.v_wind = V_WIND_INIT
        self.edf_pct = self.v_wind / V_MAX * 100
        self.sim_time = 0.0
        self.hist_z.clear()
        self.hist_t.clear()
        for p in (self.pid_pitch, self.pid_yaw, self.pid_roll,
                  self.pid_x, self.pid_y, self.pid_z):
            p.reset()

    # ── derivatives ──────────────────────────────────────────
    def _deriv(self, s, fins, v_wind):
        x, y, z     = s[0], s[1], s[2]
        vx, vy, vz  = s[3], s[4], s[5]
        roll, pitch, yaw = s[6], s[7], s[8]
        wx, wy, wz  = s[9], s[10], s[11]
        dp, dy_f, dr = fins

        # effective wind speed with height-loss gradient
        v_eff = v_wind * max(1 - K_LOSS * z / TUBE_H, 0.05)
        v_rel = v_eff - vz

        # ─── forces ───
        Fz_grav = -ROCKET_M * G
        Fz_drag = 0.5 * RHO * v_rel * abs(v_rel) * CD_AXIAL * ROCKET_A

        # ── Bernoulli eccentric-annulus wall force ──
        # Two competing terms:
        #  1) Inviscid Bernoulli suction (TOWARD wall, destabilising)
        #     Derived from pressure integration around eccentric annulus:
        #     F_b = C_turb · π ρ v_ann² R H ε / (1-ε²)^1.5
        #  2) Viscous / wake restoring (TOWARD centre, stabilising)
        #     Boundary-layer pressure recovery on the wide-gap side:
        #     F_v = C_visc · ½ρ v² A ε
        # Net = F_b − F_v; positive = toward wall.
        r_lat = math.sqrt(x * x + y * y)
        Fx_w = Fy_w = 0.0
        if r_lat > 1e-7:
            eps = min(r_lat / GAP, 0.97)
            v_ann = v_eff * V_RATIO
            denom = max((1 - eps * eps) ** 1.5, 0.02)
            qw = 0.5 * RHO * v_eff * v_eff
            # ① inviscid Bernoulli suction → toward wall
            F_bern = (C_BERN_TURB * math.pi * RHO * v_ann * v_ann
                      * ROCKET_R * ROCKET_H * eps / denom)
            # ② viscous / wake restoring → toward centre
            F_visc = C_VISC_REST * qw * ROCKET_A * eps
            # ③ squeeze-film lubrication (narrow-gap viscous) → toward centre
            A_lat = ROCKET_H * 2 * ROCKET_R
            e1 = max(1 - eps, 0.03)
            F_squeeze = C_SQUEEZE * qw * A_lat * eps * eps * eps / (e1 * e1)
            F_net = F_bern - F_visc - F_squeeze
            Fx_w = F_net * x / r_lat
            Fy_w = F_net * y / r_lat

        # ── lateral aero damping (quadratic + linear) ──
        A_lat = ROCKET_H * 2 * ROCKET_R
        Fx_d = (-C_LAT_DAMP * 0.5 * RHO * vx * abs(vx) * CD_LATERAL * A_lat
                - C_LAT_LIN * vx)
        Fy_d = (-C_LAT_DAMP * 0.5 * RHO * vy * abs(vy) * CD_LATERAL * A_lat
                - C_LAT_LIN * vy)

        # ── fin normal force ──
        # CN(δ) = CL_α·sin δ·cos δ  +  CD_n·sin²δ
        #         ↑ lift (circulation)    ↑ drag (projected-area)
        # At small δ: ≈ CL_α·δ   (lift dominates)
        # At large δ: ≈ CD_n·sin²δ (projected-area drag dominates)
        qf = 0.5 * RHO * v_rel * v_rel
        s_dp, c_dp = math.sin(dp), math.cos(dp)
        s_dy, c_dy = math.sin(dy_f), math.cos(dy_f)
        s_dr, c_dr = math.sin(dr), math.cos(dr)
        CN_p = CL_ALPHA * s_dp * c_dp + CD_N_FIN * s_dp * s_dp
        CN_y = CL_ALPHA * s_dy * c_dy + CD_N_FIN * s_dy * s_dy
        CN_r = CL_ALPHA * s_dr * c_dr + CD_N_FIN * s_dr * s_dr
        # keep sign: CN should flip with δ
        CN_p = math.copysign(CN_p, dp)
        CN_y = math.copysign(CN_y, dy_f)
        CN_r = math.copysign(CN_r, dr)

        Fx_fin = 2 * qf * CN_p * FIN_A
        Fy_fin = 2 * qf * CN_y * FIN_A

        # ── tilt-to-lateral coupling ──
        Fx_tilt = Fz_drag * math.sin(pitch)
        Fy_tilt = Fz_drag * math.sin(yaw)

        Fx = Fx_w + Fx_d + Fx_fin + Fx_tilt
        Fy = Fy_w + Fy_d + Fy_fin + Fy_tilt
        Fz = Fz_grav + Fz_drag

        # ─── torques ───
        Kf_p = 2 * qf * abs(CN_p) * FIN_A * FIN_ARM
        Kf_y = 2 * qf * abs(CN_y) * FIN_A * FIN_ARM
        Ty = -math.copysign(Kf_p, dp)     # pitch
        Tx = -math.copysign(Kf_y, dy_f)   # yaw
        Tz = -4 * qf * abs(CN_r) * FIN_A * ROCKET_R * math.copysign(1, dr)  # roll

        qs = 0.5 * RHO * v_eff * v_eff
        Ty += C_STATIC * qs * ROCKET_A * (ROCKET_H / 2) * pitch
        Tx += C_STATIC * qs * ROCKET_A * (ROCKET_H / 2) * yaw

        Tx -= C_ANG_DAMP * wx
        Ty -= C_ANG_DAMP * wy
        Tz -= C_ANG_DAMP * wz

        ds = np.zeros(12)
        ds[0:3] = vx, vy, vz
        ds[3] = Fx / ROCKET_M
        ds[4] = Fy / ROCKET_M
        ds[5] = Fz / ROCKET_M
        ds[6], ds[7], ds[8] = wz, wy, wx
        ds[9]  = Tx / I_YAW
        ds[10] = Ty / I_PITCH
        ds[11] = Tz / I_ROLL
        return ds

    # ── RK4 step ─────────────────────────────────────────────
    def _rk4(self, dt):
        s = self.state
        f = self.fins
        v = self.v_wind
        k1 = self._deriv(s, f, v)
        k2 = self._deriv(s + 0.5 * dt * k1, f, v)
        k3 = self._deriv(s + 0.5 * dt * k2, f, v)
        k4 = self._deriv(s + dt * k3, f, v)
        self.state = s + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

    # ── clamp state (hard walls) ─────────────────────────────
    def _clamp(self):
        s = self.state
        r = math.sqrt(s[0]**2 + s[1]**2)
        if r > GAP * 0.99:
            scale = GAP * 0.99 / max(r, 1e-9)
            s[0] *= scale
            s[1] *= scale
            s[3] *= 0.3
            s[4] *= 0.3
        s[2] = np.clip(s[2], 0.01, TUBE_H - 0.01)
        if s[2] <= 0.011 or s[2] >= TUBE_H - 0.011:
            s[5] *= -0.2
        for i in (6, 7, 8):
            s[i] = np.clip(s[i], -math.pi / 4, math.pi / 4)

    # ── PID update ───────────────────────────────────────────
    def _update_pid(self, dt):
        s = self.state
        pitch_cmd = -self.pid_x.step(s[0], dt)
        yaw_cmd   = -self.pid_y.step(s[1], dt)

        self.fins[0] = float(np.clip(
            self.pid_pitch.step(s[self.PITCH] - pitch_cmd, dt),
            -MAX_FIN, MAX_FIN))
        self.fins[1] = float(np.clip(
            self.pid_yaw.step(s[self.YAW] - yaw_cmd, dt),
            -MAX_FIN, MAX_FIN))
        self.fins[2] = float(np.clip(
            self.pid_roll.step(s[self.ROLL], dt),
            -MAX_FIN, MAX_FIN))

        if self.height_hold:
            dv = self.pid_z.step(self.z_target - s[self.Z], dt)
            self.v_wind = float(np.clip(V_WIND_INIT + dv, 0, V_MAX))
            self.edf_pct = self.v_wind / V_MAX * 100

    # ── display forces cache ─────────────────────────────────
    def _cache_forces(self):
        s = self.state
        v_eff = self.v_wind * max(1 - K_LOSS * s[2] / TUBE_H, 0.05)
        v_rel = v_eff - s[5]
        self.dsp_Fg = -ROCKET_M * G
        self.dsp_Fd = 0.5 * RHO * v_rel * abs(v_rel) * CD_AXIAL * ROCKET_A
        r = math.sqrt(s[0]**2 + s[1]**2)
        if r > 1e-7:
            eps = min(r / GAP, 0.97)
            v_ann = v_eff * V_RATIO
            denom = max((1 - eps * eps) ** 1.5, 0.02)
            qw = 0.5 * RHO * v_eff ** 2
            self.dsp_Fb = (C_BERN_TURB * math.pi * RHO * v_ann ** 2
                           * ROCKET_R * ROCKET_H * eps / denom)
            A_lat = ROCKET_H * 2 * ROCKET_R
            e1 = max(1 - eps, 0.03)
            self.dsp_Fv = (C_VISC_REST * qw * ROCKET_A * eps
                           + C_SQUEEZE * qw * A_lat * eps**3 / e1**2)
            F_net = self.dsp_Fb - self.dsp_Fv
            self.dsp_Fw = np.array([F_net * s[0] / r, F_net * s[1] / r])
        else:
            self.dsp_Fb = 0.0
            self.dsp_Fv = 0.0
            self.dsp_Fw = np.zeros(2)
        qf = 0.5 * RHO * v_rel ** 2
        dp, dy_f = self.fins[0], self.fins[1]
        s_dp, c_dp = math.sin(dp), math.cos(dp)
        s_dy, c_dy = math.sin(dy_f), math.cos(dy_f)
        CN_p = CL_ALPHA * s_dp * c_dp + CD_N_FIN * s_dp * s_dp
        CN_y = CL_ALPHA * s_dy * c_dy + CD_N_FIN * s_dy * s_dy
        CN_p = math.copysign(CN_p, dp)
        CN_y = math.copysign(CN_y, dy_f)
        self.dsp_Ffin = np.array([2 * qf * CN_p * FIN_A,
                                   2 * qf * CN_y * FIN_A])

    # ── perturbation ─────────────────────────────────────────
    def perturb(self):
        self.state[3] += np.random.uniform(-0.35, 0.35)
        self.state[4] += np.random.uniform(-0.35, 0.35)
        self.state[5] += np.random.uniform(-0.3, 0.3)
        self.state[7] += np.random.uniform(-0.08, 0.08)
        self.state[8] += np.random.uniform(-0.08, 0.08)
        self.state[9] += np.random.uniform(-1.5, 1.5)
        self.state[10] += np.random.uniform(-1.5, 1.5)

    # ── full sim step ────────────────────────────────────────
    def update(self):
        if self.paused:
            return
        for _ in range(STEPS_PER_FRAME):
            if self.pid_on:
                self._update_pid(PHYS_DT)
            self._rk4(PHYS_DT)
            self._clamp()
            self.sim_time += PHYS_DT
        self._cache_forces()
        # history
        self.hist_t.append(self.sim_time)
        self.hist_z.append(self.state[self.Z])
        max_pts = 600
        if len(self.hist_t) > max_pts:
            self.hist_t = self.hist_t[-max_pts:]
            self.hist_z = self.hist_z[-max_pts:]
        # particles
        rx, rz = self.state[0], self.state[2]
        for p in self.particles:
            p.update(self.v_wind, PHYS_DT * STEPS_PER_FRAME, rx, rz)

    # ============================================================
    #  Rendering
    # ============================================================

    def _draw_tube(self):
        scr = self.scr
        # tube interior
        rect = pygame.Rect(TUBE_LX, TUBE_TOP_Y, TUBE_PIX_W, TUBE_PIX_H)
        pygame.draw.rect(scr, TUBE_INNER, rect)
        # walls
        wall_w = 5
        pygame.draw.rect(scr, TUBE_CLR,
                         (TUBE_LX - wall_w, TUBE_TOP_Y - wall_w,
                          wall_w, TUBE_PIX_H + 2 * wall_w))
        pygame.draw.rect(scr, TUBE_CLR,
                         (TUBE_RX, TUBE_TOP_Y - wall_w,
                          wall_w, TUBE_PIX_H + 2 * wall_w))
        # wall highlights
        for yo in range(TUBE_TOP_Y, SIDE_BOT_Y, 4):
            a = max(0, 60 - abs(yo - (TUBE_TOP_Y + TUBE_PIX_H // 2)) // 8)
            c = (90 + a, 140 + a, 200 + min(a, 55))
            pygame.draw.line(scr, c, (TUBE_LX - 2, yo), (TUBE_LX - 2, yo))
            pygame.draw.line(scr, c, (TUBE_RX + 2, yo), (TUBE_RX + 2, yo))

        # EDF housing at bottom
        edf_h = 28
        edf_rect = pygame.Rect(TUBE_LX - wall_w, SIDE_BOT_Y,
                                TUBE_PIX_W + 2 * wall_w, edf_h)
        pygame.draw.rect(scr, EDF_CLR, edf_rect, border_radius=4)
        pygame.draw.rect(scr, TUBE_CLR, edf_rect, 2, border_radius=4)
        # EDF label
        lbl = self.font_sm.render("EDF 5\"", True, DIM_CLR)
        scr.blit(lbl, (SIDE_CX - lbl.get_width()//2, SIDE_BOT_Y + 6))

        # height scale ticks
        for cm in range(0, 61, 10):
            z_m = cm / 100
            _, sy = phys_to_side(0, z_m)
            pygame.draw.line(scr, DIM_CLR, (TUBE_RX + 8, sy), (TUBE_RX + 18, sy), 1)
            t = self.font_sm.render(f"{cm}", True, DIM_CLR)
            scr.blit(t, (TUBE_RX + 21, sy - t.get_height()//2))

    def _draw_particles(self):
        scr = self.scr
        for p in self.particles:
            sx, sy = phys_to_side(p.x, p.z)
            if TUBE_LX < sx < TUBE_RX and TUBE_TOP_Y < sy < SIDE_BOT_Y:
                brightness = min(255, 120 + int(p.z / TUBE_H * 80))
                c = (50, int(brightness * 0.7), brightness)
                pygame.draw.circle(scr, c, (sx, sy), p.sz)

    def _draw_wind_arrows(self):
        """Animated upward-flow arrows below the rocket."""
        scr = self.scr
        t = self.sim_time
        n = 5
        spacing = TUBE_PIX_W // (n + 1)
        cycle = 30
        for i in range(n):
            sx = TUBE_LX + spacing * (i + 1)
            phase = (t * 60 + i * 7) % cycle
            base_y = SIDE_BOT_Y - 8
            sy = int(base_y - phase * 2.2)
            if sy < TUBE_TOP_Y:
                continue
            length = max(6, int(self.edf_pct / 100 * 18))
            alpha = max(50, int(200 * self.edf_pct / 100))
            c = (70, min(255, 150 + int(alpha*0.3)), min(255, 200 + int(alpha*0.2)))
            pygame.draw.line(scr, c, (sx, sy + length), (sx, sy), 2)
            pygame.draw.line(scr, c, (sx, sy), (sx - 3, sy + 5), 1)
            pygame.draw.line(scr, c, (sx, sy), (sx + 3, sy + 5), 1)

    def _draw_rocket(self):
        scr = self.scr
        s = self.state
        cx, cz = s[0], s[2]
        pitch = s[self.PITCH]
        sx, sy = phys_to_side(cx, cz)

        rw = int(ROCKET_R * SIDE_SCALE)       # half-width px
        rh = int(ROCKET_H * SIDE_SCALE / 2)   # half-height px

        # body (with slight tilt indication)
        tilt_px = int(pitch * rh * 0.5)
        body_pts = [
            (sx - rw, sy + rh + tilt_px),
            (sx + rw, sy + rh - tilt_px),
            (sx + rw, sy - rh - tilt_px),
            (sx - rw, sy - rh + tilt_px),
        ]
        pygame.draw.polygon(scr, ROCKET_CLR, body_pts)
        pygame.draw.polygon(scr, ROCKET_EDGE, body_pts, 2)

        # center-of-gravity marker
        pygame.draw.circle(scr, (255, 255, 255), (sx, sy), 3)
        pygame.draw.circle(scr, ROCKET_EDGE, (sx, sy), 3, 1)

        # ── fins (side view: left & right pair) ──
        fin_px_l = int(FIN_L * SIDE_SCALE / 2)
        fin_px_w = int(FIN_W * SIDE_SCALE)
        fin_base_y = sy + rh + tilt_px - fin_px_l

        dp_deg = math.degrees(self.fins[0])
        deflect_px = int(self.fins[0] * fin_px_w * 0.8)

        # left fin
        lf = [
            (sx - rw, fin_base_y - fin_px_l),
            (sx - rw - fin_px_w, fin_base_y - fin_px_l + deflect_px),
            (sx - rw - fin_px_w, fin_base_y + fin_px_l + deflect_px),
            (sx - rw, fin_base_y + fin_px_l),
        ]
        pygame.draw.polygon(scr, FIN_CLR, lf)
        pygame.draw.polygon(scr, (140, 150, 170), lf, 1)
        # right fin
        rf = [
            (sx + rw, fin_base_y - fin_px_l),
            (sx + rw + fin_px_w, fin_base_y - fin_px_l - deflect_px),
            (sx + rw + fin_px_w, fin_base_y + fin_px_l - deflect_px),
            (sx + rw, fin_base_y + fin_px_l),
        ]
        pygame.draw.polygon(scr, FIN_CLR, rf)
        pygame.draw.polygon(scr, (140, 150, 170), rf, 1)

    def _draw_forces(self):
        if not self.show_forces:
            return
        scr = self.scr
        s = self.state
        sx, sy = phys_to_side(s[0], s[2])
        F_SCALE = 120   # pixels per Newton

        # gravity (down)
        fg_len = int(abs(self.dsp_Fg) * F_SCALE)
        if fg_len > 2:
            pygame.draw.line(scr, F_GRAV_CLR, (sx - 8, sy), (sx - 8, sy + fg_len), 2)
            pygame.draw.polygon(scr, F_GRAV_CLR, [
                (sx - 8, sy + fg_len),
                (sx - 12, sy + fg_len - 6),
                (sx - 4, sy + fg_len - 6)])

        # drag (up)
        fd_len = int(abs(self.dsp_Fd) * F_SCALE)
        if fd_len > 2:
            pygame.draw.line(scr, F_DRAG_CLR, (sx + 8, sy), (sx + 8, sy - fd_len), 2)
            pygame.draw.polygon(scr, F_DRAG_CLR, [
                (sx + 8, sy - fd_len),
                (sx + 4, sy - fd_len + 6),
                (sx + 12, sy - fd_len + 6)])

        # wall force (lateral arrow)
        fw_mag = math.sqrt(self.dsp_Fw[0]**2 + self.dsp_Fw[1]**2)
        if fw_mag > 0.001:
            fw_px = int(self.dsp_Fw[0] * F_SCALE * 3)
            fw_px = np.clip(fw_px, -60, 60)
            pygame.draw.line(scr, F_WALL_CLR, (sx, sy + 15),
                             (sx + fw_px, sy + 15), 2)

    def _draw_top_view(self):
        scr = self.scr
        s = self.state
        cx, cy_off = TOP_CX, TOP_CY

        # background circle (tube)
        pygame.draw.circle(scr, TUBE_INNER, (cx, cy_off), TOP_R)
        pygame.draw.circle(scr, TUBE_CLR, (cx, cy_off), TOP_R, 2)

        # gap boundary (dashed)
        gap_r = int(GAP * TOP_SCALE)
        for ang in range(0, 360, 8):
            a = math.radians(ang)
            x1 = cx + int(gap_r * math.cos(a))
            y1 = cy_off - int(gap_r * math.sin(a))
            x2 = cx + int((gap_r + 3) * math.cos(a))
            y2 = cy_off - int((gap_r + 3) * math.sin(a))
            pygame.draw.line(scr, (50, 70, 90), (x1, y1), (x2, y2), 1)

        # rocket body (circle)
        rocket_r = int(ROCKET_R * TOP_SCALE)
        rx_s, ry_s = phys_to_top(s[0], s[1])
        pygame.draw.circle(scr, ROCKET_CLR, (rx_s, ry_s), rocket_r)
        pygame.draw.circle(scr, ROCKET_EDGE, (rx_s, ry_s), rocket_r, 2)

        # fins (4 lines extending radially)
        fin_inner = rocket_r
        fin_outer = rocket_r + int(FIN_W * TOP_SCALE)
        roll = s[self.ROLL]
        for i in range(4):
            ang = roll + i * math.pi / 2
            x1 = rx_s + int(fin_inner * math.cos(ang))
            y1 = ry_s - int(fin_inner * math.sin(ang))
            x2 = rx_s + int(fin_outer * math.cos(ang))
            y2 = ry_s - int(fin_outer * math.sin(ang))
            pygame.draw.line(scr, FIN_CLR, (x1, y1), (x2, y2), 3)

        # centre marker
        pygame.draw.circle(scr, (255, 255, 255), (cx, cy_off), 2)

        # label
        lbl = self.font_sm.render("Top View (x-y)", True, DIM_CLR)
        scr.blit(lbl, (cx - lbl.get_width()//2, cy_off + TOP_R + 6))

    def _draw_info_panel(self):
        scr = self.scr
        s = self.state

        # semi-transparent panel
        panel = pygame.Surface((270, 750), pygame.SRCALPHA)
        panel.fill((PANEL_BG[0], PANEL_BG[1], PANEL_BG[2], 215))
        scr.blit(panel, (8, 50))

        x0, y = 18, 58
        gap = 19

        def put(txt, clr=TEXT_CLR, font=None):
            nonlocal y
            f = font or self.font
            s = f.render(txt, True, clr)
            scr.blit(s, (x0, y))
            y += gap

        def section(title):
            nonlocal y
            y += 4
            put(title, CYAN, self.font_lg)
            y += 2

        # title
        section("STATE")
        put(f" x  {s[0]*100:+7.2f} cm")
        put(f" y  {s[1]*100:+7.2f} cm")
        put(f" z  {s[2]*100:+7.2f} cm")
        put(f" vz {s[5]:+7.3f} m/s")

        section("ATTITUDE")
        put(f" roll  {math.degrees(s[6]):+6.2f}°")
        put(f" pitch {math.degrees(s[7]):+6.2f}°")
        put(f" yaw   {math.degrees(s[8]):+6.2f}°")

        section("FORCES")
        put(f" Weight {self.dsp_Fg:+.4f} N", F_GRAV_CLR)
        put(f" Drag   {self.dsp_Fd:+.4f} N", F_DRAG_CLR)
        net = self.dsp_Fg + self.dsp_Fd
        put(f" Net-z  {net:+.4f} N", YELLOW)
        section("WALL  (Bernoulli)")
        put(f" Bern(→wall) {self.dsp_Fb:+.4f} N", RED)
        put(f" Visc(→ctr)  {self.dsp_Fv:+.4f} N", GREEN)
        fw_net = self.dsp_Fb - self.dsp_Fv
        clr = RED if fw_net > 0 else GREEN
        put(f" Net-lat     {fw_net:+.4f} N", clr)

        section("CONTROL")
        mode_str = "PID ON" if self.pid_on else "MANUAL"
        mode_clr = GREEN if self.pid_on else RED
        put(f" Mode     {mode_str}", mode_clr)
        hh = "ON" if self.height_hold else "OFF"
        put(f" H-hold   {hh}", GREEN if self.height_hold else DIM_CLR)
        put(f" EDF      {self.edf_pct:5.1f} %")
        put(f" v_wind   {self.v_wind:5.2f} m/s")
        put(f" v_hover  {V_HOVER:5.2f} m/s", DIM_CLR)

        section("FIN DEFLECTION")
        put(f" δ_pitch {math.degrees(self.fins[0]):+6.2f}°", F_FIN_CLR)
        put(f" δ_yaw   {math.degrees(self.fins[1]):+6.2f}°", F_FIN_CLR)
        put(f" δ_roll  {math.degrees(self.fins[2]):+6.2f}°", F_FIN_CLR)

        put(f" time  {self.sim_time:7.2f} s", DIM_CLR)

    def _draw_z_history(self):
        """Small z-height chart on the right panel."""
        scr = self.scr
        if len(self.hist_t) < 3:
            return
        ox, oy = 920, 340
        w, h = 350, 150

        panel = pygame.Surface((w + 10, h + 40), pygame.SRCALPHA)
        panel.fill((PANEL_BG[0], PANEL_BG[1], PANEL_BG[2], 200))
        scr.blit(panel, (ox - 5, oy - 25))

        lbl = self.font_sm.render("Height z(t)  [cm]", True, DIM_CLR)
        scr.blit(lbl, (ox, oy - 20))

        # axes
        pygame.draw.line(scr, DIM_CLR, (ox, oy), (ox, oy + h), 1)
        pygame.draw.line(scr, DIM_CLR, (ox, oy + h), (ox + w, oy + h), 1)

        z_vals = np.array(self.hist_z[-w:])
        t_vals = np.array(self.hist_t[-w:])
        z_cm = z_vals * 100
        z_min = max(0, z_cm.min() - 2)
        z_max = min(60, z_cm.max() + 2)
        if z_max - z_min < 4:
            mid = (z_max + z_min) / 2
            z_min, z_max = mid - 2, mid + 2

        # target line
        zt_cm = self.z_target * 100
        if z_min < zt_cm < z_max:
            yt = int(oy + h - (zt_cm - z_min) / (z_max - z_min) * h)
            pygame.draw.line(scr, (60, 100, 60), (ox, yt), (ox + w, yt), 1)

        # plot
        n = len(z_cm)
        dx = w / max(n - 1, 1)
        pts = []
        for i in range(n):
            px = int(ox + i * dx)
            py = int(oy + h - (z_cm[i] - z_min) / (z_max - z_min) * h)
            py = np.clip(py, oy, oy + h)
            pts.append((px, py))
        if len(pts) > 1:
            pygame.draw.lines(scr, GREEN, False, pts, 2)

        # y-axis labels
        for v in (z_min, z_max, (z_min + z_max) / 2):
            py = int(oy + h - (v - z_min) / (z_max - z_min) * h)
            t = self.font_sm.render(f"{v:.0f}", True, DIM_CLR)
            scr.blit(t, (ox - t.get_width() - 4, py - t.get_height()//2))

    def _draw_help(self):
        scr = self.scr
        lines = [
            "UP/DN: EDF pwr | P: perturb | 1: PID on | 2: PID off | "
            "H: height hold | F: forces | R: reset | SPACE: pause | ESC: quit",
            "PID off → W/S: pitch  A/D: yaw  Q/E: roll",
        ]
        for i, ln in enumerate(lines):
            t = self.font_sm.render(ln, True, DIM_CLR)
            scr.blit(t, (12, SCR_H - 36 + i * 17))

    def _draw_param_box(self):
        """Physical parameters summary on the right."""
        scr = self.scr
        ox, oy = 920, 520
        panel = pygame.Surface((360, 270), pygame.SRCALPHA)
        panel.fill((PANEL_BG[0], PANEL_BG[1], PANEL_BG[2], 200))
        scr.blit(panel, (ox - 5, oy - 5))

        y = oy
        gap = 17

        def put(txt, clr=DIM_CLR):
            nonlocal y
            s = self.font_sm.render(txt, True, clr)
            scr.blit(s, (ox, y))
            y += gap

        put("─── Physical Parameters ───", TEXT_CLR)
        put(f"Tube    ∅{TUBE_R*200:.0f} cm × {TUBE_H*100:.0f} cm")
        put(f"Rocket  ∅{ROCKET_R*200:.0f} cm × {ROCKET_H*100:.0f} cm  "
            f"{ROCKET_M*1000:.0f} g")
        put(f"Gap     {GAP*100:.1f} cm")
        put(f"Cd_axial  {CD_AXIAL:.2f}   Cd_lat  {CD_LATERAL:.2f}")
        put(f"C_bern  {C_BERN_TURB:.3f}  C_visc  {C_VISC_REST:.2f}")
        put(f"Fin  {FIN_L*1000:.0f}×{FIN_W*1000:.0f} mm  "
            f"AR={FIN_AR:.2f}  arm {FIN_ARM*100:.1f} cm")
        put(f"CL_α  {CL_ALPHA:.2f} /rad (low-AR)  "
            f"δ_max {math.degrees(MAX_FIN):.0f}°")
        put(f"I_pitch {I_PITCH:.2e}  I_roll {I_ROLL:.2e}  kg·m²")
        y += 4
        put("─── PID Gains ───", TEXT_CLR)
        put(f"Att  Kp={self.pid_pitch.kp:.1f}  "
            f"Ki={self.pid_pitch.ki:.1f}  "
            f"Kd={self.pid_pitch.kd:.2f}")
        put(f"Pos  Kp={self.pid_x.kp:.1f}  "
            f"Ki={self.pid_x.ki:.1f}  "
            f"Kd={self.pid_x.kd:.1f}")
        put(f"Alt  Kp={self.pid_z.kp:.1f}  "
            f"Ki={self.pid_z.ki:.1f}  "
            f"Kd={self.pid_z.kd:.1f}")
        y += 4
        put(f"ρ = {RHO} kg/m³   g = {G} m/s²")

    def render(self):
        self.scr.fill(BG)

        # title
        title = self.font_tl.render(
            "Wind Tunnel Rocket — EDF Hover Simulation", True, TEXT_CLR)
        self.scr.blit(title, (SCR_W//2 - title.get_width()//2, 10))

        self._draw_tube()
        self._draw_wind_arrows()
        self._draw_particles()
        self._draw_rocket()
        self._draw_forces()
        self._draw_top_view()
        self._draw_info_panel()
        self._draw_z_history()
        self._draw_param_box()
        self._draw_help()

        if self.paused:
            ps = self.font_lg.render("▌▌ PAUSED", True, YELLOW)
            self.scr.blit(ps, (SCR_W//2 - ps.get_width()//2, 42))

        pygame.display.flip()

    # ============================================================
    #  Input handling
    # ============================================================
    def handle_events(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return False
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    return False
                elif ev.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif ev.key == pygame.K_r:
                    self.reset()
                elif ev.key == pygame.K_p:
                    self.perturb()
                elif ev.key == pygame.K_1:
                    self.pid_on = True
                elif ev.key == pygame.K_2:
                    self.pid_on = False
                    self.fins[:] = 0
                elif ev.key == pygame.K_h:
                    self.height_hold = not self.height_hold
                elif ev.key == pygame.K_f:
                    self.show_forces = not self.show_forces

        # continuous key input
        keys = pygame.key.get_pressed()
        step_edf = 0.4
        if keys[pygame.K_UP]:
            self.edf_pct = min(100, self.edf_pct + step_edf)
            if not self.height_hold:
                self.v_wind = self.edf_pct / 100 * V_MAX
        if keys[pygame.K_DOWN]:
            self.edf_pct = max(0, self.edf_pct - step_edf)
            if not self.height_hold:
                self.v_wind = self.edf_pct / 100 * V_MAX

        if not self.pid_on:
            fin_rate = math.radians(2)
            if keys[pygame.K_w]:
                self.fins[0] = min(MAX_FIN, self.fins[0] + fin_rate)
            if keys[pygame.K_s]:
                self.fins[0] = max(-MAX_FIN, self.fins[0] - fin_rate)
            if keys[pygame.K_d]:
                self.fins[1] = min(MAX_FIN, self.fins[1] + fin_rate)
            if keys[pygame.K_a]:
                self.fins[1] = max(-MAX_FIN, self.fins[1] - fin_rate)
            if keys[pygame.K_e]:
                self.fins[2] = min(MAX_FIN, self.fins[2] + fin_rate)
            if keys[pygame.K_q]:
                self.fins[2] = max(-MAX_FIN, self.fins[2] - fin_rate)

        return True

    # ── main loop ────────────────────────────────────────────
    def run(self):
        running = True
        while running:
            running = self.handle_events()
            self.update()
            self.render()
            self.clock.tick(FPS)
        pygame.quit()
        sys.exit()


# ============================================================
#  Entry point
# ============================================================
if __name__ == "__main__":
    sim = WindTunnelSim()
    sim.run()
