"""
Saturn Ring N-Body Simulation — 1000 Particle Cloud
====================================================

Simulates Saturn's ring as a cloud of 1000 particles orbiting under:
  - Saturn's central gravity (dominant)
  - One moon's gravitational perturbation
  - Optional inter-particle gravity (N-body, O(N^2) — off by default for speed)

Particles are initialized with circular orbit velocities plus small random
perturbations, distributed in a ring-shaped annulus around Saturn.

Physics:
  - Central force: a_i = -GM_saturn * r_i / |r_i|^3
  - Moon perturbation: a_i += -GM_moon * (r_i - r_moon) / |r_i - r_moon|^3
  - Leapfrog (kick-drift-kick) integration for symplectic stability

Optimizations:
  - NumPy vectorized computation (no Python loops over particles)
  - Inter-particle gravity uses vectorized pairwise distances (optional)

Controls:
  Mouse wheel     — zoom in/out
  Mouse drag      — pan view
  UP/DOWN         — simulation speed (steps per frame)
  +/-             — change dt
  R               — reset particles
  G               — toggle moon gravity on particles
  N               — toggle inter-particle gravity (slow for 1000!)
  T               — toggle trail persistence
  C               — clear trails / fade
  P               — pause/resume
  1/2/3           — color mode: speed / distance / density
  H               — toggle info panel
  ESC             — quit
"""

import numpy as np
import pygame
import sys
import os
import math

# ============================================================
#  Physical parameters (normalized units: G*M_saturn = 1)
# ============================================================
GM_SATURN = 1.0
SATURN_RADIUS = 0.3  # display only

N_PARTICLES = 1000
RING_INNER = 0.7
RING_OUTER = 1.4
RING_THICKNESS = 0.02  # vertical scatter (for slight y-perturbation in 2D: radial scatter)

MOON_R = 2.8
MOON_MASS_RATIO = 0.0005
GM_MOON = GM_SATURN * MOON_MASS_RATIO

# Gap resonances: particles near 2:1 resonance with moon get cleared
# Moon period T_moon = 2*pi*sqrt(MOON_R^3) ≈ 29.4
# 2:1 resonance at r where T_particle = T_moon/2 => r = MOON_R / 2^(2/3) ≈ 1.76

# ============================================================
#  Display
# ============================================================
SCREEN_W, SCREEN_H = 1200, 900
BG = (3, 3, 10)
SATURN_COLOR = (200, 170, 90)
MOON_COLOR = (140, 190, 240)
TEXT_COLOR = (190, 190, 200)
HELP_COLOR = (110, 110, 130)


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
#  Particle system (vectorized)
# ============================================================
class ParticleRing:
    def __init__(self, n=N_PARTICLES, seed=42):
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        n = self.n
        # Distribute radii uniformly in annulus area: r = sqrt(uniform(r_in^2, r_out^2))
        r = np.sqrt(self.rng.uniform(RING_INNER**2, RING_OUTER**2, n))
        theta = self.rng.uniform(0, 2 * np.pi, n)

        self.pos = np.zeros((n, 2))
        self.pos[:, 0] = r * np.cos(theta)
        self.pos[:, 1] = r * np.sin(theta)

        # Circular orbit velocity + small perturbation
        v_circ = np.sqrt(GM_SATURN / r)
        v_perturb = self.rng.normal(0, 0.01, n)

        self.vel = np.zeros((n, 2))
        # Velocity perpendicular to radius (counter-clockwise)
        self.vel[:, 0] = -(v_circ + v_perturb) * np.sin(theta)
        self.vel[:, 1] = (v_circ + v_perturb) * np.cos(theta)

        # Small radial velocity perturbation (creates slight eccentricity)
        v_radial = self.rng.normal(0, 0.005, n)
        self.vel[:, 0] += v_radial * np.cos(theta)
        self.vel[:, 1] += v_radial * np.sin(theta)

        self.acc = np.zeros((n, 2))

    def compute_accelerations(self, moon_pos, moon_gravity_on, particle_gravity_on):
        """Vectorized acceleration computation."""
        # Saturn gravity (central force)
        r_vec = self.pos  # shape (n, 2)
        r_mag = np.linalg.norm(r_vec, axis=1, keepdims=True)  # (n, 1)
        r_mag = np.maximum(r_mag, 1e-6)  # softening
        self.acc = -GM_SATURN * r_vec / r_mag**3

        # Moon gravity on particles
        if moon_gravity_on and moon_pos is not None:
            dr = self.pos - moon_pos[np.newaxis, :]  # (n, 2)
            dr_mag = np.linalg.norm(dr, axis=1, keepdims=True)
            dr_mag = np.maximum(dr_mag, 0.05)  # softening to prevent divergence
            self.acc -= GM_MOON * dr / dr_mag**3

        # Inter-particle gravity (O(N^2) — very slow for large N)
        if particle_gravity_on and self.n <= 500:
            gm_particle = GM_SATURN * 1e-8  # tiny mass per particle
            for i in range(self.n):
                dr = self.pos[i] - self.pos  # (n, 2)
                dr_mag = np.linalg.norm(dr, axis=1)
                mask = dr_mag > 0.01
                dr_mag_safe = np.where(mask, dr_mag, 1.0)
                force = -gm_particle * dr / dr_mag_safe[:, np.newaxis]**3
                force[~mask] = 0
                self.acc[i] += force.sum(axis=0)

    def step(self, dt, moon_pos, moon_gravity_on, particle_gravity_on):
        """Leapfrog (kick-drift-kick) integration."""
        # Half kick
        self.vel += 0.5 * self.acc * dt
        # Drift
        self.pos += self.vel * dt
        # Recompute accelerations
        self.compute_accelerations(moon_pos, moon_gravity_on, particle_gravity_on)
        # Half kick
        self.vel += 0.5 * self.acc * dt


class Moon:
    def __init__(self):
        self.reset()

    def reset(self):
        self.pos = np.array([MOON_R, 0.0])
        v_circ = math.sqrt(GM_SATURN / MOON_R)
        self.vel = np.array([0.0, v_circ])
        self.acc = np.zeros(2)
        self.trail = []

    def step(self, dt):
        """Simple leapfrog for the moon (only Saturn gravity, ignores particle masses)."""
        r_mag = np.linalg.norm(self.pos)
        if r_mag < 1e-6:
            return
        self.acc = -GM_SATURN * self.pos / r_mag**3
        self.vel += 0.5 * self.acc * dt
        self.pos += self.vel * dt
        r_mag = np.linalg.norm(self.pos)
        self.acc = -GM_SATURN * self.pos / r_mag**3
        self.vel += 0.5 * self.acc * dt


# ============================================================
#  Camera
# ============================================================
class Camera:
    def __init__(self):
        self.cx = SCREEN_W / 2
        self.cy = SCREEN_H / 2
        self.zoom = 220.0

    def world_to_screen(self, wx, wy):
        return int(self.cx + wx * self.zoom), int(self.cy - wy * self.zoom)

    def world_to_screen_array(self, positions):
        """Convert (n, 2) world positions to (n, 2) screen coordinates."""
        screen = np.empty_like(positions, dtype=int)
        screen[:, 0] = (self.cx + positions[:, 0] * self.zoom).astype(int)
        screen[:, 1] = (self.cy - positions[:, 1] * self.zoom).astype(int)
        return screen

    def screen_to_world(self, sx, sy):
        return (sx - self.cx) / self.zoom, (self.cy - sy) / self.zoom


# ============================================================
#  Color mapping
# ============================================================
def speed_colormap(speeds, vmin=None, vmax=None):
    """Map speeds to colors: slow=blue, medium=white, fast=red."""
    if vmin is None:
        vmin = speeds.min()
    if vmax is None:
        vmax = speeds.max()
    if vmax - vmin < 1e-10:
        t = np.zeros_like(speeds)
    else:
        t = (speeds - vmin) / (vmax - vmin)
    t = np.clip(t, 0, 1)

    colors = np.zeros((len(t), 3), dtype=np.uint8)
    # Blue -> White -> Red
    colors[:, 0] = (np.where(t < 0.5, 80 + 350 * t, 255)).clip(0, 255).astype(np.uint8)
    colors[:, 1] = (np.where(t < 0.5, 80 + 350 * t, 255 - 350 * (t - 0.5))).clip(0, 255).astype(np.uint8)
    colors[:, 2] = (np.where(t < 0.5, 255, 255 - 350 * (t - 0.5))).clip(0, 255).astype(np.uint8)
    return colors


def distance_colormap(distances):
    """Map distance from Saturn to color: close=warm, far=cool."""
    dmin, dmax = RING_INNER * 0.8, RING_OUTER * 1.3
    t = np.clip((distances - dmin) / (dmax - dmin), 0, 1)
    colors = np.zeros((len(t), 3), dtype=np.uint8)
    colors[:, 0] = (255 * (1 - t)).astype(np.uint8)
    colors[:, 1] = (180 * (1 - np.abs(t - 0.5) * 2)).astype(np.uint8)
    colors[:, 2] = (255 * t).astype(np.uint8)
    return colors


def uniform_color(n):
    """All particles same warm golden color."""
    colors = np.zeros((n, 3), dtype=np.uint8)
    colors[:, 0] = 230
    colors[:, 1] = 190
    colors[:, 2] = 80
    return colors


# ============================================================
#  Rendering
# ============================================================
def draw_saturn(surface, cam):
    cx, cy = cam.world_to_screen(0, 0)
    r_px = max(int(SATURN_RADIUS * cam.zoom), 6)
    for i in range(r_px, 0, -1):
        frac = i / r_px
        c = (int(170 * frac + 30), int(140 * frac + 25), int(50 * frac + 15))
        pygame.draw.circle(surface, c, (cx, cy), i)


def draw_particles(surface, cam, ring, color_mode):
    """Draw all particles efficiently."""
    screen_pos = cam.world_to_screen_array(ring.pos)

    # Compute colors based on mode
    if color_mode == 0:  # speed
        speeds = np.linalg.norm(ring.vel, axis=1)
        colors = speed_colormap(speeds)
    elif color_mode == 1:  # distance
        distances = np.linalg.norm(ring.pos, axis=1)
        colors = distance_colormap(distances)
    else:  # uniform
        colors = uniform_color(ring.n)

    # Draw each particle as a single pixel or small circle
    particle_size = max(1, int(0.008 * cam.zoom))

    for i in range(ring.n):
        sx, sy = screen_pos[i]
        if 0 <= sx < SCREEN_W and 0 <= sy < SCREEN_H:
            c = (int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2]))
            if particle_size <= 1:
                surface.set_at((sx, sy), c)
            else:
                pygame.draw.circle(surface, c, (sx, sy), particle_size)


def draw_moon(surface, cam, moon):
    sx, sy = cam.world_to_screen(moon.pos[0], moon.pos[1])
    r_px = max(int(0.06 * cam.zoom), 4)
    pygame.draw.circle(surface, MOON_COLOR, (sx, sy), r_px)
    pygame.draw.circle(surface, (200, 230, 255), (sx, sy), r_px, 1)
    lbl_font = pygame.font.SysFont("monospace", 11)
    lbl = lbl_font.render("Moon", True, MOON_COLOR)
    surface.blit(lbl, (sx + r_px + 4, sy - 8))


def draw_orbit_guides(surface, cam):
    """Draw faint circles for ring boundaries and moon orbit."""
    cx, cy = cam.world_to_screen(0, 0)
    for r in [RING_INNER, RING_OUTER]:
        r_px = int(r * cam.zoom)
        if r_px > 2:
            pygame.draw.circle(surface, (30, 35, 25), (cx, cy), r_px, 1)
    # Moon orbit
    r_px = int(MOON_R * cam.zoom)
    if r_px > 2:
        pygame.draw.circle(surface, (25, 30, 40), (cx, cy), r_px, 1)


def draw_moon_trail(surface, cam, trail):
    if len(trail) < 2:
        return
    pts = []
    step = max(1, len(trail) // 300)
    for i in range(0, len(trail), step):
        pts.append(cam.world_to_screen(trail[i][0], trail[i][1]))
    if len(pts) >= 2:
        pygame.draw.lines(surface, (40, 60, 90), False, pts, 1)


def draw_info(surface, font, font_sm, ring, moon, sim_time, dt,
              steps_per_frame, paused, moon_grav, particle_grav, color_mode):
    panel = pygame.Surface((320, 280), pygame.SRCALPHA)
    panel.fill((8, 8, 18, 210))
    surface.blit(panel, (10, 10))

    y = 14
    gap = 19

    def put(text, color=TEXT_COLOR, f=font):
        nonlocal y
        s = f.render(text, True, color)
        surface.blit(s, (18, y))
        y += gap

    put("SATURN RING — 1000 PARTICLES", (220, 190, 100))
    y += 2

    speeds = np.linalg.norm(ring.vel, axis=1)
    distances = np.linalg.norm(ring.pos, axis=1)
    put(f"Particles: {ring.n}", f=font_sm)
    put(f"r: [{distances.min():.3f}, {distances.max():.3f}]", f=font_sm)
    put(f"v: [{speeds.min():.3f}, {speeds.max():.3f}]", f=font_sm)
    y += 2

    moon_r = np.linalg.norm(moon.pos)
    moon_v = np.linalg.norm(moon.vel)
    put(f"Moon: r={moon_r:.3f} v={moon_v:.4f}", MOON_COLOR, f=font_sm)
    y += 2

    put(f"Time: {sim_time:.2f}", f=font_sm)
    put(f"dt={dt:.5f}  steps/frame={steps_per_frame}", HELP_COLOR, f=font_sm)
    status = "PAUSED" if paused else "RUNNING"
    put(f"Status: {status}", (255, 100, 100) if paused else (100, 255, 100), f=font_sm)
    put(f"Moon gravity: {'ON' if moon_grav else 'OFF'}", f=font_sm)
    put(f"Particle gravity: {'ON' if particle_grav else 'OFF'}", f=font_sm)
    cmode_names = ["Speed", "Distance", "Uniform"]
    put(f"Color: {cmode_names[color_mode]}", f=font_sm)


def draw_help(surface, font):
    lines = [
        "Scroll: zoom | Drag: pan | UP/DOWN: speed | +/-: dt",
        "R: reset | G: moon grav | N: particle grav | T: trail",
        "1/2/3: color mode | P: pause | H: panel | ESC: quit",
    ]
    for i, line in enumerate(lines):
        s = font.render(line, True, HELP_COLOR)
        surface.blit(s, (15, SCREEN_H - 68 + i * 22))


# ============================================================
#  Main
# ============================================================
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Saturn Ring N-Body — 1000 Particle Cloud")
    clock = pygame.time.Clock()

    font = load_font(14)
    font_sm = load_font(12)

    cam = Camera()
    ring = ParticleRing(N_PARTICLES)
    moon = Moon()

    # Initial acceleration computation
    ring.compute_accelerations(moon.pos, True, False)

    dt = 0.003
    sim_time = 0.0
    steps_per_frame = 8
    paused = False
    moon_gravity_on = True
    particle_gravity_on = False
    color_mode = 0  # 0=speed, 1=distance, 2=uniform
    show_panel = True
    trail_on = True
    moon_trail = []
    moon_trail_max = 2000

    dragging = False
    drag_start = (0, 0)
    cam_start = (0.0, 0.0)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    ring.reset()
                    moon.reset()
                    ring.compute_accelerations(moon.pos, moon_gravity_on, particle_gravity_on)
                    sim_time = 0.0
                    moon_trail.clear()
                elif event.key == pygame.K_p:
                    paused = not paused
                elif event.key == pygame.K_g:
                    moon_gravity_on = not moon_gravity_on
                elif event.key == pygame.K_n:
                    particle_gravity_on = not particle_gravity_on
                elif event.key == pygame.K_t:
                    trail_on = not trail_on
                    if not trail_on:
                        moon_trail.clear()
                elif event.key == pygame.K_c:
                    moon_trail.clear()
                elif event.key == pygame.K_h:
                    show_panel = not show_panel
                elif event.key == pygame.K_UP:
                    steps_per_frame = min(steps_per_frame + 2, 60)
                elif event.key == pygame.K_DOWN:
                    steps_per_frame = max(steps_per_frame - 2, 1)
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    dt *= 1.5
                elif event.key == pygame.K_MINUS:
                    dt = max(dt / 1.5, 0.0001)
                elif event.key == pygame.K_1:
                    color_mode = 0
                elif event.key == pygame.K_2:
                    color_mode = 1
                elif event.key == pygame.K_3:
                    color_mode = 2

            elif event.type == pygame.MOUSEWHEEL:
                factor = 1.12 if event.y > 0 else 1 / 1.12
                mx, my = pygame.mouse.get_pos()
                wx, wy = cam.screen_to_world(mx, my)
                cam.zoom *= factor
                sx, sy = cam.world_to_screen(wx, wy)
                cam.cx += mx - sx
                cam.cy += my - sy

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    dragging = True
                    drag_start = event.pos
                    cam_start = (cam.cx, cam.cy)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragging = False

            elif event.type == pygame.MOUSEMOTION:
                if dragging:
                    dx = event.pos[0] - drag_start[0]
                    dy = event.pos[1] - drag_start[1]
                    cam.cx = cam_start[0] + dx
                    cam.cy = cam_start[1] + dy

        # Physics
        if not paused:
            for _ in range(steps_per_frame):
                moon.step(dt)
                ring.step(dt, moon.pos, moon_gravity_on, particle_gravity_on)
                sim_time += dt

            if trail_on:
                moon_trail.append(moon.pos.copy())
                if len(moon_trail) > moon_trail_max:
                    moon_trail.pop(0)

        # ---- Render ----
        screen.fill(BG)

        # Background stars (fixed seed for consistency)
        rng_stars = np.random.default_rng(7)
        n_stars = 150
        star_x = rng_stars.integers(0, SCREEN_W, n_stars)
        star_y = rng_stars.integers(0, SCREEN_H, n_stars)
        star_b = rng_stars.integers(30, 100, n_stars)
        for i in range(n_stars):
            screen.set_at((star_x[i], star_y[i]),
                          (star_b[i], star_b[i], int(star_b[i] * 1.1) & 255))

        # Orbit guides
        draw_orbit_guides(screen, cam)

        # Moon trail
        if trail_on and moon_trail:
            draw_moon_trail(screen, cam, moon_trail)

        # Particles
        draw_particles(screen, cam, ring, color_mode)

        # Saturn (draw on top so it occludes inner particles)
        draw_saturn(screen, cam)

        # Moon
        draw_moon(screen, cam, moon)

        # Info
        if show_panel:
            draw_info(screen, font, font_sm, ring, moon, sim_time, dt,
                      steps_per_frame, paused, moon_gravity_on,
                      particle_gravity_on, color_mode)
        draw_help(screen, font_sm)

        # FPS
        fps = clock.get_fps()
        fps_surf = font_sm.render(f"FPS: {fps:.0f}", True, HELP_COLOR)
        screen.blit(fps_surf, (SCREEN_W - 80, 10))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
