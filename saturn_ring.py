"""
Saturn Ring Simulation — Saturn + Ring Particle + Moon
======================================================

A minimal N-body gravitational simulation showing:
  - Saturn (central massive body, stationary)
  - One ring particle orbiting Saturn
  - One moon (satellite) orbiting Saturn at a larger radius

Physics:
  Gravitational acceleration on body i from body j:
    a_i = -G * M_j * (r_i - r_j) / |r_i - r_j|^3

  Integration: Velocity-Verlet (symplectic, energy-conserving)

The simulation demonstrates:
  - Stable circular/elliptical orbits
  - Gravitational perturbation of the ring particle by the moon
  - Orbital resonance effects when parameters are tuned

Controls:
  Mouse wheel    — zoom in/out
  Mouse drag     — pan the view
  UP/DOWN        — speed up / slow down simulation
  R              — reset to initial conditions
  T              — toggle orbit trails
  G              — toggle gravity from moon on particle
  P              — pause/resume
  SPACE          — single step (when paused)
  ESC            — quit
"""

import numpy as np
import pygame
import sys
import os
import math

# ============================================================
#  Physical constants (scaled units for visualization)
#  Using units: distance in 10^6 m, mass in 10^24 kg, time in hours
# ============================================================
G_REAL = 6.674e-11  # m^3 kg^-1 s^-2

# We use a simplified unit system where G*M_saturn = some convenient number
# Saturn mass ~ 5.683e26 kg, ring radius ~ 1e8 m
# Let's set G*M = 1.0 and scale distances so ring is at r~1, moon at r~3
GM_SATURN = 1.0
SATURN_RADIUS_DISPLAY = 0.4  # display radius of Saturn (not physical)

# Ring particle: circular orbit at r = 1.0
RING_R = 1.0
RING_V = math.sqrt(GM_SATURN / RING_R)  # circular orbit speed

# Moon: circular orbit at r = 3.0
MOON_R = 3.0
MOON_V = math.sqrt(GM_SATURN / MOON_R)
MOON_MASS_RATIO = 0.0002  # moon mass / saturn mass (small perturbation)
GM_MOON = GM_SATURN * MOON_MASS_RATIO

# ============================================================
#  Display constants
# ============================================================
SCREEN_W, SCREEN_H = 1100, 800
BG = (5, 5, 15)
SATURN_COLOR = (210, 180, 100)
SATURN_RING_COLOR = (180, 160, 110, 80)
PARTICLE_COLOR = (255, 200, 80)
MOON_COLOR = (160, 200, 240)
TRAIL_PARTICLE = (255, 180, 50)
TRAIL_MOON = (100, 160, 220)
GRID_COLOR = (20, 20, 35)
TEXT_COLOR = (200, 200, 210)
HELP_COLOR = (120, 120, 140)
ORBIT_GUIDE = (40, 40, 60)


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
#  Simulation state
# ============================================================
class Body:
    def __init__(self, pos, vel, gm, radius_display, color, name):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.acc = np.zeros(2, dtype=float)
        self.gm = gm  # G * mass of this body (for gravitational pull on others)
        self.radius_display = radius_display
        self.color = color
        self.name = name
        self.trail = []
        self.trail_max = 3000


def initial_conditions():
    """Create initial state: ring particle + moon in circular orbits."""
    particle = Body(
        pos=[RING_R, 0.0],
        vel=[0.0, RING_V],
        gm=0.0,  # massless particle
        radius_display=0.03,
        color=PARTICLE_COLOR,
        name="Ring Particle"
    )
    moon = Body(
        pos=[MOON_R, 0.0],
        vel=[0.0, MOON_V],
        gm=GM_MOON,
        radius_display=0.12,
        color=MOON_COLOR,
        name="Moon"
    )
    return particle, moon


def compute_accelerations(bodies, moon_gravity_on):
    """Compute gravitational accelerations on each body."""
    for b in bodies:
        b.acc = np.zeros(2, dtype=float)

        # Gravity from Saturn (at origin)
        r = b.pos
        r_mag = np.linalg.norm(r)
        if r_mag > 1e-8:
            b.acc -= GM_SATURN * r / r_mag**3

    # Mutual gravity between bodies
    if moon_gravity_on:
        for i in range(len(bodies)):
            for j in range(len(bodies)):
                if i == j:
                    continue
                r_ij = bodies[i].pos - bodies[j].pos
                r_mag = np.linalg.norm(r_ij)
                if r_mag > 1e-8:
                    bodies[i].acc -= bodies[j].gm * r_ij / r_mag**3


def step_verlet(bodies, dt, moon_gravity_on):
    """Velocity-Verlet integration step."""
    # Half-step velocity
    for b in bodies:
        b.vel += 0.5 * b.acc * dt

    # Full-step position
    for b in bodies:
        b.pos += b.vel * dt

    # Recompute accelerations at new position
    compute_accelerations(bodies, moon_gravity_on)

    # Half-step velocity again
    for b in bodies:
        b.vel += 0.5 * b.acc * dt


def compute_orbital_elements(body):
    """Compute basic orbital elements for info display."""
    r = np.linalg.norm(body.pos)
    v = np.linalg.norm(body.vel)
    energy = 0.5 * v**2 - GM_SATURN / r  # specific orbital energy
    # Semi-major axis: a = -GM / (2*E) for bound orbits
    if abs(energy) > 1e-12 and energy < 0:
        a = -GM_SATURN / (2 * energy)
    else:
        a = float('inf')
    # Angular momentum (scalar, z-component)
    Lz = body.pos[0] * body.vel[1] - body.pos[1] * body.vel[0]
    # Eccentricity
    if a != float('inf') and a > 0:
        e = math.sqrt(max(0, 1 - Lz**2 / (GM_SATURN * a)))
    else:
        e = 0.0
    # Period
    if a > 0 and a != float('inf'):
        T = 2 * math.pi * math.sqrt(a**3 / GM_SATURN)
    else:
        T = float('inf')
    return r, v, energy, a, e, Lz, T


# ============================================================
#  Rendering
# ============================================================
class Camera:
    def __init__(self):
        self.offset_x = SCREEN_W / 2
        self.offset_y = SCREEN_H / 2
        self.zoom = 120.0  # pixels per unit distance

    def world_to_screen(self, wx, wy):
        sx = int(self.offset_x + wx * self.zoom)
        sy = int(self.offset_y - wy * self.zoom)
        return sx, sy

    def screen_to_world(self, sx, sy):
        wx = (sx - self.offset_x) / self.zoom
        wy = (self.offset_y - sy) / self.zoom
        return wx, wy


def draw_saturn(surface, cam):
    """Draw Saturn with decorative rings."""
    cx, cy = cam.world_to_screen(0, 0)
    r_px = max(int(SATURN_RADIUS_DISPLAY * cam.zoom), 8)

    # Saturn body (gradient-like concentric circles)
    for i in range(r_px, 0, -1):
        frac = i / r_px
        c = (
            int(180 * frac + 30),
            int(150 * frac + 30),
            int(60 * frac + 20),
        )
        pygame.draw.circle(surface, c, (cx, cy), i)

    # Decorative ring bands (display only, not the simulated particle)
    overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
    for ring_r in [0.7, 0.8, 0.9, 1.0, 1.1]:
        rr_px = int(ring_r * cam.zoom)
        if rr_px > 2:
            alpha = 30 if ring_r < 0.9 else 20
            pygame.draw.circle(overlay, (180, 160, 110, alpha), (cx, cy), rr_px, 1)
    surface.blit(overlay, (0, 0))


def draw_orbit_guide(surface, cam, radius, color=ORBIT_GUIDE):
    """Draw a faint circle showing the expected circular orbit."""
    cx, cy = cam.world_to_screen(0, 0)
    r_px = int(radius * cam.zoom)
    if r_px > 2:
        pygame.draw.circle(surface, color, (cx, cy), r_px, 1)


def draw_body(surface, cam, body):
    sx, sy = cam.world_to_screen(body.pos[0], body.pos[1])
    r_px = max(int(body.radius_display * cam.zoom), 3)
    pygame.draw.circle(surface, body.color, (sx, sy), r_px)
    pygame.draw.circle(surface, (255, 255, 255), (sx, sy), r_px, 1)


def draw_trail(surface, cam, trail, color):
    if len(trail) < 2:
        return
    n = len(trail)
    # Draw with fading alpha
    step = max(1, n // 500)
    pts = []
    for i in range(0, n, step):
        sx, sy = cam.world_to_screen(trail[i][0], trail[i][1])
        pts.append((sx, sy))
    if len(pts) >= 2:
        # Fade from dim to bright
        segments = min(len(pts) - 1, 200)
        start_idx = max(0, len(pts) - 1 - segments)
        for i in range(start_idx, len(pts) - 1):
            frac = (i - start_idx) / max(segments, 1)
            c = (
                int(color[0] * (0.2 + 0.8 * frac)),
                int(color[1] * (0.2 + 0.8 * frac)),
                int(color[2] * (0.2 + 0.8 * frac)),
            )
            pygame.draw.line(surface, c, pts[i], pts[i + 1], 1)


def draw_velocity_arrow(surface, cam, body, scale=0.3):
    """Draw velocity vector as an arrow."""
    sx, sy = cam.world_to_screen(body.pos[0], body.pos[1])
    vx_px = body.vel[0] * cam.zoom * scale
    vy_px = -body.vel[1] * cam.zoom * scale
    end_x = sx + int(vx_px)
    end_y = sy + int(vy_px)
    pygame.draw.line(surface, body.color, (sx, sy), (end_x, end_y), 2)
    # Small arrowhead
    length = math.sqrt(vx_px**2 + vy_px**2)
    if length > 5:
        ux, uy = vx_px / length, vy_px / length
        px, py = -uy, ux
        hs = 6
        tip = (end_x, end_y)
        left = (int(tip[0] - hs * ux + hs * 0.4 * px),
                int(tip[1] - hs * uy + hs * 0.4 * py))
        right = (int(tip[0] - hs * ux - hs * 0.4 * px),
                 int(tip[1] - hs * uy - hs * 0.4 * py))
        pygame.draw.polygon(surface, body.color, [tip, left, right])


def draw_info_panel(surface, font, font_sm, particle, moon, sim_time,
                    dt, paused, trail_on, moon_grav_on):
    panel = pygame.Surface((340, 400), pygame.SRCALPHA)
    panel.fill((10, 10, 20, 210))
    surface.blit(panel, (10, 10))

    y = 16
    gap = 20

    def put(text, color=TEXT_COLOR, f=font):
        nonlocal y
        s = f.render(text, True, color)
        surface.blit(s, (20, y))
        y += gap

    put("SATURN RING SIMULATION", (220, 200, 130))
    y += 4

    # Particle info
    r_p, v_p, E_p, a_p, e_p, L_p, T_p = compute_orbital_elements(particle)
    put("Ring Particle:", PARTICLE_COLOR)
    put(f"  r={r_p:.4f}  v={v_p:.4f}", f=font_sm)
    put(f"  a={a_p:.4f}  e={e_p:.6f}", f=font_sm)
    put(f"  E={E_p:.6f}  L={L_p:.4f}", f=font_sm)
    put(f"  T={T_p:.3f}", f=font_sm)
    y += 4

    # Moon info
    r_m, v_m, E_m, a_m, e_m, L_m, T_m = compute_orbital_elements(moon)
    put("Moon:", MOON_COLOR)
    put(f"  r={r_m:.4f}  v={v_m:.4f}", f=font_sm)
    put(f"  a={a_m:.4f}  e={e_m:.6f}", f=font_sm)
    put(f"  T={T_m:.3f}", f=font_sm)
    y += 4

    # Resonance ratio
    if T_p > 0 and T_m > 0 and T_p != float('inf') and T_m != float('inf'):
        ratio = T_m / T_p
        put(f"Period ratio (moon/particle): {ratio:.3f}", HELP_COLOR, f=font_sm)
    y += 4

    put(f"Time: {sim_time:.2f}  dt={dt:.5f}", HELP_COLOR, f=font_sm)
    status = "PAUSED" if paused else "RUNNING"
    put(f"Status: {status}", (255, 100, 100) if paused else (100, 255, 100), f=font_sm)
    put(f"Trail: {'ON' if trail_on else 'OFF'}  Moon grav: {'ON' if moon_grav_on else 'OFF'}",
        HELP_COLOR, f=font_sm)


def draw_help(surface, font):
    lines = [
        "Scroll: zoom | Drag: pan | UP/DOWN: speed | R: reset",
        "T: trail | G: moon gravity | P: pause | SPACE: step | ESC: quit",
    ]
    for i, line in enumerate(lines):
        s = font.render(line, True, HELP_COLOR)
        surface.blit(s, (15, SCREEN_H - 50 + i * 22))


# ============================================================
#  Main loop
# ============================================================
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Saturn Ring — Particle + Moon Gravity Simulation")
    clock = pygame.time.Clock()

    font = load_font(15)
    font_sm = load_font(12)

    cam = Camera()
    particle, moon = initial_conditions()
    bodies = [particle, moon]
    compute_accelerations(bodies, True)

    dt = 0.002
    sim_time = 0.0
    steps_per_frame = 10
    paused = False
    trail_on = True
    moon_gravity_on = True
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
                    particle, moon = initial_conditions()
                    bodies = [particle, moon]
                    compute_accelerations(bodies, moon_gravity_on)
                    sim_time = 0.0
                elif event.key == pygame.K_p:
                    paused = not paused
                elif event.key == pygame.K_SPACE:
                    # Single step when paused
                    if paused:
                        for _ in range(steps_per_frame):
                            step_verlet(bodies, dt, moon_gravity_on)
                            sim_time += dt
                        for b in bodies:
                            if trail_on:
                                b.trail.append(b.pos.copy())
                                if len(b.trail) > b.trail_max:
                                    b.trail.pop(0)
                elif event.key == pygame.K_t:
                    trail_on = not trail_on
                    if not trail_on:
                        for b in bodies:
                            b.trail.clear()
                elif event.key == pygame.K_g:
                    moon_gravity_on = not moon_gravity_on
                elif event.key == pygame.K_UP:
                    steps_per_frame = min(steps_per_frame + 5, 100)
                elif event.key == pygame.K_DOWN:
                    steps_per_frame = max(steps_per_frame - 5, 1)

            elif event.type == pygame.MOUSEWHEEL:
                # Zoom
                factor = 1.15 if event.y > 0 else 1 / 1.15
                mx, my = pygame.mouse.get_pos()
                # Zoom toward mouse position
                wx, wy = cam.screen_to_world(mx, my)
                cam.zoom *= factor
                new_sx, new_sy = cam.world_to_screen(wx, wy)
                cam.offset_x += mx - new_sx
                cam.offset_y += my - new_sy

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    dragging = True
                    drag_start = event.pos
                    cam_start = (cam.offset_x, cam.offset_y)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragging = False

            elif event.type == pygame.MOUSEMOTION:
                if dragging:
                    dx = event.pos[0] - drag_start[0]
                    dy = event.pos[1] - drag_start[1]
                    cam.offset_x = cam_start[0] + dx
                    cam.offset_y = cam_start[1] + dy

        # Physics update
        if not paused:
            for _ in range(steps_per_frame):
                step_verlet(bodies, dt, moon_gravity_on)
                sim_time += dt

            for b in bodies:
                if trail_on:
                    b.trail.append(b.pos.copy())
                    if len(b.trail) > b.trail_max:
                        b.trail.pop(0)

        # ---- Rendering ----
        screen.fill(BG)

        # Background stars
        rng = np.random.default_rng(42)
        for _ in range(80):
            sx = int(rng.uniform(0, SCREEN_W))
            sy = int(rng.uniform(0, SCREEN_H))
            brightness = int(rng.uniform(40, 120))
            screen.set_at((sx, sy), (brightness, brightness, brightness))

        # Orbit guides
        draw_orbit_guide(screen, cam, RING_R, (40, 50, 30))
        draw_orbit_guide(screen, cam, MOON_R, (30, 40, 50))

        # Trails
        if trail_on:
            draw_trail(screen, cam, particle.trail, TRAIL_PARTICLE)
            draw_trail(screen, cam, moon.trail, TRAIL_MOON)

        # Saturn
        draw_saturn(screen, cam)

        # Bodies
        draw_body(screen, cam, moon)
        draw_body(screen, cam, particle)

        # Velocity arrows
        draw_velocity_arrow(screen, cam, particle, scale=0.2)
        draw_velocity_arrow(screen, cam, moon, scale=0.2)

        # Gravitational force direction indicator (particle <- moon)
        if moon_gravity_on:
            r_pm = moon.pos - particle.pos
            r_pm_mag = np.linalg.norm(r_pm)
            if r_pm_mag > 0.01:
                direction = r_pm / r_pm_mag
                sp = cam.world_to_screen(particle.pos[0], particle.pos[1])
                arrow_len = 15
                end = (sp[0] + int(direction[0] * arrow_len),
                       sp[1] - int(direction[1] * arrow_len))
                pygame.draw.line(screen, (255, 100, 100), sp, end, 1)

        # Labels
        for b in bodies:
            sx, sy = cam.world_to_screen(b.pos[0], b.pos[1])
            lbl = font_sm.render(b.name, True, b.color)
            screen.blit(lbl, (sx + 10, sy - 15))

        # Info panel
        draw_info_panel(screen, font, font_sm, particle, moon,
                        sim_time, dt, paused, trail_on, moon_gravity_on)
        draw_help(screen, font_sm)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
