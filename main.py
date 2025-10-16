import pygame
import numpy as np
import sounddevice as sd
import threading
import queue
import math
import time
import random
import os
from datetime import datetime

# Simple water ripple simulation using a discrete wave equation
# - Mouse click/drag creates disturbances
# - Microphone audio level modulates ripple intensity and color
# - Day/night cycle and basic particle splashes

WIDTH, HEIGHT = 800, 600
DAMPING = 0.995

# Audio capture queue
audio_q = queue.Queue()


def audio_callback(indata, frames, time_info, status):
    volume = np.linalg.norm(indata) / frames
    try:
        audio_q.put_nowait(volume)
    except queue.Full:
        pass


class Ripples:
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.current = np.zeros((h, w), dtype=np.float32)
        self.previous = np.zeros_like(self.current)

    def step(self):
        # discrete wave equation
        lap = (
            np.roll(self.current, 1, axis=0)
            + np.roll(self.current, -1, axis=0)
            + np.roll(self.current, 1, axis=1)
            + np.roll(self.current, -1, axis=1)
            - 4 * self.current
        )
        new = 2 * self.current - self.previous + 0.5 * lap
        new *= DAMPING
        self.previous = self.current
        self.current = new

    def disturb(self, x, y, magnitude=1.0):
        if 0 <= x < self.w and 0 <= y < self.h:
            self.current[y, x] += magnitude


class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.vx = random.uniform(-2, 2)
        self.vy = random.uniform(-5, -1)
        self.life = 60
        self.color = color

    def step(self):
        self.vy += 0.2
        self.x += self.vx
        self.y += self.vy
        self.life -= 1


def run():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    ripples = Ripples(WIDTH, HEIGHT)
    particles = []
    # UI / visual flags
    normal_shading = True
    audio_enabled = True
    intensity = 1.0
    paused = False

    # font for HUD
    try:
        font = pygame.font.SysFont("Arial", 16)
    except Exception:
        pygame.font.init()
        font = pygame.font.SysFont("Arial", 16)

    # audio stream
    stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=44100)
    stream.start()

    running = True
    last_mouse = None
    day_time = 0.0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                ripples.disturb(mx, my, magnitude=50)
                for _ in range(8):
                    particles.append(Particle(mx, my, (255, 255, 255)))
            elif event.type == pygame.MOUSEMOTION and pygame.mouse.get_pressed()[0]:
                mx, my = pygame.mouse.get_pos()
                if last_mouse:
                    dx = mx - last_mouse[0]
                    dy = my - last_mouse[1]
                    mag = min(100, math.hypot(dx, dy))
                    ripples.disturb(mx, my, magnitude=mag * 0.5)
                last_mouse = (mx, my)
            elif event.type == pygame.MOUSEBUTTONUP:
                last_mouse = None
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m:
                    normal_shading = not normal_shading
                elif event.key == pygame.K_a:
                    audio_enabled = not audio_enabled
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    intensity = min(5.0, intensity + 0.1)
                elif event.key == pygame.K_MINUS:
                    intensity = max(0.1, intensity - 0.1)
                elif event.key == pygame.K_p:
                    paused = not paused
                elif event.key == pygame.K_o:
                    fname = f"ripple_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    pygame.image.save(screen, fname)
                    print(f"Saved {fname}")

        # audio influence
        audio_level = 0.0
        try:
            while True:
                audio_level = max(audio_level, audio_q.get_nowait())
        except queue.Empty:
            pass

        # day/night cycle
        day_time += 0.005
        t = (math.sin(day_time) + 1) / 2  # 0..1
        bg_color = (
            int(30 + 100 * (1 - t)),
            int(60 + 120 * (1 - t)),
            int(90 + 165 * t),
        )
        screen.fill(bg_color)

        # step simulation
        ripples.step()

        # render ripples with color mapping and optional normal-based shading
        height_map = ripples.current  # float32

        # simple color palettes for day/night interpolation
        day_palette = np.array([200, 230, 255], dtype=np.float32)
        night_palette = np.array([20, 30, 60], dtype=np.float32)
        tcol = t  # from day/night cycle earlier
        base_col = (1 - tcol) * day_palette + tcol * night_palette

        # compute normals from height map
        if normal_shading:
            gy, gx = np.gradient(height_map)
            nx = -gx
            ny = -gy
            nz = np.ones_like(nx) * 0.5
            norm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-8
            nx /= norm
            ny /= norm
            nz /= norm
            # light direction
            lx, ly, lz = 0.5, -0.5, 1.0
            lnorm = math.sqrt(lx * lx + ly * ly + lz * lz)
            lx /= lnorm; ly /= lnorm; lz /= lnorm
            dot = np.clip(nx * lx + ny * ly + nz * lz, 0.0, 1.0)
        else:
            dot = np.ones_like(height_map) * 0.7

        # map height to a color tint
        # normalize height to -1..1
        hnorm = np.clip(height_map * intensity / 10.0, -1.0, 1.0)
        tint = np.stack([
            60 * hnorm,  # red shift
            80 * hnorm,  # green shift
            120 * (-hnorm)  # blue opposite
        ], axis=2)

        # build RGB image
        base_rgb = np.ones((HEIGHT, WIDTH, 3), dtype=np.float32) * base_col.reshape((1, 1, 3))
        shaded = base_rgb + tint + (dot[..., None] * 80.0)
        shaded = np.clip(shaded, 0, 255).astype(np.uint8)
        surface = pygame.surfarray.make_surface(shaded.swapaxes(0, 1))
        screen.blit(surface, (0, 0))

        # particles
        for p in particles[:]:
            p.step()
            if p.life <= 0:
                particles.remove(p)
            else:
                pygame.draw.circle(screen, p.color, (int(p.x), int(p.y)), max(1, p.life // 10))

        # audio-driven disturbances
        if audio_enabled and audio_level > 0.01:
            # add random ripple influenced by audio
            for _ in range(int(min(10, audio_level * 200))):
                x = random.randint(0, WIDTH - 1)
                y = random.randint(0, HEIGHT - 1)
                ripples.disturb(x, y, magnitude=audio_level * 200 * intensity)

        # draw HUD
        hud_lines = [
            f"Normal shading (M): {'ON' if normal_shading else 'OFF'}",
            f"Audio (A): {'ON' if audio_enabled else 'OFF'}  Level: {audio_level:.3f}",
            f"Intensity (+/-): {intensity:.2f}",
            "Pause (P), Save (O), Quit (close window)"
        ]
        y0 = 8
        for line in hud_lines:
            txt = font.render(line, True, (255, 255, 255))
            screen.blit(txt, (8, y0))
            y0 += 20

        pygame.display.flip()
        clock.tick(60)

    stream.stop()
    stream.close()
    pygame.quit()


if __name__ == '__main__':
    run()
