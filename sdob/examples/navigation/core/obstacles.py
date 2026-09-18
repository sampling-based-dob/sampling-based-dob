"""Obstacle primitives used by the navigation toy simulations."""

import torch


class DynamicObstacle:
    """Circular obstacle with linear motion and boundary bouncing."""

    def __init__(self, pos, radius, velocity, bounds, device="cpu"):
        self.pos = torch.tensor(pos, dtype=torch.float32, device=device)
        self.velocity = torch.tensor(velocity, dtype=torch.float32, device=device)
        self.radius = radius
        self.bounds = bounds  # [x_min, x_max, y_min, y_max]
        self.device = device

    def step(self, dt):
        self.pos = self.pos + self.velocity * dt
        for i in range(2):
            lo, hi = self.bounds[i * 2], self.bounds[i * 2 + 1]
            if self.pos[i] < lo:
                self.pos[i] = lo
                self.velocity[i] *= -1
            elif self.pos[i] > hi:
                self.pos[i] = hi
                self.velocity[i] *= -1

    def to_dict(self):
        return {"pos": self.pos.clone(), "radius": self.radius}


class ChaoticObstacle(DynamicObstacle):
    """Circular obstacle with deterministic time-varying velocity.

    The motion remains fully observable through the standard ``to_dict`` API,
    but the obstacle's velocity oscillates around a base velocity.  This makes
    dense navigation scenes less regular while keeping the toy simulator
    deterministic and reproducible.
    """

    def __init__(
        self,
        pos,
        radius,
        velocity,
        bounds,
        velocity_amp,
        velocity_freq,
        phase=(0.0, 0.0),
        device="cpu",
    ):
        super().__init__(pos, radius, velocity, bounds, device)
        self.base_velocity = self.velocity.clone()
        self.velocity_amp = torch.tensor(
            velocity_amp, dtype=torch.float32, device=device
        )
        self.velocity_freq = torch.tensor(
            velocity_freq, dtype=torch.float32, device=device
        )
        self.phase = torch.tensor(phase, dtype=torch.float32, device=device)
        self.t = 0.0

    def step(self, dt):
        self.t += dt
        t = torch.tensor(self.t, dtype=torch.float32, device=self.device)
        oscillation = self.velocity_amp * torch.stack(
            [
                torch.sin(self.velocity_freq[0] * t + self.phase[0]),
                torch.cos(self.velocity_freq[1] * t + self.phase[1]),
            ]
        )
        self.velocity = self.base_velocity + oscillation
        self.pos = self.pos + self.velocity * dt

        for i in range(2):
            lo, hi = self.bounds[i * 2], self.bounds[i * 2 + 1]
            if self.pos[i] < lo:
                self.pos[i] = lo
                self.base_velocity[i] *= -1
                self.velocity[i] *= -1
            elif self.pos[i] > hi:
                self.pos[i] = hi
                self.base_velocity[i] *= -1
                self.velocity[i] *= -1
