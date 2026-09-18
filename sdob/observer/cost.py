"""Local cost disturbance estimation from repeated state-cost queries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .base import BaseObserver

CostQuery = Callable[[torch.Tensor], torch.Tensor]


def compact_query_kernel(distance: torch.Tensor, bandwidth: float | torch.Tensor) -> torch.Tensor:
    """Compact Wendland kernel for local query-field interpolation."""
    radius = distance / torch.as_tensor(
        bandwidth,
        dtype=distance.dtype,
        device=distance.device,
    ).clamp_min(1.0e-4)
    support = torch.clamp(1.0 - radius, min=0.0)
    return support.pow(4) * (4.0 * radius + 1.0)


def interpolate_query_values(
    query_positions: torch.Tensor,
    field: dict[str, torch.Tensor] | "CostDisturbanceField",
    value_key: str = "query_velocities",
) -> torch.Tensor | None:
    """Interpolate sampled vector values at arbitrary query states."""
    if isinstance(field, CostDisturbanceField):
        points = field.query_states
        weights = field.query_weights
        values = field.velocities
        bandwidth = field.interpolation_bandwidth
    else:
        points = field.get("query_points")
        weights = field.get("query_weights")
        values = field.get(value_key)
        bandwidth = field.get("query_bandwidth")
    if points is None or weights is None or values is None or bandwidth is None:
        return None
    if points.numel() == 0 or values.shape != points.shape:
        return None
    points = points.to(device=query_positions.device, dtype=query_positions.dtype)
    weights = weights.to(device=query_positions.device, dtype=query_positions.dtype)
    values = values.to(device=query_positions.device, dtype=query_positions.dtype)
    distances = torch.norm(query_positions.unsqueeze(-2) - points, dim=-1)
    local_weights = compact_query_kernel(distances, bandwidth) * weights
    mass = local_weights.sum(dim=-1, keepdim=True)
    interpolated = (local_weights.unsqueeze(-1) * values).sum(dim=-2) / mass.clamp_min(1.0e-8)
    return torch.where(mass > 1.0e-8, interpolated, torch.zeros_like(interpolated))


@dataclass(frozen=True)
class CostDisturbanceConfig:
    dt: float
    max_displacement: tuple[float, ...]
    candidate_steps: int = 9
    neighborhood_radius: float = 0.5
    interpolation_bandwidth: float = 0.5
    displacement_penalty: float = 0.01
    device: str = "cpu"
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if not self.max_displacement or any(value <= 0.0 for value in self.max_displacement):
            raise ValueError("max_displacement must contain positive values")
        if self.candidate_steps < 3:
            raise ValueError("candidate_steps must be at least three")
        if self.neighborhood_radius <= 0.0 or self.interpolation_bandwidth <= 0.0:
            raise ValueError("kernel radii must be positive")


@dataclass(frozen=True)
class CostDisturbanceField:
    query_states: torch.Tensor
    velocities: torch.Tensor
    query_weights: torch.Tensor
    residuals: torch.Tensor
    interpolation_bandwidth: float
    dt: float

    def velocity_at(self, query_states: torch.Tensor) -> torch.Tensor:
        query_states = torch.as_tensor(
            query_states,
            dtype=self.query_states.dtype,
            device=self.query_states.device,
        )
        values = interpolate_query_values(query_states, self)
        if values is None:
            return torch.zeros_like(query_states)
        return values

    def transport_cost(
        self,
        query_states: torch.Tensor,
        horizon_step: int,
        current_cost: CostQuery,
        *,
        step_dt: float | None = None,
    ) -> torch.Tensor:
        """Evaluate the current cost after iterative backward transport."""
        transported_queries = self.backtrace_queries(
            query_states,
            horizon_step,
            step_dt=step_dt,
        )
        return current_cost(transported_queries)

    def backtrace_queries(
        self,
        query_states: torch.Tensor,
        horizon_step: int,
        *,
        step_dt: float | None = None,
    ) -> torch.Tensor:
        """Backtrace query states along the estimated local cost motion."""
        if horizon_step < 0:
            raise ValueError("horizon_step must be nonnegative")
        integration_dt = self.dt if step_dt is None else float(step_dt)
        if integration_dt <= 0.0:
            raise ValueError("step_dt must be positive")
        query_states = torch.as_tensor(
            query_states,
            dtype=self.query_states.dtype,
            device=self.query_states.device,
        )
        transported_queries = query_states.clone()
        for _ in range(horizon_step):
            transported_queries = (
                transported_queries
                - integration_dt * self.velocity_at(transported_queries)
            )
        return transported_queries


class CostDisturbanceObserver(BaseObserver):
    """Estimate local query-space motion by matching consecutive cost queries."""

    def __init__(self, config: CostDisturbanceConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = config.dtype
        self.last_field: CostDisturbanceField | None = None

    def reset(self) -> None:
        self.last_field = None

    @torch.no_grad()
    def update(
        self,
        query_states: torch.Tensor,
        previous_cost: CostQuery,
        current_cost: CostQuery,
    ) -> CostDisturbanceField:
        query_states = torch.as_tensor(query_states, dtype=self.dtype, device=self.device)
        if query_states.ndim != 2:
            raise ValueError("query_states must have shape [num_queries, query_dim]")
        query_dim = query_states.shape[1]
        if query_dim != len(self.config.max_displacement):
            raise ValueError("query dimension must match max_displacement")
        candidates = self._candidate_displacements()
        current_values = current_cost(query_states).reshape(-1)
        shifted = query_states[:, None, :] - candidates[None, :, :]
        previous_values = previous_cost(shifted.reshape(-1, query_dim)).reshape(
            query_states.shape[0],
            candidates.shape[0],
        )
        point_loss = (previous_values - current_values[:, None]).square()
        distances = torch.cdist(query_states, query_states)
        neighborhood = torch.exp(
            -0.5 * (distances / float(self.config.neighborhood_radius)).square()
        )
        local_loss = neighborhood @ point_loss
        local_loss = local_loss / neighborhood.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
        local_loss = local_loss + float(self.config.displacement_penalty) * candidates.square().sum(dim=1)
        best = torch.argmin(local_loss, dim=1)
        displacements = candidates[best]
        residuals = current_values - previous_cost(query_states).reshape(-1)
        weights = residuals.abs()
        if float(weights.max()) <= 1.0e-8:
            displacements.zero_()
            weights = torch.ones_like(weights)
        else:
            weights = weights / weights.max().clamp_min(1.0e-8)
        field = CostDisturbanceField(
            query_states=query_states,
            velocities=displacements / float(self.config.dt),
            query_weights=weights,
            residuals=residuals,
            interpolation_bandwidth=float(self.config.interpolation_bandwidth),
            dt=float(self.config.dt),
        )
        self.last_field = field
        return field

    def _candidate_displacements(self) -> torch.Tensor:
        axes = [
            torch.linspace(
                -float(limit),
                float(limit),
                self.config.candidate_steps,
                dtype=self.dtype,
                device=self.device,
            )
            for limit in self.config.max_displacement
        ]
        if len(axes) == 1:
            return axes[0].unsqueeze(-1)
        grids = torch.meshgrid(*axes, indexing="ij")
        return torch.stack([grid.reshape(-1) for grid in grids], dim=-1)
