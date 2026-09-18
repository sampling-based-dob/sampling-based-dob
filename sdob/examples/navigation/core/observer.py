"""Cost disturbance observer based on regular state-cost queries."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as torch_functional


def compact_query_kernel(distance, bandwidth):
    """Compact Wendland kernel used for local flow interpolation."""
    radius = distance / torch.as_tensor(
        bandwidth,
        dtype=distance.dtype,
        device=distance.device,
    ).clamp(min=1e-4)
    support = torch.clamp(1.0 - radius, min=0.0)
    return support.pow(4) * (4.0 * radius + 1.0)


def interpolate_query_values(query_positions, cost_flow, value_key):
    """Interpolate a vector query field at arbitrary world-frame positions."""
    points = cost_flow.get("query_points")
    weights = cost_flow.get("query_weights")
    values = cost_flow.get(value_key)
    bandwidth = cost_flow.get("query_bandwidth")
    if (
        points is None
        or weights is None
        or values is None
        or bandwidth is None
        or points.numel() == 0
        or values.shape != points.shape
    ):
        return None

    points = points.to(device=query_positions.device, dtype=query_positions.dtype)
    weights = weights.to(device=query_positions.device, dtype=query_positions.dtype)
    values = values.to(device=query_positions.device, dtype=query_positions.dtype)
    distance = torch.norm(
        query_positions.unsqueeze(-2) - points,
        dim=-1,
    )
    local_weights = compact_query_kernel(distance, bandwidth) * weights
    mass = local_weights.sum(dim=-1, keepdim=True)
    interpolated = (
        local_weights.unsqueeze(-1) * values
    ).sum(dim=-2) / mass.clamp(min=1e-8)
    return torch.where(
        mass > 1e-8,
        interpolated,
        torch.zeros_like(interpolated),
    )


class SamplingDisturbanceObserver:
    """Estimate a local cost velocity field from repeated cost queries."""

    def __init__(
        self,
        num_samples=1000,
        lambda_=1.0,
        sigma=None,
        alpha=0.5,
        alpha_dot=None,
        scale=1.0,
        obstacle_decay=3.0,
        obstacle_margin=0.3,
        num_clusters=3,
        query_bandwidth=None,
        lookahead_dist=1.5,
        n_lookahead_centers=4,
        perception_radius=None,
        sdob_radius=None,
        dt=0.05,
        flow_order=1,
        flow_velocity_alpha=1.0,
        flow_accel_alpha=0.6,
        flow_accel_clip=0.75,
        flow_displacement_clip=0.15,
        cost_gradient_reg=1e-4,
        cost_match_grid_size=31,
        cost_match_velocity_penalty=0.02,
        cost_match_smoothness=0.35,
        cost_match_smoothness_radius=None,
        device='cpu',
    ):
        if flow_order not in (1, 2):
            raise ValueError(f"flow_order must be 1 or 2, got {flow_order!r}")
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        self.N = num_samples
        self.lambda_ = lambda_
        self.device = device
        self.alpha = alpha
        self.alpha_dot = alpha_dot if alpha_dot is not None else alpha
        self.scale = scale
        self.obstacle_decay = obstacle_decay
        self.obstacle_margin = obstacle_margin
        self.K = num_clusters
        self.query_bandwidth = query_bandwidth
        self.lookahead_dist = lookahead_dist
        self.n_lookahead_centers = n_lookahead_centers
        self.perception_radius = perception_radius
        self.sdob_radius = sdob_radius
        if (
            self.perception_radius is not None
            and self.sdob_radius is not None
            and self.sdob_radius > self.perception_radius
        ):
            raise ValueError("sdob_radius must not exceed perception_radius")
        self.dt = float(dt)
        self.flow_order = flow_order
        self.flow_velocity_alpha = flow_velocity_alpha
        self.flow_accel_alpha = flow_accel_alpha
        self.flow_accel_clip = flow_accel_clip
        self.flow_displacement_clip = flow_displacement_clip
        self.cost_gradient_reg = cost_gradient_reg
        self.cost_match_grid_size = cost_match_grid_size
        self.cost_match_velocity_penalty = cost_match_velocity_penalty
        self.cost_match_smoothness = cost_match_smoothness
        self.cost_match_smoothness_radius = cost_match_smoothness_radius
        self.sigma = torch.as_tensor(
            sigma if sigma is not None else [0.5, 0.5, 0.15],
            device=device,
        )

        self.d_hat = torch.zeros(num_clusters, 3, device=device)
        self.d_hat_dot = torch.zeros(num_clusters, 3, device=device)
        self.d_hat_prev = torch.zeros(num_clusters, 3, device=device)
        self.cost_rate = torch.tensor(0.0, device=device)
        self.cost_accel = torch.tensor(0.0, device=device)
        self.prev_cost_rate = torch.tensor(0.0, device=device)

        self.prev_obstacles = None
        self.prev_query_points = torch.zeros(0, 2, device=device)
        self.prev_query_displacements = torch.zeros(0, 2, device=device)
        self.prev_query_accelerations = torch.zeros(0, 2, device=device)
        self.prev_query_bandwidth = torch.tensor(0.5, device=device)

    def _eval_obstacle_cost(self, positions, obstacles):
        original_shape = positions.shape[:-1]
        flat_positions = positions.reshape(-1, 2)
        costs = torch.zeros(
            flat_positions.shape[0],
            device=self.device,
            dtype=flat_positions.dtype,
        )
        for obstacle in obstacles:
            obstacle_position = obstacle['pos'][:2].to(
                device=self.device,
                dtype=flat_positions.dtype,
            )
            distance = torch.norm(
                flat_positions - obstacle_position,
                dim=-1,
            ).clamp(min=1e-6)
            costs += torch.exp(
                -self.obstacle_decay
                * (distance - obstacle['radius'] - self.obstacle_margin)
            )
        return costs.reshape(original_shape)

    def _regular_query_grid(self, state):
        side = max(3, int(math.sqrt(max(int(self.N), 9))))
        extent = self.sdob_radius
        if extent is None or extent <= 0.0:
            extent = self.perception_radius
        if extent is None or extent <= 0.0:
            extent = max(float(self.lookahead_dist), float(self.sigma[:2].mean()), 0.5)
        axis = torch.linspace(
            -float(extent),
            float(extent),
            side,
            dtype=state.dtype,
            device=self.device,
        )
        grid_x, grid_y = torch.meshgrid(axis, axis, indexing='ij')
        offsets = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)
        grid_shape = (side, side)
        if self.sdob_radius is not None and self.sdob_radius > 0.0:
            offsets = offsets[
                torch.norm(offsets, dim=-1) <= float(self.sdob_radius) + 1e-6
            ]
            grid_shape = None
        spacing = 2.0 * float(extent) / max(side - 1, 1)
        bandwidth = self.query_bandwidth
        if bandwidth is None:
            bandwidth = max(2.25 * spacing, 1e-3)
        return state[:2].unsqueeze(0) + offsets, grid_shape, float(bandwidth)

    def _query_previous_flow(self, query_points, values):
        if (
            self.prev_query_points.shape[0] == 0
            or values is None
            or values.shape[0] == 0
        ):
            return torch.zeros_like(query_points)
        flow = {
            'query_points': self.prev_query_points,
            'query_weights': torch.ones(
                self.prev_query_points.shape[0],
                dtype=query_points.dtype,
                device=self.device,
            ),
            'query_bandwidth': self.prev_query_bandwidth,
            'values': values,
        }
        interpolated = interpolate_query_values(query_points, flow, 'values')
        return interpolated if interpolated is not None else torch.zeros_like(query_points)

    def _candidate_cost_flow_displacements(self, dtype):
        axis = torch.linspace(
            -self.flow_displacement_clip,
            self.flow_displacement_clip,
            max(int(self.cost_match_grid_size), 3),
            dtype=dtype,
            device=self.device,
        )
        dx, dy = torch.meshgrid(axis, axis, indexing='ij')
        return torch.stack([dx.reshape(-1), dy.reshape(-1)], dim=-1)

    def _local_match_loss(self, query_points, point_loss, grid_shape):
        if grid_shape is not None and grid_shape[0] * grid_shape[1] == point_loss.shape[0]:
            height, width = grid_shape
            loss_image = point_loss.transpose(0, 1).reshape(
                1,
                point_loss.shape[1],
                height,
                width,
            )
            local_loss = torch_functional.avg_pool2d(
                loss_image,
                kernel_size=5,
                stride=1,
                padding=2,
                count_include_pad=False,
            )
            return local_loss.reshape(point_loss.shape[1], -1).transpose(0, 1)
        radius = self.cost_match_smoothness_radius
        if radius is None:
            radius = float(self.sigma[:2].mean().detach().cpu().item())
        distance = torch.cdist(query_points, query_points)
        neighbor_kernel = torch.exp(
            -0.5 * (distance / max(float(radius), 1e-4)).square()
        )
        return (
            neighbor_kernel @ point_loss
        ) / neighbor_kernel.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    def _smooth_cost_flow_motion(self, query_points, displacements):
        if query_points.shape[0] <= 1 or self.cost_match_smoothness <= 0.0:
            return displacements
        radius = self.cost_match_smoothness_radius
        if radius is None:
            radius = float(self.sigma[:2].mean().detach().cpu().item())
        distance = torch.cdist(query_points, query_points)
        kernel = compact_query_kernel(distance, radius)
        smooth = (
            kernel @ displacements
        ) / kernel.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        alpha = torch.as_tensor(
            self.cost_match_smoothness,
            dtype=query_points.dtype,
            device=self.device,
        ).clamp(min=0.0, max=1.0)
        return (1.0 - alpha) * displacements + alpha * smooth

    def _estimate_cost_flow_motion(self, query_points, obstacles, grid_shape=None):
        """Estimate per-step query displacement by local cost matching."""
        if self.prev_obstacles is None:
            return torch.zeros_like(query_points), torch.zeros_like(query_points)

        candidates = self._candidate_cost_flow_displacements(query_points.dtype)
        current_cost = self._eval_obstacle_cost(query_points, obstacles).reshape(-1, 1)
        shifted_queries = query_points.unsqueeze(1) - candidates.view(1, -1, 2)
        previous_cost = self._eval_obstacle_cost(
            shifted_queries.reshape(-1, 2),
            self.prev_obstacles,
        ).reshape(query_points.shape[0], -1)
        point_loss = (previous_cost - current_cost).square()
        match_loss = self._local_match_loss(query_points, point_loss, grid_shape)
        velocity_penalty = self.cost_match_velocity_penalty * candidates.square().sum(-1)
        best_index = torch.argmin(match_loss + velocity_penalty.unsqueeze(0), dim=1)
        raw_displacement = self._smooth_cost_flow_motion(
            query_points,
            candidates[best_index],
        )

        previous_displacement = self._query_previous_flow(
            query_points,
            self.prev_query_displacements,
        )
        displacement = (
            self.flow_velocity_alpha * raw_displacement
            + (1.0 - self.flow_velocity_alpha) * previous_displacement
        )
        norm = displacement.norm(dim=-1, keepdim=True)
        displacement *= torch.clamp(
            self.flow_displacement_clip / norm.clamp(min=1e-8),
            max=1.0,
        )

        acceleration = torch.zeros_like(displacement)
        if self.flow_order == 2:
            raw_acceleration = displacement - previous_displacement
            previous_acceleration = self._query_previous_flow(
                query_points,
                self.prev_query_accelerations,
            )
            acceleration = (
                self.flow_accel_alpha * raw_acceleration
                + (1.0 - self.flow_accel_alpha) * previous_acceleration
            )
            accel_norm = acceleration.norm(dim=-1, keepdim=True)
            acceleration *= torch.clamp(
                self.flow_accel_clip / accel_norm.clamp(min=1e-8),
                max=1.0,
            )
        return displacement, acceleration

    def _obstacles_within_radius(self, obstacles, state, radius):
        if radius is None:
            return obstacles
        radius_tensor = torch.as_tensor(
            radius,
            dtype=state.dtype,
            device=self.device,
        )
        return [
            obstacle
            for obstacle in obstacles
            if torch.norm(obstacle['pos'][:2] - state[:2]) - obstacle['radius']
            <= radius_tensor
        ]

    def _perceived_obstacles(self, obstacles, state):
        return self._obstacles_within_radius(
            obstacles,
            state,
            self.perception_radius,
        )

    def _sdob_obstacles(self, obstacles, state):
        return self._obstacles_within_radius(
            obstacles,
            state,
            self.sdob_radius,
        )

    def _observation_is_continuous(self, perceived):
        if self.prev_obstacles is None or len(perceived) != len(self.prev_obstacles):
            return False
        if not perceived:
            return True
        current_positions = torch.stack(
            [obstacle['pos'][:2] for obstacle in perceived],
            dim=0,
        )
        previous_positions = torch.stack(
            [obstacle['pos'][:2] for obstacle in self.prev_obstacles],
            dim=0,
        ).to(device=current_positions.device, dtype=current_positions.dtype)
        distances = torch.cdist(current_positions, previous_positions)
        continuity_limit = 2.0 * float(self.flow_displacement_clip) + 1e-6
        return bool(
            torch.all(distances.min(dim=1).values <= continuity_limit)
            and torch.all(distances.min(dim=0).values <= continuity_limit)
        )

    def _store_observation(
        self,
        perceived,
        query_points,
        query_displacements,
        query_accelerations,
        bandwidth,
    ):
        self.prev_obstacles = [
            {'pos': obstacle['pos'].clone(), 'radius': obstacle['radius']}
            for obstacle in perceived
        ]
        self.prev_query_points = query_points.clone()
        self.prev_query_displacements = query_displacements.clone()
        self.prev_query_accelerations = query_accelerations.clone()
        self.prev_query_bandwidth = torch.as_tensor(bandwidth, device=self.device)

    def _result(
        self,
        query_points,
        query_weights,
        query_velocities,
        query_accelerations,
        query_residuals,
        query_bandwidth,
    ):
        return {
            'flows': self.d_hat.clone(),
            'flow_dots': self.d_hat_dot.clone(),
            'rate': self.cost_rate.clone(),
            'accel': self.cost_accel.clone(),
            'query_points': query_points,
            'query_weights': query_weights,
            'query_velocities': query_velocities,
            'query_accelerations': query_accelerations,
            'query_residuals': query_residuals,
            'query_bandwidth': torch.as_tensor(
                query_bandwidth,
                dtype=query_points.dtype,
                device=self.device,
            ),
            'dt': self.dt,
        }

    def estimate(self, state, obstacles):
        """Estimate the local query velocity field at the current update."""
        perceived = self._perceived_obstacles(obstacles, state)
        sdob_obstacles = self._sdob_obstacles(perceived, state)
        query_points, grid_shape, bandwidth = self._regular_query_grid(state)
        zeros = torch.zeros(query_points.shape[0], device=self.device, dtype=state.dtype)
        zero_vectors = torch.zeros_like(query_points)

        observation_changed = not self._observation_is_continuous(perceived)
        if observation_changed or not sdob_obstacles:
            self.d_hat.zero_()
            self.d_hat_dot.zero_()
            self.d_hat_prev.zero_()
            self.cost_rate.zero_()
            self.cost_accel.zero_()
            self.prev_cost_rate.zero_()
            self._store_observation(
                perceived,
                query_points,
                zero_vectors,
                zero_vectors,
                bandwidth,
            )
            return self._result(
                query_points,
                zeros,
                zero_vectors,
                zero_vectors,
                zeros,
                bandwidth,
            )

        current_cost = self._eval_obstacle_cost(query_points, perceived)
        previous_cost = self._eval_obstacle_cost(query_points, self.prev_obstacles)
        residuals = current_cost - previous_cost
        weights = residuals.abs()

        if float(weights.max()) > 1e-8:
            displacements, displacement_accelerations = self._estimate_cost_flow_motion(
                query_points,
                perceived,
                grid_shape=grid_shape,
            )
        else:
            displacements = zero_vectors
            displacement_accelerations = zero_vectors

        velocities = displacements / self.dt
        accelerations = displacement_accelerations / (self.dt * self.dt)
        representative = torch.zeros_like(self.d_hat)
        count = min(self.K, query_points.shape[0])
        if count and float(weights.max()) > 1e-8:
            indices = torch.topk(weights, count).indices
            representative[:count, :2] = velocities[indices]
        previous_representative = self.d_hat.clone()
        self.d_hat = self.alpha * representative + (1.0 - self.alpha) * self.d_hat
        derivative = (self.d_hat - previous_representative) / self.dt
        self.d_hat_dot = (
            self.alpha_dot * derivative
            + (1.0 - self.alpha_dot) * self.d_hat_dot
        )

        robot_now = self._eval_obstacle_cost(state[:2].reshape(1, 2), perceived).squeeze()
        robot_previous = self._eval_obstacle_cost(
            state[:2].reshape(1, 2),
            self.prev_obstacles,
        ).squeeze()
        new_rate = (robot_now - robot_previous) / self.dt
        new_accel = (new_rate - self.prev_cost_rate) / self.dt
        self.cost_accel = self.alpha * new_accel + (1.0 - self.alpha) * self.cost_accel
        self.prev_cost_rate = self.cost_rate.clone()
        self.cost_rate = self.alpha * new_rate + (1.0 - self.alpha) * self.cost_rate

        self._store_observation(
            perceived,
            query_points,
            displacements,
            displacement_accelerations,
            bandwidth,
        )
        return self._result(
            query_points,
            weights,
            velocities,
            accelerations,
            residuals,
            bandwidth,
        )

    def get_magnitude(self):
        return torch.max(torch.norm(self.d_hat[:, :2], dim=-1)).item()

    def reset(self):
        self.d_hat.zero_()
        self.d_hat_dot.zero_()
        self.d_hat_prev.zero_()
        self.cost_rate.zero_()
        self.cost_accel.zero_()
        self.prev_cost_rate.zero_()
        self.prev_obstacles = None
        self.prev_query_points = torch.zeros(0, 2, device=self.device)
        self.prev_query_displacements = torch.zeros(0, 2, device=self.device)
        self.prev_query_accelerations = torch.zeros(0, 2, device=self.device)
        self.prev_query_bandwidth = torch.tensor(0.5, device=self.device)



