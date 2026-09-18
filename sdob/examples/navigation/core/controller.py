"""MPPI controller for a differential-drive navigation toy problem.

The controller can consume a cost-flow disturbance field estimated by SDOB.
When provided, MPPI can either add the estimated cost-flow increase as a
separate term or synthesize it into a time-indexed obstacle cost map by
back-tracing cost queries along the estimated cost-flow velocity.
"""

import torch
import math

from .observer import compact_query_kernel, interpolate_query_values


class DiffDriveDynamics:
    """Differential drive robot dynamics model."""

    def __init__(self, dt=0.05, device='cpu'):
        self.dt = dt
        self.device = device

    def step(self, state, control, disturbance=None):
        x = state[:, 0]
        y = state[:, 1]
        theta = state[:, 2]
        v = control[:, 0]
        omega = control[:, 1]

        x_next = x + v * torch.cos(theta) * self.dt
        y_next = y + v * torch.sin(theta) * self.dt
        theta_next = theta + omega * self.dt

        next_state = torch.stack([x_next, y_next, theta_next], dim=-1)

        if disturbance is not None:
            next_state = next_state + disturbance

        return next_state


class MPPIController:
    """MPPI controller with an optional SDOB cost-query velocity field.

    Baseline MPPI scores sampled rollouts with goal, heading, control, and
    current-obstacle costs.  When cost_flow is provided, the controller can
        synthesize a future obstacle cost map for each rollout step or add the
        observed cost-change field as a separate penalty.
    """

    def __init__(
        self,
        dynamics,
        horizon=30,
        num_samples=1000,
        lambda_=1.0,
        control_dim=2,
        v_limits=(0.0, 1.0),
        omega_limits=(-2.0, 2.0),
        sigma_v=0.3,
        sigma_omega=0.5,
        goal_weight=10.0,
        running_goal_weight=1.0,
        obstacle_weight=100.0,
        obstacle_decay=3.0,
        obstacle_margin=0.3,
        heading_weight=3.0,
        backward_penalty=10.0,
        control_weight=0.0,
        tail_goal_weight=0.0,
        tail_goal_steps=8,
        flow_weight=50.0,
        flow_mode='raw',
        flow_synthesis='map_update',
        flow_advection=True,
        flow_transport_gain=1.0,
        adaptive_temperature=False,
        eta_min=0.05,
        eta_max=0.1,
        lambda_decrease=0.9,
        lambda_increase=1.2,
        lambda_min=1e-3,
        lambda_max=1e3,
        risk_weight=0.0,
        device='cpu',
    ):
        valid_flow_modes = {'raw', 'mean', 'sqrt', 'log'}
        if flow_mode not in valid_flow_modes:
            raise ValueError(
                f"flow_mode must be one of {sorted(valid_flow_modes)}, got {flow_mode!r}"
            )
        valid_flow_synthesis = {'additive', 'map_update'}
        if flow_synthesis not in valid_flow_synthesis:
            raise ValueError(
                f"flow_synthesis must be one of {sorted(valid_flow_synthesis)}, "
                f"got {flow_synthesis!r}"
            )

        self.dynamics = dynamics
        self.horizon = horizon
        self.K = num_samples
        self.lambda_ = lambda_
        self.control_dim = control_dim
        self.v_limits = v_limits
        self.omega_limits = omega_limits
        self.device = device

        self.goal_weight = goal_weight
        self.running_goal_weight = running_goal_weight
        self.obstacle_weight = obstacle_weight
        self.obstacle_decay = obstacle_decay
        self.obstacle_margin = obstacle_margin
        self.heading_weight = heading_weight
        self.backward_penalty = backward_penalty
        self.control_weight = control_weight
        self.tail_goal_weight = tail_goal_weight
        self.tail_goal_steps = tail_goal_steps
        self.flow_weight = flow_weight
        self.flow_mode = flow_mode
        self.flow_synthesis = flow_synthesis
        self.flow_advection = flow_advection
        self.flow_transport_gain = flow_transport_gain
        self.adaptive_temperature = adaptive_temperature
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.lambda_decrease = lambda_decrease
        self.lambda_increase = lambda_increase
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.risk_weight = risk_weight
        self.last_eta = None
        self.lambda_history = [float(lambda_)]

        self.sigma = torch.tensor([sigma_v, sigma_omega], device=device)
        self.U = torch.zeros(horizon, control_dim, device=device)

    def compute_cost(self, trajectories, controls, goal, obstacles, cost_flow=None):
        """Compute rollout costs with an optional SDOB query velocity field.

        Baseline MPPI is recovered by setting cost_flow=None.
        """
        K, T_plus_1, _ = trajectories.shape
        all_pos = trajectories[:, :, :2]  # [K, T+1, 2]

        # Terminal goal cost
        goal_pos = goal[:2].unsqueeze(0)
        terminal_pos = trajectories[:, -1, :2]
        goal_cost = torch.sum((terminal_pos - goal_pos) ** 2, dim=-1) * self.goal_weight

        # Running goal cost
        running_goal = torch.sum(
            (all_pos - goal_pos.unsqueeze(0)) ** 2, dim=-1
        )
        running_goal_cost = torch.mean(running_goal, dim=-1) * self.running_goal_weight

        # Heading-toward-goal alignment cost (running)
        # Penalises trajectories where robot heading is misaligned with goal direction.
        # Pushes MPPI to prefer turning toward the goal over reversing.
        goal_vec  = goal_pos.unsqueeze(0) - all_pos              # [K, T+1, 2]
        goal_dist = goal_vec.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        goal_dir  = goal_vec / goal_dist                          # unit vector
        heading_vec = torch.stack(
            [torch.cos(trajectories[:, :, 2]),
             torch.sin(trajectories[:, :, 2])], dim=-1)           # [K, T+1, 2]
        # 0 = perfectly aligned, 2 = facing away
        heading_misalign = 1.0 - (heading_vec * goal_dir).sum(-1) # [K, T+1]
        heading_cost = heading_misalign.mean(-1) * self.heading_weight

        # Backward velocity penalty — discourages reversing
        backward_cost = (
            torch.clamp(-controls[:, :, 0], min=0.0) ** 2
        ).sum(-1) * self.backward_penalty
        control_cost = (controls ** 2).sum(dim=(1, 2)) * self.control_weight

        # Tail goal cost encourages sampled rollouts to remain near the goal
        # over the final few predicted states, rather than only touching the
        # terminal target ball at the last rollout state.
        tail_goal_cost = torch.zeros(K, device=self.device)
        if self.tail_goal_weight > 0.0 and self.tail_goal_steps > 0:
            n_tail = min(self.tail_goal_steps, T_plus_1)
            tail_pos = trajectories[:, -n_tail:, :2]
            tail_err = torch.sum(
                (tail_pos - goal_pos.unsqueeze(0)) ** 2,
                dim=-1,
            )
            tail_goal_cost = tail_err.mean(dim=-1) * self.tail_goal_weight

        # Obstacle cost map at the current planning instant.  The baseline
        # planner evaluates this same map along every future rollout state.
        obstacle_density = torch.zeros(K, T_plus_1, device=self.device)
        for obs in obstacles:
            obs_pos = obs['pos'][:2]  # [2]
            obs_r = obs['radius']
            dist = torch.norm(all_pos - obs_pos.unsqueeze(0).unsqueeze(0), dim=-1)
            margin = dist - obs_r - self.obstacle_margin
            raw_cost = torch.exp(-self.obstacle_decay * margin)
            obstacle_density += raw_cost
        obstacle_cost = obstacle_density.sum(dim=-1) * self.obstacle_weight

        # SDOB transports the current cost query with the locally interpolated
        # velocity field: C_syn(x, tau) = C_now(x - tau * v_hat(x)).
        flow_cost = torch.zeros(K, device=self.device)
        if cost_flow is not None:
            points = cost_flow.get('query_points')
            weights = cost_flow.get('query_weights')
            bandwidth = cost_flow.get('query_bandwidth')
            velocities = cost_flow.get('query_velocities')
            accelerations = cost_flow.get('query_accelerations')

            if points is not None and points.shape[0] > 0:
                points = points.to(device=self.device, dtype=all_pos.dtype)
                weights = weights.to(device=self.device, dtype=all_pos.dtype)
                tau = torch.arange(
                    T_plus_1,
                    device=self.device,
                    dtype=all_pos.dtype,
                ) * self.dynamics.dt

                if self.flow_synthesis == 'map_update' and self.obstacle_weight > 0.0:
                    local_velocity = (
                        interpolate_query_values(
                            all_pos,
                            cost_flow,
                            'query_velocities',
                        )
                        if self.flow_advection
                        else None
                    )
                    if local_velocity is None:
                        local_velocity = torch.zeros_like(all_pos)
                    local_motion = tau.view(1, -1, 1) * local_velocity
                    local_acceleration = (
                        interpolate_query_values(
                            all_pos,
                            cost_flow,
                            'query_accelerations',
                        )
                        if self.flow_advection
                        else None
                    )
                    if local_acceleration is not None:
                        local_motion = (
                            local_motion
                            + 0.5
                            * tau.square().view(1, -1, 1)
                            * local_acceleration
                        )
                    query_pos = all_pos - self.flow_transport_gain * local_motion

                    synthesized_density = torch.zeros(K, T_plus_1, device=self.device)
                    for obs in obstacles:
                        obs_pos = obs['pos'][:2]
                        obs_r = obs['radius']
                        dist = torch.norm(
                            query_pos - obs_pos.unsqueeze(0).unsqueeze(0),
                            dim=-1,
                        )
                        margin = dist - obs_r - self.obstacle_margin
                        synthesized_density += torch.exp(-self.obstacle_decay * margin)
                    obstacle_cost = synthesized_density.sum(dim=-1) * self.obstacle_weight
                else:
                    advected_points = points.unsqueeze(0).expand(T_plus_1, -1, -1)
                    if self.flow_advection and velocities is not None:
                        velocities = velocities.to(device=self.device, dtype=all_pos.dtype)
                        advected_points = (
                            advected_points
                            + tau.view(-1, 1, 1) * velocities.unsqueeze(0)
                        )
                        if accelerations is not None:
                            accelerations = accelerations.to(
                                device=self.device,
                                dtype=all_pos.dtype,
                            )
                            advected_points = (
                                advected_points
                                + 0.5
                                * tau.square().view(-1, 1, 1)
                                * accelerations.unsqueeze(0)
                            )

                    distance = torch.norm(
                        all_pos.unsqueeze(2) - advected_points.unsqueeze(0),
                        dim=-1,
                    )
                    kernel = compact_query_kernel(distance, bandwidth)
                    raw_values = (kernel * weights.view(1, 1, -1)).sum(-1)
                    if self.flow_mode == 'mean':
                        kernel_mass = kernel.sum(-1)
                        field_values = raw_values / (kernel_mass + 1e-6)
                    elif self.flow_mode == 'sqrt':
                        kernel_mass = kernel.sum(-1)
                        field_values = raw_values / torch.sqrt(kernel_mass + 1.0)
                    elif self.flow_mode == 'log':
                        field_values = torch.log1p(raw_values)
                    else:
                        field_values = raw_values

                    flow_increment = field_values * tau.unsqueeze(0)
                    flow_cost = flow_increment.sum(-1) * self.flow_weight

        total = (goal_cost + running_goal_cost + heading_cost
                 + backward_cost + control_cost + tail_goal_cost
                 + obstacle_cost + flow_cost)
        return total

    def step(
        self,
        state,
        goal,
        obstacles,
        cost_flow=None,
        state_disturbance=None,
        obstacle_scenarios=None,
    ):
        """Compute one MPPI control step."""
        epsilon = torch.randn(self.K, self.horizon, self.control_dim, device=self.device)
        epsilon = epsilon * self.sigma.unsqueeze(0).unsqueeze(0)

        U_perturbed = self.U.unsqueeze(0) + epsilon
        U_perturbed[:, :, 0] = torch.clamp(U_perturbed[:, :, 0], *self.v_limits)
        U_perturbed[:, :, 1] = torch.clamp(U_perturbed[:, :, 1], *self.omega_limits)

        trajectories = self._rollout(
            state,
            U_perturbed,
            state_disturbance=state_disturbance,
        )
        if obstacle_scenarios is None:
            costs = self.compute_cost(
                trajectories,
                U_perturbed,
                goal,
                obstacles,
                cost_flow=cost_flow,
            )
        else:
            scenario_costs = [
                self.compute_cost(
                    trajectories,
                    U_perturbed,
                    goal,
                    scenario,
                    cost_flow=cost_flow,
                )
                for scenario in obstacle_scenarios
            ]
            scenario_stack = torch.stack(scenario_costs, dim=0)
            costs = scenario_stack.mean(dim=0)
            if self.risk_weight > 0.0 and scenario_stack.shape[0] > 1:
                costs = costs + self.risk_weight * scenario_stack.std(
                    dim=0,
                    unbiased=False,
                )

        beta = torch.min(costs)
        weights = torch.exp(-(costs - beta) / self.lambda_)
        weights = weights / torch.sum(weights)

        self.U = torch.sum(
            weights.unsqueeze(-1).unsqueeze(-1) * U_perturbed, dim=0
        )
        self._adapt_temperature_from_weights(weights)

        u_opt = self.U[0].clone()
        self.U = torch.roll(self.U, -1, dims=0)
        self.U[-1] = 0.0

        return u_opt

    def _adapt_temperature_from_weights(self, weights):
        """Update lambda for the next MPPI step using ESS-ratio eta."""
        if not self.adaptive_temperature:
            return

        ess = 1.0 / torch.sum(weights ** 2).clamp(min=1e-12)
        eta = ess / weights.numel()
        self.last_eta = float(eta.detach().cpu().item())

        if self.last_eta > self.eta_max:
            self.lambda_ *= self.lambda_decrease
        elif self.last_eta < self.eta_min:
            self.lambda_ *= self.lambda_increase

        self.lambda_ = float(min(max(self.lambda_, self.lambda_min), self.lambda_max))
        self.lambda_history.append(self.lambda_)

    def _rollout(self, state, controls, state_disturbance=None):
        """Roll out trajectories from current state."""
        K, T, _ = controls.shape
        states = [state.unsqueeze(0).expand(K, -1)]

        for t in range(T):
            next_s = self.dynamics.step(
                states[-1],
                controls[:, t],
                disturbance=state_disturbance,
            )
            states.append(next_s)

        return torch.stack(states, dim=1)
