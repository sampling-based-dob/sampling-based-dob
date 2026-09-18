import unittest

import torch

from sdob.examples.navigation.core import DiffDriveDynamics, MPPIController


class FlowModeTests(unittest.TestCase):
    def _controller(
        self,
        flow_mode,
        flow_synthesis='additive',
        obstacle_weight=0.0,
        flow_weight=1.0,
        flow_advection=True,
        flow_transport_gain=1.0,
        **kwargs,
    ):
        return MPPIController(
            DiffDriveDynamics(dt=1.0, device='cpu'),
            horizon=1,
            num_samples=1,
            flow_mode=flow_mode,
            flow_synthesis=flow_synthesis,
            flow_advection=flow_advection,
            flow_weight=flow_weight,
            flow_transport_gain=flow_transport_gain,
            goal_weight=0.0,
            running_goal_weight=0.0,
            obstacle_weight=obstacle_weight,
            heading_weight=0.0,
            backward_penalty=0.0,
            device='cpu',
            **kwargs,
        )

    def _flow_cost(
        self,
        flow_mode,
        flow_synthesis='additive',
        obstacle_weight=0.0,
        flow_weight=1.0,
    ):
        controller = self._controller(
            flow_mode,
            flow_synthesis=flow_synthesis,
            obstacle_weight=obstacle_weight,
            flow_weight=flow_weight,
        )
        trajectories = torch.zeros(1, 2, 3)
        controls = torch.zeros(1, 1, 2)
        goal = torch.zeros(3)
        cost_flow = {
            'query_points': torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
            'query_weights': torch.tensor([2.0, 4.0]),
            'query_bandwidth': torch.tensor(1.0),
        }
        return controller.compute_cost(
            trajectories, controls, goal, [], cost_flow=cost_flow
        ).item()

    def test_raw_flow_mode_uses_unnormalized_query_weight_sum(self):
        self.assertAlmostEqual(self._flow_cost('raw'), 6.0, places=5)

    def test_mean_flow_mode_normalizes_by_local_kernel_mass(self):
        self.assertAlmostEqual(self._flow_cost('mean'), 3.0, places=5)

    def test_map_update_advects_cost_queries_with_estimated_flow(self):
        controller = self._controller(
            'mean',
            flow_synthesis='map_update',
            obstacle_weight=1.0,
            obstacle_decay=1.0,
            obstacle_margin=0.0,
            flow_transport_gain=1.0,
        )
        trajectories = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
        controls = torch.zeros(1, 1, 2)
        goal = torch.zeros(3)
        obstacles = [{'pos': torch.tensor([0.0, 0.0]), 'radius': 0.0}]
        cost_flow = {
            'query_points': torch.tensor([[1.0, 0.0]]),
            'query_weights': torch.tensor([1.0]),
            'query_velocities': torch.tensor([[1.0, 0.0]]),
            'query_bandwidth': torch.tensor(1.0),
        }

        cost = controller.compute_cost(
            trajectories,
            controls,
            goal,
            obstacles,
            cost_flow=cost_flow,
        )

        expected = torch.exp(torch.tensor(-1.0)).item() + 1.0
        self.assertAlmostEqual(cost.item(), expected, places=5)

    def test_flow_advection_transports_query_weights_over_rollout_steps(self):
        trajectories = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
        controls = torch.zeros(1, 1, 2)
        goal = torch.zeros(3)
        cost_flow = {
            'query_points': torch.tensor([[0.0, 0.0]]),
            'query_weights': torch.tensor([1.0]),
            'query_velocities': torch.tensor([[1.0, 0.0]]),
            'query_bandwidth': torch.tensor(1.0),
        }

        advective = self._controller(
            'raw',
            flow_synthesis='additive',
            flow_weight=1.0,
            flow_advection=True,
        ).compute_cost(trajectories, controls, goal, [], cost_flow=cost_flow)
        static = self._controller(
            'raw',
            flow_synthesis='additive',
            flow_weight=1.0,
            flow_advection=False,
        ).compute_cost(trajectories, controls, goal, [], cost_flow=cost_flow)

        self.assertAlmostEqual(advective.item(), 1.0, places=5)
        self.assertLess(static.item(), advective.item())

    def test_flow_advection_uses_second_order_query_acceleration(self):
        trajectories = torch.tensor([[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]])
        controls = torch.zeros(1, 1, 2)
        goal = torch.zeros(3)
        cost_flow = {
            'query_points': torch.tensor([[0.0, 0.0]]),
            'query_weights': torch.tensor([1.0]),
            'query_velocities': torch.tensor([[1.0, 0.0]]),
            'query_accelerations': torch.tensor([[1.0, 0.0]]),
            'query_bandwidth': torch.tensor(1.0),
        }

        second_order = self._controller(
            'raw',
            flow_synthesis='additive',
            flow_weight=1.0,
            flow_advection=True,
        ).compute_cost(trajectories, controls, goal, [], cost_flow=cost_flow)

        self.assertAlmostEqual(second_order.item(), 1.0, places=5)

    def test_control_effort_cost_penalizes_sampled_inputs(self):
        controller = self._controller(
            'raw',
            flow_synthesis='additive',
            flow_weight=0.0,
            control_weight=2.0,
        )
        trajectories = torch.zeros(1, 2, 3)
        controls = torch.tensor([[[1.0, 2.0]]])
        goal = torch.zeros(3)

        cost = controller.compute_cost(trajectories, controls, goal, [])

        self.assertAlmostEqual(cost.item(), 10.0, places=5)

    def test_tail_goal_cost_penalizes_last_predicted_states(self):
        controller = self._controller(
            'raw',
            flow_synthesis='additive',
            flow_weight=0.0,
            tail_goal_weight=3.0,
            tail_goal_steps=1,
        )
        trajectories = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
        controls = torch.zeros(1, 1, 2)
        goal = torch.tensor([1.0, 0.0, 0.0])

        cost = controller.compute_cost(trajectories, controls, goal, [])

        self.assertAlmostEqual(cost.item(), 3.0, places=5)


if __name__ == '__main__':
    unittest.main()
