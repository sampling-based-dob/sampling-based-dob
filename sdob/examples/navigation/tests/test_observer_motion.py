import unittest

import torch

from sdob.examples.navigation.core import SamplingDisturbanceObserver


class ObserverMotionTests(unittest.TestCase):
    def test_estimate_returns_a_deterministic_grid_without_kde_fields(self):
        observer = SamplingDisturbanceObserver(
            num_samples=25,
            perception_radius=1.0,
            dt=0.1,
            obstacle_decay=2.0,
            obstacle_margin=0.0,
            device='cpu',
        )
        state = torch.tensor([0.0, 0.0, 0.0])
        previous = [{'pos': torch.tensor([0.0, 0.0]), 'radius': 0.25}]
        current = [{'pos': torch.tensor([0.1, 0.0]), 'radius': 0.25}]

        initial = observer.estimate(state, previous)
        estimate = observer.estimate(state, current)

        self.assertEqual(tuple(initial['query_points'].shape), (25, 2))
        self.assertTrue(torch.equal(initial['query_points'], estimate['query_points']))
        self.assertFalse(any(key.startswith('kde_') for key in estimate))
        self.assertTrue(
            {
                'query_points',
                'query_weights',
                'query_velocities',
                'query_accelerations',
                'query_residuals',
                'query_bandwidth',
            }.issubset(estimate)
        )

    def test_query_grid_keeps_cost_increase_and_decrease(self):
        observer = SamplingDisturbanceObserver(
            num_samples=81,
            perception_radius=1.2,
            dt=0.1,
            obstacle_decay=3.0,
            obstacle_margin=0.0,
            device='cpu',
        )
        state = torch.tensor([0.0, 0.0, 0.0])
        observer.estimate(
            state,
            [{'pos': torch.tensor([-0.1, 0.0]), 'radius': 0.3}],
        )

        estimate = observer.estimate(
            state,
            [{'pos': torch.tensor([0.1, 0.0]), 'radius': 0.3}],
        )

        residuals = estimate['query_residuals']
        self.assertLess(float(residuals.min()), 0.0)
        self.assertGreater(float(residuals.max()), 0.0)
        self.assertTrue(torch.allclose(estimate['query_weights'], residuals.abs()))

    def test_two_radius_observer_warms_up_new_outer_obstacle_before_flow(self):
        observer = SamplingDisturbanceObserver(
            num_samples=81,
            perception_radius=3.2,
            sdob_radius=2.6,
            dt=0.1,
            obstacle_decay=3.0,
            obstacle_margin=0.0,
            flow_displacement_clip=0.3,
            device='cpu',
        )
        state = torch.tensor([0.0, 0.0, 0.0])

        observer.estimate(state, [])
        appeared = observer.estimate(
            state,
            [{'pos': torch.tensor([3.0, 0.0]), 'radius': 0.2}],
        )
        still_outer = observer.estimate(
            state,
            [{'pos': torch.tensor([2.9, 0.0]), 'radius': 0.2}],
        )
        entered_inner = observer.estimate(
            state,
            [{'pos': torch.tensor([2.75, 0.0]), 'radius': 0.2}],
        )

        self.assertTrue(torch.equal(appeared['query_velocities'], torch.zeros_like(appeared['query_velocities'])))
        self.assertTrue(torch.equal(still_outer['query_velocities'], torch.zeros_like(still_outer['query_velocities'])))
        active = entered_inner['query_weights'] > 1e-8
        self.assertTrue(torch.any(active))
        self.assertLess(float(entered_inner['query_velocities'][active, 0].mean()), 0.0)

    def test_sdob_query_points_stay_inside_inner_radius(self):
        observer = SamplingDisturbanceObserver(
            num_samples=121,
            perception_radius=3.2,
            sdob_radius=2.6,
            device='cpu',
        )
        state = torch.tensor([1.0, -2.0, 0.0])

        estimate = observer.estimate(state, [])
        radii = torch.norm(estimate['query_points'] - state[:2], dim=-1)

        self.assertLessEqual(float(radii.max()), 2.6 + 1e-6)

    def test_same_count_obstacle_replacement_is_not_treated_as_motion(self):
        observer = SamplingDisturbanceObserver(
            num_samples=81,
            perception_radius=3.2,
            sdob_radius=2.6,
            dt=0.1,
            flow_displacement_clip=0.3,
            device='cpu',
        )
        state = torch.tensor([0.0, 0.0, 0.0])
        observer.estimate(
            state,
            [
                {'pos': torch.tensor([1.0, 0.0]), 'radius': 0.2},
                {'pos': torch.tensor([2.0, 0.0]), 'radius': 0.2},
            ],
        )

        estimate = observer.estimate(
            state,
            [
                {'pos': torch.tensor([1.05, 0.0]), 'radius': 0.2},
                {'pos': torch.tensor([-2.0, 0.0]), 'radius': 0.2},
            ],
        )

        self.assertTrue(
            torch.equal(
                estimate['query_velocities'],
                torch.zeros_like(estimate['query_velocities']),
            )
        )

    def test_cost_flow_motion_matches_shifted_cost_queries_without_gradients(self):
        observer = SamplingDisturbanceObserver(
            obstacle_decay=1.0,
            obstacle_margin=0.0,
            flow_displacement_clip=0.5,
            device='cpu',
        )
        observer.prev_obstacles = [
            {'pos': torch.tensor([0.0, 0.0]), 'radius': 0.5},
        ]

        sample_pos = torch.tensor([
            [0.8, -0.2],
            [0.8, 0.2],
            [1.0, -0.2],
            [1.0, 0.2],
            [1.2, -0.2],
            [1.2, 0.2],
        ])
        obstacles = [
            {'pos': torch.tensor([0.2, 0.0]), 'radius': 0.5},
        ]

        def fail_if_gradient_is_used(*_args, **_kwargs):
            raise AssertionError("query matching should not use cost gradients")

        observer._eval_obstacle_cost_and_gradient = fail_if_gradient_is_used

        displacements, accelerations = observer._estimate_cost_flow_motion(
            sample_pos,
            obstacles,
        )

        self.assertTrue(torch.all(displacements[:, 0] > 0.05))
        self.assertTrue(torch.all(torch.abs(displacements[:, 1]) < 0.08))
        self.assertTrue(torch.allclose(accelerations, torch.zeros_like(accelerations)))

    def test_second_order_cost_flow_uses_previous_estimated_flow(self):
        observer = SamplingDisturbanceObserver(
            flow_order=2,
            flow_accel_alpha=1.0,
            flow_accel_clip=10.0,
            obstacle_decay=1.0,
            obstacle_margin=0.0,
            device='cpu',
        )
        observer.prev_obstacles = [
            {'pos': torch.tensor([0.0, 0.0]), 'radius': 0.5},
        ]
        observer.prev_query_points = torch.tensor([[1.0, 0.0]])
        observer.prev_query_displacements = torch.tensor([[0.05, 0.0]])
        observer.prev_query_accelerations = torch.zeros(1, 2)
        observer.prev_query_bandwidth = torch.tensor(1.0)

        sample_pos = torch.tensor([
            [0.8, -0.2],
            [0.8, 0.2],
            [1.0, -0.2],
            [1.0, 0.2],
            [1.2, -0.2],
            [1.2, 0.2],
        ])
        obstacles = [
            {'pos': torch.tensor([0.2, 0.0]), 'radius': 0.5},
        ]

        displacements, accelerations = observer._estimate_cost_flow_motion(
            sample_pos,
            obstacles,
        )

        self.assertTrue(torch.all(displacements[:, 0] > 0.05))
        self.assertTrue(torch.all(accelerations[:, 0] > 0.0))
        self.assertTrue(torch.all(torch.abs(accelerations[:, 1]) < 0.08))

    def test_cost_flow_motion_resolves_small_per_step_obstacle_motion(self):
        observer = SamplingDisturbanceObserver(
            obstacle_decay=5.5,
            obstacle_margin=0.05,
            device='cpu',
        )
        observer.prev_obstacles = [
            {'pos': torch.tensor([2.25, 5.30]), 'radius': 0.48},
        ]
        obstacles = [
            {'pos': torch.tensor([2.209, 5.314]), 'radius': 0.48},
        ]
        sample_pos = torch.tensor([
            [1.8, 5.0],
            [2.0, 5.1],
            [2.2, 5.3],
            [2.4, 5.4],
            [1.8, 5.2],
            [2.0, 5.3],
            [2.2, 5.5],
            [2.4, 5.6],
        ])

        displacements, _ = observer._estimate_cost_flow_motion(sample_pos, obstacles)
        mean_displacement = displacements.mean(0)

        self.assertLess(mean_displacement[0].item(), -0.02)
        self.assertGreater(mean_displacement[1].item(), 0.005)
        self.assertLess(torch.norm(mean_displacement - torch.tensor([-0.041, 0.014])).item(), 0.02)


if __name__ == '__main__':
    unittest.main()
