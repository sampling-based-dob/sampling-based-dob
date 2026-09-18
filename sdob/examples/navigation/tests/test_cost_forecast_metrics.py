from __future__ import annotations

import math
import unittest

import torch

from sdob.examples.navigation.benchmark import (
    cost_flow_evaluation_points,
    cost_forecast_errors,
)


class CostForecastMetricTests(unittest.TestCase):
    def test_exact_translational_flow_reduces_next_step_cost_error(self) -> None:
        previous = [{"pos": torch.tensor([0.0, 0.0]), "radius": 0.5}]
        current = [{"pos": torch.tensor([0.1, 0.0]), "radius": 0.5}]
        query_points = torch.tensor(
            [
                [0.6, 0.0],
                [0.1, 0.5],
                [-0.4, 0.0],
                [0.1, -0.5],
            ],
            dtype=torch.float32,
        )
        flow = {
            "query_points": query_points.clone(),
            "query_weights": torch.ones(len(query_points)),
            "query_velocities": torch.tensor([[1.0, 0.0]]).repeat(len(query_points), 1),
            "query_accelerations": torch.zeros(len(query_points), 2),
            "query_bandwidth": torch.tensor(0.4),
            "dt": 0.1,
        }

        nominal, compensated = cost_forecast_errors(
            previous,
            current,
            flow,
            query_points=query_points,
            obstacle_decay=5.5,
            obstacle_margin=0.05,
        )

        self.assertGreater(nominal, 0.0)
        self.assertLess(compensated, nominal * 0.05)

    def test_flow_evaluation_queries_are_offset_from_estimation_centers(self) -> None:
        centers = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        flow = {"query_points": centers}

        points = cost_flow_evaluation_points(flow, offset=0.1)

        self.assertEqual(tuple(points.shape), (8, 2))
        for center in centers:
            self.assertFalse(torch.any(torch.all(torch.isclose(points, center), dim=1)))

    def test_missing_flow_keeps_compensated_error_undefined(self) -> None:
        obstacle = [{"pos": torch.tensor([0.0, 0.0]), "radius": 0.5}]
        nominal, compensated = cost_forecast_errors(
            obstacle,
            obstacle,
            None,
            query_points=torch.tensor([[0.5, 0.0]]),
            obstacle_decay=5.5,
            obstacle_margin=0.05,
        )

        self.assertAlmostEqual(nominal, 0.0)
        self.assertTrue(math.isnan(compensated))


if __name__ == "__main__":
    unittest.main()
