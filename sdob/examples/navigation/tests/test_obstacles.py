import unittest

import torch

from sdob.examples.navigation.core import ChaoticObstacle


class ChaoticObstacleTests(unittest.TestCase):
    def test_zero_oscillation_matches_constant_velocity_obstacle(self):
        obstacle = ChaoticObstacle(
            pos=[0.0, 0.0],
            radius=0.5,
            velocity=[1.0, -0.5],
            bounds=[-10.0, 10.0, -10.0, 10.0],
            velocity_amp=[0.0, 0.0],
            velocity_freq=[1.0, 1.0],
            phase=[0.0, 0.0],
            device='cpu',
        )

        obstacle.step(2.0)

        self.assertTrue(torch.allclose(obstacle.pos, torch.tensor([2.0, -1.0])))
        self.assertTrue(torch.allclose(obstacle.velocity, torch.tensor([1.0, -0.5])))

    def test_to_dict_exposes_current_position_and_radius(self):
        obstacle = ChaoticObstacle(
            pos=[1.0, 2.0],
            radius=0.4,
            velocity=[0.0, 0.0],
            bounds=[-10.0, 10.0, -10.0, 10.0],
            velocity_amp=[0.0, 0.0],
            velocity_freq=[1.0, 1.0],
            phase=[0.0, 0.0],
            device='cpu',
        )

        data = obstacle.to_dict()

        self.assertTrue(torch.allclose(data['pos'], torch.tensor([1.0, 2.0])))
        self.assertEqual(data['radius'], 0.4)


if __name__ == '__main__':
    unittest.main()
