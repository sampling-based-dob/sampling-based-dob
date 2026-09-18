import unittest

import torch

from sdob.examples.navigation.core import DiffDriveDynamics, MPPIController


class AdaptiveTemperatureTests(unittest.TestCase):
    def _controller(self, eta_min=0.3, eta_max=0.8):
        return MPPIController(
            DiffDriveDynamics(dt=1.0, device='cpu'),
            horizon=1,
            num_samples=4,
            lambda_=1.0,
            adaptive_temperature=True,
            eta_min=eta_min,
            eta_max=eta_max,
            device='cpu',
        )

    def test_reduces_lambda_when_eta_is_above_maximum(self):
        controller = self._controller(eta_min=0.3, eta_max=0.8)
        weights = torch.full((4,), 0.25)

        controller._adapt_temperature_from_weights(weights)

        self.assertAlmostEqual(controller.last_eta, 1.0, places=5)
        self.assertAlmostEqual(controller.lambda_, 0.9, places=5)

    def test_increases_lambda_when_eta_is_below_minimum(self):
        controller = self._controller(eta_min=0.4, eta_max=0.8)
        weights = torch.tensor([1.0, 0.0, 0.0, 0.0])

        controller._adapt_temperature_from_weights(weights)

        self.assertAlmostEqual(controller.last_eta, 0.25, places=5)
        self.assertAlmostEqual(controller.lambda_, 1.2, places=5)

    def test_keeps_lambda_when_eta_is_inside_band(self):
        controller = self._controller(eta_min=0.2, eta_max=0.8)
        weights = torch.tensor([0.5, 0.5, 0.0, 0.0])

        controller._adapt_temperature_from_weights(weights)

        self.assertAlmostEqual(controller.last_eta, 0.5, places=5)
        self.assertAlmostEqual(controller.lambda_, 1.0, places=5)


if __name__ == '__main__':
    unittest.main()
