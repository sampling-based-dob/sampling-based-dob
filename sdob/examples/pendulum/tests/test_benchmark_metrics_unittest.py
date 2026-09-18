from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from sdob.examples.pendulum.benchmark import run_suite
from sdob.examples.pendulum.dynamics import PendulumConfig
from sdob.examples.pendulum.interfaces import ControllerBase


class ConstantController(ControllerBase):
    name = "constant"

    def reset(self) -> None:
        return None

    def act(self, state: np.ndarray, step: int) -> float:
        return 0.0


class BenchmarkTimingMetricTests(unittest.TestCase):
    def test_run_suite_records_per_step_and_summary_planning_time(self) -> None:
        result = run_suite(
            controllers=[ConstantController()],
            config=PendulumConfig(dt=0.05),
            steps=4,
            initial_state=np.asarray([0.2, 0.0]),
            disturbance_name="constant_bias",
            disturbance_value=0.1,
        )

        self.assertIn("planning_time_ms", result.trajectories)
        self.assertIn("planning_time_mean_ms", result.summary)
        self.assertIn("planning_time_p95_ms", result.summary)
        self.assertTrue((result.trajectories["planning_time_ms"] >= 0.0).all())

    def test_run_suite_synchronizes_cuda_before_and_after_each_timed_plan(self) -> None:
        with mock.patch(
            "sdob.examples.pendulum.benchmark._synchronize_device"
        ) as synchronize:
            run_suite(
                controllers=[ConstantController()],
                config=PendulumConfig(dt=0.05),
                steps=3,
                initial_state=np.asarray([0.2, 0.0]),
                disturbance_name="constant_bias",
                disturbance_value=0.1,
                device="cuda",
            )

        self.assertEqual(synchronize.call_count, 6)
        synchronize.assert_has_calls([mock.call("cuda")] * 6)


if __name__ == "__main__":
    unittest.main()
