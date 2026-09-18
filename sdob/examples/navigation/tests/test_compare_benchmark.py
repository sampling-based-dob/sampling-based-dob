import csv
import importlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from sdob.examples.navigation.benchmark import (
    COLLISION_TOL,
    DEFAULT_CONTROLLER_NAMES,
    CompareConfig,
    CompareRunner,
    _shift_obstacle_dicts,
    create_obstacles,
)
from sdob.examples.navigation.config import load_experiment_config


class CompareBenchmarkTests(unittest.TestCase):
    def test_collision_tolerance_does_not_hide_geometric_overlap(self):
        self.assertLessEqual(COLLISION_TOL, 1.0e-6)

    def test_default_controller_suite_has_required_baselines(self):
        self.assertEqual(
            DEFAULT_CONTROLLER_NAMES,
            ["MPPI", "MPPI+DOB", "MPPI+SDOB", "Tube-MPPI", "RMPPI"],
        )

    def test_compare_runner_writes_artifacts_and_required_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CompareConfig(
                scenario="four",
                steps=4,
                horizon=3,
                num_samples=8,
                output_dir=Path(tmpdir),
                make_gif=False,
                device="cpu",
                seed=7,
            )

            with mock.patch(
                "sdob.examples.navigation.benchmark._synchronize_device"
            ) as synchronize:
                output = CompareRunner(config).run()

            self.assertTrue(output.summary_csv.exists())
            self.assertTrue(output.trajectories_csv.exists())
            self.assertTrue(output.figure_path.exists())
            self.assertIsNone(output.gif_path)
            self.assertEqual(
                [result.name for result in output.results],
                DEFAULT_CONTROLLER_NAMES,
            )

            with output.summary_csv.open(newline="") as f:
                rows = list(csv.DictReader(f))

            with output.trajectories_csv.open(newline="") as f:
                trajectory_rows = list(csv.DictReader(f))

            self.assertEqual(len(rows), len(DEFAULT_CONTROLLER_NAMES))
            required = {
                "controller",
                "final_error",
                "min_error",
                "tail_error",
                "mean_safety",
                "min_safety",
                "collision",
                "first_hit_step",
                "first_hit_time",
                "cost_flow_estimation_error",
                "state_effect_estimation_error",
                "success",
                "collision_before_goal",
                "terminal_step",
                "path_length",
                "path_ratio",
                "control_tv",
                "cost_forecast_nominal_mae",
                "cost_forecast_sdob_mae",
                "cost_forecast_reduction",
                "planning_time_mean_ms",
                "planning_time_p95_ms",
                "lambda_final",
                "eta_last",
            }
            self.assertTrue(required.issubset(rows[0].keys()))
            self.assertEqual(len(trajectory_rows), config.steps * len(DEFAULT_CONTROLLER_NAMES))
            self.assertAlmostEqual(float(trajectory_rows[0]["time"]), config.dt)
            self.assertTrue(
                {
                    "cost_forecast_nominal_mae",
                    "cost_forecast_sdob_mae",
                    "planning_time_ms",
                }.issubset(trajectory_rows[0].keys())
            )
            self.assertEqual(
                synchronize.call_count,
                2 * config.steps * len(DEFAULT_CONTROLLER_NAMES),
            )
            synchronize.assert_has_calls(
                [mock.call(config.device)] * synchronize.call_count
            )

    def test_available_scenarios_include_four_eight_and_chaotic(self):
        for scenario in ("four", "eight", "chaotic"):
            obstacles = create_obstacles(scenario, device="cpu")
            self.assertGreaterEqual(len(obstacles), 1)

        self.assertEqual(len(create_obstacles("four", device="cpu")), 4)
        self.assertEqual(len(create_obstacles("4obs", device="cpu")), 4)
        self.assertEqual(len(create_obstacles("eight", device="cpu")), 8)
        self.assertEqual(len(create_obstacles("chaotic", device="cpu")), 8)

    def test_eight_obstacle_scenario_separates_eighth_crossing_motion(self):
        obstacles = create_obstacles("eight", device="cpu")
        expected_positions = torch.tensor(
            [
                [7.2, -0.4],
                [-0.4, 2.8],
                [8.4, 3.2],
                [3.2, -0.8],
                [2.2, 8.8],
                [-0.6, 6.6],
                [6.5, 9.0],
                [9.6, 7.6],
            ]
        )
        expected_velocities = torch.tensor(
            [
                [-0.65, 0.82],
                [0.85, 0.35],
                [-0.82, 0.28],
                [0.28, 0.90],
                [0.35, -0.75],
                [0.95, -0.25],
                [-0.45, -0.72],
                [-0.80, -0.20],
            ]
        )
        torch.testing.assert_close(
            torch.stack([obstacle.pos for obstacle in obstacles]),
            expected_positions,
        )
        torch.testing.assert_close(
            torch.stack([obstacle.velocity for obstacle in obstacles]),
            expected_velocities,
        )
        minimum_clearance = float("inf")
        eighth_minimum_clearance = float("inf")

        for _ in range(500):
            for first_i, first in enumerate(obstacles):
                for second in obstacles[first_i + 1 :]:
                    clearance = (
                        torch.norm(first.pos - second.pos).item()
                        - float(first.radius)
                        - float(second.radius)
                    )
                    minimum_clearance = min(minimum_clearance, clearance)
            for other in obstacles[:-1]:
                clearance = (
                    torch.norm(obstacles[-1].pos - other.pos).item()
                    - float(obstacles[-1].radius)
                    - float(other.radius)
                )
                eighth_minimum_clearance = min(
                    eighth_minimum_clearance,
                    clearance,
                )
            for obstacle in obstacles:
                obstacle.step(0.05)

        self.assertLess(minimum_clearance, 0.0)
        self.assertGreater(eighth_minimum_clearance, 0.2)

    def test_yaml_config_matches_inverted_pendulum_style_entrypoint(self):
        config = load_experiment_config("sdob/examples/navigation/configs/default.yaml")

        self.assertEqual(config.scenario, "four")
        self.assertEqual(config.disturbance, "constant_input")
        self.assertFalse(config.skip_gif)
        self.assertGreaterEqual(config.steps, 500)
        self.assertLessEqual(config.max_gif_frames, 220)

    def test_four_and_eight_configs_write_to_scene_specific_output_dirs(self):
        config_4 = load_experiment_config("sdob/examples/navigation/configs/compare_4obs.yaml")
        config_8 = load_experiment_config("sdob/examples/navigation/configs/compare_8obs.yaml")

        self.assertEqual(config_4.scenario, "four")
        self.assertEqual(config_8.scenario, "eight")
        self.assertEqual(config_4.output_dir.name, "4obs")
        self.assertEqual(config_8.output_dir.name, "8obs")
        self.assertGreaterEqual(config_4.steps, 500)
        self.assertEqual(config_8.steps, 600)
        self.assertAlmostEqual(config_8.steps * config_8.dt, 30.0)
        self.assertFalse(config_4.skip_gif)
        self.assertFalse(config_8.skip_gif)
        self.assertGreater(config_4.perception_radius, 0.0)
        self.assertGreater(config_8.perception_radius, 0.0)
        self.assertLess(config_4.sdob_radius, config_4.perception_radius)
        self.assertLess(config_8.sdob_radius, config_8.perception_radius)

    def test_sdob_radius_must_not_exceed_robot_perception_radius(self):
        with self.assertRaisesRegex(ValueError, "sdob_radius"):
            CompareConfig(perception_radius=3.2, sdob_radius=3.3)

    def test_sinusoidal_slip_disturbance_is_applied_in_world_frame(self):
        config = CompareConfig(
            scenario="four",
            steps=2,
            dt=0.25,
            disturbance="sinusoidal_slip",
            slip_amplitude=0.02,
            slip_frequency=1.0,
            device="cpu",
        )
        runner = CompareRunner(config)
        disturbance = runner._state_disturbance(runner.start, step_i=1)

        self.assertNotAlmostEqual(float(disturbance[:2].norm()), 0.0)
        self.assertAlmostEqual(float(disturbance[2]), 0.0)

    def test_sinusoidal_input_disturbance_is_applied_in_input_channel(self):
        config = CompareConfig(
            scenario="four",
            steps=2,
            dt=0.25,
            disturbance="sinusoidal_input",
            input_disturbance_amplitude=(0.08, 0.16),
            input_disturbance_frequency=0.5,
            device="cpu",
        )
        runner = CompareRunner(config)
        state = runner.start
        action = state.new_tensor([0.3, -0.2])
        input_disturbance = runner._input_disturbance(step_i=1)
        executed_action = runner._executed_action(action, input_disturbance)

        self.assertNotAlmostEqual(float(input_disturbance.norm()), 0.0)
        self.assertEqual(tuple(input_disturbance.shape), (2,))
        self.assertTrue(float(executed_action[0]) >= 0.0)

        state_disturbance = runner._state_disturbance(state, step_i=1)
        self.assertAlmostEqual(float(state_disturbance.norm()), 0.0)

    def test_constant_input_disturbance_is_applied_in_input_channel(self):
        config = CompareConfig(
            scenario="four",
            steps=2,
            disturbance="constant_input",
            input_disturbance_bias=(0.12, 0.24),
            device="cpu",
        )
        runner = CompareRunner(config)
        state = runner.start
        action = state.new_tensor([0.3, -0.2])
        input_disturbance = runner._input_disturbance(step_i=0)
        executed_action = runner._executed_action(action, input_disturbance)

        self.assertAlmostEqual(float(input_disturbance[0]), 0.12, places=6)
        self.assertAlmostEqual(float(input_disturbance[1]), 0.24, places=6)
        self.assertAlmostEqual(float(executed_action[0]), 0.42, places=6)
        self.assertAlmostEqual(float(executed_action[1]), 0.04, places=6)

        state_disturbance = runner._state_disturbance(state, step_i=0)
        self.assertAlmostEqual(float(state_disturbance.norm()), 0.0)

    def test_sdob_cost_observer_uses_local_perception_radius(self):
        config = CompareConfig(
            scenario="four",
            perception_radius=1.0,
            sdob_radius=0.8,
            lookahead_dist=0.0,
            num_cost_queries=24,
            device="cpu",
        )
        runner = CompareRunner(config)
        dynamics = runner._make_dynamics()
        agent = runner._make_agent("MPPI+SDOB", dynamics)
        near = [
            {
                "pos": runner.start[:2] + runner.start.new_tensor([0.4, 0.0]),
                "radius": 0.2,
            }
        ]
        far = [
            {
                "pos": runner.start[:2] + runner.start.new_tensor([5.0, 0.0]),
                "radius": 0.2,
            }
        ]

        self.assertEqual(len(agent.cost_observer._perceived_obstacles(near, runner.start)), 1)
        self.assertEqual(len(agent.cost_observer._perceived_obstacles(far, runner.start)), 0)

    def test_robot_perception_filters_obstacles_before_control(self):
        runner = CompareRunner(
            CompareConfig(perception_radius=3.2, sdob_radius=2.6, device="cpu")
        )
        obstacles = [
            {"pos": torch.tensor([2.0, 0.0]), "radius": 0.2},
            {"pos": torch.tensor([5.0, 0.0]), "radius": 0.2},
        ]

        perceived = runner._perceived_obstacles(runner.start, obstacles)

        self.assertEqual(len(perceived), 1)
        torch.testing.assert_close(perceived[0]["pos"], obstacles[0]["pos"])

    def test_rmppi_scenario_shift_accepts_empty_perception(self):
        self.assertEqual(_shift_obstacle_dicts([], 0.1, -0.1), [])

    def test_cost_flow_error_ignores_queries_without_observed_cost_change(self):
        runner = CompareRunner(CompareConfig(device="cpu", dt=0.1))
        previous = [{"pos": torch.tensor([0.0, 0.0]), "radius": 0.2}]
        current = [{"pos": torch.tensor([0.1, 0.0]), "radius": 0.2}]
        cost_flow = {
            "query_points": torch.tensor([[0.0, 0.0], [10.0, 10.0]]),
            "query_weights": torch.tensor([1.0, 0.0]),
            "query_velocities": torch.tensor([[1.0, 0.0], [100.0, 100.0]]),
            "dt": 0.1,
        }

        error = runner._cost_flow_error(cost_flow, previous, current)

        self.assertAlmostEqual(error, 0.0, places=6)

    def test_all_controllers_use_same_obstacle_margin_for_fair_comparison(self):
        config = CompareConfig(scenario="four", device="cpu")
        runner = CompareRunner(config)
        dynamics = runner._make_dynamics()

        for name in DEFAULT_CONTROLLER_NAMES:
            agent = runner._make_agent(name, dynamics)
            self.assertAlmostEqual(agent.planner.obstacle_margin, 0.05)

    def test_run_compare_entrypoints_expose_main(self):
        top_level = importlib.import_module("sdob.examples.navigation.run_compare")

        self.assertTrue(callable(top_level.main))


if __name__ == "__main__":
    unittest.main()
