import math
import subprocess
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from sdob.examples.pendulum.benchmark import make_controllers, run_suite
from sdob.examples.pendulum.animate import animate_trajectories, controller_layout_order, disturbance_text, pendulum_tip
from sdob.examples.pendulum.controllers import DOBMPPIController, RMPPIController, TubeMPPIController, disturbance_forecast
from sdob.examples.pendulum.dynamics import PendulumConfig, pendulum_step_np
from sdob.examples.pendulum.factory import make_controllers as make_controller_suite
from sdob.examples.pendulum.interfaces import ControllerBase
from sdob.examples.pendulum.mppi import BaseMPPI, MPPIConfig, TorchMPPI
from sdob.examples.pendulum.tracking import lqr_tracking_gain_np, tracking_feedback_control_np
from sdob.examples.pendulum.observers import ClassicalDOBObserver, DisturbanceEstimate, ParticleSDOBObserver
from sdob.examples.pendulum.workspace import BenchmarkWorkspace, ExperimentConfig, load_experiment_config


def test_particle_sdob_observer_recovers_matched_torque_bias():
    config = PendulumConfig(dt=0.05)
    observer = ParticleSDOBObserver(config, d_min=-2.0, d_max=2.0, num_particles=121, beta=1.0)
    state = np.array([0.25, -0.1], dtype=float)
    action = np.array([0.35], dtype=float)
    true_disturbance = 0.7

    feedback = pendulum_step_np(state, action, config, disturbance=true_disturbance)
    estimate = observer.update(state, action, feedback)

    assert math.isclose(estimate.mean, true_disturbance, abs_tol=0.08)
    assert estimate.sigma < 0.2


def test_particle_sdob_observer_estimates_disturbance_rate_from_successive_feedback():
    config = PendulumConfig(dt=0.05)
    observer = ParticleSDOBObserver(config, d_min=-2.0, d_max=2.0, num_particles=401, beta=1.0, rate_limit=3.0)
    state = np.array([0.25, -0.1], dtype=float)
    action = np.array([0.35], dtype=float)

    first_feedback = pendulum_step_np(state, action, config, disturbance=0.1)
    first = observer.update(state, action, first_feedback)
    second_feedback = pendulum_step_np(state, action, config, disturbance=0.2)
    second = observer.update(state, action, second_feedback)

    assert math.isclose(first.rate, 0.0, abs_tol=1.0e-8)
    assert math.isclose(second.mean, 0.2, abs_tol=0.03)
    assert math.isclose(second.rate, 2.0, abs_tol=0.35)


def test_particle_sdob_observer_clamps_default_disturbance_rate():
    config = PendulumConfig(dt=0.05)
    observer = ParticleSDOBObserver(config, d_min=-2.0, d_max=2.0, num_particles=401, beta=1.0)
    state = np.array([0.25, -0.1], dtype=float)
    action = np.array([0.35], dtype=float)

    observer.update(state, action, pendulum_step_np(state, action, config, disturbance=-1.0))
    estimate = observer.update(state, action, pendulum_step_np(state, action, config, disturbance=1.0))

    assert abs(estimate.rate) <= 1.2 + 1.0e-8


def test_classical_dob_skips_update_when_velocity_feedback_is_saturated():
    config = PendulumConfig.gym_pendulum()
    observer = ClassicalDOBObserver(config)
    observer.mean = 0.7
    observer.rate = 0.2
    observer.initialized = True
    state = np.array([2.5, config.max_speed], dtype=float)
    feedback = pendulum_step_np(state, action=2.0, config=config, disturbance=0.7)

    estimate = observer.update(state, 2.0, feedback)

    assert math.isclose(estimate.mean, 0.7, abs_tol=1.0e-8)
    assert math.isclose(estimate.rate, 0.2, abs_tol=1.0e-8)


def test_pendulum_tip_uses_upright_zero_angle():
    assert np.allclose(pendulum_tip(0.0, length=1.0), np.array([0.0, 1.0]))
    assert np.allclose(pendulum_tip(np.pi / 2.0, length=2.0), np.array([2.0, 0.0]), atol=1.0e-8)


def test_animation_controller_order_matches_comparison_order():
    names = ["RMPPI", "MPPI+SDOB", "Tube-MPPI", "MPPI", "MPPI+DOB", "MPPI (no disturbance)"]
    assert controller_layout_order(names) == [
        "MPPI (no disturbance)",
        "MPPI",
        "MPPI+DOB",
        "MPPI+SDOB",
        "Tube-MPPI",
        "RMPPI",
    ]


def test_disturbance_text_shows_true_and_estimated_values():
    row = {"disturbance": 0.75, "d_hat": 0.62}
    text = disturbance_text(row)

    assert "d_true=+0.75" in text
    assert "d_hat=+0.62" in text
    assert "err=-0.13" in text


def test_disturbance_text_marks_missing_estimate():
    row = {"disturbance": 0.75, "d_hat": np.nan}
    text = disturbance_text(row)

    assert "d_true=+0.75" in text
    assert "d_hat=n/a" in text


def test_gym_pendulum_config_matches_pendulum_v1_limits():
    config = PendulumConfig.gym_pendulum()

    assert config.dynamics_model == "gym"
    assert math.isclose(config.dt, 0.05)
    assert math.isclose(config.max_torque, 2.0)
    assert math.isclose(config.max_speed, 8.0)


def test_gym_pendulum_step_matches_reference_dynamics():
    config = PendulumConfig.gym_pendulum()
    state = np.array([0.25, -0.1], dtype=float)
    action = np.array([0.35], dtype=float)
    disturbance = 0.2

    feedback = pendulum_step_np(state, action, config, disturbance=disturbance)

    expected_dot = state[1] + (
        3.0 * config.gravity / (2.0 * config.length) * np.sin(state[0])
        + 3.0 * (float(action[0]) + disturbance) / (config.mass * config.length**2)
    ) * config.dt
    expected_dot = float(np.clip(expected_dot, -config.max_speed, config.max_speed))
    expected_theta = float((state[0] + expected_dot * config.dt + np.pi) % (2.0 * np.pi) - np.pi)
    assert np.allclose(feedback, np.array([expected_theta, expected_dot]))


def test_project_exposes_mppi_and_tracking_modules():
    assert issubclass(TorchMPPI, BaseMPPI)
    assert callable(lqr_tracking_gain_np)
    assert callable(tracking_feedback_control_np)


def test_factory_returns_controller_interface_instances():
    controllers = make_controller_suite(PendulumConfig(dt=0.05), num_samples=8, horizon=3, seed=5)

    assert [controller.name for controller in controllers] == ["MPPI", "MPPI+DOB", "MPPI+SDOB", "Tube-MPPI", "RMPPI"]
    assert all(isinstance(controller, ControllerBase) for controller in controllers)


def test_direct_run_script_is_available():
    script = Path("sdob/examples/pendulum/run_compare.py")

    assert script.exists()


def test_package_import_does_not_eagerly_load_pandas():
    code = "import sys; import sdob.examples.pendulum; print('pandas' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)

    assert result.stdout.strip() == "False"


def test_workspace_runs_experiment_lifecycle(tmp_path):
    config = ExperimentConfig(
        output_dir=tmp_path / "workspace_smoke",
        steps=4,
        num_samples=8,
        horizon=3,
        skip_gif=True,
    )
    workspace = BenchmarkWorkspace(config)

    output = workspace.run()

    assert output.summary_path.exists()
    assert output.trajectory_path.exists()
    assert output.plot_path.exists()
    assert output.gif_path is None
    assert set(output.result.summary["controller"]) == {
        "MPPI (no disturbance)",
        "MPPI",
        "MPPI+DOB",
        "MPPI+SDOB",
        "Tube-MPPI",
        "RMPPI",
    }
    baseline = output.result.summary[output.result.summary["controller"] == "MPPI (no disturbance)"].iloc[0]
    assert math.isclose(float(baseline["disturbance_value"]), 0.0)
    assert (
        output.result.trajectories[output.result.trajectories["controller"] == "MPPI (no disturbance)"][
            "disturbance"
        ]
        == 0.0
    ).all()


def test_workspace_forwards_action_cost_to_pendulum():
    config = ExperimentConfig(action_cost=0.006, skip_gif=True)
    workspace = BenchmarkWorkspace(config)

    assert math.isclose(workspace.pendulum.action_cost, 0.006)


def test_workspace_can_render_gif_from_saved_trajectory(tmp_path):
    config = ExperimentConfig(
        output_dir=tmp_path / "workspace_gif_smoke",
        steps=3,
        num_samples=4,
        horizon=2,
        style="schematic",
        dpi=45,
        skip_gif=False,
    )

    output = BenchmarkWorkspace(config).run()

    assert output.gif_path is not None
    assert output.gif_path.exists()


def test_gym_animation_uses_native_rgb_renderer(tmp_path):
    csv_path = tmp_path / "trajectory.csv"
    csv_path.write_text(
        "\n".join(
            [
                "controller,step,time,theta,theta_abs,theta_dot,action,disturbance,d_hat",
                "RMPPI,0,0.00,0.35,0.35,0.00,0.20,0.75,0.70",
                "RMPPI,1,0.05,0.30,0.30,-0.10,-0.10,0.75,0.72",
            ]
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "native.gif"

    result = animate_trajectories(csv_path, output_path, fps=8, stride=1, style="gym")

    assert result.exists()
    from PIL import Image

    with Image.open(result) as image:
        assert image.size == (300, 412)


def test_experiment_config_loads_from_yaml(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "\n".join(
            [
                "output_dir: outputs/pendulum/yaml_smoke",
                "steps: 6",
                "num_samples: 16",
                "horizon: 4",
                "lambda_: 0.7",
                "noise_sigma: 0.9",
                "adaptive_temperature: true",
                "eta_min: 0.02",
                "eta_max: 0.15",
                "action_cost: 0.004",
                "disturbance_compensation_cost: 0.12",
                "device: cuda",
                "disturbance: step_bias",
                "disturbance_value: 0.25",
                "skip_gif: true",
            ]
        ),
        encoding="utf-8",
    )

    config = load_experiment_config(path)

    assert config.output_dir == Path("outputs/pendulum/yaml_smoke")
    assert config.steps == 6
    assert config.num_samples == 16
    assert config.horizon == 4
    assert math.isclose(config.lambda_, 0.7)
    assert math.isclose(config.noise_sigma, 0.9)
    assert config.adaptive_temperature is True
    assert math.isclose(config.eta_min, 0.02)
    assert math.isclose(config.eta_max, 0.15)
    assert math.isclose(config.action_cost, 0.004)
    assert math.isclose(config.disturbance_compensation_cost, 0.12)
    assert config.device == "cuda"
    assert config.disturbance == "step_bias"
    assert math.isclose(config.disturbance_value, 0.25)
    assert config.skip_gif is True


def test_workspace_forwards_device_to_controller_factory():
    config = ExperimentConfig(device="cuda")

    with mock.patch(
        "sdob.examples.pendulum.workspace.make_controllers",
        return_value=[],
    ) as make_controller_mock:
        BenchmarkWorkspace(config)

    assert make_controller_mock.call_args.kwargs["device"] == "cuda"


def test_all_controller_variants_run_small_benchmark():
    config = PendulumConfig(dt=0.05)
    controllers = make_controllers(config, num_samples=64, horizon=10, seed=3)

    result = run_suite(
        controllers=controllers,
        config=config,
        steps=12,
        initial_state=np.array([0.35, 0.0], dtype=float),
        disturbance_name="constant_bias",
        disturbance_value=0.5,
        seed=11,
    )

    names = set(result.summary["controller"])
    assert names == {"MPPI", "MPPI+DOB", "MPPI+SDOB", "Tube-MPPI", "RMPPI"}
    assert result.trajectories["controller"].nunique() == 5
    assert result.summary["total_cost"].notna().all()


def test_benchmark_summary_reports_disturbance_estimation_error():
    config = PendulumConfig(dt=0.05)
    controller = DOBMPPIController(config, MPPIConfig(horizon=3, num_samples=8, seed=41))

    result = run_suite(
        controllers=[controller],
        config=config,
        steps=4,
        initial_state=np.array([0.2, 0.0], dtype=float),
        disturbance_name="constant_bias",
        disturbance_value=0.5,
        seed=0,
    )

    row = result.summary.iloc[0]
    assert "d_mae" in result.summary.columns
    assert "d_rmse" in result.summary.columns
    assert "d_dot_mae" in result.summary.columns
    assert "planning_time_mean_ms" in result.summary.columns
    assert "planning_time_p95_ms" in result.summary.columns
    assert "planning_time_ms" in result.trajectories.columns
    assert not math.isnan(float(row["d_mae"]))
    assert float(row["planning_time_mean_ms"]) >= 0.0


def test_mppi_adaptive_temperature_uses_effective_sample_ratio():
    config = PendulumConfig(dt=0.05)
    planner = TorchMPPI(
        config,
        MPPIConfig(
            horizon=3,
            num_samples=4,
            lambda_=1.0,
            adaptive_temperature=True,
            eta_min=0.3,
            eta_max=0.8,
        ),
    )

    planner._adapt_temperature_from_weights(torch.full((4,), 0.25, dtype=torch.double))
    assert math.isclose(planner.last_eta, 1.0, abs_tol=1.0e-8)
    assert math.isclose(planner.lambda_, 0.9, abs_tol=1.0e-8)

    planner._adapt_temperature_from_weights(torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.double))
    assert math.isclose(planner.last_eta, 0.25, abs_tol=1.0e-8)
    assert math.isclose(planner.lambda_, 1.08, abs_tol=1.0e-8)


def test_gaussian_likelihood_ratio_matches_reference_formula():
    config = PendulumConfig(dt=0.05)
    planner = TorchMPPI(
        config,
        MPPIConfig(
            horizon=2,
            num_samples=1,
            lambda_=0.5,
            noise_sigma=2.0,
            sampling_control_cost=3.0,
            likelihood_alpha=0.2,
        ),
    )
    planner.U[:, 0] = torch.tensor([0.5, -0.25], dtype=torch.double)
    actions = torch.tensor([[[1.25], [-0.75]]], dtype=torch.double)

    cost = planner._likelihood_ratio_cost(actions)

    expected = 0.5 * 0.5 * (1.0 - 0.2) * 3.0 * (
        0.5 * (0.5 - 2.0 * 1.25) / 4.0
        + (-0.25) * (-0.25 - 2.0 * -0.75) / 4.0
    )
    assert torch.allclose(cost.cpu(), torch.tensor([expected], dtype=torch.double))


def test_gaussian_feedback_cost_matches_reference_formula():
    config = PendulumConfig(dt=0.05)
    planner = TorchMPPI(
        config,
        MPPIConfig(
            horizon=2,
            num_samples=1,
            lambda_=0.5,
            noise_sigma=2.0,
            sampling_control_cost=3.0,
            likelihood_alpha=0.2,
        ),
    )
    feedback = torch.tensor([[[0.6], [-0.2]]], dtype=torch.double)

    cost = planner._feedback_sampling_cost(feedback)

    expected = 0.5 * 0.5 * (1.0 - 0.2) * 3.0 * ((0.6 * 0.6 + -0.2 * -0.2) / 4.0)
    assert torch.allclose(cost.cpu(), torch.tensor([expected], dtype=torch.double))


def test_mppi_rollout_cost_penalizes_action_changes_when_smoothness_is_enabled():
    config = PendulumConfig(
        dt=0.0,
        gravity=0.0,
        damping=0.0,
        angle_cost=0.0,
        velocity_cost=0.0,
        action_cost=0.0,
        terminal_angle_cost=0.0,
        terminal_velocity_cost=0.0,
        action_smoothness_cost=2.0,
    )
    planner = TorchMPPI(config, MPPIConfig(horizon=3, num_samples=2, seed=5))
    state = np.array([0.0, 0.0], dtype=float)
    constant_actions = torch.zeros((1, 3, 1), dtype=torch.double)
    changing_actions = torch.tensor([[[1.0], [-1.0], [1.0]]], dtype=torch.double)

    constant_cost = float(planner._rollout_costs(state, constant_actions, disturbance=0.0)[0])
    changing_cost = float(planner._rollout_costs(state, changing_actions, disturbance=0.0)[0])

    assert math.isclose(constant_cost, 0.0, abs_tol=1.0e-8)
    assert changing_cost > constant_cost


def test_mppi_rollout_cost_prefers_steady_disturbance_compensating_action():
    config = PendulumConfig(
        dt=0.0,
        gravity=0.0,
        damping=0.0,
        angle_cost=0.0,
        velocity_cost=0.0,
        action_cost=0.0,
        action_smoothness_cost=0.0,
        terminal_angle_cost=0.0,
        terminal_velocity_cost=0.0,
        disturbance_compensation_cost=1.0,
    )
    planner = TorchMPPI(config, MPPIConfig(horizon=3, num_samples=2, seed=5))
    state = np.array([0.0, 0.0], dtype=float)
    zero_actions = torch.zeros((1, 3, 1), dtype=torch.double)
    compensating_actions = torch.full((1, 3, 1), -0.75, dtype=torch.double)

    zero_cost = float(planner._rollout_costs(state, zero_actions, disturbance=0.75)[0])
    compensating_cost = float(planner._rollout_costs(state, compensating_actions, disturbance=0.75)[0])

    assert compensating_cost < zero_cost


def test_mppi_rollout_cost_uses_disturbance_sequence_across_horizon():
    config = PendulumConfig(
        dt=1.0,
        gravity=0.0,
        damping=0.0,
        max_torque=5.0,
        max_speed=100.0,
        angle_cost=0.0,
        velocity_cost=0.0,
        action_cost=0.0,
        action_smoothness_cost=0.0,
        terminal_angle_cost=0.0,
        terminal_velocity_cost=1.0,
    )
    planner = TorchMPPI(config, MPPIConfig(horizon=2, num_samples=1, seed=5))
    state = np.array([0.0, 0.0], dtype=float)
    actions = torch.zeros((1, 2, 1), dtype=torch.double)

    cost = float(planner._rollout_costs(state, actions, disturbance=torch.tensor([0.0, 1.0]))[0])

    assert math.isclose(cost, 1.0, abs_tol=1.0e-8)


def test_mppi_disturbance_compensation_cost_is_inactive_without_disturbance():
    config = PendulumConfig(
        dt=0.0,
        gravity=0.0,
        damping=0.0,
        angle_cost=0.0,
        velocity_cost=0.0,
        action_cost=0.0,
        action_smoothness_cost=0.0,
        terminal_angle_cost=0.0,
        terminal_velocity_cost=0.0,
        disturbance_compensation_cost=1.0,
    )
    planner = TorchMPPI(config, MPPIConfig(horizon=3, num_samples=2, seed=6))
    state = np.array([0.0, 0.0], dtype=float)
    nonzero_actions = torch.ones((1, 3, 1), dtype=torch.double)

    cost = float(planner._rollout_costs(state, nonzero_actions, disturbance=0.0)[0])

    assert cost == 0.0


def test_mppi_recenters_nominal_actions_when_rollout_disturbance_changes():
    config = PendulumConfig(dt=0.05, max_torque=2.0)
    planner = TorchMPPI(config, MPPIConfig(horizon=4, num_samples=8, seed=7))
    planner.U.fill_(0.1)

    planner._recenter_action_prior_for_disturbance(0.75)

    expected = torch.full((4, 1), -0.65, dtype=torch.double)
    assert torch.allclose(planner.U.cpu(), expected)
    assert math.isclose(planner.last_rollout_disturbance, 0.75, abs_tol=1.0e-8)


def test_mppi_shift_repeats_previous_terminal_action():
    config = PendulumConfig(dt=0.05, max_torque=2.0)
    planner = TorchMPPI(config, MPPIConfig(horizon=4, num_samples=8, seed=8))
    planner.U[:, 0] = torch.tensor([-0.7, -0.6, -0.5, -0.4], dtype=torch.double)
    planner.last_rollout_disturbance = 0.75

    planner._shift_nominal()

    expected = torch.tensor([-0.6, -0.5, -0.4, -0.4], dtype=torch.double).view(4, 1)
    assert torch.allclose(planner.U.cpu(), expected)


def test_mppi_shift_can_be_deferred_when_nominal_stride_is_zero():
    config = PendulumConfig(dt=0.05, max_torque=2.0)
    planner = TorchMPPI(config, MPPIConfig(horizon=4, num_samples=8, seed=9))
    planner.U[:, 0] = torch.tensor([-0.7, -0.6, -0.5, -0.4], dtype=torch.double)

    planner.shift_nominal(0)

    expected = torch.tensor([-0.7, -0.6, -0.5, -0.4], dtype=torch.double).view(4, 1)
    assert torch.allclose(planner.U.cpu(), expected)


def test_augmented_mppi_keeps_nominal_action_separate_from_clipped_action():
    config = PendulumConfig(dt=0.05, max_torque=2.0)
    planner = TorchMPPI(config, MPPIConfig(horizon=4, num_samples=8, noise_sigma=0.0, seed=11))
    planner.U[:, 0] = 0.4

    plan = planner.command_augmented(
        actual_state=np.array([0.0, 0.0], dtype=float),
        nominal_state=np.array([1.0, 0.0], dtype=float),
        tracking_gain=np.array([10.0, 0.0], dtype=float),
        tracking_cost_weight=0.0,
        value_function_threshold=100.0,
    )

    assert math.isclose(plan.action, config.max_torque, abs_tol=1.0e-8)
    assert plan.nominal_action is not None
    assert math.isclose(plan.nominal_action, 0.4, abs_tol=1.0e-8)
    assert plan.real_action is not None


def test_augmented_mppi_executes_real_cost_sequence_not_nominal_sequence():
    config = PendulumConfig(dt=0.05, max_torque=2.0)
    planner = TorchMPPI(config, MPPIConfig(horizon=1, num_samples=2, noise_sigma=0.0, lambda_=0.05, seed=12))
    planner.U[:, 0] = 0.0
    planner._sample_noise = lambda: torch.tensor([[[-1.0]], [[1.0]]], dtype=torch.double)

    def fake_augmented_costs(*args, **kwargs):
        mixed_nominal_cost = torch.tensor([0.0, 100.0], dtype=torch.double)
        real_closed_loop_cost = torch.tensor([100.0, 0.0], dtype=torch.double)
        real_closed_loop_actions = torch.tensor([[[0.2]], [[1.4]]], dtype=torch.double)
        return mixed_nominal_cost, real_closed_loop_cost, real_closed_loop_actions

    planner._augmented_rollout_costs = fake_augmented_costs

    plan = planner.command_augmented(
        actual_state=np.array([0.0, 0.0], dtype=float),
        nominal_state=np.array([0.0, 0.0], dtype=float),
        tracking_gain=np.array([0.0, 0.0], dtype=float),
        tracking_cost_weight=0.0,
        value_function_threshold=100.0,
    )

    assert plan.nominal_action is not None
    assert plan.nominal_action < -0.9
    assert plan.real_action is not None
    assert plan.real_action > 0.9
    assert plan.action > 0.9


def test_augmented_mppi_nominal_and_real_weights_share_same_temperature():
    config = PendulumConfig(dt=0.05, max_torque=2.0)
    planner = TorchMPPI(config, MPPIConfig(horizon=1, num_samples=2, noise_sigma=0.0, lambda_=0.5, seed=14))
    planner.U[:, 0] = 0.0
    planner._sample_noise = lambda: torch.tensor([[[-1.0]], [[1.0]]], dtype=torch.double)

    def fake_augmented_costs(*args, **kwargs):
        mixed_nominal_cost = torch.tensor([0.0, 1.0], dtype=torch.double)
        real_closed_loop_cost = torch.tensor([1.0, 0.0], dtype=torch.double)
        real_closed_loop_actions = torch.tensor([[[-1.0]], [[1.0]]], dtype=torch.double)
        return mixed_nominal_cost, real_closed_loop_cost, real_closed_loop_actions

    logged_lambdas: list[float] = []
    original_weights_from_costs = planner._weights_from_costs

    def logged_weights(costs):
        logged_lambdas.append(planner.lambda_)
        return original_weights_from_costs(costs)

    planner._augmented_rollout_costs = fake_augmented_costs
    planner._weights_from_costs = logged_weights
    planner._adapt_temperature_from_weights = lambda weights: setattr(planner, "lambda_", 99.0)

    planner.command_augmented(
        actual_state=np.array([0.0, 0.0], dtype=float),
        nominal_state=np.array([0.0, 0.0], dtype=float),
        tracking_gain=np.array([0.0, 0.0], dtype=float),
        tracking_cost_weight=0.0,
        value_function_threshold=100.0,
    )

    assert logged_lambdas[:2] == [0.5, 0.5]


def test_dob_mppi_uses_estimate_inside_rollout_model_not_direct_input_subtraction():
    class DummyPlanner:
        def __init__(self):
            self.config = MPPIConfig(horizon=3, num_samples=4)
            self.rollout_disturbance = None

        def command(self, state, rollout_disturbance=0.0, disturbance_scenarios=None):
            self.rollout_disturbance = rollout_disturbance
            return 0.4

    config = PendulumConfig(dt=0.05, max_torque=2.0)
    controller = DOBMPPIController(config, MPPIConfig(horizon=3, num_samples=4, seed=13))
    dummy_planner = DummyPlanner()
    controller.planner = dummy_planner
    controller.estimate = DisturbanceEstimate(mean=0.7, sigma=0.1, measurement=0.7)

    action = controller.act(np.array([0.2, 0.0], dtype=float), step=0)

    assert np.allclose(dummy_planner.rollout_disturbance, np.array([0.7, 0.7, 0.7]))
    assert math.isclose(action, 0.4, abs_tol=1.0e-8)


def test_dob_mppi_forecasts_disturbance_rate_inside_rollout_model():
    class DummyPlanner:
        def __init__(self, horizon):
            self.config = MPPIConfig(horizon=horizon, num_samples=4)
            self.rollout_disturbance = None

        def command(self, state, rollout_disturbance=0.0, disturbance_scenarios=None):
            self.rollout_disturbance = rollout_disturbance
            return 0.4

    config = PendulumConfig(dt=0.05, max_torque=2.0)
    controller = DOBMPPIController(config, MPPIConfig(horizon=3, num_samples=4, seed=13))
    dummy_planner = DummyPlanner(horizon=3)
    controller.planner = dummy_planner
    controller.estimate = DisturbanceEstimate(mean=0.5, sigma=0.1, measurement=0.5, rate=2.0)

    action = controller.act(np.array([0.2, 0.0], dtype=float), step=0)

    assert np.allclose(dummy_planner.rollout_disturbance, np.array([0.5, 0.6, 0.7]))
    assert math.isclose(action, 0.4, abs_tol=1.0e-8)


def test_disturbance_forecast_uses_local_rate_window_not_full_horizon_extrapolation():
    config = PendulumConfig(dt=0.05, max_torque=10.0)
    estimate = DisturbanceEstimate(mean=0.0, sigma=0.1, measurement=0.0, rate=10.0)

    forecast = disturbance_forecast(estimate, config, horizon=8)

    assert np.allclose(forecast, np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 2.5, 2.5]))


def test_tube_mppi_resets_nominal_state_when_actual_solution_is_acceptable():
    config = PendulumConfig(dt=0.05)
    controller = TubeMPPIController(
        config,
        MPPIConfig(horizon=5, num_samples=32, seed=17),
        reset_threshold=1.0e6,
    )
    controller.reset()
    controller.nominal_state = np.array([1.0, 0.0], dtype=float)
    controller.initialized = True
    actual_state = np.array([0.15, 0.0], dtype=float)

    controller.act(actual_state, step=4)

    assert np.allclose(controller.nominal_state, actual_state)
    assert controller.info()["used_actual_reset"] == 1.0


def test_tube_mppi_default_keeps_nominal_tube_without_unconstrained_reset():
    config = PendulumConfig(dt=0.05)
    controller = TubeMPPIController(
        config,
        MPPIConfig(horizon=5, num_samples=32, seed=18),
    )
    controller.reset()
    controller.nominal_state = np.array([1.0, 0.0], dtype=float)
    controller.initialized = True
    actual_state = np.array([0.15, 0.0], dtype=float)

    controller.act(actual_state, step=4)

    assert not np.allclose(controller.nominal_state, actual_state)
    assert controller.info()["used_actual_reset"] == 0.0


def test_robust_baselines_do_not_use_dob_compensation_cost():
    config = PendulumConfig.gym_pendulum(disturbance_compensation_cost=3.0)
    mppi_config = MPPIConfig(horizon=5, num_samples=16, seed=19)

    tube = TubeMPPIController(config, mppi_config)
    rmppi = RMPPIController(config, mppi_config)

    assert math.isclose(config.disturbance_compensation_cost, 3.0)
    assert math.isclose(tube.pendulum.disturbance_compensation_cost, 0.0)
    assert math.isclose(tube.nominal_planner.pendulum.disturbance_compensation_cost, 0.0)
    assert math.isclose(tube.actual_planner.pendulum.disturbance_compensation_cost, 0.0)
    assert math.isclose(rmppi.pendulum.disturbance_compensation_cost, 0.0)
    assert math.isclose(rmppi.planner.pendulum.disturbance_compensation_cost, 0.0)


def test_rmppi_uses_augmented_nominal_real_tracking_feedback():
    config = PendulumConfig(dt=0.05)
    controller = RMPPIController(
        config,
        MPPIConfig(horizon=5, num_samples=32, seed=23),
        disturbance_bound=0.4,
        kp=3.0,
        kd=0.5,
    )
    controller.reset()
    controller.nominal_state = np.array([0.8, 0.0], dtype=float)
    controller.initialized = True
    actual_state = np.array([0.2, 0.0], dtype=float)

    action = controller.act(actual_state, step=3)
    info = controller.info()

    assert abs(info["tracking_feedback"]) > 0.1
    expected = np.clip(
        info["real_action"] + info["tracking_feedback"],
        -config.max_torque,
        config.max_torque,
    )
    assert math.isclose(action, expected, abs_tol=1.0e-8)


def test_rmppi_does_not_use_disturbance_scenario_enumeration():
    class DummyPlanner:
        def __init__(self):
            self.kwargs = None

        def reset(self):
            pass

        def first_nominal_action(self):
            return 0.0

        def shift_nominal(self, steps=1):
            self.shift_steps = steps

        def command_augmented(self, **kwargs):
            self.kwargs = kwargs
            return type("Plan", (), {"action": 0.3, "nominal_action": 0.2, "real_action": 0.25, "cost": 0.0})()

    config = PendulumConfig(dt=0.05)
    controller = RMPPIController(
        config,
        MPPIConfig(horizon=5, num_samples=16, seed=24),
        free_energy_threshold=6.0,
        value_function_threshold=40.0,
    )
    dummy_planner = DummyPlanner()
    controller.planner = dummy_planner

    controller.act(np.array([0.2, 0.0], dtype=float), step=0)

    assert dummy_planner.kwargs is not None
    assert "disturbance_scenarios" not in dummy_planner.kwargs
    assert math.isclose(dummy_planner.kwargs["value_function_threshold"], 40.0)


def test_tube_mppi_lqr_tracking_can_replace_pd_feedback():
    config = PendulumConfig(dt=0.05)
    controller = TubeMPPIController(
        config,
        MPPIConfig(horizon=5, num_samples=32, seed=29),
        kp=0.0,
        kd=0.0,
        reset_threshold=-1.0e9,
        tracking_mode="lqr",
    )
    controller.reset()
    controller.nominal_state = np.array([0.8, 0.0], dtype=float)
    controller.initialized = True
    actual_state = np.array([0.2, 0.0], dtype=float)

    controller.act(actual_state, step=3)
    info = controller.info()

    assert abs(info["tracking_feedback"]) > 0.1
    assert info["tracking_mode"] == "lqr"
    assert abs(info["tracking_gain_theta"]) > 0.1


def test_rmppi_soft_nominal_selection_prefers_propagated_state_when_costs_tie():
    config = PendulumConfig(dt=0.05)
    controller = RMPPIController(
        config,
        MPPIConfig(horizon=5, num_samples=16, seed=31),
        free_energy_threshold=10.0,
        allow_real_nominal_projection=True,
        nominal_projection_penalty=1.0,
        nominal_jump_penalty=0.0,
    )
    controller.nominal_state = np.array([0.0, 0.0], dtype=float)
    feedback = np.array([2.0, 0.0], dtype=float)
    propagated = np.array([1.0, 0.0], dtype=float)
    controller._candidate_free_energy = lambda candidate: 0.0

    selected = controller._select_nominal_state(feedback, propagated)

    assert np.allclose(selected, propagated)


def test_rmppi_soft_nominal_selection_can_move_toward_feedback_when_cost_improves():
    config = PendulumConfig(dt=0.05)
    controller = RMPPIController(
        config,
        MPPIConfig(horizon=5, num_samples=16, seed=31),
        free_energy_threshold=10.0,
        allow_real_nominal_projection=True,
        nominal_projection_penalty=1.0,
        nominal_jump_penalty=0.0,
    )
    controller.nominal_state = np.array([0.0, 0.0], dtype=float)
    feedback = np.array([2.0, 0.0], dtype=float)
    propagated = np.array([1.0, 0.0], dtype=float)

    def candidate_free_energy(candidate):
        theta = float(candidate[0])
        if math.isclose(theta, 1.25, abs_tol=1.0e-8):
            return 0.0
        if math.isclose(theta, 1.0, abs_tol=1.0e-8):
            return 1.0
        return 5.0

    controller._candidate_free_energy = candidate_free_energy

    selected = controller._select_nominal_state(feedback, propagated)

    assert np.allclose(selected, np.array([1.25, 0.0]))


def test_rmppi_info_reports_selected_nominal_candidate():
    config = PendulumConfig(dt=0.05)
    controller = RMPPIController(
        config,
        MPPIConfig(horizon=5, num_samples=16, seed=31),
        free_energy_threshold=0.2,
        allow_real_nominal_projection=True,
    )
    controller.nominal_state = np.array([0.0, 0.0], dtype=float)
    controller.last_nominal_action = 0.0

    def candidate_free_energy(candidate):
        theta = float(candidate[0])
        if math.isclose(theta, 0.5, abs_tol=1.0e-8):
            return 0.1
        return 10.0

    controller._candidate_free_energy = candidate_free_energy

    controller.observe(
        state=np.array([0.0, 0.0], dtype=float),
        action=0.0,
        feedback=np.array([1.0, 0.0], dtype=float),
        step=0,
    )
    info = controller.info()

    assert math.isclose(info["selected_nominal_index"], 6.0)
    assert math.isclose(info["selected_nominal_stride"], 1.0)
    assert math.isclose(info["selected_nominal_cost"], 0.1)
    assert info["selected_nominal_source"] == "propagated_to_feedback"


def test_benchmark_records_rmppi_nominal_selection():
    config = PendulumConfig(dt=0.05)
    controller = RMPPIController(
        config,
        MPPIConfig(horizon=4, num_samples=8, seed=34),
    )

    result = run_suite(
        [controller],
        config,
        steps=1,
        initial_state=np.array([0.2, 0.0], dtype=float),
        disturbance_name="constant_bias",
        disturbance_value=0.1,
        seed=0,
    )

    row = result.trajectories.iloc[0]
    assert "selected_nominal_index" in result.trajectories.columns
    assert not math.isnan(float(row["selected_nominal_index"]))
    assert "selected_nominal_source" in result.trajectories.columns


def test_rmppi_nominal_propagation_uses_absolute_free_energy_threshold():
    config = PendulumConfig(dt=0.05)
    controller = RMPPIController(
        config,
        MPPIConfig(horizon=5, num_samples=16, seed=32),
        free_energy_threshold=1.0,
        num_nominal_candidates=3,
    )
    controller.nominal_state = np.array([0.0, 0.0], dtype=float)
    feedback = np.array([2.0, 0.0], dtype=float)
    propagated = np.array([1.0, 0.0], dtype=float)

    def candidate_free_energy(candidate):
        theta = float(candidate[0])
        if math.isclose(theta, 2.0, abs_tol=1.0e-8):
            return 1.2
        return 0.5

    controller._candidate_free_energy = candidate_free_energy

    selected = controller._select_nominal_state(feedback, propagated)

    assert np.allclose(selected, propagated)


def test_rmppi_default_nominal_selection_uses_soft_score_not_hard_projection():
    config = PendulumConfig(dt=0.05)
    controller = RMPPIController(
        config,
        MPPIConfig(horizon=5, num_samples=16, seed=32),
        free_energy_threshold=1.0e6,
        num_nominal_candidates=9,
    )
    controller.nominal_state = np.array([0.0, 0.0], dtype=float)
    controller._candidate_free_energy = lambda candidate: 0.0

    selected = controller._select_nominal_state(
        feedback=np.array([2.0, 0.0], dtype=float),
        propagated=np.array([1.0, 0.0], dtype=float),
    )

    assert np.allclose(selected, np.array([1.0, 0.0]))
    assert controller.info()["allow_real_nominal_projection"] == 1.0
    assert controller.info()["nominal_projection_penalty"] > 0.0


def test_rmppi_nominal_state_candidates_use_reference_line_search():
    config = PendulumConfig(dt=0.05)
    controller = RMPPIController(
        config,
        MPPIConfig(horizon=5, num_samples=16, seed=32),
        num_nominal_candidates=9,
    )
    controller.nominal_state = np.array([0.0, 0.0], dtype=float)

    candidates = controller._nominal_state_candidates(
        feedback=np.array([2.0, 0.0], dtype=float),
        propagated=np.array([1.0, 0.0], dtype=float),
    )

    expected_theta = np.linspace(0.0, 2.0, 9)
    assert len(candidates) == 9
    assert np.allclose([candidate[0] for candidate in candidates], expected_theta)


def test_rmppi_selected_candidate_controls_next_importance_sampler_shift():
    config = PendulumConfig(dt=0.05)
    controller = RMPPIController(
        config,
        MPPIConfig(horizon=4, num_samples=16, noise_sigma=0.0, seed=33),
        free_energy_threshold=0.2,
    )
    controller.planner.U[:, 0] = torch.tensor([-0.7, -0.6, -0.5, -0.4], dtype=torch.double)
    controller.nominal_state = np.array([0.0, 0.0], dtype=float)
    controller.initialized = True

    controller._candidate_free_energy = lambda candidate: 10.0
    controller.observe(
        state=np.array([0.0, 0.0], dtype=float),
        action=0.0,
        feedback=np.array([0.1, 0.0], dtype=float),
        step=0,
    )
    controller.act(np.array([0.1, 0.0], dtype=float), step=1)
    assert math.isclose(float(controller.info()["nominal_stride"]), 0.0)
    assert math.isclose(float(controller.last_nominal_action), -0.7, abs_tol=1.0e-8)

    controller.planner.U[:, 0] = torch.tensor([-0.7, -0.6, -0.5, -0.4], dtype=torch.double)
    controller.nominal_state = np.array([0.0, 0.0], dtype=float)
    controller._candidate_free_energy = lambda candidate: 0.0
    controller.observe(
        state=np.array([0.0, 0.0], dtype=float),
        action=0.0,
        feedback=np.array([0.1, 0.0], dtype=float),
        step=1,
    )
    controller.act(np.array([0.1, 0.0], dtype=float), step=2)
    assert math.isclose(float(controller.info()["nominal_stride"]), 1.0)
    assert math.isclose(float(controller.last_nominal_action), -0.6, abs_tol=1.0e-8)
