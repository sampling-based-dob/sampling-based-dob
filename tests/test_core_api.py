from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

from sdob.controller import MPPIConfig, MPPIController
from sdob.config import load_config
from sdob.observer import (
    CostDisturbanceConfig,
    CostDisturbanceField,
    CostDisturbanceObserver,
    SamplingDisturbanceObserver,
    StateEffectConfig,
    StateEffectObserver,
)


def test_simulation_tasks_are_packaged_as_examples() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    example_names = (
        "sdob.examples.pendulum",
        "sdob.examples.navigation",
        "sdob.examples.reacher",
    )

    for name in example_names:
        assert (repository_root / Path(*name.split("."))).is_dir()


class IntegratorModel:
    def step(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        disturbance: torch.Tensor | None = None,
    ) -> torch.Tensor:
        next_state = state + action
        if disturbance is not None:
            next_state = next_state + disturbance
        return next_state


def quadratic_trajectory_cost(
    trajectories: torch.Tensor,
    controls: torch.Tensor,
    context: dict[str, torch.Tensor],
) -> torch.Tensor:
    target = context["target"]
    tracking = (trajectories[:, 1:] - target).square().sum(dim=(1, 2))
    effort = 0.01 * controls.square().sum(dim=(1, 2))
    return tracking + effort


def test_generic_mppi_moves_integrator_toward_target() -> None:
    controller = MPPIController(
        IntegratorModel(),
        quadratic_trajectory_cost,
        MPPIConfig(
            horizon=6,
            num_samples=2048,
            action_dim=1,
            noise_sigma=(0.8,),
            action_min=(-1.0,),
            action_max=(1.0,),
            temperature=0.4,
            seed=7,
        ),
    )

    result = controller.plan(
        torch.tensor([0.0]),
        context={"target": torch.tensor([3.0])},
    )

    assert result.action.shape == (1,)
    assert float(result.action[0]) > 0.4
    assert torch.allclose(result.action, result.control_sequence[0])
    assert result.trajectories.shape == (2048, 7, 1)
    assert torch.isclose(result.weights.sum(), torch.tensor(1.0))


def test_state_effect_observer_recovers_constant_bias() -> None:
    observer = StateEffectObserver(
        IntegratorModel(),
        StateEffectConfig(
            disturbance_dim=1,
            num_samples=201,
            disturbance_min=(-1.0,),
            disturbance_max=(1.0,),
            prediction_weight=(1.0,),
            regularization_weight=(0.0,),
            temperature=0.01,
            filter_alpha=1.0,
        ),
    )

    estimate = observer.update(
        previous_state=torch.tensor([0.0]),
        applied_action=torch.tensor([0.2]),
        feedback_state=torch.tensor([0.65]),
    )

    assert torch.allclose(estimate.value, torch.tensor([0.45]), atol=0.02)
    assert estimate.candidates.shape == (201, 1)
    assert estimate.weights.shape == (201,)


def test_cost_disturbance_observer_recovers_translation_and_transports_cost() -> None:
    observer = CostDisturbanceObserver(
        CostDisturbanceConfig(
            dt=0.1,
            max_displacement=(0.3,),
            candidate_steps=13,
            neighborhood_radius=0.35,
            interpolation_bandwidth=0.4,
            displacement_penalty=0.0,
        )
    )
    query_states = torch.linspace(-0.8, 0.8, 33).unsqueeze(-1)

    def previous_cost(query: torch.Tensor) -> torch.Tensor:
        return torch.exp(-((query[..., 0] - 0.0) / 0.16).square())

    def current_cost(query: torch.Tensor) -> torch.Tensor:
        return torch.exp(-((query[..., 0] - 0.2) / 0.16).square())

    field = observer.update(query_states, previous_cost, current_cost)
    central = field.velocity_at(torch.tensor([[0.2]]))
    future_cost = field.transport_cost(
        torch.tensor([[0.4]]),
        horizon_step=1,
        current_cost=current_cost,
    )

    assert torch.allclose(central, torch.tensor([[2.0]]), atol=0.25)
    assert float(future_cost[0]) > 0.9


def test_cost_transport_can_use_a_prediction_step_smaller_than_observer_dt() -> None:
    field = CostDisturbanceField(
        query_states=torch.tensor([[0.0]]),
        velocities=torch.tensor([[2.0]]),
        query_weights=torch.tensor([1.0]),
        residuals=torch.tensor([1.0]),
        interpolation_bandwidth=100.0,
        dt=16.0,
    )

    backtraced = field.backtrace_queries(
        torch.tensor([[10.0]]),
        horizon_step=2,
        step_dt=1.0,
    )

    assert torch.allclose(backtraced, torch.tensor([[6.0]]))


def test_sdob_facade_exposes_both_observer_channels() -> None:
    state_observer = StateEffectObserver(
        IntegratorModel(),
        StateEffectConfig(
            disturbance_dim=1,
            num_samples=11,
            disturbance_min=(-0.5,),
            disturbance_max=(0.5,),
        ),
    )
    cost_observer = CostDisturbanceObserver(
        CostDisturbanceConfig(dt=0.1, max_displacement=(0.1,), candidate_steps=3)
    )

    observer = SamplingDisturbanceObserver(
        state_effect=state_observer,
        cost=cost_observer,
    )

    assert observer.state_effect is state_observer
    assert observer.cost is cost_observer

    state_estimate = observer.update_state(
        previous_state=torch.tensor([0.0]),
        applied_action=torch.tensor([0.1]),
        feedback_state=torch.tensor([0.2]),
    )
    queries = torch.linspace(-0.2, 0.2, 5).unsqueeze(-1)
    cost_field = observer.update_cost(
        queries,
        lambda query: query[..., 0].square(),
        lambda query: (query[..., 0] - 0.1).square(),
    )

    assert state_estimate.value.shape == (1,)
    assert cost_field.velocities.shape == queries.shape


def test_packaged_default_configs_can_be_loaded() -> None:
    controller = load_config("controller/mppi.yaml")
    state_observer = load_config("observer/state_effect.yaml")
    cost_observer = load_config("observer/cost.yaml")

    assert controller["target"] == "sdob.controller.MPPIController"
    assert state_observer["target"] == "sdob.observer.StateEffectObserver"
    assert cost_observer["target"] == "sdob.observer.CostDisturbanceObserver"


def test_installation_diagnostic_runs_a_core_smoke_test() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "sdob.diagnostics"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Core smoke test: OK" in result.stdout
