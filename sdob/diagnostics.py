"""Installation and core-runtime diagnostic."""

from __future__ import annotations

import platform

import torch

from .controller import MPPIConfig, MPPIController
from .observer import StateEffectConfig, StateEffectObserver


class _Integrator:
    def step(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        disturbance: torch.Tensor | None = None,
    ) -> torch.Tensor:
        result = state + action
        return result if disturbance is None else result + disturbance


def _cost(
    trajectories: torch.Tensor,
    controls: torch.Tensor,
    context: dict[str, torch.Tensor],
) -> torch.Tensor:
    return (trajectories[:, 1:] - context["target"]).square().sum(dim=(1, 2))


def main() -> None:
    model = _Integrator()
    controller = MPPIController(
        model,
        _cost,
        MPPIConfig(
            horizon=3,
            num_samples=64,
            action_dim=1,
            noise_sigma=(0.5,),
            action_min=(-1.0,),
            action_max=(1.0,),
            seed=0,
        ),
    )
    plan = controller.plan(torch.tensor([0.0]), {"target": torch.tensor([1.0])})
    observer = StateEffectObserver(
        model,
        StateEffectConfig(
            disturbance_dim=1,
            num_samples=51,
            disturbance_min=(-0.5,),
            disturbance_max=(0.5,),
            filter_alpha=1.0,
        ),
    )
    estimate = observer.update(
        previous_state=torch.tensor([0.0]),
        applied_action=torch.tensor([0.1]),
        feedback_state=torch.tensor([0.3]),
    )
    if not torch.isfinite(plan.action).all() or not torch.isfinite(estimate.value).all():
        raise RuntimeError("core smoke test returned a non-finite value")

    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print("Core smoke test: OK")


if __name__ == "__main__":
    main()
