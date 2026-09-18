"""Sampled state-effect disturbance estimation from a completed transition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from sdob.controller import gibbs_weights
from sdob.controller.base import RolloutModel

from .base import BaseObserver

ResidualFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
AdmissibilityCost = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _vector(
    value: float | Sequence[float],
    size: int,
    name: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(value, (int, float)):
        values = (float(value),) * size
    else:
        values = tuple(float(item) for item in value)
        if len(values) != size:
            raise ValueError(f"{name} must contain {size} values")
    return torch.tensor(values, dtype=dtype, device=device)


def score_state_effect_candidates(
    candidates: torch.Tensor,
    predictions: torch.Tensor,
    feedback_state: torch.Tensor,
    *,
    temperature: float,
    prediction_weight: torch.Tensor,
    regularization_weight: torch.Tensor,
    residual_function: ResidualFunction | None = None,
    admissibility_cost: AdmissibilityCost | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Score candidate disturbances and return residuals, losses, and weights."""
    feedback = feedback_state.unsqueeze(0).expand_as(predictions)
    residuals = (
        residual_function(feedback, predictions)
        if residual_function is not None
        else feedback - predictions
    )
    prediction_loss = (residuals * prediction_weight).square().sum(dim=-1)
    regularization = (candidates * regularization_weight).square().sum(dim=-1)
    losses = prediction_loss + regularization
    if admissibility_cost is not None:
        losses = losses + admissibility_cost(candidates, predictions).reshape(-1)
    return residuals, losses, gibbs_weights(losses, temperature)


@dataclass(frozen=True)
class StateEffectConfig:
    disturbance_dim: int
    num_samples: int = 256
    disturbance_min: float | tuple[float, ...] = -1.0
    disturbance_max: float | tuple[float, ...] = 1.0
    prediction_weight: float | tuple[float, ...] = 1.0
    regularization_weight: float | tuple[float, ...] = 0.0
    temperature: float = 0.05
    filter_alpha: float = 0.5
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    seed: int = 0

    def __post_init__(self) -> None:
        if self.disturbance_dim <= 0 or self.num_samples <= 0:
            raise ValueError("disturbance_dim and num_samples must be positive")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if not 0.0 <= self.filter_alpha <= 1.0:
            raise ValueError("filter_alpha must be between zero and one")


@dataclass(frozen=True)
class StateEffectEstimate:
    value: torch.Tensor
    raw_value: torch.Tensor
    candidates: torch.Tensor
    predictions: torch.Tensor
    residuals: torch.Tensor
    losses: torch.Tensor
    weights: torch.Tensor


class StateEffectObserver(BaseObserver):
    """Estimate a disturbance by sampled forward evaluation of one transition."""

    def __init__(
        self,
        model: RolloutModel,
        config: StateEffectConfig,
        *,
        residual_function: ResidualFunction | None = None,
        admissibility_cost: AdmissibilityCost | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = config.dtype
        self.residual_function = residual_function
        self.admissibility_cost = admissibility_cost
        self.generator = torch.Generator(device=self.device).manual_seed(config.seed)
        self.lower = _vector(
            config.disturbance_min,
            config.disturbance_dim,
            "disturbance_min",
            device=self.device,
            dtype=self.dtype,
        )
        self.upper = _vector(
            config.disturbance_max,
            config.disturbance_dim,
            "disturbance_max",
            device=self.device,
            dtype=self.dtype,
        )
        if torch.any(self.lower >= self.upper):
            raise ValueError("disturbance_min must be smaller than disturbance_max")
        self.regularization_weight = _vector(
            config.regularization_weight,
            config.disturbance_dim,
            "regularization_weight",
            device=self.device,
            dtype=self.dtype,
        )
        self.value = torch.zeros(config.disturbance_dim, dtype=self.dtype, device=self.device)

    def reset(self) -> None:
        self.value.zero_()

    @torch.no_grad()
    def update(
        self,
        previous_state: torch.Tensor,
        applied_action: torch.Tensor,
        feedback_state: torch.Tensor,
    ) -> StateEffectEstimate:
        previous_state = torch.as_tensor(previous_state, dtype=self.dtype, device=self.device).reshape(-1)
        applied_action = torch.as_tensor(applied_action, dtype=self.dtype, device=self.device).reshape(-1)
        feedback_state = torch.as_tensor(feedback_state, dtype=self.dtype, device=self.device).reshape(-1)
        candidates = self._sample_candidates()
        states = previous_state.unsqueeze(0).expand(self.config.num_samples, -1)
        actions = applied_action.unsqueeze(0).expand(self.config.num_samples, -1)
        predictions = self.model.step(states, actions, candidates)
        prediction_weight = _vector(
            self.config.prediction_weight,
            feedback_state.numel(),
            "prediction_weight",
            device=self.device,
            dtype=self.dtype,
        )
        residuals, losses, weights = score_state_effect_candidates(
            candidates,
            predictions,
            feedback_state,
            temperature=self.config.temperature,
            prediction_weight=prediction_weight,
            regularization_weight=self.regularization_weight,
            residual_function=self.residual_function,
            admissibility_cost=self.admissibility_cost,
        )
        raw_value = torch.sum(weights[:, None] * candidates, dim=0)
        alpha = float(self.config.filter_alpha)
        self.value = ((1.0 - alpha) * self.value + alpha * raw_value).detach()
        return StateEffectEstimate(
            value=self.value.clone(),
            raw_value=raw_value,
            candidates=candidates,
            predictions=predictions,
            residuals=residuals,
            losses=losses,
            weights=weights,
        )

    def _sample_candidates(self) -> torch.Tensor:
        if self.config.disturbance_dim == 1:
            return torch.linspace(
                float(self.lower[0]),
                float(self.upper[0]),
                self.config.num_samples,
                dtype=self.dtype,
                device=self.device,
            ).unsqueeze(-1)
        unit = torch.rand(
            self.config.num_samples,
            self.config.disturbance_dim,
            generator=self.generator,
            dtype=self.dtype,
            device=self.device,
        )
        candidates = self.lower + unit * (self.upper - self.lower)
        candidates[0] = self.value
        if self.config.num_samples > 1:
            candidates[1].zero_()
        return candidates
