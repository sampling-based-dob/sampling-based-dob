"""Container for the two SDOB observation channels."""

from __future__ import annotations

import torch

from .cost import CostDisturbanceField, CostDisturbanceObserver, CostQuery
from .state_effect import StateEffectEstimate, StateEffectObserver


class SamplingDisturbanceObserver:
    """Expose state-effect and cost disturbance observers through one object."""

    def __init__(
        self,
        *,
        state_effect: StateEffectObserver | None = None,
        cost: CostDisturbanceObserver | None = None,
    ) -> None:
        if state_effect is None and cost is None:
            raise ValueError("at least one observer channel must be configured")
        self.state_effect = state_effect
        self.cost = cost

    def reset(self) -> None:
        if self.state_effect is not None:
            self.state_effect.reset()
        if self.cost is not None:
            self.cost.reset()

    def update_state(
        self,
        previous_state: torch.Tensor,
        applied_action: torch.Tensor,
        feedback_state: torch.Tensor,
    ) -> StateEffectEstimate:
        if self.state_effect is None:
            raise RuntimeError("the state-effect channel is not configured")
        return self.state_effect.update(previous_state, applied_action, feedback_state)

    def update_cost(
        self,
        query_states: torch.Tensor,
        previous_cost: CostQuery,
        current_cost: CostQuery,
    ) -> CostDisturbanceField:
        if self.cost is None:
            raise RuntimeError("the cost disturbance channel is not configured")
        return self.cost.update(query_states, previous_cost, current_cost)
