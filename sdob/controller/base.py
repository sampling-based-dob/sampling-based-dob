"""Interfaces used by sampling-based controllers."""

from __future__ import annotations

from typing import Any, Protocol

import torch


class RolloutModel(Protocol):
    """Batched one-step model required by the generic controller."""

    def step(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        disturbance: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the next state for each state-action pair."""


class TrajectoryCost(Protocol):
    """Score a batch of state and control trajectories."""

    def __call__(
        self,
        trajectories: torch.Tensor,
        controls: torch.Tensor,
        context: Any,
    ) -> torch.Tensor:
        """Return one scalar cost per sampled trajectory."""
