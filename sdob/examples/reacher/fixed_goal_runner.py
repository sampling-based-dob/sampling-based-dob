from __future__ import annotations

import csv
import hashlib
import math
import os
import sys
import time
import traceback
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sdob.controller import effective_sample_size, gibbs_weights

from .methods import feedback_type_for_method
from .config import ExperimentConfig
from .reacher_protocol import FORMAL_SUCCESS_THRESHOLD_RAD, seed_streams


def _synchronize_device(device: str) -> None:
    import torch

    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)


def _release_device_cache(device: str) -> None:
    import gc
    import torch

    gc.collect()
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()


def _prepare_device_runtime(device: str) -> None:
    import importlib
    import torch

    if torch.device(device).type == "cuda":
        importlib.import_module("torch._inductor")


def _prepare_reacher_runtime_imports() -> None:
    import importlib

    for module_name in ("cffi", "pycparser", "glfw", "mujoco", "dm_control.mjcf"):
        importlib.import_module(module_name)


REACHER_FIXED_GOAL_CASES = [
    ("MPPI without disturbance", "WorldModel-MPPI", False),
    ("MPPI", "WorldModel-MPPI", True),
    ("WorldModel+SDOB", "WorldModel+SDOB", True),
    ("Tube-MPPI", "Tube-MPPI", True),
    ("RMPPI", "Robust-MPPI", True),
]



def _sdob_forecast(observer, order: int) -> np.ndarray:
    if order not in (0, 1):
        raise ValueError("SDOB forecast order must be 0 or 1")
    return observer.forecast(include_derivative=(order == 1))




@dataclass(frozen=True)
class FixedGoalOutput:
    comparison_video: Path
    comparison_image: Path
    summary_csv: Path
    trajectories_csv: Path
    diagnostics_csv: Path

    @classmethod
    def from_output_dir(
        cls,
        output_dir: Path,
        diagnostics_name: str = "disturbance_diagnostics.csv",
    ) -> "FixedGoalOutput":
        output_dir = Path(output_dir)
        return cls(
            comparison_video=output_dir / "comparison.mp4",
            comparison_image=output_dir / "comparison.png",
            summary_csv=output_dir / "summary.csv",
            trajectories_csv=output_dir / "trajectories.csv",
            diagnostics_csv=output_dir / diagnostics_name,
        )


class _ImageNetTensorTransform:
    """Small image transform fallback for world-model rollouts."""

    def __init__(self, size: int):
        import torch

        self.size = int(size)
        self.mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)

    def __call__(self, image: Any):
        import torch
        import torch.nn.functional as F

        tensor = torch.as_tensor(image)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 3 and tensor.shape[-1] in (1, 3, 4) and tensor.shape[0] not in (1, 3, 4):
            tensor = tensor.permute(2, 0, 1)
        if tensor.ndim != 3:
            raise ValueError(f"expected image tensor with 3 dims, got shape {tuple(tensor.shape)}")
        if tensor.shape[0] == 4:
            tensor = tensor[:3]
        elif tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        tensor = tensor.to(dtype=torch.float32)
        if tensor.numel() and float(tensor.max()) > 2.0:
            tensor = tensor / 255.0
        mean = self.mean.to(device=tensor.device)
        std = self.std.to(device=tensor.device)
        tensor = (tensor - mean) / std
        if self.size > 0 and tuple(tensor.shape[-2:]) != (self.size, self.size):
            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=(self.size, self.size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return tensor


def _patch_transformers_lightweight_imports() -> None:
    """Avoid optional Transformers branches unused by the ViT world model."""

    import importlib
    import sys
    import types

    try:
        importlib.import_module("transformers.generation")
        importlib.import_module("transformers.integrations.flex_attention")
        importlib.import_module("transformers.loss.loss_utils")
        return
    except ImportError:
        pass

    generation = types.ModuleType("transformers.generation")

    class CompileConfig:
        def __init__(self, *args: Any, **kwargs: Any):
            del args, kwargs

    class GenerationConfig:
        def __init__(self, *args: Any, **kwargs: Any):
            del args, kwargs

    class GenerationMixin:
        pass

    generation.CompileConfig = CompileConfig
    generation.GenerationConfig = GenerationConfig
    generation.GenerationMixin = GenerationMixin
    sys.modules.setdefault("transformers.generation", generation)

    flex_attention = types.ModuleType("transformers.integrations.flex_attention")

    def flex_attention_forward(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise RuntimeError("flex attention is disabled for SDOB world-model evaluation")

    flex_attention.flex_attention_forward = flex_attention_forward
    sys.modules.setdefault("transformers.integrations.flex_attention", flex_attention)

    loss_utils = types.ModuleType("transformers.loss.loss_utils")
    loss_utils.LOSS_MAPPING = {}
    sys.modules.setdefault("transformers.loss.loss_utils", loss_utils)


def patch_transformer_configs_for_checkpoint(module: Any) -> int:
    """Fill compatibility fields missing from older serialized configs.

    The local le-wm Reacher checkpoint can deserialize into a ViTConfig that
    contains private fields such as ``_output_attentions`` but lacks public
    fields accessed by the installed Transformers forward pass. This function
    patches only missing config attributes and returns the number of configs
    touched.
    """

    patched = 0
    modules = module.modules() if hasattr(module, "modules") else [module]
    for submodule in modules:
        config = getattr(submodule, "config", None)
        if config is None:
            continue
        changed = False
        defaults = {
            "output_attentions": getattr(config, "_output_attentions", False),
            "output_hidden_states": getattr(config, "_output_hidden_states", False),
            "torchscript": False,
            "return_dict": getattr(config, "return_dict", True),
        }
        for name, value in defaults.items():
            if not hasattr(config, name):
                config.__dict__[name] = value
                changed = True
        if changed:
            patched += 1
    return patched


class ActionBiasCostModel:
    """World-model cost wrapper that synthesizes an input-equivalent bias."""

    def __init__(
        self,
        base_model: Any,
        action_dim: int,
        synthesis_gain: float = 1.0,
        bound_penalty_weight: float = 2.0,
        action_reg_weight: float = 0.0,
        smoothness_reg_weight: float = 0.0,
        reacher_progress_weight: float = 0.0,
        state_progress_weight: float = 0.0,
        state_goal_cost_weight: float = 0.0,
        state_goal_step_scale: float = 5.0,
        state_goal_cost_steps: int = 1,
        trajectory_cost_weight: float = 0.0,
        trajectory_cost_steps: int = 0,
        trajectory_cost_mode: str = "add",
    ):
        self.base_model = base_model
        self.action_dim = int(action_dim)
        self.synthesis_gain = float(synthesis_gain)
        self.bound_penalty_weight = float(bound_penalty_weight)
        self.action_reg_weight = float(action_reg_weight)
        self.smoothness_reg_weight = float(smoothness_reg_weight)
        self.reacher_progress_weight = float(reacher_progress_weight)
        self.state_progress_weight = float(state_progress_weight)
        self.state_goal_cost_weight = float(state_goal_cost_weight)
        self.state_goal_step_scale = float(state_goal_step_scale)
        self.state_goal_cost_steps = int(state_goal_cost_steps)
        self.trajectory_cost_weight = float(trajectory_cost_weight)
        self.trajectory_cost_steps = int(trajectory_cost_steps)
        self.trajectory_cost_mode = str(trajectory_cost_mode)
        self._bias = np.zeros(self.action_dim, dtype=np.float32)
        self.last_info_keys: tuple[str, ...] = ()
        self.last_reacher_progress_available = False

    def parameters(self):
        return self.base_model.parameters()

    def set_bias(self, bias: np.ndarray) -> None:
        self._bias = np.asarray(bias, dtype=np.float32).reshape(self.action_dim)

    def get_cost(self, info_dict: dict, action_candidates):
        import torch

        self.last_info_keys = tuple(sorted(info_dict.keys()))
        self.last_reacher_progress_available = False
        bias = torch.as_tensor(
            np.tile(
                self.synthesis_gain * self._bias,
                action_candidates.shape[-1] // self.action_dim,
            ),
            dtype=action_candidates.dtype,
            device=action_candidates.device,
        )
        rollout_actions = torch.clamp(action_candidates + bias, -1.0, 1.0)
        costs = self.base_model.get_cost(info_dict, rollout_actions)
        trajectory_cost = self._trajectory_cost(info_dict)
        if trajectory_cost is not None:
            if self.trajectory_cost_mode == "replace":
                costs = self.trajectory_cost_weight * trajectory_cost
            else:
                costs = costs + self.trajectory_cost_weight * trajectory_cost
        if self.bound_penalty_weight <= 0:
            return costs + self._regularization_cost(info_dict, rollout_actions)
        violation = torch.relu(torch.abs(action_candidates) - 1.0)
        penalty = violation.square().flatten(start_dim=2).mean(dim=2)
        return (
            costs
            + self.bound_penalty_weight * penalty
            + self._regularization_cost(info_dict, rollout_actions)
        )

    def _trajectory_cost(self, info_dict: dict):
        if self.trajectory_cost_weight <= 0.0 or self.trajectory_cost_steps <= 0:
            return None
        pred = info_dict.get("predicted_emb")
        goal = info_dict.get("goal_emb")
        if pred is None or goal is None:
            return None
        tail_steps = min(self.trajectory_cost_steps, pred.shape[-2])
        if tail_steps <= 0:
            return None
        goal = goal[..., -1:, :].expand_as(pred)
        tail_error = pred[..., -tail_steps:, :] - goal[..., -tail_steps:, :].detach()
        return tail_error.square().sum(dim=-1).mean(dim=-1)

    def _regularization_cost(self, info_dict: dict, action_candidates):
        import torch

        reg = torch.zeros(
            action_candidates.shape[:2],
            dtype=action_candidates.dtype,
            device=action_candidates.device,
        )
        if (
            self.action_reg_weight <= 0.0
            and self.smoothness_reg_weight <= 0.0
            and self.reacher_progress_weight <= 0.0
            and self.state_progress_weight <= 0.0
            and self.state_goal_cost_weight <= 0.0
        ):
            return reg

        block = max(1, action_candidates.shape[-1] // self.action_dim)
        actions = action_candidates.reshape(
            *action_candidates.shape[:-1],
            block,
            self.action_dim,
        )
        if self.action_reg_weight > 0.0:
            reg = reg + self.action_reg_weight * actions.square().mean(dim=(-1, -2, -3))
        if self.smoothness_reg_weight > 0.0:
            flat = actions.reshape(actions.shape[0], actions.shape[1], -1, self.action_dim)
            if flat.shape[2] > 1:
                diffs = flat[:, :, 1:] - flat[:, :, :-1]
                reg = reg + self.smoothness_reg_weight * diffs.square().mean(dim=(-1, -2))
        if self.reacher_progress_weight > 0.0:
            progress = self._reacher_progress(info_dict, actions)
            if progress is not None:
                reg = reg - self.reacher_progress_weight * progress
        if self.state_progress_weight > 0.0:
            progress = self._state_progress(info_dict, actions)
            if progress is not None:
                reg = reg - self.state_progress_weight * progress
        if self.state_goal_cost_weight > 0.0:
            state_cost = self._state_goal_cost(info_dict, actions)
            if state_cost is not None:
                reg = reg + self.state_goal_cost_weight * state_cost
        return reg

    def _reacher_progress(self, info_dict: dict, actions):
        import torch

        finger = self._info_vector(info_dict.get("finger_pos"), actions, dim=2)
        target = self._info_vector(info_dict.get("target_pos"), actions, dim=2)
        if finger is None or target is None:
            return None
        self.last_reacher_progress_available = True
        direction = target - finger
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        mean_action = actions[..., :2].mean(dim=(2, 3))
        return (mean_action * direction).sum(dim=-1)

    def _state_progress(self, info_dict: dict, actions):
        state = self._info_vector(info_dict.get("state"), actions, dim=self.action_dim)
        target = self._info_vector(info_dict.get("goal_state"), actions, dim=self.action_dim)
        if state is None or target is None:
            return None
        direction = target - state
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        mean_action = actions.mean(dim=(2, 3))
        return (mean_action * direction).sum(dim=-1)

    def _state_goal_cost(self, info_dict: dict, actions):
        state = self._info_vector(info_dict.get("state"), actions, dim=self.action_dim)
        target = self._info_vector(info_dict.get("goal_state"), actions, dim=self.action_dim)
        if state is None or target is None:
            return None
        steps = min(max(1, self.state_goal_cost_steps), actions.shape[2])
        rollout_delta = actions[:, :, :steps].sum(dim=(2, 3)) * self.state_goal_step_scale
        predicted = state + rollout_delta
        return (predicted - target).norm(dim=-1)

    def _info_vector(self, value: Any, actions, dim: int):
        import torch

        if value is None:
            return None
        if not torch.is_tensor(value):
            value = torch.as_tensor(value, dtype=actions.dtype, device=actions.device)
        else:
            value = value.to(dtype=actions.dtype, device=actions.device)
        while value.ndim > 3:
            value = value[..., 0, :]
        if value.ndim == 2:
            value = value[:, None, :]
        if value.shape[-1] != dim:
            return None
        if value.shape[1] == 1 and actions.shape[1] != 1:
            value = value.expand(-1, actions.shape[1], -1)
        return value








class StateEffectActionObserver:
    """Small input-equivalent observer based only on feedback state changes."""

    def __init__(
        self,
        action_dim: int,
        gain: float = 0.12,
        alpha: float = 0.45,
        max_abs: float = 0.35,
        derivative_gain: float = 0.25,
    ):
        self.action_dim = int(action_dim)
        self.gain = float(gain)
        self.alpha = float(alpha)
        self.max_abs = float(max_abs)
        self.derivative_gain = float(derivative_gain)
        self.d_hat = np.zeros(self.action_dim, dtype=np.float64)
        self.d_dot_hat = np.zeros(self.action_dim, dtype=np.float64)
        self._prev_qvel: np.ndarray | None = None
        self._prev_d_hat = self.d_hat.copy()
        self.last_innovation = self.d_hat.copy()

    def reset(self, qvel: np.ndarray) -> None:
        self.d_hat[:] = 0.0
        self.d_dot_hat[:] = 0.0
        self._prev_d_hat = self.d_hat.copy()
        self._prev_qvel = np.asarray(qvel, dtype=np.float64).copy()
        self.last_innovation = self.d_hat.copy()

    def update(self, action: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        qvel = np.asarray(qvel, dtype=np.float64).reshape(self.action_dim)
        action = np.asarray(action, dtype=np.float64).reshape(self.action_dim)
        if self._prev_qvel is None:
            self._prev_qvel = qvel.copy()
            return self.d_hat

        observed_effect = (qvel - self._prev_qvel) / max(self.gain, 1e-6)
        innovation = np.clip(observed_effect - action, -self.max_abs, self.max_abs)
        self.last_innovation = innovation.copy()
        new_hat = self.alpha * self.d_hat + (1.0 - self.alpha) * innovation
        self.d_dot_hat = new_hat - self._prev_d_hat
        self._prev_d_hat = self.d_hat.copy()
        self.d_hat = np.clip(new_hat, -self.max_abs, self.max_abs)
        self._prev_qvel = qvel.copy()
        return self.d_hat

    def update_innovation(self, innovation: np.ndarray) -> np.ndarray:
        innovation = np.asarray(innovation, dtype=np.float64).reshape(self.action_dim)
        innovation = np.clip(innovation, -self.max_abs, self.max_abs)
        self.last_innovation = innovation.copy()
        new_hat = self.alpha * self.d_hat + (1.0 - self.alpha) * innovation
        self.d_dot_hat = new_hat - self._prev_d_hat
        self._prev_d_hat = self.d_hat.copy()
        self.d_hat = np.clip(new_hat, -self.max_abs, self.max_abs)
        return self.d_hat

    def forecast(self, include_derivative: bool) -> np.ndarray:
        if not include_derivative:
            return self.d_hat.copy()
        return np.clip(
            self.d_hat + self.derivative_gain * self.d_dot_hat,
            -self.max_abs,
            self.max_abs,
        )


class InnovationActionObserver:
    """Simple observers over latent LS action-equivalent innovations."""

    def __init__(
        self,
        action_dim: int,
        mode: str,
        alpha: float = 0.4,
        window_size: int = 5,
        max_norm: float = 0.5,
        derivative_gain: float = 0.0,
    ):
        if mode not in {"direct", "ema", "window"}:
            raise ValueError(f"unknown innovation observer mode: {mode}")
        self.action_dim = int(action_dim)
        self.mode = mode
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.window_size = max(1, int(window_size))
        self.max_norm = float(max_norm)
        self.derivative_gain = float(derivative_gain)
        self.d_hat = np.zeros(self.action_dim, dtype=np.float64)
        self.d_dot_hat = np.zeros(self.action_dim, dtype=np.float64)
        self.last_innovation = self.d_hat.copy()
        self.last_raw_mean = self.d_hat.copy()
        self.last_ess = ""
        self._window: deque[np.ndarray] = deque(maxlen=self.window_size)

    def reset(self, _: np.ndarray | None = None) -> None:
        self.d_hat[:] = 0.0
        self.d_dot_hat[:] = 0.0
        self.last_innovation = self.d_hat.copy()
        self.last_raw_mean = self.d_hat.copy()
        self._window.clear()

    def update_innovation(self, innovation: np.ndarray) -> np.ndarray:
        innovation = np.asarray(innovation, dtype=np.float64).reshape(self.action_dim)
        innovation = self._clip_norm(innovation[None, :])[0]
        self.last_innovation = innovation.copy()
        old_hat = self.d_hat.copy()
        if self.mode == "direct":
            new_hat = innovation
        elif self.mode == "ema":
            new_hat = (1.0 - self.alpha) * self.d_hat + self.alpha * innovation
        else:
            self._window.append(innovation.copy())
            new_hat = np.mean(np.asarray(list(self._window), dtype=np.float64), axis=0)
        new_hat = self._clip_norm(new_hat[None, :])[0]
        self.last_raw_mean = new_hat.copy()
        self.d_dot_hat = new_hat - old_hat
        self.d_hat = new_hat
        return self.d_hat

    def forecast(self, include_derivative: bool) -> np.ndarray:
        if not include_derivative:
            return self.d_hat.copy()
        return self._clip_norm((self.d_hat + self.derivative_gain * self.d_dot_hat)[None, :])[0]

    def _clip_norm(self, values: np.ndarray) -> np.ndarray:
        if self.max_norm <= 0.0:
            return values
        norms = np.linalg.norm(values, axis=-1, keepdims=True)
        scale = np.minimum(1.0, self.max_norm / np.maximum(norms, 1e-12))
        return values * scale


class ParticleMeanActionObserver:
    """Particle belief over an input-equivalent action disturbance."""

    def __init__(
        self,
        action_dim: int,
        num_particles: int = 256,
        particle_std: float = 0.35,
        temperature: float = 0.05,
        process_noise: float = 0.002,
        prior_weight: float = 0.5,
        magnitude_prior_weight: float = 0.02,
        resample_ess_fraction: float = 0.3,
        filter_alpha: float = 0.1,
        max_norm: float = 0.25,
        seed: int = 0,
    ):
        self.action_dim = int(action_dim)
        self.num_particles = int(num_particles)
        self.particle_std = float(particle_std)
        self.temperature = float(temperature)
        self.process_noise = float(process_noise)
        self.prior_weight = float(prior_weight)
        self.magnitude_prior_weight = float(magnitude_prior_weight)
        self.resample_ess_fraction = float(resample_ess_fraction)
        self.filter_alpha = float(np.clip(filter_alpha, 0.0, 1.0))
        self.max_norm = float(max_norm)
        self.rng = np.random.default_rng(seed)
        self.d_hat = np.zeros(self.action_dim, dtype=np.float64)
        self.d_dot_hat = np.zeros(self.action_dim, dtype=np.float64)
        self._prev_d_hat = self.d_hat.copy()
        self.particles = np.zeros((self.num_particles, self.action_dim), dtype=np.float64)
        self.weights = np.full(self.num_particles, 1.0 / self.num_particles, dtype=np.float64)
        self.last_ess = float(self.num_particles)
        self.last_raw_mean = self.d_hat.copy()
        self.last_innovation = self.d_hat.copy()

    def reset(self, _: np.ndarray | None = None) -> None:
        self.d_hat[:] = 0.0
        self.d_dot_hat[:] = 0.0
        self._prev_d_hat = self.d_hat.copy()
        self.particles = self.rng.normal(
            loc=0.0,
            scale=max(self.particle_std, 1e-8),
            size=(self.num_particles, self.action_dim),
        )
        self.particles = self._clip_norm(self.particles)
        self.weights.fill(1.0 / self.num_particles)
        self.last_ess = float(self.num_particles)
        self.last_raw_mean = self.d_hat.copy()
        self.last_innovation = self.d_hat.copy()

    def update_innovation(self, innovation: np.ndarray) -> np.ndarray:
        innovation = np.asarray(innovation, dtype=np.float64).reshape(self.action_dim)
        innovation = self._clip_norm(innovation[None, :])[0]
        self.last_innovation = innovation.copy()
        proposal_std = max(self.particle_std, self.process_noise, 1e-8)
        self.particles = self.d_hat[None, :] + self.rng.normal(
            loc=0.0,
            scale=proposal_std,
            size=self.particles.shape,
        )
        self.particles = self._clip_norm(self.particles)
        self.particles[0] = innovation
        if self.num_particles > 1:
            self.particles[1] = self.d_hat.copy()

        measurement_loss = ((self.particles - innovation[None, :]) ** 2).sum(axis=1)
        prior_loss = ((self.particles - self.d_hat[None, :]) ** 2).sum(axis=1)
        magnitude_loss = (self.particles**2).sum(axis=1)
        loss = (
            measurement_loss
            + self.prior_weight * prior_loss
            + self.magnitude_prior_weight * magnitude_loss
        )
        scaled = loss - np.min(loss)
        logits = -scaled / max(self.temperature, 1e-8)
        logits = logits - np.max(logits)
        weights = np.exp(logits)
        weights_sum = float(weights.sum())
        if weights_sum <= 0.0 or not np.isfinite(weights_sum):
            weights = np.full(self.num_particles, 1.0 / self.num_particles, dtype=np.float64)
        else:
            weights = weights / weights_sum
        self.weights = weights
        self.last_ess = float(1.0 / np.maximum(np.sum(weights**2), 1e-12))
        self.last_raw_mean = weights @ self.particles

        new_hat = (1.0 - self.filter_alpha) * self.d_hat + self.filter_alpha * self.last_raw_mean
        new_hat = self._clip_norm(new_hat[None, :])[0]
        self.d_dot_hat = new_hat - self.d_hat
        self._prev_d_hat = self.d_hat.copy()
        self.d_hat = new_hat

        if self.last_ess < self.resample_ess_fraction * self.num_particles:
            indices = self.rng.choice(self.num_particles, size=self.num_particles, p=self.weights)
            self.particles = self.particles[indices]
            self.weights.fill(1.0 / self.num_particles)
        return self.d_hat

    def forecast(self, include_derivative: bool) -> np.ndarray:
        if not include_derivative:
            return self.d_hat.copy()
        forecast = self.d_hat + self.d_dot_hat
        return self._clip_norm(forecast[None, :])[0]

    def _clip_norm(self, values: np.ndarray) -> np.ndarray:
        if self.max_norm <= 0.0:
            return values
        norms = np.linalg.norm(values, axis=-1, keepdims=True)
        scale = np.minimum(1.0, self.max_norm / np.maximum(norms, 1e-12))
        return values * scale


class DisturbanceMPPIObserver:
    """Direct MPPI-style observer over input-equivalent disturbances.

    Unlike ``ParticleMeanActionObserver``, the particle likelihood is evaluated
    by forward world-model rollouts, not by distance to an LS innovation.
    """

    def __init__(
        self,
        action_dim: int,
        num_particles: int = 256,
        particle_std: float = 0.15,
        temperature: float = 0.01,
        process_noise: float = 0.002,
        prior_weight: float = 0.05,
        magnitude_prior_weight: float = 0.0,
        resample_ess_fraction: float = 0.3,
        filter_alpha: float = 0.4,
        max_norm: float = 0.5,
        seed: int = 0,
    ):
        self.action_dim = int(action_dim)
        self.num_particles = int(num_particles)
        self.particle_std = float(particle_std)
        self.temperature = float(temperature)
        self.process_noise = float(process_noise)
        self.prior_weight = float(prior_weight)
        self.magnitude_prior_weight = float(magnitude_prior_weight)
        self.resample_ess_fraction = float(resample_ess_fraction)
        self.filter_alpha = float(np.clip(filter_alpha, 0.0, 1.0))
        self.max_norm = float(max_norm)
        self.rng = np.random.default_rng(seed)
        self.d_hat = np.zeros(self.action_dim, dtype=np.float64)
        self.d_dot_hat = np.zeros(self.action_dim, dtype=np.float64)
        self._prev_d_hat = self.d_hat.copy()
        self.particles = np.zeros((self.num_particles, self.action_dim), dtype=np.float64)
        self.weights = np.full(self.num_particles, 1.0 / self.num_particles, dtype=np.float64)
        self.last_ess = float(self.num_particles)
        self.last_raw_mean = self.d_hat.copy()
        self.last_innovation = self.d_hat.copy()
        self.last_forward_loss = math.nan

    def reset(self, _: np.ndarray | None = None) -> None:
        self.d_hat[:] = 0.0
        self.d_dot_hat[:] = 0.0
        self._prev_d_hat = self.d_hat.copy()
        self.particles = self.rng.normal(
            loc=0.0,
            scale=max(self.particle_std, 1e-8),
            size=(self.num_particles, self.action_dim),
        )
        self.particles = self._clip_norm(self.particles)
        self.weights.fill(1.0 / self.num_particles)
        self.last_ess = float(self.num_particles)
        self.last_raw_mean = self.d_hat.copy()
        self.last_innovation = self.d_hat.copy()
        self.last_forward_loss = math.nan

    def update_from_rollout(
        self,
        rollout_fn: Any,
        action_block_plan: np.ndarray,
        feedback_latent: np.ndarray,
        proposal_center: np.ndarray | None = None,
        seed_disturbances: tuple[np.ndarray, ...] = (),
    ) -> np.ndarray:
        nominal = np.asarray(action_block_plan, dtype=np.float64)
        if nominal.ndim != 2 or nominal.shape[1] != self.action_dim:
            raise ValueError("action_block_plan must have shape (action_block, action_dim)")
        feedback = np.asarray(feedback_latent, dtype=np.float64).reshape(-1)

        center = self.d_hat if proposal_center is None else np.asarray(proposal_center, dtype=np.float64)
        center = self._clip_norm(center.reshape(1, self.action_dim))[0]
        proposal_std = max(self.particle_std, self.process_noise, 1e-8)
        self.particles = center[None, :] + self.rng.normal(
            loc=0.0,
            scale=proposal_std,
            size=self.particles.shape,
        )
        self.particles = self._clip_norm(self.particles)
        anchors = [self.d_hat.copy(), np.zeros(self.action_dim, dtype=np.float64)]
        if proposal_center is not None:
            anchors.append(center.copy())
        anchors.extend(seed_disturbances)
        for idx, anchor in enumerate(anchors[: self.num_particles]):
            self.particles[idx] = self._clip_norm(np.asarray(anchor, dtype=np.float64).reshape(1, self.action_dim))[0]

        candidate_blocks = np.clip(nominal[None, :, :] + self.particles[:, None, :], -1.0, 1.0)
        embeddings = np.asarray(
            rollout_fn(candidate_blocks.reshape(self.num_particles, -1)),
            dtype=np.float64,
        )
        if embeddings.ndim != 2 or embeddings.shape[0] != self.num_particles:
            raise ValueError("rollout_fn must return shape (num_particles, latent_dim)")

        residual = embeddings - feedback[None, :]
        measurement_loss = np.mean(residual**2, axis=1)
        prior_loss = ((self.particles - self.d_hat[None, :]) ** 2).sum(axis=1)
        magnitude_loss = (self.particles**2).sum(axis=1)
        loss = (
            measurement_loss
            + self.prior_weight * prior_loss
            + self.magnitude_prior_weight * magnitude_loss
        )
        best = int(np.argmin(loss))
        self.last_innovation = self.particles[best].copy()
        self.last_forward_loss = float(measurement_loss[best])

        scaled = loss - np.min(loss)
        logits = -scaled / max(self.temperature, 1e-8)
        logits = logits - np.max(logits)
        weights = np.exp(logits)
        weights_sum = float(weights.sum())
        if weights_sum <= 0.0 or not np.isfinite(weights_sum):
            weights = np.full(self.num_particles, 1.0 / self.num_particles, dtype=np.float64)
        else:
            weights = weights / weights_sum
        self.weights = weights
        self.last_ess = float(1.0 / np.maximum(np.sum(weights**2), 1e-12))
        self.last_raw_mean = weights @ self.particles

        new_hat = (1.0 - self.filter_alpha) * self.d_hat + self.filter_alpha * self.last_raw_mean
        new_hat = self._clip_norm(new_hat[None, :])[0]
        self.d_dot_hat = new_hat - self.d_hat
        self._prev_d_hat = self.d_hat.copy()
        self.d_hat = new_hat

        if self.last_ess < self.resample_ess_fraction * self.num_particles:
            indices = self.rng.choice(self.num_particles, size=self.num_particles, p=self.weights)
            self.particles = self.particles[indices]
            self.weights.fill(1.0 / self.num_particles)
        return self.d_hat

    def update_innovation(self, innovation: np.ndarray) -> np.ndarray:
        self.last_innovation = self._clip_norm(
            np.asarray(innovation, dtype=np.float64).reshape(1, self.action_dim)
        )[0]
        return self.d_hat.copy()

    def forecast(self, include_derivative: bool) -> np.ndarray:
        if not include_derivative:
            return self.d_hat.copy()
        forecast = self.d_hat + self.d_dot_hat
        return self._clip_norm(forecast[None, :])[0]

    def _clip_norm(self, values: np.ndarray) -> np.ndarray:
        if self.max_norm <= 0.0:
            return values
        norms = np.linalg.norm(values, axis=-1, keepdims=True)
        scale = np.minimum(1.0, self.max_norm / np.maximum(norms, 1e-12))
        return values * scale


class StabilizedMPPISolver:
    """Local MPPI variant matching the earlier stable Reacher experiments."""

    def __init__(
        self,
        model: Any,
        config: ExperimentConfig,
        batch_size: int = 1,
        device: str = "cpu",
        seed: int = 1234,
    ):
        import torch

        self.model = model
        self.config = config
        self.batch_size = int(batch_size)
        self.num_samples = int(config.num_samples)
        self.topk = int(config.topk)
        self.var_scale = float(config.var_scale)
        self.n_steps = int(config.cem_steps)
        self.temperature = float(config.mppi_temperature)
        self.device = device
        self.torch_gen = torch.Generator(device=device).manual_seed(seed)
        self.last_eta: float | None = None
        try:
            self._dtype = next(model.parameters()).dtype
        except (AttributeError, StopIteration):
            self._dtype = torch.float32

    def configure(self, *, action_space: Any, n_envs: int, config: Any) -> None:
        self._action_space = action_space
        self._n_envs = int(n_envs)
        self._config = config
        self._action_dim = int(np.prod(action_space.shape[1:]))
        self._configured = True

    @property
    def n_envs(self) -> int:
        return self._n_envs

    @property
    def action_dim(self) -> int:
        return self._action_dim * self._config.action_block

    @property
    def horizon(self) -> int:
        return self._config.horizon

    @property
    def dtype(self):
        return self._dtype

    def __call__(self, *args: Any, **kwargs: Any) -> dict:
        return self.solve(*args, **kwargs)

    def init_action_distrib(
        self,
        n_envs: int,
        actions: Any | None = None,
    ):
        import torch

        var = self.var_scale * torch.ones(
            [n_envs, self.horizon, self.action_dim],
            dtype=self.dtype,
        )
        mean = (
            torch.zeros([n_envs, 0, self.action_dim], dtype=self.dtype)
            if actions is None
            else actions
        )
        remaining = self.horizon - mean.shape[1]
        if remaining > 0:
            if mean.shape[1] > 0:
                new_mean = mean[:, -1:, :].expand(n_envs, remaining, self.action_dim).clone()
            else:
                new_mean = torch.zeros(
                    [n_envs, remaining, self.action_dim],
                    dtype=self.dtype,
                    device=mean.device,
                )
            mean = torch.cat([mean, new_mean], dim=1)
        return mean, var

    def solve(self, info_dict: dict, init_action: Any | None = None) -> dict:
        import torch

        outputs = {"costs": [], "mean": [], "var": []}
        total_envs = len(next(iter(info_dict.values())))
        mean, var = self.init_action_distrib(total_envs, init_action)
        mean = mean.to(device=self.device, dtype=self.dtype)
        var = var.to(device=self.device, dtype=self.dtype)
        low, high = self._action_bounds(mean.device)

        for start_idx in range(0, total_envs, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total_envs)
            current_bs = end_idx - start_idx
            batch_mean = mean[start_idx:end_idx]
            batch_var = var[start_idx:end_idx]
            expanded_infos = self._expand_infos(info_dict, start_idx, end_idx, current_bs)
            final_batch_cost = None

            for step in range(max(1, self.n_steps)):
                noise = torch.randn(
                    current_bs,
                    self.num_samples,
                    self.horizon,
                    self.action_dim,
                    generator=self.torch_gen,
                    device=self.device,
                    dtype=self.dtype,
                )
                decay = self.config.mppi_noise_decay ** step
                sigma = torch.clamp(
                    batch_var * decay,
                    min=float(self.config.mppi_min_variance),
                )
                candidates = batch_mean.unsqueeze(1) + noise * sigma.unsqueeze(1)
                candidates[:, 0] = batch_mean
                if self.config.mppi_action_clamp:
                    candidates = torch.max(torch.min(candidates, high), low)

                costs = self.model.get_cost(expanded_infos, candidates)
                relevant_candidates, relevant_costs = self._select_relevant(candidates, costs)
                weights = self._weights(relevant_costs)
                candidate_mean = (
                    weights.unsqueeze(-1).unsqueeze(-1) * relevant_candidates
                ).sum(dim=1)
                rho = float(np.clip(self.config.mppi_mean_update_rate, 0.0, 1.0))
                batch_mean = (1.0 - rho) * batch_mean + rho * candidate_mean
                if self.config.mppi_action_clamp:
                    batch_mean = torch.max(torch.min(batch_mean, high), low)
                self._adapt_temperature(weights)
                final_batch_cost = relevant_costs.mean(dim=1).cpu().tolist()

            mean[start_idx:end_idx] = batch_mean
            outputs["costs"].extend(final_batch_cost or [])

        outputs["actions"] = mean.detach().cpu()
        outputs["mean"] = [mean.detach().cpu()]
        outputs["var"] = [var.detach().cpu()]
        return outputs

    def _expand_infos(
        self,
        info_dict: dict,
        start_idx: int,
        end_idx: int,
        current_bs: int,
    ) -> dict:
        import torch

        expanded_infos = {}
        for key, value in info_dict.items():
            value_batch = value[start_idx:end_idx]
            if torch.is_tensor(value):
                target_dtype = self.dtype if value_batch.is_floating_point() else None
                value_batch = (
                    value_batch.to(device=self.device, dtype=target_dtype)
                    .unsqueeze(1)
                    .expand(current_bs, self.num_samples, *value_batch.shape[1:])
                )
            elif isinstance(value, np.ndarray):
                value_batch = np.repeat(value_batch[:, None, ...], self.num_samples, axis=1)
            expanded_infos[key] = value_batch
        return expanded_infos

    def _select_relevant(self, candidates: Any, costs: Any) -> tuple[Any, Any]:
        import torch

        if self.topk > 0 and self.topk < self.num_samples:
            topk_vals, topk_inds = torch.topk(costs, k=self.topk, dim=1, largest=False)
            batch_indices = (
                torch.arange(costs.shape[0], device=costs.device)
                .unsqueeze(1)
                .expand(-1, self.topk)
            )
            return candidates[batch_indices, topk_inds], topk_vals
        return candidates, costs

    def _weights(self, costs: Any):
        import torch

        scaled = costs - costs.min(dim=1, keepdim=True)[0]
        if self.config.mppi_cost_normalize:
            scale = scaled.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
            scaled = scaled / scale
        return gibbs_weights(scaled, max(self.temperature, 1e-8), dim=1)

    def _adapt_temperature(self, weights: Any) -> None:
        if not self.config.adaptive_temperature:
            return
        eta = effective_sample_size(weights, dim=1) / weights.shape[1]
        self.last_eta = float(eta.mean().detach().cpu())
        if self.last_eta > self.config.eta_max:
            self.temperature *= self.config.mppi_temperature_decrease
        elif self.last_eta < self.config.eta_min:
            self.temperature *= self.config.mppi_temperature_increase
        self.temperature = float(
            min(
                max(self.temperature, self.config.mppi_min_temperature),
                self.config.mppi_max_temperature,
            )
        )

    def _action_bounds(self, device: Any) -> tuple[Any, Any]:
        import torch

        low = np.asarray(self._action_space.low)
        high = np.asarray(self._action_space.high)
        if low.ndim > 1 and low.shape[0] == self._n_envs:
            low = low[0]
            high = high[0]
        low = np.tile(low.reshape(-1), self._config.action_block)
        high = np.tile(high.reshape(-1), self._config.action_block)
        low_t = torch.as_tensor(low, dtype=self.dtype, device=device).view(1, 1, -1)
        high_t = torch.as_tensor(high, dtype=self.dtype, device=device).view(1, 1, -1)
        return low_t, high_t


class FixedGoalRunner:
    """Dataset-free Reacher evaluation with rendered goals and a frozen LeWM."""

    def __init__(
        self,
        config: ExperimentConfig,
        reacher_cases: Iterable[tuple[str, str, bool]] | None = None,
        *,
        fail_fast_on_case_error: bool = False,
    ):
        self.config = config
        self.reacher_cases = tuple(
            REACHER_FIXED_GOAL_CASES if reacher_cases is None else reacher_cases
        )
        self.fail_fast_on_case_error = bool(fail_fast_on_case_error)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


    def run(self) -> FixedGoalOutput:
        if self.config.task != "reacher":
            raise ValueError("This release supports only Reacher")
        from .runtime import ensure_runtime_paths
        ensure_runtime_paths(self.config)
        return self._run_reacher()

    def _run_reacher(self) -> FixedGoalOutput:
        _prepare_reacher_runtime_imports()

        import torch

        _prepare_device_runtime(self.config.device)
        output = FixedGoalOutput.from_output_dir(self.output_dir)

        summary_rows: list[dict[str, Any]] = []
        trajectory_rows: list[dict[str, Any]] = []
        diagnostic_rows: list[dict[str, Any]] = []
        videos: list[tuple[str, list[np.ndarray]]] = []
        reference_frame: np.ndarray | None = None

        for case_index, (case_name, method, disturbed) in enumerate(self.reacher_cases):
            try:
                with torch.inference_mode():
                    case = self._run_reacher_case(case_index, case_name, method, disturbed)
            finally:
                _release_device_cache(self.config.device)
            summary = case["summary"]
            if self.fail_fast_on_case_error and summary.get("status") != "ok":
                detail = f": {summary['error']}" if summary.get("error") else ""
                raise RuntimeError(
                    f"Reacher case {case_name!r} returned status "
                    f"{summary.get('status')!r}{detail}"
                )
            summary_rows.append(summary)
            trajectory_rows.extend(case["trajectory"])
            diagnostic_rows.extend(case["diagnostics"])
            videos.append((case_name, case["frames"]))
            if reference_frame is None:
                reference_frame = case.get("reference_frame")

        if reference_frame is not None:
            videos.append(("Reference", [reference_frame]))

        _write_csv(output.summary_csv, summary_rows)
        _write_csv(output.trajectories_csv, trajectory_rows)
        _write_csv(output.diagnostics_csv, diagnostic_rows)
        if self.config.write_media:
            _write_comparison_video(output.comparison_video, videos, fps=15)
            _write_comparison_image(output.comparison_image, videos)
        return output

    def _run_reacher_case(
        self,
        _case_index: int,
        case_name: str,
        method: str,
        disturbed: bool,
    ) -> dict[str, Any]:
        import stable_worldmodel as swm
        import torch

        _patch_transformers_lightweight_imports()
        effective_task_seed = (
            self.config.seed if self.config.task_seed is None else self.config.task_seed
        )
        streams = seed_streams(effective_task_seed)
        feedback_rng = None
        if method == "Tube-MPPI":
            feedback_rng = np.random.default_rng(streams.tube_feedback_seed)
        elif method == "Robust-MPPI":
            feedback_rng = np.random.default_rng(streams.robust_feedback_seed)
        model = swm.policy.AutoCostModel(
            self.config.world_model_name,
            cache_dir=os.environ.get(
                "STABLEWM_HOME", str(Path.home() / ".stable_worldmodel")
            ),
        )
        patch_transformer_configs_for_checkpoint(model)
        model = model.to(self.config.device).eval()

        cost_model: Any = ActionBiasCostModel(
            model,
            action_dim=2,
            synthesis_gain=0.0,
            bound_penalty_weight=2.0,
            action_reg_weight=self.config.action_reg_weight,
            smoothness_reg_weight=self.config.smoothness_reg_weight,
            reacher_progress_weight=self.config.reacher_progress_weight,
            state_progress_weight=self.config.state_progress_weight,
            state_goal_cost_weight=self.config.state_goal_cost_weight,
            state_goal_step_scale=self.config.state_goal_step_scale,
            state_goal_cost_steps=self.config.state_goal_cost_steps,
            trajectory_cost_weight=self.config.trajectory_cost_weight,
            trajectory_cost_steps=self.config.trajectory_cost_steps,
            trajectory_cost_mode=self.config.trajectory_cost_mode,
        )
        observer: (
            StateEffectActionObserver
            | InnovationActionObserver
            | ParticleMeanActionObserver
            | DisturbanceMPPIObserver
            | None
        ) = None
        if method in {"WorldModel+DOB", "WorldModel+SDOB"}:
            if method == "WorldModel+SDOB":
                cost_model.synthesis_gain = self.config.sdob_action_prior_rollout_gain
                if self.config.dob_observer_mode in {"particle_mean", "particle_sharp"}:
                    particle_std = self.config.dob_action_prior_particle_std
                    temperature = self.config.dob_action_prior_particle_temperature
                    prior_weight = self.config.dob_action_prior_particle_prior_weight
                    if self.config.dob_observer_mode == "particle_sharp":
                        particle_std = min(particle_std, 0.15)
                        temperature = min(temperature, 0.01)
                        prior_weight = min(prior_weight, 0.05)
                    observer = ParticleMeanActionObserver(
                        action_dim=2,
                        num_particles=self.config.dob_action_prior_particle_samples,
                        particle_std=particle_std,
                        temperature=temperature,
                        process_noise=self.config.dob_action_prior_particle_process_noise,
                        prior_weight=prior_weight,
                        magnitude_prior_weight=(
                            self.config.dob_action_prior_particle_magnitude_prior_weight
                        ),
                        resample_ess_fraction=(
                            self.config.dob_action_prior_particle_resample_ess_fraction
                        ),
                        filter_alpha=self.config.dob_action_prior_filter_alpha,
                        max_norm=self.config.dob_action_prior_max_norm,
                        seed=streams.sdob_observer_seed,
                    )
                elif self.config.dob_observer_mode == "innovation_direct":
                    observer = InnovationActionObserver(action_dim=2, mode="direct", max_norm=self.config.dob_action_prior_max_norm)
                elif self.config.dob_observer_mode == "innovation_ema":
                    observer = InnovationActionObserver(
                        action_dim=2,
                        mode="ema",
                        alpha=self.config.dob_action_prior_filter_alpha,
                        max_norm=self.config.dob_action_prior_max_norm,
                    )
                elif self.config.dob_observer_mode == "window_mean":
                    observer = InnovationActionObserver(
                        action_dim=2,
                        mode="window",
                        window_size=self.config.dob_action_prior_window_size,
                        max_norm=self.config.dob_action_prior_max_norm,
                    )
                elif self.config.dob_observer_mode == "disturbance_mppi":
                    observer = DisturbanceMPPIObserver(
                        action_dim=2,
                        num_particles=self.config.dob_action_prior_particle_samples,
                        particle_std=self.config.dob_action_prior_particle_std,
                        temperature=self.config.dob_action_prior_particle_temperature,
                        process_noise=self.config.dob_action_prior_particle_process_noise,
                        prior_weight=self.config.dob_action_prior_particle_prior_weight,
                        magnitude_prior_weight=(
                            self.config.dob_action_prior_particle_magnitude_prior_weight
                        ),
                        resample_ess_fraction=(
                            self.config.dob_action_prior_particle_resample_ess_fraction
                        ),
                        filter_alpha=self.config.dob_action_prior_filter_alpha,
                        max_norm=self.config.dob_action_prior_max_norm,
                        seed=streams.sdob_observer_seed,
                    )
                else:
                    raise ValueError(f"unsupported Reacher SDOB observer mode: {self.config.dob_observer_mode}")
            else:
                cost_model.synthesis_gain = 0.55
                observer = StateEffectActionObserver(action_dim=2)
        elif method in {"Tube-MPPI", "Robust-MPPI"}:
            observer_seed = (
                streams.tube_observer_seed
                if method == "Tube-MPPI"
                else streams.robust_observer_seed
            )
            observer = DisturbanceMPPIObserver(
                action_dim=2,
                num_particles=self.config.dob_action_prior_particle_samples,
                particle_std=self.config.dob_action_prior_particle_std,
                temperature=self.config.dob_action_prior_particle_temperature,
                process_noise=self.config.dob_action_prior_particle_process_noise,
                prior_weight=self.config.dob_action_prior_particle_prior_weight,
                magnitude_prior_weight=self.config.dob_action_prior_particle_magnitude_prior_weight,
                resample_ess_fraction=self.config.dob_action_prior_particle_resample_ess_fraction,
                filter_alpha=self.config.dob_action_prior_filter_alpha,
                max_norm=self.config.dob_action_prior_max_norm,
                seed=observer_seed,
            )

        plan_config = swm.PlanConfig(
            horizon=self.config.horizon,
            receding_horizon=self.config.receding_horizon,
            action_block=self.config.action_block,
        )
        solver = StabilizedMPPISolver(
            model=cost_model,
            config=self.config,
            batch_size=1,
            device=self.config.device,
            seed=streams.planner_seed,
        )
        transform = _ImageNetTensorTransform(self.config.img_size)
        policy = swm.policy.WorldModelPolicy(
            solver=solver,
            config=plan_config,
            transform={"pixels": transform, "goal": transform},
        )
        world = swm.World(
            self.config.env_name,
            num_envs=1,
            image_shape=(self.config.img_size, self.config.img_size),
            max_episode_steps=self.config.steps + 5,
            task="qpos_match",
        )

        frames: list[np.ndarray] = []
        trajectory_rows: list[dict[str, Any]] = []
        diagnostic_rows: list[dict[str, Any]] = []
        errors: list[float] = []
        planning_times_ms: list[float] = []
        control_cost = 0.0
        runtime_error: str | None = None
        reference_frame: np.ndarray | None = None
        terminated_once = False
        initial_qpos = np.full(2, np.nan, dtype=np.float64)
        target_qpos_metadata = np.full(2, np.nan, dtype=np.float64)
        goal_image_sha256 = ""
        start_time = time.perf_counter()

        try:
            world.set_policy(policy)
            world.reset(seed=streams.task_seed)
            base = world.envs.envs[0].unwrapped
            qpos0 = base.env.physics.data.qpos.copy()
            qvel0 = base.env.physics.data.qvel.copy()
            initial_qpos[: min(2, qpos0.size)] = qpos0[:2]
            _disable_dmcontrol_early_termination(base)
            target_delta = np.asarray(self.config.target_delta, dtype=np.float64).reshape(-1)
            if target_delta.size < qpos0.size:
                target_delta = np.pad(target_delta, (0, qpos0.size - target_delta.size))
            target_qpos = qpos0 + target_delta[: qpos0.size]
            target_qpos = np.clip(target_qpos, -math.pi, math.pi)
            target_qpos_metadata[: min(2, target_qpos.size)] = target_qpos[:2]
            base.set_target_qpos(target_qpos)

            base.set_state(target_qpos, np.zeros_like(qvel0))
            goal_img = base.render(
                width=self.config.img_size,
                height=self.config.img_size,
            ).copy()
            goal_image_sha256 = hashlib.sha256(
                np.ascontiguousarray(goal_img).tobytes()
            ).hexdigest()
            reference_frame = goal_img.copy()
            base.set_state(qpos0, qvel0)
            _install_goal(world, goal_img, target_qpos)
            if observer is not None:
                observer.reset(np.zeros(2, dtype=np.float64))

            pending_latent_prediction: dict[str, np.ndarray] | None = None

            for step in range(self.config.steps):
                _install_goal(world, goal_img, target_qpos)
                _synchronize_device(self.config.device)
                planning_started = time.perf_counter()
                nominal_action = _as_numpy_action(world._get_actions()[0]).astype(np.float64, copy=True)
                control_action = clip_action_to_bounds(nominal_action)
                if method in {"Tube-MPPI", "Robust-MPPI"}:
                    rollout_info = _copy_rollout_info(world.infos)
                    goal_latent = _last_latent_from_info(rollout_info, "goal_emb")
                    action_dim = nominal_action.size
                    disturbance_forecast = (
                        observer.forecast(include_derivative=True)
                        if observer is not None
                        else np.zeros(action_dim, dtype=np.float64)
                    )
                    base_feedback_action = _tube_compensating_action(
                        nominal_action,
                        disturbance_forecast,
                        gain=self.config.dob_action_prior_rollout_gain,
                        max_correction_norm=self.config.dob_action_prior_max_norm,
                    )

                    if method == "Tube-MPPI":
                        nominal_block = planned_action_block(
                            policy,
                            nominal_action,
                            action_dim=action_dim,
                            action_block=self.config.action_block,
                        )
                        nominal_endpoint = _latent_rollout_embeddings(
                            model,
                            policy,
                            rollout_info,
                            nominal_block.reshape(1, -1),
                        )[0]

                        def latent_score(samples: np.ndarray) -> np.ndarray:
                            action_blocks = _action_blocks_from_samples(
                                policy,
                                samples,
                                action_dim=action_dim,
                                action_block=self.config.action_block,
                            )
                            disturbed_blocks = _disturbed_action_blocks(
                                action_blocks,
                                disturbance_forecast.reshape(1, -1),
                            )[0]
                            endpoints = _latent_rollout_embeddings(
                                model,
                                policy,
                                rollout_info,
                                disturbed_blocks.reshape(samples.shape[0], -1),
                            )
                            tracking_cost = np.mean(
                                (endpoints - nominal_endpoint[None, :]) ** 2,
                                axis=1,
                            )
                            feedback_prior = 0.02 * np.mean(
                                (samples - base_feedback_action[None, :]) ** 2,
                                axis=1,
                            )
                            effort = 0.01 * np.mean(samples * samples, axis=1)
                            return tracking_cost + feedback_prior + effort
                    else:
                        branches, branch_weights = _robust_disturbance_branches(
                            disturbance_forecast,
                            max_norm=self.config.dob_action_prior_max_norm,
                        )

                        def latent_score(samples: np.ndarray) -> np.ndarray:
                            action_blocks = _action_blocks_from_samples(
                                policy,
                                samples,
                                action_dim=action_dim,
                                action_block=self.config.action_block,
                            )
                            disturbed_blocks = _disturbed_action_blocks(action_blocks, branches)
                            num_branches, num_samples = disturbed_blocks.shape[:2]
                            endpoints = _latent_rollout_embeddings(
                                model,
                                policy,
                                rollout_info,
                                disturbed_blocks.reshape(num_branches * num_samples, -1),
                            ).reshape(num_branches, num_samples, -1)
                            if goal_latent is None:
                                branch_costs = np.zeros((num_branches, num_samples), dtype=np.float64)
                            else:
                                branch_costs = np.mean(
                                    (endpoints - goal_latent.reshape(1, 1, -1)) ** 2,
                                    axis=2,
                                )
                            robust_cost = _combine_robust_branch_costs(
                                branch_costs,
                                branch_weights=branch_weights,
                                risk_weight=0.35,
                            )
                            likelihood_ratio_penalty = 0.04 * np.mean(
                                (samples - nominal_action[None, :]) ** 2,
                                axis=1,
                            )
                            effort = 0.015 * np.mean(samples * samples, axis=1)
                            return robust_cost + likelihood_ratio_penalty + effort

                    control_action = _latent_mppi_feedback(
                        nominal_action=base_feedback_action,
                        score_fn=latent_score,
                        rng=feedback_rng,
                        robust=(method == "Robust-MPPI"),
                        num_samples=max(256, self.config.num_samples),
                        temperature=self.config.mppi_temperature,
                    )
                _synchronize_device(self.config.device)
                planning_time_ms = 1000.0 * (time.perf_counter() - planning_started)
                true_disturbance = (
                    _configured_action_disturbance(step, self.config, nominal_action.size)
                    if disturbed
                    else np.zeros_like(nominal_action)
                )
                remaining_actions = _remaining_policy_actions(policy)
                if observer is not None and _starts_action_block(remaining_actions, self.config.action_block):
                    pending_latent_prediction = estimate_latent_action_equivalent_innovation(
                        model,
                        policy,
                        world.infos,
                        planned_action_block(
                            policy,
                            control_action,
                            action_dim=nominal_action.size,
                            action_block=self.config.action_block,
                        ),
                        action_dim=nominal_action.size,
                        action_block=self.config.action_block,
                    )
                applied_action = clip_action_to_bounds(control_action + true_disturbance)
                _, rewards, terms, truncs, infos = world.envs.step(applied_action[None, ...])
                terminated_once = terminated_once or bool(terms[0])
                world.rewards, world.terminateds, world.truncateds, world.infos = (
                    rewards,
                    terms,
                    truncs,
                    infos,
                )
                _install_goal(world, goal_img, target_qpos)

                qpos = world.infos["qpos"][0, 0].astype(float)
                qvel = world.infos["qvel"][0, 0].astype(float)
                finger = world.infos["finger_pos"][0, 0].astype(float)
                target = world.infos["target_pos"][0, 0].astype(float)
                qpos_error = float(np.max(np.abs(qpos - target_qpos)))
                finger_target_error = float(np.linalg.norm(finger - target))
                error = qpos_error
                errors.append(error)
                control_cost += float(np.dot(control_action, control_action))

                d_hat = np.zeros_like(nominal_action)
                d_dot_hat = np.zeros_like(nominal_action)
                latent_residual_norm = ""
                latent_sensitivity_norm = ""
                latent_sensitivity_cond = ""
                innovation_diag = np.full_like(nominal_action, np.nan, dtype=np.float64)
                forecast_bias = np.zeros_like(nominal_action)
                if observer is not None:
                    _synchronize_device(self.config.device)
                    observer_started = time.perf_counter()
                    if pending_latent_prediction is not None and _ends_action_block(
                        remaining_actions,
                        self.config.action_block,
                    ):
                        feedback_latent = encode_feedback_latent(model, policy, world.infos)
                        innovation = solve_action_equivalent_bias(
                            pending_latent_prediction["state"],
                            pending_latent_prediction["sensitivity"],
                            feedback_latent,
                        )
                        innovation_diag = innovation.copy()
                        if isinstance(observer, DisturbanceMPPIObserver):
                            d_hat = observer.update_from_rollout(
                                rollout_fn=lambda action_vectors: _latent_rollout_embeddings(
                                    model,
                                    policy,
                                    pending_latent_prediction["info_dict"],
                                    action_vectors,
                                ),
                                action_block_plan=pending_latent_prediction["action_block_plan"],
                                feedback_latent=feedback_latent,
                                proposal_center=None,
                                seed_disturbances=(innovation,),
                            ).copy()
                        else:
                            d_hat = observer.update_innovation(innovation).copy()
                        latent_residual_norm = float(
                            np.linalg.norm(feedback_latent - pending_latent_prediction["state"])
                        )
                        latent_sensitivity_norm = float(
                            np.linalg.norm(pending_latent_prediction["sensitivity"])
                        )
                        try:
                            latent_sensitivity_cond = float(
                                np.linalg.cond(pending_latent_prediction["sensitivity"])
                            )
                        except np.linalg.LinAlgError:
                            latent_sensitivity_cond = float("inf")
                        pending_latent_prediction = None
                    else:
                        d_hat = observer.d_hat.copy()
                        innovation_diag = getattr(observer, "last_innovation", innovation_diag).copy()
                    d_dot_hat = observer.d_dot_hat.copy()
                    if method == "WorldModel+SDOB":
                        forecast_bias = _sdob_forecast(
                            observer,
                            self.config.dob_action_prior_forecast_order,
                        )
                        cost_model.set_bias(forecast_bias)
                    else:
                        forecast_bias = observer.forecast(
                            include_derivative=(method in {"Tube-MPPI", "Robust-MPPI"})
                        )
                    _synchronize_device(self.config.device)
                    planning_time_ms += 1000.0 * (time.perf_counter() - observer_started)

                planning_times_ms.append(planning_time_ms)

                trajectory_rows.append(
                    {
                        "case": case_name,
                        "method": method,
                        "step": step,
                        "error": error,
                        "qpos_0": qpos[0],
                        "qpos_1": qpos[1],
                        "qvel_0": qvel[0],
                        "qvel_1": qvel[1],
                        "target_qpos_0": target_qpos[0],
                        "target_qpos_1": target_qpos[1],
                        "qpos_error": qpos_error,
                        "finger_target_error": finger_target_error,
                        "finger_x": finger[0],
                        "finger_y": finger[1],
                        "target_x": target[0],
                        "target_y": target[1],
                        "u_nom_0": nominal_action[0],
                        "u_nom_1": nominal_action[1],
                        "u_ctrl_0": control_action[0],
                        "u_ctrl_1": control_action[1],
                        "u_applied_0": applied_action[0],
                        "u_applied_1": applied_action[1],
                        "planning_time_ms": planning_time_ms,
                    }
                )
                diagnostic_rows.append(
                    {
                        "case": case_name,
                        "method": method,
                        "step": step,
                        "task_seed": streams.task_seed,
                        "initial_qpos_0": initial_qpos[0],
                        "initial_qpos_1": initial_qpos[1],
                        "target_qpos_0": target_qpos_metadata[0],
                        "target_qpos_1": target_qpos_metadata[1],
                        "goal_image_sha256": goal_image_sha256,
                        "d_true_0": true_disturbance[0],
                        "d_true_1": true_disturbance[1],
                        "d_hat_0": d_hat[0],
                        "d_hat_1": d_hat[1],
                        "d_dot_hat_0": d_dot_hat[0],
                        "d_dot_hat_1": d_dot_hat[1],
                        "innovation_0": innovation_diag[0],
                        "innovation_1": innovation_diag[1],
                        "observer_raw_mean_0": getattr(observer, "last_raw_mean", ["", ""])[0]
                        if observer is not None
                        else "",
                        "observer_raw_mean_1": getattr(observer, "last_raw_mean", ["", ""])[1]
                        if observer is not None
                        else "",
                        "forecast_bias_0": forecast_bias[0],
                        "forecast_bias_1": forecast_bias[1],
                        "u_nom_0": nominal_action[0],
                        "u_nom_1": nominal_action[1],
                        "u_ctrl_0": control_action[0],
                        "u_ctrl_1": control_action[1],
                        "u_applied_0": applied_action[0],
                        "u_applied_1": applied_action[1],
                        "observer_type": type(observer).__name__ if observer is not None else "none",
                        "particle_ess": getattr(observer, "last_ess", ""),
                        "observer_forward_loss": getattr(observer, "last_forward_loss", ""),
                        "observer_space": "latent" if observer is not None else "none",
                        "latent_residual_norm": latent_residual_norm,
                        "latent_sensitivity_norm": latent_sensitivity_norm,
                        "latent_sensitivity_cond": latent_sensitivity_cond,
                        "solver_info_keys": "|".join(getattr(cost_model, "last_info_keys", ())),
                        "reacher_progress_available": getattr(
                            cost_model,
                            "last_reacher_progress_available",
                            False,
                        ),
                        "disturbed": disturbed,
                        "terminated": bool(terms[0]),
                        "truncated": bool(truncs[0]),
                    }
                )
                frames.append(np.asarray(world.infos["pixels"][0, -1]).copy())
        except Exception as exc:  # Keep failures explicit in CSV instead of fabricating results.
            runtime_error = f"{exc!r}\n{traceback.format_exc(limit=6)}"
        finally:
            world.close()

        final_error = errors[-1] if errors else float("nan")
        min_error = min(errors) if errors else float("nan")
        tail = errors[-min(20, len(errors)) :] if errors else []
        tail_error = float(np.mean(tail)) if tail else float("nan")
        planning_time_mean_ms = (
            float(np.mean(planning_times_ms)) if planning_times_ms else float("nan")
        )
        planning_time_p95_ms = (
            float(np.quantile(planning_times_ms, 0.95))
            if planning_times_ms
            else float("nan")
        )
        for row in diagnostic_rows:
            row["final_tracking_error"] = final_error
            row["tail_tracking_error"] = tail_error
            row["min_tracking_error"] = min_error
        summary = {
            "case": case_name,
            "method": method,
            "env_name": self.config.env_name,
            "status": "ok" if runtime_error is None else "failed",
            "error": runtime_error or "",
            "disturbed": disturbed,
            "steps": len(errors),
            "final_error": final_error,
            "min_error": min_error,
            "tail_error": tail_error,
            "success": bool(
                runtime_error is None
                and len(errors) == self.config.steps
                and tail_error < FORMAL_SUCCESS_THRESHOLD_RAD
            ),
            "terminated_once": terminated_once,
            "control_cost": control_cost,
            "runtime_sec": time.perf_counter() - start_time,
            "planning_time_mean_ms": planning_time_mean_ms,
            "planning_time_p95_ms": planning_time_p95_ms,
            "feedback_type": feedback_type_for_method(method),
            "task_seed": streams.task_seed,
            "planner_seed": streams.planner_seed,
            "sdob_observer_seed": streams.sdob_observer_seed,
            "tube_observer_seed": streams.tube_observer_seed,
            "robust_observer_seed": streams.robust_observer_seed,
            "tube_feedback_seed": streams.tube_feedback_seed,
            "robust_feedback_seed": streams.robust_feedback_seed,
            "initial_qpos_0": initial_qpos[0],
            "initial_qpos_1": initial_qpos[1],
            "target_qpos_0": target_qpos_metadata[0],
            "target_qpos_1": target_qpos_metadata[1],
            "goal_image_sha256": goal_image_sha256,
        }
        return {
            "summary": summary,
            "trajectory": trajectory_rows,
            "diagnostics": diagnostic_rows,
            "frames": frames,
            "reference_frame": reference_frame,
        }






def _install_goal(world: Any, goal_img: np.ndarray, target_qpos: np.ndarray) -> None:
    world.infos["goal"] = goal_img[None, None].copy()
    world.infos["goal_qpos"] = target_qpos[None, None].copy()












def _disable_dmcontrol_early_termination(base_env: Any) -> None:
    task = getattr(getattr(base_env, "env", None), "task", None)
    if task is not None and hasattr(task, "get_termination"):
        task.get_termination = lambda physics: None


def clip_action_to_bounds(action: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0).copy()


def _clip_vector_norm(vector: np.ndarray, max_norm: float | None) -> np.ndarray:
    out = np.asarray(vector, dtype=np.float64).reshape(-1).copy()
    if max_norm is None or float(max_norm) <= 0.0:
        return out
    norm = float(np.linalg.norm(out))
    if norm <= float(max_norm) or norm <= 1e-12:
        return out
    return out * (float(max_norm) / norm)


def _tube_compensating_action(
    nominal_action: np.ndarray,
    disturbance_estimate: np.ndarray,
    gain: float = 1.0,
    max_correction_norm: float | None = None,
) -> np.ndarray:
    nominal = np.asarray(nominal_action, dtype=np.float64).reshape(-1)
    disturbance = np.asarray(disturbance_estimate, dtype=np.float64).reshape(nominal.shape)
    correction = _clip_vector_norm(-float(gain) * disturbance, max_correction_norm)
    return clip_action_to_bounds(nominal + correction)


def _action_blocks_from_samples(
    policy: Any,
    samples: np.ndarray,
    action_dim: int,
    action_block: int,
) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float64)
    return np.asarray(
        [
            planned_action_block(
                policy,
                sample,
                action_dim=action_dim,
                action_block=action_block,
            )
            for sample in samples
        ],
        dtype=np.float64,
    )


def _disturbed_action_blocks(
    action_blocks: np.ndarray,
    disturbance_branches: np.ndarray,
) -> np.ndarray:
    blocks = np.asarray(action_blocks, dtype=np.float64)
    if blocks.ndim != 3:
        raise ValueError("action_blocks must have shape (num_samples, action_block, action_dim)")
    branches = np.asarray(disturbance_branches, dtype=np.float64)
    if branches.ndim == 1:
        branches = branches.reshape(1, -1)
    if branches.ndim != 2 or branches.shape[1] != blocks.shape[2]:
        raise ValueError("disturbance_branches must have shape (num_branches, action_dim)")
    return np.clip(blocks[None, :, :, :] + branches[:, None, None, :], -1.0, 1.0)


def _robust_disturbance_branches(
    disturbance_estimate: np.ndarray,
    max_norm: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    estimate = _clip_vector_norm(disturbance_estimate, max_norm)
    if estimate.size == 0 or float(np.linalg.norm(estimate)) < 1e-9:
        return estimate.reshape(1, -1), np.ones(1, dtype=np.float64)
    branches = np.asarray(
        [
            np.zeros_like(estimate),
            estimate,
            0.5 * estimate,
            1.5 * estimate,
        ],
        dtype=np.float64,
    )
    branches = np.asarray([_clip_vector_norm(branch, max_norm) for branch in branches], dtype=np.float64)
    weights = np.asarray([0.12, 0.58, 0.20, 0.10], dtype=np.float64)
    return branches, weights / weights.sum()


def _combine_robust_branch_costs(
    branch_costs: np.ndarray,
    branch_weights: np.ndarray | None = None,
    risk_weight: float = 0.25,
) -> np.ndarray:
    costs = np.asarray(branch_costs, dtype=np.float64)
    if costs.ndim != 2:
        raise ValueError("branch_costs must have shape (num_branches, num_samples)")
    if branch_weights is None:
        weights = np.full(costs.shape[0], 1.0 / costs.shape[0], dtype=np.float64)
    else:
        weights = np.asarray(branch_weights, dtype=np.float64).reshape(-1)
        if weights.size != costs.shape[0]:
            raise ValueError("branch_weights must match branch_costs first dimension")
        total = float(weights.sum())
        weights = np.full(costs.shape[0], 1.0 / costs.shape[0], dtype=np.float64) if total <= 0 else weights / total
    expected = weights @ costs
    spread = np.max(costs, axis=0) - expected
    return expected + float(risk_weight) * spread


def _sinusoidal_action_bias(step: int, config: ExperimentConfig, action_dim: int) -> np.ndarray:
    phase = 2.0 * math.pi * config.disturbance_frequency * step
    base = config.disturbance_amplitude * math.sin(phase)
    pattern = np.ones(action_dim, dtype=np.float64)
    if action_dim >= 2:
        pattern[1] = -0.65
    return base * pattern


def _configured_action_disturbance(
    step: int,
    config: ExperimentConfig,
    action_dim: int,
) -> np.ndarray:
    if config.disturbance_mode == "action_bias":
        bias = np.asarray(config.disturb_action_bias, dtype=np.float64).reshape(-1)
        if bias.size < action_dim:
            bias = np.pad(bias, (0, action_dim - bias.size))
        return bias[:action_dim].copy()
    if config.disturbance_mode == "sinusoidal_action_bias":
        return _sinusoidal_action_bias(step, config, action_dim)
    return np.zeros(action_dim, dtype=np.float64)


def _latent_mppi_feedback(
    nominal_action: np.ndarray,
    score_fn: Any,
    rng: np.random.Generator,
    robust: bool,
    num_samples: int = 256,
    temperature: float = 0.08,
    noise_scale: float | None = None,
) -> np.ndarray:
    nominal = np.asarray(nominal_action, dtype=np.float64).reshape(-1)
    num_samples = max(2, int(num_samples))
    scale = float(noise_scale if noise_scale is not None else (0.32 if robust else 0.24))
    samples = nominal[None, :] + rng.normal(
        loc=0.0,
        scale=scale,
        size=(num_samples, nominal.size),
    )
    samples[0] = nominal
    samples = np.clip(samples, -1.0, 1.0)

    scores = np.asarray(score_fn(samples), dtype=np.float64).reshape(-1)
    if scores.shape[0] != num_samples:
        raise ValueError("latent MPPI feedback scorer must return one score per sample")
    finite = np.isfinite(scores)
    if not finite.any():
        return nominal.copy()
    max_finite = float(np.max(scores[finite]))
    scores = np.where(finite, scores, max_finite + 1.0)
    scaled = scores - float(np.min(scores))
    logits = -scaled / max(float(temperature), 1e-8)
    logits = logits - float(np.max(logits))
    weights = np.exp(logits)
    weight_sum = float(weights.sum())
    if weight_sum <= 0.0 or not np.isfinite(weight_sum):
        return samples[int(np.argmin(scores))].copy()
    weights = weights / weight_sum
    return np.clip(weights @ samples, -1.0, 1.0)


def _last_latent_from_info(info_dict: dict, key: str) -> np.ndarray | None:
    try:
        import torch

        value = info_dict.get(key)
        if value is None:
            return None
        if torch.is_tensor(value):
            arr = np.asarray(value.detach().cpu().tolist(), dtype=np.float64)
        else:
            arr = np.asarray(value)
    except Exception:
        value = info_dict.get(key)
        if value is None:
            return None
        arr = np.asarray(value)
    if arr.ndim < 1:
        return None
    return np.asarray(arr.reshape(-1, arr.shape[-1])[-1], dtype=np.float64)








def planned_action_block(
    policy: Any,
    current_action: np.ndarray,
    action_dim: int,
    action_block: int,
) -> np.ndarray:
    action_dim = int(action_dim)
    action_block = max(1, int(action_block))
    block = [np.asarray(current_action, dtype=np.float64).reshape(action_dim)]
    buffers = getattr(policy, "_action_buffer", None)
    if buffers:
        for item in list(buffers[0])[: action_block - 1]:
            block.append(_as_numpy_action(item).reshape(action_dim))
    while len(block) < action_block:
        block.append(block[-1].copy())
    return np.asarray(block[:action_block], dtype=np.float64)


def estimate_latent_action_equivalent_innovation(
    model: Any,
    policy: Any,
    info_dict: dict,
    action_block_plan: np.ndarray,
    action_dim: int,
    action_block: int,
    eps: float = 0.08,
) -> dict[str, np.ndarray]:
    action_dim = int(action_dim)
    action_block = max(1, int(action_block))
    nominal = np.asarray(action_block_plan, dtype=np.float64).reshape(action_block, action_dim)
    samples = [nominal]
    deltas: list[float] = []
    for col in range(action_dim):
        perturbed = nominal.copy()
        perturbed[:, col] = np.clip(perturbed[:, col] + eps, -1.0, 1.0)
        delta = float(np.mean(perturbed[:, col] - nominal[:, col]))
        if abs(delta) < 1e-9:
            perturbed[:, col] = np.clip(nominal[:, col] - eps, -1.0, 1.0)
            delta = float(np.mean(perturbed[:, col] - nominal[:, col]))
        samples.append(perturbed)
        deltas.append(delta)

    action_vectors = np.asarray([sample.reshape(-1) for sample in samples], dtype=np.float64)
    embeddings = _latent_rollout_embeddings(model, policy, info_dict, action_vectors)
    nominal_embedding = embeddings[0]
    sensitivity = np.zeros((nominal_embedding.size, action_dim), dtype=np.float64)
    for col, delta in enumerate(deltas):
        if abs(delta) >= 1e-9:
            sensitivity[:, col] = (embeddings[col + 1] - nominal_embedding) / delta
    return {
        "state": nominal_embedding,
        "sensitivity": sensitivity,
        "action_block_plan": nominal.copy(),
        "info_dict": _copy_rollout_info(info_dict),
    }


def _copy_rollout_info(info_dict: dict) -> dict:
    import torch

    copied: dict[str, Any] = {}
    for key, value in info_dict.items():
        if torch.is_tensor(value):
            copied[key] = value.detach().clone()
        elif isinstance(value, np.ndarray):
            copied[key] = value.copy()
        elif isinstance(value, (list, tuple)):
            copied[key] = type(value)(
                item.copy() if isinstance(item, np.ndarray) else item for item in value
            )
        else:
            copied[key] = value
    return copied


def encode_feedback_latent(model: Any, policy: Any, info_dict: dict) -> np.ndarray:
    import torch

    device, dtype = _model_device_dtype(model)
    prepared = policy._prepare_info(info_dict)
    pixels = prepared.get("pixels")
    if pixels is None or not torch.is_tensor(pixels):
        raise ValueError("image-only SDOB requires tensor pixels after policy preprocessing")
    pixels = pixels[:1].to(device=device, dtype=dtype)
    with torch.inference_mode():
        encoded = model.encode({"pixels": pixels})
    return np.asarray(encoded["emb"][0, -1].detach().cpu().tolist(), dtype=np.float64)


def solve_action_equivalent_bias(
    predicted_state: np.ndarray,
    sensitivity: np.ndarray,
    feedback_state: np.ndarray,
) -> np.ndarray:
    residual = np.asarray(feedback_state, dtype=np.float64) - np.asarray(predicted_state, dtype=np.float64)
    sensitivity = np.asarray(sensitivity, dtype=np.float64)
    if sensitivity.ndim != 2 or np.linalg.norm(sensitivity) < 1e-9:
        return np.zeros(sensitivity.shape[1] if sensitivity.ndim == 2 else 0, dtype=np.float64)
    estimate, *_ = np.linalg.lstsq(sensitivity, residual, rcond=None)
    return np.asarray(estimate, dtype=np.float64)


def _latent_rollout_embeddings(
    model: Any,
    policy: Any,
    info_dict: dict,
    action_vectors: np.ndarray,
) -> np.ndarray:
    import torch

    device, dtype = _model_device_dtype(model)
    prepared = policy._prepare_info(info_dict)
    pixels = prepared.get("pixels")
    if pixels is None or not torch.is_tensor(pixels):
        raise ValueError("image-only SDOB requires tensor pixels after policy preprocessing")

    action_vectors = np.asarray(action_vectors, dtype=np.float64)
    if action_vectors.ndim != 2:
        raise ValueError("action_vectors must have shape (num_samples, action_dim)")
    num_samples, flat_dim = action_vectors.shape
    history_len = int(pixels.shape[1]) if pixels.ndim >= 2 else 1
    candidates = torch.as_tensor(action_vectors, dtype=dtype, device=device)
    candidates = candidates.view(1, num_samples, 1, flat_dim).expand(
        1,
        num_samples,
        history_len,
        flat_dim,
    ).clone()

    expanded: dict[str, Any] = {}
    for key, value in prepared.items():
        if torch.is_tensor(value):
            value = value[:1].to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
            expanded[key] = value.unsqueeze(1).expand(
                1,
                num_samples,
                *value.shape[1:],
            ).clone()
        elif isinstance(value, np.ndarray):
            expanded[key] = np.repeat(value[:1, None, ...], num_samples, axis=1)
        else:
            expanded[key] = value
    if "action" not in expanded:
        expanded["action"] = torch.zeros_like(candidates)

    with torch.inference_mode():
        model.get_cost(expanded, candidates)
    return np.asarray(expanded["predicted_emb"][0, :, -1, :].detach().cpu().tolist(), dtype=np.float64)


def _model_device_dtype(model: Any):
    import torch

    try:
        param = next(model.parameters())
        return param.device, param.dtype
    except (AttributeError, StopIteration):
        return torch.device("cpu"), torch.float32


def _as_numpy_action(action: Any) -> np.ndarray:
    if all(callable(getattr(action, name, None)) for name in ("detach", "cpu", "tolist")):
        return np.asarray(action.detach().cpu().tolist(), dtype=np.float64)
    if isinstance(action, np.ndarray):
        return np.asarray(action.tolist(), dtype=np.float64)
    return np.asarray(action, dtype=np.float64)


def _remaining_policy_actions(policy: Any) -> int:
    buffers = getattr(policy, "_action_buffer", None)
    if not buffers:
        return 0
    return len(buffers[0])


def _starts_action_block(remaining_actions: int, action_block: int) -> bool:
    action_block = max(1, int(action_block))
    return (int(remaining_actions) + 1) % action_block == 0


def _ends_action_block(remaining_actions: int, action_block: int) -> bool:
    action_block = max(1, int(action_block))
    return int(remaining_actions) % action_block == 0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_comparison_video(
    path: Path,
    videos: list[tuple[str, list[np.ndarray]]],
    fps: int,
) -> None:
    valid = [(name, frames) for name, frames in videos if frames]
    if not valid:
        return
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    max_len = max(len(frames) for _, frames in valid)
    with imageio.get_writer(str(path), fps=fps, codec="libx264", macro_block_size=1) as writer:
        for idx in range(max_len):
            panel_frames = [
                frames[min(idx, len(frames) - 1)]
                for _, frames in valid
            ]
            writer.append_data(_make_panel(panel_frames, [name for name, _ in valid]))


def _write_comparison_image(path: Path, videos: list[tuple[str, list[np.ndarray]]]) -> None:
    valid = [(name, frames) for name, frames in videos if frames]
    if not valid:
        return
    import imageio.v2 as imageio

    panel = _make_panel([frames[-1] for _, frames in valid], [name for name, _ in valid])
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, panel)


def _make_panel(frames: list[np.ndarray], labels: list[str]) -> np.ndarray:
    from PIL import Image, ImageDraw

    rendered = []
    for frame, label in zip(frames, labels):
        img = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
        canvas = Image.new("RGB", (img.width, img.height + 24), "white")
        canvas.paste(img, (0, 24))
        draw = ImageDraw.Draw(canvas)
        draw.text((6, 4), label, fill=(20, 20, 20))
        rendered.append(canvas)
    width = sum(img.width for img in rendered)
    height = max(img.height for img in rendered)
    out = Image.new("RGB", (width, height), "white")
    x = 0
    for img in rendered:
        out.paste(img, (x, 0))
        x += img.width
    return np.asarray(out)
