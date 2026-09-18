"""Inverted pendulum benchmark for MPPI, DOB, and SDOB variants."""

__all__ = [
    "BaseMPPI",
    "BenchmarkWorkspace",
    "ControllerBase",
    "ExperimentConfig",
    "load_experiment_config",
    "MPPIConfig",
    "MPPIPlan",
    "PendulumEnvRunner",
    "RunnerOutput",
    "TorchMPPI",
    "make_controllers",
]


def __getattr__(name: str):
    if name == "ExperimentConfig":
        from .config import ExperimentConfig

        return ExperimentConfig
    if name == "load_experiment_config":
        from .config import load_experiment_config

        return load_experiment_config
    if name in {"PendulumEnvRunner", "RunnerOutput"}:
        from .env_runner import PendulumEnvRunner, RunnerOutput

        return {"PendulumEnvRunner": PendulumEnvRunner, "RunnerOutput": RunnerOutput}[name]
    if name == "make_controllers":
        from .factory import make_controllers

        return make_controllers
    if name == "ControllerBase":
        from .interfaces import ControllerBase

        return ControllerBase
    if name in {"BaseMPPI", "MPPIConfig", "MPPIPlan", "TorchMPPI"}:
        from .mppi import BaseMPPI, MPPIConfig, MPPIPlan, TorchMPPI

        return {
            "BaseMPPI": BaseMPPI,
            "MPPIConfig": MPPIConfig,
            "MPPIPlan": MPPIPlan,
            "TorchMPPI": TorchMPPI,
        }[name]
    if name == "BenchmarkWorkspace":
        from .workspace import BenchmarkWorkspace

        return BenchmarkWorkspace
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
