"""Guard against dependence on the research checkout's working directory."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("task", ["navigation", "pendulum", "reacher"])
def test_cli_help_outside_checkout(task, tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", f"sdob.examples.{task}.run_compare", "--help"],
        cwd=tmp_path, capture_output=True, text=True, timeout=30, check=True,
    )
    assert "--config" in result.stdout
    assert "--device" in result.stdout


def test_missing_reacher_sources_produce_setup_instruction(monkeypatch, tmp_path):
    from sdob.examples.reacher.runtime import ensure_runtime_paths

    monkeypatch.setenv("SDOB_SWM_ROOT", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError, match="scripts/setup_reacher.py"):
        ensure_runtime_paths()
