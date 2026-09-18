# Local validation

Validated on 2026-09-18 with Python 3.10.18, PyTorch 2.7.1+cu118,
NumPy 1.26.4, and Transformers 4.56.2. The project-local test environment
reused an existing PyTorch installation; this is not a claim of a fresh-machine
installation test.

- 305 tests passed, including core observer/controller behavior, Reacher protocol,
  compensation direction, navigation, pendulum, and installed CLI checks.
- Navigation: five controllers, eight CPU steps, CSV and PNG output.
- Pendulum: five controllers plus nominal reference, eight CPU steps; native
  Gymnasium GIF rendering also checked with a four-step run.
- Reacher: all five cases returned `status=ok` for ten steps on an RTX 4080 Laptop
  GPU, with 50 trajectory rows, 50 diagnostic rows, MP4, and PNG outputs.
- All three installed commands ran from outside the source checkout.
- Source distribution and wheel built successfully; the wheel was installed to
  another location and ran a pendulum comparison with its bundled YAML config.
- Upstream setup was run twice to verify repeatability.

These are runtime and regression checks, not full-duration convergence tests or
reproduction of the paper's multi-seed results. Generated outputs and checkpoint
files are local artifacts and are excluded from version control.

## Commands used for the principal checks

```bash
python -m pytest -q
sdob-navigation --device cpu --steps 8 --horizon 5 --num-samples 32 --skip-gif
sdob-pendulum --device cpu --steps 8 --horizon 5 --num-samples 32 --skip-gif
MUJOCO_GL=egl sdob-reacher --device cuda:1 --steps 10 --num-samples 8 --cem-steps 1
python -m build
```

Tests used `OMP_NUM_THREADS=1`. On the validation workstation, existing ROS
pytest plugins and user-site packages were disabled with
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, `PYTHONNOUSERSITE=1`, and an empty `PYTHONPATH`.
The compatible RTX 4080 was CUDA device 1 on that machine; choose the appropriate
device for your own installation.
