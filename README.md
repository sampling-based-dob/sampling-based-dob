<h1 align="center">SDOB</h1>
<p align="center"><strong>Tackling Sim-to-Real Mismatch Through Sampling-Based Disturbance Observers: From Analytical Models to Learned World Models</strong></p>

<p align="center">
  <a href="https://sampling-based-dob.github.io/">Project page</a> ·
  <a href="https://sampling-based-dob.github.io/static/papers/sdob-paper.pdf">Paper</a> ·
  <a href="README_zh-CN.md">简体中文</a> ·
  <a href="LICENSE">MIT License</a>
</p>
<p align="center"><code>Python 3.10–3.11</code> &nbsp; <code>PyTorch</code> &nbsp; <code>3 simulation examples</code></p>

<p align="center">
  <a href="docs/images/overview.png"><img src="docs/images/overview.png" width="100%" alt="SDOB overview: analytical and learned models, state-effect and cost disturbance channels, and online compensation."></a>
</p>
<p align="center"><sub>Feedback-driven compensation at state-prediction and cost-query interfaces.</sub></p>

SDOB estimates task-relevant mismatch from successive feedback and applies the
correction to future model rollouts and cost evaluation. It uses forward model
and cost queries, without analytical model inversion or model retraining when
the required interfaces are available.

This initial release provides a reusable MPPI/observer library and **three
standalone simulation examples**:

| Example | Predictive model | Compensation | Entry point |
| :--- | :--- | :--- | :--- |
| [Mobile robot](#navigation) | Differential-drive dynamics + local obstacle costs | State effect + cost-field motion | `sdob-navigation` |
| [Inverted pendulum](#pendulum) | Torque-limited pendulum dynamics | Matched torque disturbance | `sdob-pendulum` |
| [LeWM Reacher](#reacher) | Frozen image-based world model | Equivalent action bias | `sdob-reacher` |

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#navigation">Navigation</a> ·
  <a href="#pendulum">Pendulum</a> ·
  <a href="#reacher">Reacher</a> ·
  <a href="#evaluation">Evaluation</a> ·
  <a href="#development">Development</a>
</p>

<details>
<summary><strong>How SDOB fits into the control loop</strong></summary>

<p align="center">
  <a href="docs/images/control-flow.png"><img src="docs/images/control-flow.png" width="100%" alt="Paper computation flow: feedback updates disturbance estimates, which compensate sampled state rollouts and cost scores before the weighted control update."></a>
</p>

1. **Observe.** Compare the completed prediction with newly available feedback.
2. **Estimate.** Score disturbance candidates using forward evaluations and combine them through sampling weights.
3. **Compensate.** Use the estimates in subsequent state rollouts and cost queries before updating the control sequence.

The figure illustrates the broader method in the paper. This release contains
the three simulation examples listed above. Figure sources are recorded in
[docs/images/README.md](docs/images/README.md).

</details>

<a id="quick-start"></a>

## Quick start

Python **3.10** is recommended. From this checkout:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
sdob-check
```

Start with either CPU example:

```bash
sdob-navigation
sdob-pendulum
```

For GPU execution, install a matching PyTorch build using the
[official instructions](https://pytorch.org/get-started/locally/), then select a
device with `--device cuda:0`. Reacher has a separate [setup step](#reacher).

<details>
<summary><strong>Short CPU checks</strong> — verify the pipeline in a few steps</summary>

```bash
sdob-navigation --device cpu --steps 8 --horizon 5 --num-samples 32 --skip-gif
sdob-pendulum --device cpu --steps 8 --horizon 5 --num-samples 32 --skip-gif
```

These short runs verify execution and output generation; they are not benchmark measurements.

</details>

<a id="navigation"></a>

## 01 · Mobile robot navigation

**State-effect and cost compensation.** A differential-drive robot navigates
among eight moving obstacles under a hidden constant input bias. SDOB uses
executed-state feedback and local cost queries; the controller does not receive
obstacle velocities or future obstacle trajectories.

[![Paper navigation snapshots: five controllers in an eight-obstacle scene at six times.](docs/images/navigation.png)](docs/images/navigation.png)

*Paper snapshots under combined state and cost disturbances. The original figure's
“MPPI+MDOB” label corresponds to MPPI+SDOB in this release.*

```bash
sdob-navigation
```

| Configuration | Output directory | Generated files |
| :--- | :--- | :--- |
| [compare_8obs.yaml](sdob/examples/navigation/configs/compare_8obs.yaml) | `outputs/navigation/compare/8obs/` | Summary CSV, trajectories CSV, PNG, optional GIF |

The comparison includes **MPPI · MPPI+DOB · MPPI+SDOB · Tube-MPPI · RMPPI**.

<a id="pendulum"></a>

## 02 · Inverted pendulum

**Matched torque compensation.** The observer estimates a torque disturbance
from feedback prediction error and injects the estimate into future pendulum
rollouts.

[![Paper inverted-pendulum snapshots comparing the nominal reference and five controllers under sinusoidal torque disturbance.](docs/images/pendulum.png)](docs/images/pendulum.png)

*Paper snapshots under sinusoidal torque bias. “MPPI+MDOB” is the earlier label
for MPPI+SDOB. The default demo below retains the constant-bias configuration.*

```bash
# Default: constant torque bias
sdob-pendulum

# Sinusoidal torque bias: d(t) = 1.2 sin(1.2t) N m
sdob-pendulum --disturbance sinusoidal_bias --disturbance-value 1.2
```

| Configuration | Output directory | Generated files |
| :--- | :--- | :--- |
| [default.yaml](sdob/examples/pendulum/configs/default.yaml) | `outputs/pendulum/direct_compare/` | Summary CSV, trajectories CSV, PNG, optional GIF |

The comparison uses the same five controller families as navigation, plus a
disturbance-free MPPI reference.

<a id="reacher"></a>

## 03 · Frozen LeWM Reacher

**Latent-feedback compensation.** A frozen LeWorldModel predicts from images,
while execution introduces a hidden action bias of `[0.30, -0.20]`. SDOB estimates
an equivalent bias from latent endpoint mismatch. The example uses the real
`swm/ReacherDMControl-v0` environment and rendered image goals, without a training
dataset.

[![Paper Reacher final observations: undisturbed MPPI, disturbed MPPI, SDOB, Tube-MPPI, RMPPI, and the goal reference.](docs/images/reacher.png)](docs/images/reacher.png)

*Representative final observations from the paper. “WorldModel+SDOB” in the code
is shown as “MPPI+MDOB” in the original figure.*

Run from the checkout:

```bash
python -m pip install -e '.[reacher]'
python scripts/setup_reacher.py
sdob-download-reacher
MUJOCO_GL=egl sdob-reacher --device cuda:0
```

| Configuration | Output directory | Generated files |
| :--- | :--- | :--- |
| [reacher.yaml](sdob/examples/reacher/configs/reacher.yaml) | `outputs/reacher/reacher_lewm/` | Summary, trajectories, observer diagnostics, optional MP4 and PNG |

The five cases are **undisturbed MPPI · disturbed MPPI · MPPI+SDOB · Tube-MPPI ·
RMPPI**. Joint coordinates establish the reset/goal and evaluate performance;
the default disturbance estimator uses image/latent feedback. Runtime failures
exit with an error.

<details>
<summary><strong>Model downloads, rendering, and a short GPU check</strong></summary>

The setup script fetches pinned public LeWM and stable-worldmodel revisions and
applies a recorded compatibility patch. Existing checkouts are preserved;
unexpected revisions cause an error. See [third-party provenance](third_party/README.md).
For sources elsewhere, set `SDOB_LEWM_ROOT` and `SDOB_SWM_ROOT`.

The [official model mirror](https://huggingface.co/quentinll/lewm-reacher) provides
the weights. The download utility converts them to
`~/.stable_worldmodel/reacher/lewm_object.ckpt`. Set `STABLEWM_HOME` to change the
cache, or use already downloaded mirror files:

```bash
sdob-download-reacher --source /path/to/reacher-mirror
```

`MUJOCO_GL=egl` enables headless rendering on Linux with working EGL drivers.
On a desktop, omit it to use the default backend. CPU planning is available with
`--device cpu`, but the full Reacher configuration is computationally expensive.

```bash
# Short runtime check across all five cases
MUJOCO_GL=egl sdob-reacher --device cuda:0 --steps 10 --num-samples 8 --cem-steps 1 --skip-media
```

Model weights and upstream repositories are downloaded separately and are not
bundled in the source package.

</details>

<a id="evaluation"></a>

## Evaluation scope

The images above come from the paper. The commands here are **single-seed demos**;
they do not reproduce its multi-trial tables by themselves.

The paper's Reacher evaluation uses **20 preselected, nominally solvable tasks**:
disturbance-free MPPI must achieve last-20-step mean joint error ≤ 0.025 rad.
Formal success requires mean tail error < 0.05 rad. The default demo seed is not
that screened task set. [reacher_protocol.py](sdob/examples/reacher/reacher_protocol.py)
records the evaluation rules and separate random streams.

Tube-MPPI and RMPPI are benchmark-scale implementations; these demos do not claim
formal robustness guarantees. See [VALIDATION.md](VALIDATION.md) for the local
runtime and regression checks performed on this release.

<a id="development"></a>

## Development

```bash
python -m pytest -q
python -m build
```

<details>
<summary><strong>Repository layout and output conventions</strong></summary>

```text
sdob/
  controller/              # Reusable MPPI
  observer/                # State-effect and cost observers
  workspace/               # Experiment lifecycle
  examples/
    navigation/            # Mobile robot and moving obstacles
    pendulum/              # Torque-disturbed inverted pendulum
    reacher/               # Frozen LeWM compensation
docs/images/               # Figures used in this README
scripts/setup_reacher.py   # Fetch pinned optional upstream sources
third_party/patches/       # Explicit runtime compatibility changes
tests/                    # Core API and Reacher protocol checks
```

Default YAML files are included as package data. Installed commands can run
outside the checkout; outputs are relative to the working directory. Use
`--output-dir` to choose another destination and `--help` to inspect each CLI.
No workspace symlinks or machine-specific paths are required.

</details>

## License and acknowledgments

SDOB code is released under the [MIT License](LICENSE). LeWM, stable-worldmodel,
and pretrained weights retain their upstream terms and attribution. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for source versions, patch scope,
model provenance, and the LeWM citation.
