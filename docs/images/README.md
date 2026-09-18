# Paper figures

These PNGs are display exports of the figures used by the current SDOB manuscript
(`2026_sdob/submit_v1.tex`). The exports were copied from the project's website
assets without changing the image content. All README references are relative,
so they work in a clone or an unpacked source distribution.

| Local image | Original manuscript figure |
| --- | --- |
| `overview.png` | `figure/sketch/mdob_introduction.pdf` |
| `control-flow.png` | `figure/sketch/control_flow.pdf` |
| `navigation.png` | `figure/exp/nav_8obs_input_const_margin005_frame_sequence.pdf` |
| `pendulum.png` | `figure/exp/pendulum_video4s_snapshot_clean.pdf` |
| `reacher.png` | `figure/exp/reacher_final_states.pdf` |

The experimental figures retain the earlier **MDOB** label; it refers to the
method named **SDOB** in this code release. The figures illustrate paper
experiments and are not outputs of this release's short validation runs.
The pendulum figure uses sinusoidal torque disturbance, while the default demo
configuration uses constant bias; both READMEs provide the sinusoidal command.

The overview and computation-flow figures describe the broader method. The
released examples remain navigation, inverted pendulum, and LeWM Reacher.
