# Optional upstream runtime

Run `python scripts/setup_reacher.py` from the checkout to populate these ignored
directories. Neither upstream repository is included in the SDOB distribution.

| Directory | Source | Pinned revision |
| --- | --- | --- |
| `le-wm/` | https://github.com/lucas-maes/le-wm | `bf04d3e8c3752ac24f3692fbc5f4cf50209fa765` |
| `stable-worldmodel/` | https://github.com/galilai-group/stable-worldmodel | `1d3d09c14b8d24f9c17dd17a2235815dc8b78623` |

LeWM includes its original MIT license. stable-worldmodel declares MIT in its
project metadata (the pinned revision has no standalone LICENSE file). The LeWM model definitions (`jepa.py`, `module.py`) are
unchanged. `patches/stable-worldmodel.patch` records the compatibility changes
used by the original Reacher experiments:

- Make optional data-format imports tolerant of unavailable extras.
- Avoid an unnecessary torchvision image wrapper in the policy.
- Accept environment wrapper arguments and handle MuJoCo/dm_control field differences.

The patch does not introduce replacement task dynamics or a proxy world model.
Do not install a different stable-worldmodel revision on top of this source path.
The setup script is repeatable; it verifies revisions and whether its patch has
already been applied. Set `SDOB_LEWM_ROOT` and `SDOB_SWM_ROOT` to locate prepared
sources elsewhere (needed when using a wheel without a source checkout).
