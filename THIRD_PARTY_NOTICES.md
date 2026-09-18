# Sources and attribution

## SDOB implementation

The core library and selected examples were extracted from the existing local
`mdob` project at revision `02b9fc24b214dbcf0c30806e0916f103211a9c66`.
The selected files contained no additional uncommitted algorithm changes.
This release updates module names and controller labels to SDOB, removes
unreleased task runners and compatibility entry points, and makes installation
and paths independent of the original workspace. It preserves the selected
controller and observer equations and demo configuration values.

## LeWorldModel

Source: https://github.com/lucas-maes/le-wm

Pinned revision: `bf04d3e8c3752ac24f3692fbc5f4cf50209fa765`.
MIT license, Copyright (c) 2026 Lucas Maes.

Please cite the upstream authors when using the frozen model:

```bibtex
@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint},
  year={2026}
}
```

Model source: https://huggingface.co/quentinll/lewm-reacher.
Pinned model revision: `62adae4b71dc474ddf8f794c476ebfe737a743ca`.
The model weights are downloaded separately and remain subject to their upstream
terms; the SDOB license does not relicense those weights. The download utility
reconstructs the upstream ViT-tiny architecture directly with Transformers and
loads the state dictionary strictly; training is not performed.

## stable-worldmodel

Source: https://github.com/galilai-group/stable-worldmodel

Pinned revision: `1d3d09c14b8d24f9c17dd17a2235815dc8b78623`.
The pinned project metadata declares the MIT license. That revision does not
contain a standalone LICENSE file; its metadata and source notices are preserved.
Local compatibility changes are distributed explicitly as
`third_party/patches/stable-worldmodel.patch`; see `third_party/README.md`.

PyTorch, Transformers, Gymnasium, dm_control, MuJoCo, and other installed
libraries retain their respective upstream licenses. Their source and binaries
are not bundled in this project.
