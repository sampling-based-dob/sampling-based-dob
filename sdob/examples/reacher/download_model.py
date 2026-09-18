"""Convert the official LeWM Reacher weights into an inference checkpoint."""

import argparse
import json
import os
from pathlib import Path

from .runtime import ensure_runtime_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Existing directory containing config.json and weights.pt")
    parser.add_argument("--cache-dir", type=Path, default=Path(os.environ.get("STABLEWM_HOME", Path.home() / ".stable_worldmodel")))
    args = parser.parse_args()
    ensure_runtime_paths()
    import torch
    from huggingface_hub import snapshot_download
    from transformers import ViTConfig, ViTModel
    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP

    source = args.source or Path(snapshot_download(
        "quentinll/lewm-reacher", revision="62adae4b71dc474ddf8f794c476ebfe737a743ca", allow_patterns=["config.json", "weights.pt"],
    ))
    config = json.loads((source / "config.json").read_text())
    # Same ViT-tiny architecture as stable_pretraining.backbone.utils.vit_hf;
    # inference does not need the upstream training framework.
    encoder_config = config["encoder"]
    if encoder_config["size"] != "tiny":
        raise ValueError("Expected the official ViT-tiny Reacher checkpoint")
    encoder = ViTModel(ViTConfig(
        hidden_size=192, num_hidden_layers=12, num_attention_heads=3,
        intermediate_size=768, patch_size=encoder_config["patch_size"],
        image_size=encoder_config["image_size"],
    ), add_pooling_layer=False, use_mask_token=False)

    def parameters(name):
        return {key: value for key, value in config[name].items() if key != "_target_"}

    def projection(name):
        values = parameters(name)
        return MLP(input_dim=values["input_dim"], output_dim=values["output_dim"],
                   hidden_dim=values["hidden_dim"], norm_fn=torch.nn.BatchNorm1d)

    model = JEPA(encoder=encoder, predictor=ARPredictor(**parameters("predictor")),
                 action_encoder=Embedder(**parameters("action_encoder")),
                 projector=projection("projector"), pred_proj=projection("pred_proj"))
    model.load_state_dict(torch.load(source / "weights.pt", map_location="cpu", weights_only=True))
    model.eval().requires_grad_(False)
    output = args.cache_dir / "reacher/lewm_object.ckpt"
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, output)
    print(f"Checkpoint: {output}")
    print(f"Run with STABLEWM_HOME={args.cache_dir.resolve()}")


if __name__ == "__main__":
    main()
