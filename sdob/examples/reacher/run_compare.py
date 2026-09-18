"""Run an actual, dataset-free frozen LeWM comparison."""

import argparse
from pathlib import Path

from .config import load_experiment_config
from .fixed_goal_runner import FixedGoalRunner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("configs") / "reacher.yaml")
    for name in ("steps", "num-samples", "cem-steps", "topk", "seed", "task-seed"):
        parser.add_argument(f"--{name}", type=int)
    parser.add_argument("--device")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--skip-media", action="store_true")
    args = vars(parser.parse_args())
    path = args.pop("config")
    skip_media = args.pop("skip_media")
    if skip_media:
        args["write_media"] = False
    config = load_experiment_config(path, overrides=args)
    output = FixedGoalRunner(config, fail_fast_on_case_error=True).run()
    print(f"Summary: {output.summary_csv}")
    print(f"Trajectories: {output.trajectories_csv}")
    print(f"Observer diagnostics: {output.diagnostics_csv}")
    if config.write_media:
        print(f"Video: {output.comparison_video}")
        print(f"Image: {output.comparison_image}")


if __name__ == "__main__":
    main()
