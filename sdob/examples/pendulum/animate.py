from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CONTROLLER_ORDER = ["MPPI (no disturbance)", "MPPI", "MPPI+DOB", "MPPI+SDOB", "Tube-MPPI", "RMPPI"]
CONTROLLER_COLORS = {
    "MPPI (no disturbance)": "#111827",
    "MPPI": "#6b7280",
    "MPPI+DOB": "#2563eb",
    "MPPI+SDOB": "#059669",
    "Tube-MPPI": "#f97316",
    "RMPPI": "#7c3aed",
}


def pendulum_tip(theta: float, length: float = 1.0) -> np.ndarray:
    """Return bob position for theta=0 as upright."""
    return np.array([length * np.sin(theta), length * np.cos(theta)], dtype=float)


def controller_layout_order(names: list[str] | np.ndarray | pd.Series) -> list[str]:
    available = list(dict.fromkeys(str(name) for name in names))
    ordered = [name for name in CONTROLLER_ORDER if name in available]
    ordered.extend(sorted(name for name in available if name not in CONTROLLER_ORDER))
    return ordered


def disturbance_text(row: dict | pd.Series) -> str:
    disturbance = row["disturbance"] if "disturbance" in row else np.nan
    d_hat = row["d_hat"] if "d_hat" in row else np.nan
    if pd.isna(disturbance):
        true_text = "d_true=n/a"
    else:
        true_text = f"d_true={float(disturbance):+.2f}"

    if pd.isna(d_hat):
        return f"{true_text}\nd_hat=n/a"

    if pd.isna(disturbance):
        return f"{true_text}\nd_hat={float(d_hat):+.2f}"

    return (
        f"{true_text}\n"
        f"d_hat={float(d_hat):+.2f}\n"
        f"err={float(d_hat - disturbance):+.2f}"
    )


def animate_trajectories(
    trajectory_csv: str | Path,
    output_path: str | Path,
    fps: int = 12,
    stride: int = 2,
    dpi: int = 130,
    trail: int = 20,
    style: str = "gym",
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
    from matplotlib.patches import Circle

    if style not in {"gym", "schematic"}:
        raise ValueError("style must be 'gym' or 'schematic'")

    trajectory_csv = Path(trajectory_csv)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(trajectory_csv)
    required = {"controller", "step", "time", "theta", "theta_abs", "theta_dot", "action"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"trajectory csv is missing columns: {sorted(missing)}")

    controllers = controller_layout_order(frame["controller"].unique())
    if not controllers:
        raise ValueError("trajectory csv does not contain any controller rows")

    by_controller = {
        name: group.sort_values("step").reset_index(drop=True)
        for name, group in frame.groupby("controller")
    }
    num_frames = min(len(by_controller[name]) for name in controllers)
    frame_ids = list(range(0, num_frames, max(1, stride)))
    if not frame_ids:
        raise ValueError("trajectory csv does not contain enough frames")

    if style == "gym":
        return _animate_native_gym_trajectories(
            controllers=controllers,
            by_controller=by_controller,
            frame_ids=frame_ids,
            output_path=output_path,
            fps=fps,
        )

    gym_env = None
    fig, axes = plt.subplots(1, len(controllers), figsize=(3.1 * len(controllers), 3.7), squeeze=False)
    axes = axes[0]
    artists = []
    for ax, name in zip(axes, controllers):
        color = CONTROLLER_COLORS.get(name, "#374151")
        ax.set_title(name, fontsize=11, fontweight="bold", color=color)
        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-1.18, 1.28)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#d1d5db")
            spine.set_linewidth(0.9)
        if style == "gym":
            ax.set_xlim(-1.0, 1.0)
            ax.set_ylim(-1.0, 1.0)
            image_artist = ax.imshow(
                np.zeros((500, 500, 3), dtype=np.uint8),
                extent=(-1.0, 1.0, -1.0, 1.0),
                zorder=1,
            )
            trail_line = None
            rod_line = None
            bob_dot = None
            pivot_dot = None
        else:
            ax.set_facecolor("#f8fafc")
            ax.plot([0.0, 0.0], [0.0, 1.02], color="#cbd5e1", lw=1.0, ls="--", zorder=1)
            ax.add_patch(Circle((0.0, 0.0), 1.04, fill=False, ec="#e5e7eb", lw=1.0, zorder=1))
            trail_line, = ax.plot([], [], color=color, lw=1.6, alpha=0.22, zorder=2)
            rod_line, = ax.plot([], [], color="#374151", lw=9.0, solid_capstyle="round", zorder=3)
            bob_dot, = ax.plot([], [], "o", color=color, ms=18, mec="white", mew=2.0, zorder=4)
            pivot_dot, = ax.plot([0.0], [0.0], "o", color="#111827", ms=8.0, mec="white", mew=1.2, zorder=5)
            image_artist = None
        info = ax.text(
            0.04,
            0.06,
            "",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.5,
            color="#111827",
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "none", "alpha": 0.78},
            zorder=6,
        )
        artists.append((image_artist, trail_line, rod_line, bob_dot, pivot_dot, info))

    fig.suptitle("Inverted Pendulum Controller Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))

    def update(frame_index: int):
        step_index = frame_ids[frame_index]
        updated = []
        for name, bundle in zip(controllers, artists):
            image_artist, trail_line, rod_line, bob_dot, pivot_dot, info = bundle
            rows = by_controller[name]
            row = rows.iloc[step_index]
            if style == "gym" and gym_env is not None and image_artist is not None:
                gym_env.unwrapped.state = np.array([float(row["theta"]), float(row["theta_dot"])], dtype=np.float64)
                image_artist.set_data(gym_env.render())
            else:
                tip = pendulum_tip(float(row["theta"]))
                rod_line.set_data([0.0, tip[0]], [0.0, tip[1]])
                bob_dot.set_data([tip[0]], [tip[1]])

                start = max(0, step_index - trail)
                theta_hist = rows.iloc[start : step_index + 1]["theta"].to_numpy(dtype=float)
                tips = np.array([pendulum_tip(theta) for theta in theta_hist])
                trail_line.set_data(tips[:, 0], tips[:, 1])

            d_hat = row["d_hat"] if "d_hat" in row.index else np.nan
            info.set_text(
                f"t={float(row['time']):.2f}s\n"
                f"|theta|={float(row['theta_abs']):.2f}\n"
                f"u={float(row['action']):+.2f}\n"
                f"{disturbance_text(row)}"
            )
            updated.extend(artist for artist in bundle if artist is not None)
        return updated

    animation = FuncAnimation(fig, update, frames=len(frame_ids), interval=1000 / fps, blit=True)
    suffix = output_path.suffix.lower()
    if suffix == ".mp4":
        try:
            writer = FFMpegWriter(fps=fps, bitrate=1800)
            animation.save(output_path, writer=writer, dpi=dpi)
        except (FileNotFoundError, RuntimeError):
            output_path = output_path.with_suffix(".gif")
            animation.save(output_path, writer=PillowWriter(fps=fps), dpi=dpi)
    else:
        if suffix != ".gif":
            output_path = output_path.with_suffix(".gif")
        animation.save(output_path, writer=PillowWriter(fps=fps), dpi=dpi)
    if gym_env is not None:
        gym_env.close()
    plt.close(fig)
    return output_path


def _make_native_gym_env():
    import os

    # This renderer only produces RGB arrays and never needs a desktop window.
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    import gymnasium as gym

    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    env.reset(seed=0)
    return env


def _native_gym_frame(env, row: pd.Series) -> np.ndarray:
    env.unwrapped.state = np.array([float(row["theta"]), float(row["theta_dot"])], dtype=np.float64)
    env.unwrapped.last_u = float(row["action"])
    return np.asarray(env.render(), dtype=np.uint8)


def _animate_native_gym_trajectories(
    controllers: list[str],
    by_controller: dict[str, pd.DataFrame],
    frame_ids: list[int],
    output_path: Path,
    fps: int,
) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() != ".gif":
        output_path = output_path.with_suffix(".gif")

    env = _make_native_gym_env()
    try:
        panel_size = 300
        title_h = 36
        info_h = 76
        gap = 8
        canvas_w = len(controllers) * panel_size + (len(controllers) - 1) * gap
        canvas_h = title_h + panel_size + info_h
        try:
            title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
            info_font = ImageFont.truetype("DejaVuSans.ttf", 13)
        except OSError:
            title_font = ImageFont.load_default()
            info_font = ImageFont.load_default()

        pil_frames: list[Image.Image] = []
        for step_index in frame_ids:
            canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
            draw = ImageDraw.Draw(canvas)
            for panel_idx, name in enumerate(controllers):
                rows = by_controller[name]
                row = rows.iloc[step_index]
                left = panel_idx * (panel_size + gap)

                color = CONTROLLER_COLORS.get(name, "#374151")
                draw.text((left + 8, 9), name, fill=color, font=title_font)

                native = Image.fromarray(_native_gym_frame(env, row))
                native = native.resize((panel_size, panel_size), Image.Resampling.LANCZOS)
                canvas.paste(native, (left, title_h))

                info = (
                    f"t={float(row['time']):.2f}s  |theta|={float(row['theta_abs']):.2f}\n"
                    f"u={float(row['action']):+.2f}  "
                    f"{disturbance_text(row).replace(chr(10), '  ')}"
                )
                draw.text((left + 8, title_h + panel_size + 8), info, fill="#111827", font=info_font)
            pil_frames.append(canvas)

        duration_ms = int(round(1000 / max(1, fps)))
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=True,
        )
    finally:
        env.close()

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Animate inverted-pendulum benchmark trajectories.")
    parser.add_argument(
        "--trajectory-csv",
        type=Path,
        default=Path("outputs/pendulum/default/trajectories.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/pendulum/default/comparison.gif"),
    )
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--dpi", type=int, default=130)
    parser.add_argument("--style", choices=["gym", "schematic"], default="gym")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = animate_trajectories(
        trajectory_csv=args.trajectory_csv,
        output_path=args.output,
        fps=args.fps,
        stride=args.stride,
        dpi=args.dpi,
        style=args.style,
    )
    print(f"Animation saved to: {output}")


if __name__ == "__main__":
    main()
