"""Fetch pinned upstream sources and apply the recorded runtime compatibility patch."""

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    ("le-wm", "https://github.com/lucas-maes/le-wm.git", "bf04d3e8c3752ac24f3692fbc5f4cf50209fa765"),
    ("stable-worldmodel", "https://github.com/galilai-group/stable-worldmodel.git", "1d3d09c14b8d24f9c17dd17a2235815dc8b78623"),
)


def main():
    for name, url, revision in SOURCES:
        destination = ROOT / "third_party" / name
        if not destination.exists():
            subprocess.run(["git", "clone", "--no-checkout", url, str(destination)], check=True)
            subprocess.run(["git", "-C", str(destination), "checkout", "--detach", revision], check=True)
        actual = subprocess.check_output(["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()
        if actual != revision:
            raise RuntimeError(f"{destination} is at {actual}; expected {revision}. Existing checkout was left intact.")
        if name == "stable-worldmodel":
            patch = str(ROOT / "third_party/patches/stable-worldmodel.patch")
            command = ["git", "-C", str(destination), "apply"]
            applied = subprocess.run(command + ["--reverse", "--check", patch], capture_output=True).returncode == 0
            if not applied:
                subprocess.run(command + ["--check", patch], check=True)
                subprocess.run(command + [patch], check=True)
        print(f"Ready: {name} @ {revision}", flush=True)
    print("Next: sdob-download-reacher")


if __name__ == "__main__":
    main()
