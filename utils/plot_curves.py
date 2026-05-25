"""Plot epoch-averaged generator and discriminator/critic losses."""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.config import CONFIG, OBJECTIVE_NAMES, ensure_output_dirs


def read_epoch_log(path: Path):
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "epoch": int(row["epoch"]),
                    "mean_d_loss": float(row["mean_d_loss"]),
                    "mean_g_loss": float(row["mean_g_loss"]),
                }
            )
    return rows


def main() -> None:
    ensure_output_dirs(CONFIG)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=150)
    found_any = False

    for objective_name in OBJECTIVE_NAMES:
        path = CONFIG.log_dir / f"{objective_name}_epoch.csv"
        if not path.exists():
            print(f"Skipping {objective_name}: log not found.")
            continue
        rows = read_epoch_log(path)
        epochs = [row["epoch"] for row in rows]
        d_losses = [row["mean_d_loss"] for row in rows]
        g_losses = [row["mean_g_loss"] for row in rows]
        axes[0].plot(epochs, d_losses, label=objective_name)
        axes[1].plot(epochs, g_losses, label=objective_name)
        found_any = True

    if not found_any:
        raise FileNotFoundError("No epoch logs found in output/logs.")

    axes[0].set_title("Discriminator / Critic Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Generator Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    output_path = CONFIG.figure_dir / "loss_curves.png"
    fig.savefig(output_path)
    print(f"Saved loss curves to {output_path}")


if __name__ == "__main__":
    main()

