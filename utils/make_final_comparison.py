"""Create a final side-by-side comparison figure for all trained objectives."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.config import CONFIG, OBJECTIVE_NAMES, ensure_output_dirs
from experiments.model import Generator


def checkpoint_path(objective_name: str) -> Path:
    latest = CONFIG.checkpoint_root / objective_name / "latest.pt"
    final_epoch = CONFIG.checkpoint_root / objective_name / f"epoch_{CONFIG.epochs:03d}.pt"
    return latest if latest.exists() else final_epoch


def load_generator(objective_name: str, device: torch.device) -> Generator:
    path = checkpoint_path(objective_name)
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint found for {objective_name}: {path}")
    state = torch.load(path, map_location=device)
    generator = Generator(CONFIG.z_dim, CONFIG.image_channels).to(device)
    generator.load_state_dict(state["generator"])
    generator.eval()
    return generator


def to_numpy_image(tensor: torch.Tensor):
    image = (tensor.detach().cpu().clamp(-1, 1) + 1.0) / 2.0
    return image.permute(1, 2, 0).numpy()


def main() -> None:
    ensure_output_dirs(CONFIG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    latent_generator = torch.Generator(device=device)
    latent_generator.manual_seed(CONFIG.fid_seed)
    z = torch.randn(
        CONFIG.final_comparison_count,
        CONFIG.z_dim,
        device=device,
        generator=latent_generator,
    )

    generated = []
    labels = []
    for objective_name in OBJECTIVE_NAMES:
        path = checkpoint_path(objective_name)
        if not path.exists():
            print(f"Skipping {objective_name}: checkpoint not found.")
            continue
        generator = load_generator(objective_name, device)
        with torch.no_grad():
            images = generator(z)
        generated.append(images)
        labels.append(objective_name)

    if not generated:
        raise FileNotFoundError("No checkpoints found for final comparison.")

    rows = len(generated)
    cols = CONFIG.final_comparison_count
    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows), dpi=150)
    if rows == 1:
        axes = [axes]

    for row_index, (label, images) in enumerate(zip(labels, generated)):
        for col_index in range(cols):
            ax = axes[row_index][col_index]
            ax.imshow(to_numpy_image(images[col_index]))
            ax.axis("off")
            if col_index == 0:
                ax.set_ylabel(label, rotation=0, labelpad=30, va="center")

    fig.tight_layout(pad=0.1)
    output_path = CONFIG.figure_dir / "final_comparison.png"
    fig.savefig(output_path)
    print(f"Saved final comparison to {output_path}")


if __name__ == "__main__":
    main()

