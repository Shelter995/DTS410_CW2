"""Generate latent-space interpolation grids for trained models."""

import sys
from pathlib import Path

import torch
from torchvision.utils import save_image

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


def main() -> None:
    ensure_output_dirs(CONFIG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    latent_generator = torch.Generator(device=device)
    latent_generator.manual_seed(CONFIG.interpolation_seed)
    z1 = torch.randn(1, CONFIG.z_dim, device=device, generator=latent_generator)
    z2 = torch.randn(1, CONFIG.z_dim, device=device, generator=latent_generator)
    alphas = torch.linspace(0.0, 1.0, CONFIG.interpolation_steps, device=device).view(-1, 1)
    z = (1.0 - alphas) * z1 + alphas * z2

    for objective_name in OBJECTIVE_NAMES:
        path = checkpoint_path(objective_name)
        if not path.exists():
            print(f"Skipping {objective_name}: checkpoint not found.")
            continue
        generator = load_generator(objective_name, device)
        with torch.no_grad():
            images = generator(z)
        output_path = CONFIG.interpolation_dir / f"{objective_name}_interp.png"
        save_image(
            images,
            output_path,
            nrow=CONFIG.interpolation_steps,
            normalize=True,
            value_range=(-1, 1),
        )
        print(f"Saved interpolation grid to {output_path}")


if __name__ == "__main__":
    main()

