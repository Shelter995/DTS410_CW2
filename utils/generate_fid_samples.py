"""Generate fixed fake image folders for FID evaluation.

The script loops over available checkpoints for vanilla/lsgan/wgan and writes
5,000 generated PNG files per model by default. The same latent seed is reused
for every model, which makes the sampling protocol controlled.
"""

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


def generate_for_objective(objective_name: str, device: torch.device) -> None:
    generator = load_generator(objective_name, device)
    output_dir = CONFIG.fid_dir / f"fake_{objective_name}_{CONFIG.fid_num_fake}"
    output_dir.mkdir(parents=True, exist_ok=True)

    latent_generator = torch.Generator(device=device)
    latent_generator.manual_seed(CONFIG.fid_seed)
    written = 0

    with torch.no_grad():
        while written < CONFIG.fid_num_fake:
            batch_size = min(CONFIG.fid_batch_size, CONFIG.fid_num_fake - written)
            z = torch.randn(
                batch_size,
                CONFIG.z_dim,
                device=device,
                generator=latent_generator,
            )
            images = generator(z)
            for index, image in enumerate(images):
                save_path = output_dir / f"{written + index:05d}.png"
                save_image(image, save_path, normalize=True, value_range=(-1, 1))
            written += batch_size

    print(f"Saved {written} fake images for {objective_name} to {output_dir}")


def main() -> None:
    ensure_output_dirs(CONFIG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for objective_name in OBJECTIVE_NAMES:
        path = checkpoint_path(objective_name)
        if not path.exists():
            print(f"Skipping {objective_name}: checkpoint not found.")
            continue
        generate_for_objective(objective_name, device)


if __name__ == "__main__":
    main()

