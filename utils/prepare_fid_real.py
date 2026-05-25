"""Prepare CelebA official test split images for FID.

This script writes normalized-and-restored PNG images into output/fid/real_test.
It uses the same center-crop and resize pipeline as training.
"""

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.config import CONFIG, ensure_output_dirs
from experiments.dataset import CelebASplitDataset


def main() -> None:
    ensure_output_dirs(CONFIG)
    output_dir = CONFIG.fid_dir / "real_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = CelebASplitDataset("test", cfg=CONFIG, preload_to_memory=False)
    loader = DataLoader(
        dataset,
        batch_size=CONFIG.fid_batch_size,
        shuffle=False,
        num_workers=CONFIG.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=CONFIG.persistent_workers and CONFIG.num_workers > 0,
    )

    saved = 0
    for images, filenames in loader:
        for image, filename in zip(images, filenames):
            save_path = output_dir / Path(filename).with_suffix(".png").name
            save_image(image, save_path, normalize=True, value_range=(-1, 1))
            saved += 1

    print(f"Saved {saved} real test images to {output_dir}")


if __name__ == "__main__":
    main()

