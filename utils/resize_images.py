"""Preprocess CelebA images into data/processed/{train,val,test}.

This utility is optional. Training can read from data/celebA directly. Use this
when you want a cached 64x64 PNG copy of each split for inspection or debugging.
"""

import sys
from pathlib import Path

from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.config import CONFIG
from experiments.dataset import read_partition_file


PARTITION_TO_SPLIT = {0: "train", 1: "val", 2: "test"}
SPLITS_TO_PROCESS = ("train", "val", "test")


def main() -> None:
    transform = transforms.Compose(
        [
            transforms.CenterCrop(CONFIG.center_crop_size),
            transforms.Resize((CONFIG.image_size, CONFIG.image_size)),
        ]
    )
    partitions = read_partition_file(CONFIG.split_file)
    processed_root = CONFIG.data_dir / "processed"
    processed_root.mkdir(parents=True, exist_ok=True)

    for partition_id, filenames in partitions.items():
        split = PARTITION_TO_SPLIT[partition_id]
        if split not in SPLITS_TO_PROCESS:
            continue
        output_dir = processed_root / split
        output_dir.mkdir(parents=True, exist_ok=True)

        for index, filename in enumerate(filenames, start=1):
            input_path = CONFIG.image_dir / filename
            output_path = output_dir / Path(filename).with_suffix(".png").name
            with Image.open(input_path) as image:
                image = image.convert("RGB")
                image = transform(image)
                image.save(output_path)
            if index % 1000 == 0:
                print(f"{split}: processed {index}/{len(filenames)} images")

    print(f"Processed images saved to {processed_root}")


if __name__ == "__main__":
    main()

