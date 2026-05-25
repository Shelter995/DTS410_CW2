"""CelebA dataset utilities for official train/val/test partitions."""

from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from experiments.config import CONFIG, TrainConfig


SPLIT_TO_PARTITION = {
    "train": 0,
    "val": 1,
    "eval": 1,
    "validation": 1,
    "test": 2,
}


def read_partition_file(split_file: Path) -> Dict[int, List[str]]:
    """Read CelebA official split file.

    Expected line format:
        000001.jpg 0

    Partition ids:
        0 = train, 1 = val/eval, 2 = test.
    """
    partitions: Dict[int, List[str]] = {0: [], 1: [], 2: []}
    with split_file.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid split line {line_number}: expected '<filename> <partition>'."
                )
            filename, partition_text = parts
            partition = int(partition_text)
            if partition not in partitions:
                raise ValueError(f"Invalid partition id {partition} on line {line_number}.")
            partitions[partition].append(filename)
    return partitions


def build_transform(cfg: TrainConfig = CONFIG) -> transforms.Compose:
    """Build the shared image preprocessing pipeline.

    All training, sampling reference images, and FID real images should use this
    same crop/resize convention so the comparison remains controlled.
    """
    return transforms.Compose(
        [
            transforms.CenterCrop(cfg.center_crop_size),
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            ),
        ]
    )


class CelebASplitDataset(Dataset):
    """CelebA Align & Cropped split dataset.

    Set preload_to_memory=True on a machine with enough RAM to avoid repeatedly
    decoding JPEGs during training. The cached tensors are already normalized to
    [-1, 1], so this is fast but can take several GB of RAM.
    """

    def __init__(
        self,
        split: str,
        cfg: TrainConfig = CONFIG,
        preload_to_memory: bool = False,
    ) -> None:
        normalized_split = split.lower()
        if normalized_split not in SPLIT_TO_PARTITION:
            valid = ", ".join(SPLIT_TO_PARTITION)
            raise ValueError(f"Unknown split '{split}'. Valid values: {valid}")

        self.cfg = cfg
        self.split = normalized_split
        self.image_dir = cfg.image_dir
        self.transform = build_transform(cfg)

        partitions = read_partition_file(cfg.split_file)
        partition_id = SPLIT_TO_PARTITION[normalized_split]
        filenames = partitions[partition_id]
        self.samples: List[Tuple[Path, str]] = [
            (self.image_dir / filename, filename) for filename in filenames
        ]

        missing = [str(path) for path, _ in self.samples[:1000] if not path.exists()]
        if missing:
            preview = "\n".join(missing[:5])
            raise FileNotFoundError(
                "Some CelebA images listed in the split file were not found. "
                f"First missing paths:\n{preview}"
            )

        self.cache = None
        if preload_to_memory:
            self.cache = [self._load_tensor(path) for path, _ in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def _load_tensor(self, path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            image = image.convert("RGB")
            return self.transform(image)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        path, filename = self.samples[index]
        if self.cache is not None:
            return self.cache[index], filename
        return self._load_tensor(path), filename


def build_dataloader(
    split: str,
    cfg: TrainConfig = CONFIG,
    shuffle: bool = True,
    preload_to_memory: bool | None = None,
) -> DataLoader:
    """Create a DataLoader with deterministic shuffling seed."""
    if preload_to_memory is None:
        preload_to_memory = (
            cfg.preload_train_to_memory if split == "train" else cfg.preload_eval_to_memory
        )

    dataset = CelebASplitDataset(
        split=split,
        cfg=cfg,
        preload_to_memory=preload_to_memory,
    )
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)
    loader_workers = 0 if preload_to_memory else cfg.num_workers

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=loader_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.persistent_workers and loader_workers > 0,
        drop_last=(split == "train"),
        generator=generator,
    )
