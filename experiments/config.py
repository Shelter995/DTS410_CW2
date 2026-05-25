"""Central configuration for the CelebA GAN experiments.

Edit this file when you want to change objectives, paths, batch size, epochs,
or evaluation settings. The scripts intentionally do not use argparse so that
all experiment settings stay in one visible place.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Change this value before running experiments/main.py.
# Valid values: "vanilla", "lsgan", "wgan".
OBJECTIVE = "vanilla"
OBJECTIVE_NAMES = ("vanilla", "lsgan", "wgan")


@dataclass(frozen=True)
class TrainConfig:
    # Dataset paths. Expected layout:
    # data/celebA/000001.jpg
    # data/list_eval_partition.txt
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    image_dir: Path = PROJECT_ROOT / "data" / "celebA"
    split_file: Path = PROJECT_ROOT / "data" / "list_eval_partition.txt"

    # Output paths.
    checkpoint_root: Path = PROJECT_ROOT / "checkpoints"
    output_root: Path = PROJECT_ROOT / "output"
    log_dir: Path = PROJECT_ROOT / "output" / "logs"
    sample_dir: Path = PROJECT_ROOT / "output" / "samples"
    fid_dir: Path = PROJECT_ROOT / "output" / "fid"
    interpolation_dir: Path = PROJECT_ROOT / "output" / "interpolation"
    figure_dir: Path = PROJECT_ROOT / "output" / "figures"

    # Reproducibility.
    seed: int = 42
    fixed_noise_seed: int = 1234
    fid_seed: int = 2024
    interpolation_seed: int = 777
    deterministic: bool = False

    # Data and preprocessing.
    image_size: int = 64
    center_crop_size: int = 178
    image_channels: int = 3
    preload_train_to_memory: bool = True
    preload_eval_to_memory: bool = False
    # When training data is preloaded, dataset.py automatically uses 0 workers
    # to avoid duplicating the in-memory cache across worker processes.
    num_workers: int = 4
    persistent_workers: bool = True

    # Training budget.
    z_dim: int = 128
    batch_size: int = 256
    epochs: int = 50
    checkpoint_epochs: Tuple[int, ...] = (10, 25, 50)
    fixed_sample_count: int = 64
    fixed_sample_nrow: int = 8

    # Labels for BCE/least-squares objectives.
    real_label: float = 1.0
    fake_label: float = 0.0

    # Evaluation and figures.
    fid_num_fake: int = 5000
    fid_batch_size: int = 256
    final_comparison_count: int = 12
    interpolation_steps: int = 11


@dataclass(frozen=True)
class ObjectiveConfig:
    name: str
    optimizer: str
    lr: float
    beta1: float = 0.5
    beta2: float = 0.999
    n_critic: int = 1
    clip_value: float = 0.0


OBJECTIVE_CONFIGS: Dict[str, ObjectiveConfig] = {
    "vanilla": ObjectiveConfig(
        name="vanilla",
        optimizer="adam",
        lr=2e-4,
        beta1=0.5,
        beta2=0.999,
    ),
    "lsgan": ObjectiveConfig(
        name="lsgan",
        optimizer="adam",
        lr=2e-4,
        beta1=0.5,
        beta2=0.999,
    ),
    "wgan": ObjectiveConfig(
        name="wgan",
        optimizer="rmsprop",
        lr=5e-5,
        n_critic=5,
        clip_value=0.01,
    ),
}


CONFIG = TrainConfig()


def get_objective_config(objective: str = OBJECTIVE) -> ObjectiveConfig:
    """Return the optimizer/loss-specific settings for one objective."""
    if objective not in OBJECTIVE_CONFIGS:
        valid = ", ".join(OBJECTIVE_CONFIGS)
        raise ValueError(f"Unknown objective '{objective}'. Valid values: {valid}")
    return OBJECTIVE_CONFIGS[objective]


def ensure_output_dirs(cfg: TrainConfig = CONFIG) -> None:
    """Create output directories used by the training and evaluation scripts."""
    dirs = [
        cfg.checkpoint_root,
        cfg.output_root,
        cfg.log_dir,
        cfg.sample_dir,
        cfg.fid_dir,
        cfg.interpolation_dir,
        cfg.figure_dir,
    ]
    for objective in OBJECTIVE_NAMES:
        dirs.append(cfg.checkpoint_root / objective)
        dirs.append(cfg.sample_dir / objective)
    for path in dirs:
        path.mkdir(parents=True, exist_ok=True)
