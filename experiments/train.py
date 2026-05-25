"""Training loop for Vanilla GAN, LSGAN, and WGAN with weight clipping."""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch
from torch import nn, optim
from torchvision.utils import save_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.config import CONFIG, OBJECTIVE, ensure_output_dirs, get_objective_config
from experiments.dataset import build_dataloader
from experiments.model import Discriminator, Generator, initialize_dcgan_weights


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Fix random seeds where practical."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(requires_grad)


def make_optimizer(model: nn.Module, objective_name: str):
    obj_cfg = get_objective_config(objective_name)
    if obj_cfg.optimizer == "adam":
        return optim.Adam(
            model.parameters(),
            lr=obj_cfg.lr,
            betas=(obj_cfg.beta1, obj_cfg.beta2),
        )
    if obj_cfg.optimizer == "rmsprop":
        return optim.RMSprop(model.parameters(), lr=obj_cfg.lr)
    raise ValueError(f"Unsupported optimizer: {obj_cfg.optimizer}")


def sample_noise(batch_size: int, z_dim: int, device: torch.device) -> torch.Tensor:
    return torch.randn(batch_size, z_dim, device=device)


def discriminator_loss(
    objective_name: str,
    discriminator: Discriminator,
    generator: Generator,
    real_images: torch.Tensor,
    bce_loss: nn.Module,
    mse_loss: nn.Module,
) -> torch.Tensor:
    batch_size = real_images.size(0)
    device = real_images.device
    z = sample_noise(batch_size, CONFIG.z_dim, device)
    fake_images = generator(z).detach()

    real_scores = discriminator(real_images)
    fake_scores = discriminator(fake_images)

    if objective_name == "vanilla":
        real_targets = torch.full_like(real_scores, CONFIG.real_label)
        fake_targets = torch.full_like(fake_scores, CONFIG.fake_label)
        return bce_loss(real_scores, real_targets) + bce_loss(fake_scores, fake_targets)

    if objective_name == "lsgan":
        real_targets = torch.full_like(real_scores, CONFIG.real_label)
        fake_targets = torch.full_like(fake_scores, CONFIG.fake_label)
        return 0.5 * (mse_loss(real_scores, real_targets) + mse_loss(fake_scores, fake_targets))

    if objective_name == "wgan":
        # Minimize fake - real, equivalent to maximizing real - fake.
        return fake_scores.mean() - real_scores.mean()

    raise ValueError(f"Unsupported objective: {objective_name}")


def generator_loss(
    objective_name: str,
    discriminator: Discriminator,
    generator: Generator,
    batch_size: int,
    device: torch.device,
    bce_loss: nn.Module,
    mse_loss: nn.Module,
) -> torch.Tensor:
    z = sample_noise(batch_size, CONFIG.z_dim, device)
    fake_images = generator(z)
    fake_scores = discriminator(fake_images)

    if objective_name == "vanilla":
        real_targets = torch.full_like(fake_scores, CONFIG.real_label)
        return bce_loss(fake_scores, real_targets)

    if objective_name == "lsgan":
        real_targets = torch.full_like(fake_scores, CONFIG.real_label)
        return 0.5 * mse_loss(fake_scores, real_targets)

    if objective_name == "wgan":
        return -fake_scores.mean()

    raise ValueError(f"Unsupported objective: {objective_name}")


def clip_critic_weights(discriminator: Discriminator, clip_value: float) -> None:
    """Apply WGAN weight clipping after each critic update."""
    for parameter in discriminator.parameters():
        parameter.data.clamp_(-clip_value, clip_value)


def write_csv_header(path: Path, fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()


def append_csv_row(path: Path, fieldnames: Iterable[str], row: Dict[str, object]) -> None:
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writerow(row)


def save_fixed_samples(
    generator: Generator,
    fixed_z: torch.Tensor,
    output_path: Path,
    nrow: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generator.eval()
    with torch.no_grad():
        samples = generator(fixed_z)
    save_image(samples, output_path, nrow=nrow, normalize=True, value_range=(-1, 1))
    generator.train()


def save_checkpoint(
    objective_name: str,
    epoch: int,
    generator: Generator,
    discriminator: Discriminator,
    optimizer_g,
    optimizer_d,
    fixed_z: torch.Tensor,
    g_steps: int,
    d_steps: int,
) -> None:
    checkpoint_dir = CONFIG.checkpoint_root / objective_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    serializable_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in CONFIG.__dict__.items()
    }
    state = {
        "objective": objective_name,
        "epoch": epoch,
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "optimizer_g": optimizer_g.state_dict(),
        "optimizer_d": optimizer_d.state_dict(),
        "fixed_z": fixed_z.detach().cpu(),
        "g_steps": g_steps,
        "d_steps": d_steps,
        "config": serializable_config,
        "objective_config": get_objective_config(objective_name).__dict__,
    }
    torch.save(state, checkpoint_dir / f"epoch_{epoch:03d}.pt")
    torch.save(state, checkpoint_dir / "latest.pt")


def train(objective_name: str = OBJECTIVE) -> None:
    cfg = CONFIG
    obj_cfg = get_objective_config(objective_name)
    ensure_output_dirs(cfg)
    set_seed(cfg.seed, deterministic=cfg.deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = build_dataloader("train", cfg=cfg, shuffle=True)

    generator = Generator(cfg.z_dim, cfg.image_channels).to(device)
    discriminator = Discriminator(cfg.image_channels).to(device)
    generator.apply(initialize_dcgan_weights)
    discriminator.apply(initialize_dcgan_weights)

    optimizer_g = make_optimizer(generator, objective_name)
    optimizer_d = make_optimizer(discriminator, objective_name)
    bce_loss = nn.BCEWithLogitsLoss()
    mse_loss = nn.MSELoss()

    fixed_generator = torch.Generator(device="cpu")
    fixed_generator.manual_seed(cfg.fixed_noise_seed)
    fixed_z = torch.randn(cfg.fixed_sample_count, cfg.z_dim, generator=fixed_generator).to(device)

    iter_log = cfg.log_dir / f"{objective_name}.csv"
    epoch_log = cfg.log_dir / f"{objective_name}_epoch.csv"
    iter_fields = [
        "epoch",
        "iteration",
        "objective",
        "d_loss",
        "g_loss",
        "d_steps",
        "g_steps",
    ]
    epoch_fields = [
        "epoch",
        "objective",
        "mean_d_loss",
        "mean_g_loss",
        "d_steps",
        "g_steps",
    ]
    write_csv_header(iter_log, iter_fields)
    write_csv_header(epoch_log, epoch_fields)

    checkpoint_epochs = set(cfg.checkpoint_epochs) | {cfg.epochs}
    g_steps = 0
    d_steps = 0

    for epoch in range(1, cfg.epochs + 1):
        epoch_d_losses = []
        epoch_g_losses = []

        for iteration, (real_images, _) in enumerate(train_loader, start=1):
            real_images = real_images.to(device, non_blocking=True)
            batch_size = real_images.size(0)

            # 1) Update discriminator/critic.
            set_requires_grad(discriminator, True)
            optimizer_d.zero_grad(set_to_none=True)
            d_loss = discriminator_loss(
                objective_name,
                discriminator,
                generator,
                real_images,
                bce_loss,
                mse_loss,
            )
            d_loss.backward()
            optimizer_d.step()
            d_steps += 1

            if objective_name == "wgan":
                clip_critic_weights(discriminator, obj_cfg.clip_value)

            # 2) Update generator. WGAN uses multiple critic updates per G update.
            g_loss_value: Optional[float] = None
            should_update_g = objective_name != "wgan" or d_steps % obj_cfg.n_critic == 0
            if should_update_g:
                set_requires_grad(discriminator, False)
                optimizer_g.zero_grad(set_to_none=True)
                g_loss = generator_loss(
                    objective_name,
                    discriminator,
                    generator,
                    batch_size,
                    device,
                    bce_loss,
                    mse_loss,
                )
                g_loss.backward()
                optimizer_g.step()
                set_requires_grad(discriminator, True)
                g_steps += 1
                g_loss_value = float(g_loss.item())
                epoch_g_losses.append(g_loss_value)

            d_loss_value = float(d_loss.item())
            epoch_d_losses.append(d_loss_value)
            append_csv_row(
                iter_log,
                iter_fields,
                {
                    "epoch": epoch,
                    "iteration": iteration,
                    "objective": objective_name,
                    "d_loss": d_loss_value,
                    "g_loss": "" if g_loss_value is None else g_loss_value,
                    "d_steps": d_steps,
                    "g_steps": g_steps,
                },
            )

        mean_d_loss = float(np.mean(epoch_d_losses))
        mean_g_loss = float(np.mean(epoch_g_losses)) if epoch_g_losses else float("nan")
        append_csv_row(
            epoch_log,
            epoch_fields,
            {
                "epoch": epoch,
                "objective": objective_name,
                "mean_d_loss": mean_d_loss,
                "mean_g_loss": mean_g_loss,
                "d_steps": d_steps,
                "g_steps": g_steps,
            },
        )

        if epoch in checkpoint_epochs:
            save_checkpoint(
                objective_name,
                epoch,
                generator,
                discriminator,
                optimizer_g,
                optimizer_d,
                fixed_z,
                g_steps,
                d_steps,
            )
            save_fixed_samples(
                generator,
                fixed_z,
                cfg.sample_dir / objective_name / f"fixed_epoch_{epoch:03d}.png",
                cfg.fixed_sample_nrow,
            )

        print(
            f"[{objective_name}] epoch {epoch:03d}/{cfg.epochs} "
            f"D={mean_d_loss:.4f} G={mean_g_loss:.4f} "
            f"D_steps={d_steps} G_steps={g_steps}"
        )


if __name__ == "__main__":
    train()
