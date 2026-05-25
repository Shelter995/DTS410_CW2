# DTS410 CW2: GAN Objectives on CelebA

This repository contains the experiment code for comparing three GAN objectives
on the CelebA Align & Cropped dataset at `64 x 64` resolution:

- Vanilla GAN
- Least-Squares GAN (LSGAN)
- Wasserstein GAN (WGAN) with weight clipping

All objectives use the same DCGAN-style generator and discriminator/critic
architecture, the same latent dimensionality (`z_dim = 128`), and the same image
preprocessing pipeline.

## Project Structure

```text
DTS410_CW2/
  assignment.md
  data/
    celebA/
      000001.jpg
      000002.jpg
      ...
    list_eval_partition.txt
  experiments/
    config.py
    dataset.py
    main.py
    model.py
    train.py
  utils/
    compute_fid.py
    generate_fid_samples.py
    make_final_comparison.py
    make_interpolation.py
    plot_curves.py
    prepare_fid_real.py
    resize_images.py
  checkpoints/
  output/
```

Large local artifacts are intentionally ignored by Git:

- raw CelebA images
- processed image caches
- model checkpoints
- generated samples
- FID folders
- output figures and logs

## Dataset Layout

The code expects the following local dataset layout:

```text
data/
  celebA/
    000001.jpg
    000002.jpg
    ...
  list_eval_partition.txt
```

The partition file should use the official CelebA format:

```text
000001.jpg 0
000002.jpg 0
000003.jpg 2
```

Partition ids are interpreted as:

| Partition | Split |
|---:|---|
| `0` | train |
| `1` | validation/eval |
| `2` | test |

The official test split is used as the real-image reference for FID.

## Configuration

All experiment settings are centralized in:

```text
experiments/config.py
```

The scripts do not use `argparse`. To switch between objectives, edit:

```python
OBJECTIVE = "vanilla"  # "vanilla", "lsgan", or "wgan"
```

Important default settings:

| Setting | Value |
|---|---:|
| image size | `64 x 64` |
| latent dimension | `128` |
| latent distribution | standard normal |
| batch size | `256` |
| epochs | `50` |
| seed | `42` |
| fixed sample grid | `8 x 8` |
| checkpoint epochs | `10`, `25`, `50` |
| FID fake samples | `5,000` |

## Model Architecture

The models follow a DCGAN-style design.

The generator maps `z in R^128` to a `64 x 64 x 3` RGB image using transposed
convolutions, batch normalization, ReLU activations, and a final `Tanh`.

The discriminator/critic maps a `64 x 64 x 3` image to one scalar output using
strided convolutions, batch normalization, and LeakyReLU activations. The final
layer does not apply sigmoid:

- Vanilla GAN uses the scalar as raw logits with `BCEWithLogitsLoss`.
- LSGAN uses the scalar as a least-squares regression score.
- WGAN uses the scalar as an unconstrained critic score.

## Objective Settings

| Objective | Loss | Optimizer | Notes |
|---|---|---|---|
| Vanilla GAN | `BCEWithLogitsLoss` | Adam, `lr=0.0002`, `betas=(0.5, 0.999)` | sigmoid is handled inside the loss |
| LSGAN | `MSELoss` | Adam, `lr=0.0002`, `betas=(0.5, 0.999)` | real label = `1`, fake label = `0` |
| WGAN | Wasserstein loss | RMSProp, `lr=0.00005` | `n_critic=5`, `clip_value=0.01` |

## Running Training

Edit `experiments/config.py` and select one objective:

```python
OBJECTIVE = "vanilla"
```

Then run:

```bash
python experiments/main.py
```

Repeat this for:

```python
OBJECTIVE = "vanilla"
OBJECTIVE = "lsgan"
OBJECTIVE = "wgan"
```

Each run writes outputs to objective-specific folders:

```text
checkpoints/vanilla/
checkpoints/lsgan/
checkpoints/wgan/

output/logs/
output/samples/
```

## Evaluation and Figures

After training all three objectives, run the utility scripts.

Prepare the real CelebA test split for FID:

```bash
python utils/prepare_fid_real.py
```

Generate 5,000 fake images per trained objective:

```bash
python utils/generate_fid_samples.py
```

Compute FID with `torch-fidelity`:

```bash
python utils/compute_fid.py
```

Plot epoch-averaged loss curves:

```bash
python utils/plot_curves.py
```

Generate latent-space interpolation grids:

```bash
python utils/make_interpolation.py
```

Generate the final side-by-side comparison figure:

```bash
python utils/make_final_comparison.py
```

## Output Files

Typical generated outputs:

```text
output/
  logs/
    vanilla.csv
    vanilla_epoch.csv
    lsgan.csv
    lsgan_epoch.csv
    wgan.csv
    wgan_epoch.csv
  samples/
    vanilla/
    lsgan/
    wgan/
  fid/
    real_test/
    fake_vanilla_5000/
    fake_lsgan_5000/
    fake_wgan_5000/
    fid_results.csv
  interpolation/
    vanilla_interp.png
    lsgan_interp.png
    wgan_interp.png
  figures/
    loss_curves.png
    final_comparison.png
```

## Notes

- No extra data augmentation is used in the main experiments.
- Images are center-cropped, resized to `64 x 64`, and normalized to `[-1, 1]`.
- The training dataset can be preloaded into memory by default to reduce image
  loading overhead on machines with sufficient RAM.
- WGAN uses weight clipping because this is the required version for the main
  coursework comparison.
