# DTS410 课程作业二小组报告

**题目：基于 CelebA 的 Vanilla GAN、LSGAN 与 WGAN 目标函数比较**  
**数据集：CelebA Align & Cropped, 64 × 64 RGB**  
**小组成员学生 ID：TODO：请在导出 PDF 前填写全部组员学生 ID**  
**代码仓库：** `Shelter995/DTS410_CW2`  
**报告版本：** 2026-05-28

> 说明：本报告按作业 Task 1-6 的评分结构组织。所有关键结果均来自 `output/` 目录下已保存的日志、样本网格、插值图、FID 结果和对比图。真实 FID 图片文件夹与每个模型的 5,000 张生成图片未放入报告，但 FID 数值、采样设置和可视化输出已保存，足以在不重新训练模型的情况下核查主要结果。

<div style="page-break-after: always;"></div>

## 摘要

本实验从零实现并训练了三种 GAN 目标函数：Vanilla GAN、Least-Squares GAN（LSGAN）和带权重裁剪的 Wasserstein GAN（WGAN）。为了保证受控比较，三种方法使用完全相同的 DCGAN-style 生成器和判别器/critic 架构，潜在向量维度均为 128，训练数据均来自 CelebA 官方训练划分，FID 使用 CelebA 官方测试划分作为真实图像参考。

实验结果显示，Vanilla GAN 在本次设置下取得最低 FID（22.66），最终样本较清晰且多样性较好；LSGAN 的 FID 为 27.39，训练曲线较平滑，但生成样本略偏平滑；WGAN with weight clipping 的 FID 为 57.87，样本中出现更多颜色块、局部伪影和不自然纹理。综合 FID、最终样本和插值结果，Vanilla GAN 在本实验预算下取得了最好的质量-稳定性平衡。

## Task 1：GAN 架构实现

### 1.1 受控实验原则

三种目标函数只改变训练目标和优化器设置，不改变网络结构。共同设置如下：

| 项目 | 设置 |
|---|---|
| 数据集 | CelebA Align & Cropped |
| 图像大小 | 64 × 64 RGB |
| 潜在向量 | $z \in \mathbb{R}^{128}$ |
| 潜在分布 | $z \sim \mathcal{N}(0, I)$ |
| 生成器 | DCGAN-style transposed convolution network |
| 判别器/critic | DCGAN-style convolution network |
| 输出归一化 | 真实图像和生成图像均使用 `[-1, 1]` 范围 |

### 1.2 生成器 G 架构

生成器输入为 128 维标准正态噪声，先 reshape 为 `(128, 1, 1)`，再通过转置卷积逐步上采样到 `64 × 64 × 3`。最后一层使用 `Tanh`，使输出范围与真实图像归一化范围一致。

| 层 | 输入尺寸 | 输出尺寸 | 通道数 | Kernel / Stride / Padding | 激活函数 |
|---|---:|---:|---:|---|---|
| Input reshape | `(N, 128)` | `(N, 128, 1, 1)` | 128 | - | - |
| ConvTranspose2d + BN | `1 × 1` | `4 × 4` | 512 | `4 / 1 / 0` | ReLU |
| ConvTranspose2d + BN | `4 × 4` | `8 × 8` | 256 | `4 / 2 / 1` | ReLU |
| ConvTranspose2d + BN | `8 × 8` | `16 × 16` | 128 | `4 / 2 / 1` | ReLU |
| ConvTranspose2d + BN | `16 × 16` | `32 × 32` | 64 | `4 / 2 / 1` | ReLU |
| ConvTranspose2d | `32 × 32` | `64 × 64` | 3 | `4 / 2 / 1` | Tanh |

### 1.3 判别器 / Critic 架构

判别器输入为 `64 × 64 × 3` 图像，经过 5 层卷积下采样为单个标量。模型最后不显式加入 sigmoid：

- Vanilla GAN：该标量作为 raw logits，送入 `BCEWithLogitsLoss`，等价于 sigmoid + BCE。
- LSGAN：该标量作为 least-squares 回归分数。
- WGAN：该标量作为 critic score，不解释为概率。

| 层 | 输入尺寸 | 输出尺寸 | 通道数 | Kernel / Stride / Padding | 激活函数 |
|---|---:|---:|---:|---|---|
| Conv2d | `64 × 64` | `32 × 32` | 64 | `4 / 2 / 1` | LeakyReLU(0.2) |
| Conv2d + BN | `32 × 32` | `16 × 16` | 128 | `4 / 2 / 1` | LeakyReLU(0.2) |
| Conv2d + BN | `16 × 16` | `8 × 8` | 256 | `4 / 2 / 1` | LeakyReLU(0.2) |
| Conv2d + BN | `8 × 8` | `4 × 4` | 512 | `4 / 2 / 1` | LeakyReLU(0.2) |
| Conv2d | `4 × 4` | `1 × 1` | 1 | `4 / 1 / 0` | None |

### 1.4 初始化

卷积层权重使用 DCGAN 常见初始化：

$$
W_{conv} \sim \mathcal{N}(0, 0.02)
$$

BatchNorm 权重初始化为：

$$
W_{BN} \sim \mathcal{N}(1, 0.02), \quad b_{BN} = 0
$$

三种目标函数从相同随机种子和相同架构出发，以减少初始化带来的随机差异。

## Task 2：三种 GAN 目标函数

### 2.1 Vanilla GAN

Vanilla GAN 的原始 minimax 目标为：

$$
\min_G \max_D V(D,G)
= \mathbb{E}_{x \sim p_{data}}[\log D(x)]
+ \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]
$$

实现中使用 `BCEWithLogitsLoss`。这意味着判别器网络本身输出 raw logits，loss 内部完成 sigmoid 和二元交叉熵计算。这样比手动 `Sigmoid + BCELoss` 更稳定。

判别器损失：

$$
\mathcal{L}_D =
\text{BCEWithLogits}(D(x), 1)
+ \text{BCEWithLogits}(D(G(z)), 0)
$$

生成器损失使用 non-saturating 形式：

$$
\mathcal{L}_G =
\text{BCEWithLogits}(D(G(z)), 1)
$$

### 2.2 LSGAN

LSGAN 将二元交叉熵替换为最小二乘损失。判别器不再输出概率，而是学习让真实图像分数接近 1、生成图像分数接近 0。

判别器损失：

$$
\mathcal{L}_D =
\frac{1}{2}\mathbb{E}_{x \sim p_{data}}[(D(x)-1)^2]
+ \frac{1}{2}\mathbb{E}_{z \sim p_z}[D(G(z))^2]
$$

生成器损失：

$$
\mathcal{L}_G =
\frac{1}{2}\mathbb{E}_{z \sim p_z}[(D(G(z))-1)^2]
$$

相比 BCE，least-squares 目标在生成样本离判别边界较远时仍可提供较直接的梯度信号，因此通常能缓解部分梯度饱和问题。本实验中 LSGAN 未使用 label smoothing，真实标签为 1，生成标签为 0。

### 2.3 WGAN with Weight Clipping

WGAN 使用 Wasserstein 距离思想，将判别器改为 critic。Critic 不输出真假概率，而是输出无界实值分数。目标为最大化真实图像和生成图像之间的 critic 分数差：

$$
\max_D \mathbb{E}_{x \sim p_{data}}[D(x)]
- \mathbb{E}_{z \sim p_z}[D(G(z))]
$$

实现中最小化：

$$
\mathcal{L}_D =
\mathbb{E}_{z \sim p_z}[D(G(z))]
- \mathbb{E}_{x \sim p_{data}}[D(x)]
$$

生成器最小化：

$$
\mathcal{L}_G =
-\mathbb{E}_{z \sim p_z}[D(G(z))]
$$

WGAN 移除 sigmoid 的原因是：critic 的输出不是概率，而是用于估计 Wasserstein 距离的实值函数。如果加入 sigmoid，输出会被限制到 `[0, 1]`，不符合 critic score 的设计，也会削弱分数差的表达能力。

WGAN 理论要求 critic 满足 Lipschitz 约束。本实验按作业要求使用 weight clipping，在每次 critic 更新后将 critic 参数裁剪到：

$$
w \in [-0.01, 0.01]
$$

该方法简单直接，但也会限制 critic 表达能力，可能导致训练不足或样本质量下降。

### 2.4 训练伪代码

```text
Algorithm: Train GAN Objective
Input: objective in {Vanilla, LSGAN, WGAN}
Initialize G and D/Critic with DCGAN weights
for epoch = 1 to 50:
    for real_batch in CelebA train loader:
        # Update D/Critic
        z ~ N(0, I)
        fake_batch = G(z).detach()
        compute D/Critic loss according to objective
        update D/Critic optimizer

        if objective is WGAN:
            clip critic weights to [-0.01, 0.01]
            update G once every 5 critic updates
        else:
            update G once every D update

        # Update G
        z ~ N(0, I)
        fake_batch = G(z)
        compute G loss according to objective
        update G optimizer

    save epoch-averaged losses
    if epoch in {10, 25, 50}:
        save checkpoint and fixed-noise sample grid
```

## Task 3：训练动态、样本演化与稳定性

### 3.1 训练设置

| 项目 | Vanilla GAN | LSGAN | WGAN |
|---|---:|---:|---:|
| Epochs | 50 | 50 | 50 |
| Batch size | 256 | 256 | 256 |
| Latent dim | 128 | 128 | 128 |
| D/Critic steps | 31,750 | 31,750 | 31,750 |
| G steps | 31,750 | 31,750 | 6,350 |
| Optimizer | Adam | Adam | RMSProp |
| Learning rate | 0.0002 | 0.0002 | 0.00005 |
| Special setting | BCE logits | MSE objective | `n_critic=5`, `clip=0.01` |

训练集每个 epoch 约 635 个 batch。Vanilla GAN 和 LSGAN 每个 batch 更新一次 D 和一次 G，因此 50 epoch 后 D/G steps 均为 31,750。WGAN 每 5 次 critic 更新后更新一次 G，因此 50 epoch 后 critic steps 为 31,750，G steps 为 6,350。

### 3.2 损失曲线

<img src="../output/figures/loss_curves.png" width="760">

**图 1：三种目标函数的 epoch-averaged D/Critic loss 与 G loss。**

观察：

- Vanilla GAN 的 G loss 初期从 6.06 快速下降到约 2-3，随后后期上升到 3.67；D loss 中后期存在震荡，说明 D/G 对抗强度持续变化。
- LSGAN 的 D loss 从 1.48 快速下降，并长期维持在较低区间；G loss 也从 5.22 降到 0.4 左右，曲线整体比 Vanilla 更平滑。
- WGAN 的 critic loss 为负值，表示真实样本 critic score 高于生成样本。其 loss 从约 -1.1 逐渐回到 -0.60，G loss 从约 0.58 降到 0.33。由于 WGAN loss 的数值含义与 BCE/MSE 不同，不能直接与 Vanilla/LSGAN 的 loss 大小比较。

### 3.3 训练日志摘要

| Objective | Epoch 10 D/Critic loss | Epoch 10 G loss | Epoch 25 D/Critic loss | Epoch 25 G loss | Epoch 50 D/Critic loss | Epoch 50 G loss |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla GAN | 0.7731 | 2.2547 | 0.6232 | 2.6154 | 0.3818 | 3.6695 |
| LSGAN | 0.1774 | 0.4195 | 0.1822 | 0.3421 | 0.1079 | 0.3991 |
| WGAN | -0.9724 | 0.5159 | -0.8361 | 0.4445 | -0.6041 | 0.3270 |

### 3.4 样本演化

以下样本网格均由固定 latent vectors 生成，因此同一 objective 内不同 epoch 的变化反映了训练过程中的生成器演化。

| Objective | Epoch 10 | Epoch 25 | Epoch 50 |
|---|---|---|---|
| Vanilla GAN | <img src="../output/samples/vanilla/fixed_epoch_010.png" width="210"> | <img src="../output/samples/vanilla/fixed_epoch_025.png" width="210"> | <img src="../output/samples/vanilla/fixed_epoch_050.png" width="210"> |
| LSGAN | <img src="../output/samples/lsgan/fixed_epoch_010.png" width="210"> | <img src="../output/samples/lsgan/fixed_epoch_025.png" width="210"> | <img src="../output/samples/lsgan/fixed_epoch_050.png" width="210"> |
| WGAN | <img src="../output/samples/wgan/fixed_epoch_010.png" width="210"> | <img src="../output/samples/wgan/fixed_epoch_025.png" width="210"> | <img src="../output/samples/wgan/fixed_epoch_050.png" width="210"> |

**图 2：三种 objective 在 epoch 10、25、50 的固定噪声样本网格。**

样本演化观察：

- Vanilla GAN 在 epoch 10 已生成清晰人脸结构，后续细节和身份多样性提升，但局部仍有不自然纹理和背景伪影。
- LSGAN 的样本整体稳定，脸部轮廓较自然，但部分样本出现平滑化倾向，细节锐度略弱于 Vanilla GAN。
- WGAN with weight clipping 能生成可识别人脸，但色块、局部形变和背景伪影更明显，说明 critic 可能受到权重裁剪限制，未能充分提供高质量梯度。

### 3.5 最终样本横向对比

<img src="../output/figures/final_comparison.png" width="760">

**图 3：最终样本对比。三行分别为 Vanilla GAN、LSGAN 和 WGAN，每种 objective 展示 12 张样本。**

最终对比中，Vanilla GAN 和 LSGAN 的人脸结构更完整，五官位置较稳定。WGAN 的样本多样性仍存在，但局部纹理、颜色和脸部边界更不稳定。结合 FID 结果，最终定性排序与量化排序基本一致。

## Task 4：FID 评估与受控比较

### 4.1 FID 流程

FID 使用 CelebA 官方测试集作为真实图像参考。流程如下：

```text
Algorithm: FID Evaluation
Input: trained final checkpoint for each objective
Prepare real images:
    read CelebA official test split where partition = 2
    center crop to 178 x 178
    resize to 64 x 64
    save as PNG into output/fid/real_test

For each objective:
    load final generator checkpoint
    sample 5,000 latent vectors z ~ N(0, I)
    generate 5,000 fake images
    save fake images into output/fid/fake_<objective>_5000
    compute FID between real_test and fake_<objective>_5000 using torch-fidelity
```

所有 objective 使用相同真实图像参考、相同 fake 样本数量、相同 latent 分布、相同图像尺寸和相同 FID 工具。

### 4.2 FID 结果

| Objective | FID ↓ | 排名 | 定性观察 |
|---|---:|---:|---|
| Vanilla GAN | **22.6565** | 1 | 最清晰，结构和多样性较好，但仍有少量伪影 |
| LSGAN | 27.3852 | 2 | 稳定、自然，但部分样本略平滑 |
| WGAN with weight clipping | 57.8744 | 3 | 伪影和颜色块更多，局部结构不稳定 |

FID 越低表示生成分布与真实测试集分布越接近。本实验中 Vanilla GAN 明显优于 LSGAN 和 WGAN，LSGAN 次之，WGAN with weight clipping 最差。

### 4.3 FID 与定性观察的一致性

FID 排名与最终样本观察基本一致。Vanilla GAN 的最终样本脸部结构、清晰度和身份多样性较好，因此取得最低 FID。LSGAN 的结果略平滑但整体稳定，FID 稍高于 Vanilla。WGAN 样本中可见更多高频噪声、颜色块和不自然脸部边界，FID 最高。

需要注意的是，FID 并不完全等价于人眼质量评价。例如某些 WGAN 样本具有较明显的人脸结构，但由于整体颜色统计、纹理分布和局部伪影偏离真实图像较大，因此 FID 仍然较高。

## Task 5：潜在空间插值与表示分析

### 5.1 插值设置

对每种 objective，采样同一对潜在向量：

$$
z_1, z_2 \sim \mathcal{N}(0, I)
$$

然后进行线性插值：

$$
z_{\alpha} = (1 - \alpha)z_1 + \alpha z_2,\quad \alpha \in [0,1]
$$

本实验使用 11 个点：

$$
\alpha = 0.0, 0.1, 0.2, \ldots, 1.0
$$

其中 0.1 到 0.9 共 9 个中间样本，满足题目要求的“至少 8 个中间插值样本”。

### 5.2 插值结果

**Vanilla GAN 插值：**

<img src="../output/interpolation/vanilla_interp.png" width="760">

Vanilla GAN 的插值从女性脸逐渐过渡到男性脸，发色、脸型和性别特征变化较连续。中间阶段没有明显突变，但局部皮肤纹理略有平滑。

**LSGAN 插值：**

<img src="../output/interpolation/lsgan_interp.png" width="760">

LSGAN 的插值较平滑，从金发女性逐渐过渡到深发女性，姿态和脸部轮廓变化自然。整体语义变化连续，但细节锐度略弱，部分中间样本呈现“柔化”效果。

**WGAN 插值：**

<img src="../output/interpolation/wgan_interp.png" width="760">

WGAN 插值也保持了大体连续的人脸变化，但前几个样本中可见较强阴影、颜色偏移和局部纹理伪影。后半段脸部结构趋于稳定，但整体质量弱于 Vanilla GAN 和 LSGAN。

### 5.3 插值比较

| Objective | 平滑性 | 语义变化 | 伪影行为 |
|---|---|---|---|
| Vanilla GAN | 较平滑 | 性别、发色、脸型变化明显且连续 | 少量局部纹理伪影 |
| LSGAN | 最平滑之一 | 发色和脸型变化自然 | 较少突变，但略偏平滑 |
| WGAN | 基本连续 | 脸部身份逐渐变化 | 阴影、颜色和局部结构伪影较明显 |

总体上，三种模型均学习到了较连续的潜在空间表示。LSGAN 插值最平滑，Vanilla GAN 在平滑性和细节之间更平衡，WGAN 虽能保持语义连续，但伪影更多。

## Task 6：讨论与反思

### 6.1 最佳目标函数平衡

综合 FID、最终样本、训练曲线和插值表现，本实验中 **Vanilla GAN** 在稳定性、样本质量和多样性之间取得了最佳平衡。它的 FID 最低，为 22.6565，最终样本中五官结构和身份多样性都较好。虽然 Vanilla GAN 的 loss 曲线比 LSGAN 更震荡，后期 G loss 上升明显，但这种对抗压力并未导致明显模式崩塌，反而在最终视觉质量上表现较好。

LSGAN 的优势是训练曲线较平滑，插值连续性好，但样本细节略少，FID 高于 Vanilla GAN。WGAN with weight clipping 理论上具有更平滑的分布距离度量，但本实验中受到 weight clipping 的表达能力限制，最终样本和 FID 都弱于另外两种方法。

### 6.2 失败案例分析

本实验中最明显的失败案例来自 **WGAN with weight clipping**。在最终样本网格中，WGAN 能生成可识别人脸，但部分样本存在：

- 局部颜色块；
- 过强阴影；
- 脸部边界不自然；
- 头发和背景混叠；
- 个别样本五官结构变形。

该问题与 WGAN 的训练动态有关。Weight clipping 通过强制将 critic 参数限制在 `[-0.01, 0.01]` 来近似 Lipschitz 约束，但这种方法较粗糙。如果裁剪范围过小，critic 表达能力会被限制，难以学习复杂的人脸分布；如果裁剪范围过大，则 Lipschitz 约束不足，训练可能不稳定。本实验采用原始 WGAN 常见设置 `clip_value=0.01`，最终结果说明该设置在 CelebA 64×64 人脸生成任务上可能限制了 critic 的有效建模能力。

### 6.3 改进方向

一个直接改进是使用 **WGAN-GP（WGAN with Gradient Penalty）**。WGAN-GP 不直接裁剪权重，而是在 loss 中加入梯度惩罚项：

$$
\lambda \mathbb{E}_{\hat{x}}
\left[
\left(\lVert \nabla_{\hat{x}}D(\hat{x}) \rVert_2 - 1\right)^2
\right]
$$

其中 $\hat{x}$ 是真实图像和生成图像之间的插值样本。该方法更直接地约束 critic 关于输入的梯度范数，使其满足 Lipschitz 条件，同时避免 weight clipping 对模型容量的过度限制。预期效果包括：

- 提高 critic 表达能力；
- 减少训练不稳定；
- 降低局部伪影；
- 改善 FID；
- 提升插值平滑性。

除 WGAN-GP 外，还可以考虑 spectral normalization、instance noise、数据增强或更长训练时间。但为了保持本作业三种主目标函数比较的可控性，这些技术未加入主实验。

## 结论

本作业实现并比较了三种 GAN 目标函数在 CelebA 64×64 人脸生成任务上的表现。实验表明，在相同 DCGAN 架构、相同预处理和相同训练预算下，Vanilla GAN 取得最佳 FID 和较好的视觉质量；LSGAN 训练曲线较稳定，插值平滑，但最终 FID 略高；WGAN with weight clipping 在本实验中表现最弱，主要原因可能是权重裁剪限制了 critic 的表达能力。

最终排序为：

$$
\text{Vanilla GAN} \;>\; \text{LSGAN} \;>\; \text{WGAN with weight clipping}
$$

其中 `>` 表示本实验中的综合表现更好，而不是理论上在所有任务中绝对优越。

## 附录：结果文件索引

| 内容 | 文件 |
|---|---|
| Loss 曲线 | `output/figures/loss_curves.png` |
| 最终对比图 | `output/figures/final_comparison.png` |
| FID 结果 | `output/fid/fid_results.csv` |
| Vanilla 样本网格 | `output/samples/vanilla/fixed_epoch_010.png`, `025.png`, `050.png` |
| LSGAN 样本网格 | `output/samples/lsgan/fixed_epoch_010.png`, `025.png`, `050.png` |
| WGAN 样本网格 | `output/samples/wgan/fixed_epoch_010.png`, `025.png`, `050.png` |
| Vanilla 插值 | `output/interpolation/vanilla_interp.png` |
| LSGAN 插值 | `output/interpolation/lsgan_interp.png` |
| WGAN 插值 | `output/interpolation/wgan_interp.png` |
| 训练日志 | `output/logs/*_epoch.csv`, `output/logs/*.csv` |

