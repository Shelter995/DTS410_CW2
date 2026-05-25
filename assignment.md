## Objective（目标）

在本次课程作业中，你们小组需要从零开始实现并训练 GAN，用于基于 CelebA 数据集的 Align & Cropped 版本（分辨率为 64 × 64）生成人脸图像。你们需要比较三种 GAN 目标函数（Vanilla GAN、LSGAN 和 WGAN），使用官方测试集上的 FID 指标评估生成质量，并分析潜在空间插值（latent space interpolation）行为。所有模型变体必须使用相同的生成器/判别器架构、相同的潜在向量维度（128），以及相同的图像预处理方式。

---

## Tasks（任务）

### Task 1: Implement the GAN Architecture [10 Marks]  
### 任务 1：实现 GAN 架构【10 分】

- 实现一个生成器 G，将 z ∈ R128 映射为一张 64 × 64 的 RGB 图像，并使用卷积架构。需要提供清晰的架构表，包括层、通道数和激活函数。【5 分】
- 实现一个适用于 64 × 64 RGB 图像的判别器 D。需要提供清晰的架构表，包括层、通道数和激活函数。【5 分】

---

### Task 2: Implement and Compare Three GAN Objectives [20 Marks]  
### 任务 2：实现并比较三种 GAN 目标函数【20 分】

- Vanilla GAN：实现标准 GAN 目标函数，并描述其 minimax 形式。【5 分】
- Least-Squares GAN (LSGAN)：实现最小二乘对抗目标函数，并解释它如何改变梯度行为。【7 分】
- Wasserstein GAN (WGAN)：实现带权重裁剪的 Wasserstein GAN 目标函数。清楚解释：
  1. 为什么移除 sigmoid 激活函数；
  2. Lipschitz 约束的作用。【8 分】

---

### Task 3: Training Dynamics, Sample Evolution, and Stability [25 Marks]  
### 任务 3：训练动态、样本演化与稳定性【25 分】

- 对每种目标函数，报告生成器和判别器的损失训练曲线。清楚评论收敛行为和不稳定性。【10 分】
- 对每种目标函数，在至少三个检查点（早期、中期、最终训练阶段）提供定性样本网格图。【10 分】
- 展示一张最终对比图，将三种目标函数生成的样本并排比较。【5 分】

---

### Task 4: FID Evaluation and Controlled Comparison [20 Marks]  
### 任务 4：FID 评估与受控比较【20 分】

FID 是强制要求的，必须使用官方测试集作为真实图像参考来计算。

- 清楚说明你的 FID 评估流程：生成样本数量（例如 5,000）、预处理步骤和实现工具。所有目标函数必须使用相同流程。【4 分】
- 用结构化对比表报告 Vanilla GAN、LSGAN 和 WGAN 的 FID。【8 分】
- 讨论 FID 排名是否与定性观察结果一致。【8 分】

---

### Task 5: Latent Space Interpolation and Representation Analysis [10 Marks]  
### 任务 5：潜在空间插值与表示分析【10 分】

- 对每种目标函数，在两个随机生成的潜在向量之间执行线性插值：

  $$
  z_{\alpha} = (1 - \alpha)z_1 + \alpha z_2, \alpha \in [0, 1]
  $$

- 每种目标函数至少生成 8 个中间插值样本。【5 分】
- 比较不同目标函数之间插值的平滑性、语义变化以及伪影（artifact）行为。【5 分】

---

### Task 6: Discussion and Reflection [15 Marks]  
### 任务 6：讨论与反思【15 分】

- 哪一种目标函数在稳定性、样本质量和多样性之间取得了最佳平衡？【5 分】
- 展示一个 GAN 训练中的明显失败案例，例如模式崩塌、伪影或不稳定插值等。解释该失败产生的原因，并将其与目标函数或训练动态联系起来。【5 分】
- 提出一种可以提升三种 GAN 性能的技术，并简要说明为什么它能够改善稳定性或降低 FID。【5 分】