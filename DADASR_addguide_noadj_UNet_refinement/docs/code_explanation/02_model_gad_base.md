# `model/gad_base.py` 代码说明书

源码位置：[model/gad_base.py](/Users/niko/Downloads/project/model/gad_base.py)

这个文件是项目的模型核心。它包含三类东西：

1. `LocalRefinementNet`：当前 refinement-only 训练时最重要的 DSM residual 预测网络。
2. `RGBDSMFeatureFusion`：把 RGB 和 LR DSM 的局部关系融合进 guide feature。
3. `GADBase` 和 diffusion：整体模型入口，负责组织 feature extraction、refinement 和 diffusion。

当前你主要使用的训练路径是：

```text
GADBase.forward()
  -> extract_features()
  -> RGBDSMFeatureFusion，可选
  -> LocalRefinementNet
  -> refinement_only=True 时直接返回 y_pred
```

## 顶部导入和常量

```python
from random import randrange

import torch
from torch import nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
```

技术解释：

- `randrange` 用于训练时随机选择 diffusion 的无梯度迭代次数。
- `torch` 和 `nn` 用于搭建网络。
- `F` 用于插值、归一化、unfold 等函数式操作。
- `segmentation_models_pytorch` 提供 `smp.Unet`。

```python
DEPTH_INPUT_CHANNELS = 1
FEATURE_DIM = 64
```

技术解释：

- DSM 是单通道，所以 depth input channel 是 1。
- 默认特征通道数是 64。

通俗解释：

```text
模型处理的高程图是单层灰度图，不是 RGB 三通道图。
默认中间特征用 64 个通道表达。
```

## `ResidualBlock`

源码位置：[model/gad_base.py:12](/Users/niko/Downloads/project/model/gad_base.py:12)

```python
class ResidualBlock(nn.Module):
```

这是一个基础残差块。

```python
self.block = nn.Sequential(
    nn.Conv2d(channels, channels, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(channels, channels, kernel_size=3, padding=1),
)
```

技术解释：

- 两个 3x3 卷积，通道数不变。
- 中间有 ReLU。
- `padding=1` 保持 H/W 不变。

```python
def forward(self, x):
    return x + self.block(x)
```

技术解释：

- 输出是输入 `x` 加上卷积块学到的变化量。
- 这就是 residual learning。

通俗解释：

```text
这个块不是完全重写特征，而是在原特征基础上学一个“修正量”。
对 DSM residual refinement 来说，这种思想和 y_bicubic + residual 是一致的。
```

## `SemanticFeatureEncoder`

源码位置：[model/gad_base.py:26](/Users/niko/Downloads/project/model/gad_base.py:26)

这个模块把 semantic masks 编码成可用于调制网络的语义特征。

```python
if semantic_channels <= 0:
    raise ValueError(...)
```

技术解释：语义特征通道数必须为正。

```python
self.encoder = nn.Sequential(
    nn.Conv2d(3, semantic_channels, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(semantic_channels, semantic_channels, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
)
```

技术解释：

- 输入通道是 3。
- 这 3 个通道分别是 building mask、road mask、valid mask。
- 输出通道是 `semantic_channels`。

```python
inputs = torch.cat(
    [semantic_masks * semantic_valid_mask, semantic_valid_mask],
    dim=1,
)
```

技术解释：

- `semantic_masks` 是 `[B,2,H,W]`。
- `semantic_valid_mask` 是 `[B,1,H,W]`。
- `semantic_masks * semantic_valid_mask` 会把 ignore 区域的 building/road 信号清零。
- 拼接后得到 `[B,3,H,W]`。

通俗解释：

```text
模型不应该相信 ignore 区域的语义。
所以先用 valid mask 把无效区域盖掉，再把 valid mask 本身也告诉网络。
```

```python
return self.encoder(inputs) * semantic_valid_mask
```

技术解释：编码后再次乘 valid mask，确保无效区域不会产生调制信号。

通俗解释：

```text
无效语义区域不仅输入被清掉，输出调制也被压成 0。
```

## `SemanticFiLMResidualBlock`

源码位置：[model/gad_base.py:50](/Users/niko/Downloads/project/model/gad_base.py:50)

这是带语义 FiLM/gate 调制的 residual block。

```python
self.film = nn.Conv2d(semantic_channels, 2 * channels, kernel_size=1)
```

技术解释：

- 从语义特征预测 `gamma` 和 `beta`。
- 输出通道是 `2 * channels`，后面一分为二。

```python
self.gate = nn.Sequential(
    nn.Conv2d(semantic_channels, channels, kernel_size=1),
    nn.Sigmoid(),
)
```

技术解释：

- 从语义特征预测 0-1 的 gate。
- gate 控制 residual block 的输出强度。

forward：

```python
gamma, beta = self.film(semantic_features).chunk(2, dim=1)
```

技术解释：把 FiLM 输出拆成两个 `[B,C,H,W]` 张量。

```python
modulated_x = x * (1.0 + torch.tanh(gamma)) + beta
```

技术解释：

- `gamma` 控制缩放。
- `beta` 控制平移。
- `tanh(gamma)` 把缩放限制在相对稳定范围。

通俗解释：

```text
语义信息不是直接预测高度，而是告诉特征：
这里像建筑，特征可以加强或削弱某些通道；
这里像道路，另一些通道可以被激活。
```

```python
gate = self.gate(semantic_features)
return x + gate * self.block(modulated_x), gate
```

技术解释：

- residual branch 先经过语义调制。
- gate 再控制 residual 改动幅度。
- 返回 block 输出和 gate，gate 会放入 aux 方便分析。

通俗解释：

```text
FiLM 决定“怎么改特征”，gate 决定“改多少”。
```

## `ConvRelu`

源码位置：[model/gad_base.py:73](/Users/niko/Downloads/project/model/gad_base.py:73)

```python
self.block = nn.Sequential(
    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
    nn.ReLU(inplace=True),
)
```

技术解释：

- 一个 3x3 卷积接 ReLU。
- `stride=1` 时保持尺寸。
- `stride=2` 时下采样。

通俗解释：

```text
这是“卷积 + 激活”的小积木，后面 encoder/decoder 会反复用。
```

## `DownBlock`

源码位置：[model/gad_base.py:86](/Users/niko/Downloads/project/model/gad_base.py:86)

```python
self.downsample = ConvRelu(channels, channels, stride=2)
self.residual_blocks = nn.Sequential(
    *[ResidualBlock(channels) for _ in range(num_blocks)]
)
```

技术解释：

- 先 stride-2 conv 把空间尺寸约减半。
- 再接若干 residual blocks。
- 默认 `num_blocks=1`。

通俗解释：

```text
DownBlock 负责进入更低分辨率，让模型看到更大范围的上下文。
```

## `UpBlock`

源码位置：[model/gad_base.py:99](/Users/niko/Downloads/project/model/gad_base.py:99)

```python
x = F.interpolate(
    x,
    size=skip.shape[-2:],
    mode='bilinear',
    align_corners=False,
)
```

技术解释：

- 把低分辨率特征上采样到 skip feature 的尺寸。
- 使用 `size=`，不是 `scale_factor=2`，因此兼容 `250` 这种不是 4 的倍数的 crop。

通俗解释：

```text
不要猜上采样后应该多大，直接对齐 skip 的尺寸。
这样 63 -> 125、125 -> 250 都不会错。
```

```python
x = torch.cat([x, skip], dim=1)
```

技术解释：在通道维拼接 decoder 特征和 encoder skip 特征。

```python
return self.residual_block(self.projection(x))
```

技术解释：

- `projection` 把 `2C` 通道压回 `C`。
- residual block 再做一次局部细化。

通俗解释：

```text
decoder 不是只靠低分辨率特征放大，还把早期高分辨率细节接回来。
```

## `LocalRefinementNet`

源码位置：[model/gad_base.py:117](/Users/niko/Downloads/project/model/gad_base.py:117)

这是 refinement-only 训练时的最终预测核心。

### 构造参数

```python
in_channels
channels=FEATURE_DIM
num_blocks=4
boundary_refinement=False
semantic_modulation=False
semantic_channels=32
```

含义：

- `in_channels`：输入 feature 的通道数。
- `channels`：内部隐藏通道数。
- `num_blocks`：bottleneck residual block 数。
- `boundary_refinement`：是否使用 flat/edge 双 residual head。
- `semantic_modulation`：是否使用语义 FiLM/gate 调制。
- `semantic_channels`：语义编码器输出通道。

### `__init__`

```python
self.in_projection = ConvRelu(in_channels, channels)
self.encoder_stage2 = DownBlock(channels)
self.encoder_stage3 = DownBlock(channels)
```

技术解释：

- `in_projection` 把外部 feature 投影到 refinement 内部通道数。
- `encoder_stage2` 下采样到约 `H/2`。
- `encoder_stage3` 下采样到约 `H/4`。

通俗解释：

```text
e1 保留原分辨率细节；
e2 看稍大范围；
e3/bottleneck 看更大上下文。
```

语义分支：

```python
if semantic_modulation:
    self.semantic_encoder = SemanticFeatureEncoder(semantic_channels)
    self.semantic_bottleneck_blocks = nn.ModuleList(
        [SemanticFiLMResidualBlock(channels, semantic_channels) for _ in range(num_blocks)]
    )
else:
    self.bottleneck_blocks = nn.Sequential(
        *[ResidualBlock(channels) for _ in range(num_blocks)]
    )
```

技术解释：

- 开启语义调制时，bottleneck 使用 `SemanticFiLMResidualBlock`。
- 不开启时，bottleneck 使用普通 `ResidualBlock`。

通俗解释：

```text
有语义时，building/road 会参与调制 bottleneck。
没有语义时，网络就是普通 U-Net residual decoder。
```

decoder：

```python
self.decoder_stage2 = UpBlock(channels)
self.decoder_stage1 = UpBlock(channels)
```

技术解释：

- decoder stage 2：从 bottleneck 上采样到 e2 尺寸并拼 skip。
- decoder stage 1：从 d2 上采样到 e1 尺寸并拼 skip。

boundary heads：

```python
if boundary_refinement:
    self.flat_residual_head = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
    self.edge_residual_head = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
    self.boundary_gate = nn.Sequential(...)
else:
    self.residual_head = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
```

技术解释：

- boundary refinement 开启时，分别预测平坦区域 residual 和边缘区域 residual。
- `boundary_gate` 决定每个像素更相信 flat head 还是 edge head。
- 关闭时只用一个 residual head。

通俗解释：

```text
平地和边缘需要的修正方式不同。
双 head 让模型可以分别处理平坦区域和建筑/道路边界。
```

### `forward`

源码位置：[model/gad_base.py:157](/Users/niko/Downloads/project/model/gad_base.py:157)

函数签名：

```python
forward(
    features,
    initial_dsm,
    boundary_prior=None,
    semantic_masks=None,
    semantic_valid_mask=None,
)
```

输入：

- `features`: `[B,C,H,W]`
- `initial_dsm`: `[B,1,H,W]`，一般是 `y_bicubic`
- `boundary_prior`: `[B,1,H,W]`，可选
- `semantic_masks`: `[B,2,H,W]`，可选
- `semantic_valid_mask`: `[B,1,H,W]`，可选

encoder：

```python
e1 = self.in_projection(features)
e2 = self.encoder_stage2(e1)
x = self.encoder_stage3(e2)
```

技术解释：

- `e1` 是原分辨率 feature。
- `e2` 是半分辨率 feature。
- `x` 是 bottleneck 输入。

通俗解释：

```text
网络先把特征压下去，获得更大感受野；
同时保存 e1/e2，后面 decoder 恢复细节时再接回来。
```

semantic modulation：

```python
semantic_masks, semantic_valid_mask = self._resize_semantic_inputs(..., x)
semantic_features = self.semantic_encoder(semantic_masks, semantic_valid_mask)
```

技术解释：

- semantic 输入 resize 到 bottleneck 尺寸。
- semantic encoder 输出 bottleneck 尺寸的语义特征。

```python
for block in self.semantic_bottleneck_blocks:
    x, gate = block(x, semantic_features)
    semantic_gates.append(gate)
```

技术解释：

- 每个 bottleneck block 都被语义特征调制。
- 每层 gate 保存到 aux。

通俗解释：

```text
语义不是在最后硬拼结果，而是在最深层特征处理时告诉网络：
这里属于建筑/道路/背景，残差生成要有所区别。
```

普通 bottleneck：

```python
x = self.bottleneck_blocks(x)
```

技术解释：不启用语义时只做普通 residual refinement。

decoder：

```python
x = self.decoder_stage2(x, e2)
x = self.decoder_stage1(x, e1)
```

技术解释：

- 先对齐 e2 尺寸并拼接。
- 再对齐 e1 尺寸并拼接。

```python
if x.shape[-2:] != initial_dsm.shape[-2:]:
    x = F.interpolate(x, size=initial_dsm.shape[-2:], mode='bilinear', align_corners=False)
```

技术解释：最终保证特征尺寸和 `initial_dsm` 完全一致。

通俗解释：

```text
即使输入尺寸因为奇偶数导致中间变化，这里也把最终输出拉回 DSM 的尺寸。
```

单 residual head：

```python
if not self.boundary_refinement:
    residual = self.residual_head(x)
    return initial_dsm + residual, residual, semantic_aux
```

技术解释：

- 直接预测 residual。
- refined DSM 等于初始 DSM 加 residual。

boundary refinement：

```python
has_boundary_prior = boundary_prior is not None
boundary_prior = self._resize_boundary_prior(boundary_prior, x)
flat_residual = self.flat_residual_head(x)
edge_residual = self.edge_residual_head(x)
```

技术解释：

- 记录用户是否真的传入 boundary prior。
- boundary prior resize 到当前特征尺寸。
- 预测 flat 和 edge 两个 residual。

```python
learned_boundary_gate = self.boundary_gate(torch.cat([x, boundary_prior], dim=1))
```

技术解释：

- 把 decoder 特征和 boundary prior 拼接。
- 网络学习一个 0-1 gate。

```python
boundary_gate = (
    0.5 * (learned_boundary_gate + boundary_prior)
    if has_boundary_prior
    else learned_boundary_gate
)
```

技术解释：

- 如果有外部 boundary prior，则把学习到的 gate 和先验平均。
- 如果没有先验，只用网络自己学到的 gate。

通俗解释：

```text
boundary_prior 是外部提醒：这里可能是边界。
learned_boundary_gate 是模型自己的判断。
两者平均，相当于“外部提示 + 模型判断”一起决定是否走 edge branch。
```

```python
residual = (1.0 - boundary_gate) * flat_residual + boundary_gate * edge_residual
```

技术解释：

- gate 接近 0：更相信 flat residual。
- gate 接近 1：更相信 edge residual。

通俗解释：

```text
每个像素都在两个 residual 之间做软选择。
不是硬切换，而是按权重混合。
```

aux：

```python
aux = {
    'flat_residual': flat_residual,
    'edge_residual': edge_residual,
    'boundary_gate': boundary_gate,
    'learned_boundary_gate': learned_boundary_gate,
    'boundary_prior': boundary_prior,
}
aux.update(semantic_aux)
return initial_dsm + residual, residual, aux
```

技术解释：

- aux 保存中间结果，方便可视化或调试。
- 返回 refined DSM、residual、aux。

## `_resize_semantic_inputs`

源码位置：[model/gad_base.py:223](/Users/niko/Downloads/project/model/gad_base.py:223)

这个函数保证 semantic 输入格式正确、尺寸正确、设备和 dtype 正确。

```python
if semantic_masks is None or semantic_valid_mask is None:
    raise ValueError(...)
```

技术解释：启用 semantic modulation 时必须提供语义输入。

```python
if semantic_masks.ndim != 4 or semantic_masks.shape[1] != 2:
    raise ValueError(...)
```

技术解释：semantic masks 必须是 `[B,2,H,W]`。

```python
if semantic_valid_mask.ndim == 3:
    semantic_valid_mask = semantic_valid_mask.unsqueeze(1)
```

技术解释：允许 `[B,H,W]` 的 valid mask，自动补成 `[B,1,H,W]`。

```python
semantic_masks = semantic_masks.to(device=reference.device, dtype=reference.dtype)
semantic_valid_mask = semantic_valid_mask.to(device=reference.device, dtype=reference.dtype)
```

技术解释：把语义输入迁移到和 bottleneck feature 一样的设备和类型。

```python
F.interpolate(..., mode='nearest')
```

技术解释：语义 mask 是类别/二值图，必须用最近邻插值。

通俗解释：

```text
building mask 不能用 bilinear 插值，否则类别边界会变成模糊小数。
```

## `_resize_boundary_prior`

源码位置：[model/gad_base.py:255](/Users/niko/Downloads/project/model/gad_base.py:255)

```python
if boundary_prior is None:
    return reference.new_zeros(reference.shape[0], 1, reference.shape[-2], reference.shape[-1])
```

技术解释：没有 boundary prior 时，用全 0 图代替。

```python
if boundary_prior.ndim == 3:
    boundary_prior = boundary_prior.unsqueeze(1)
if boundary_prior.shape[1] != 1:
    boundary_prior = boundary_prior.mean(dim=1, keepdim=True)
```

技术解释：

- 允许 `[B,H,W]` 输入。
- 如果有多通道，取均值变成单通道。

```python
F.interpolate(..., mode='bilinear')
```

技术解释：boundary prior 是连续 0-1 权重图，用 bilinear 合理。

```python
return boundary_prior.clamp(0.0, 1.0)
```

技术解释：保证 gate 先验范围在 0-1。

## `GateConv2D`

源码位置：[model/gad_base.py:273](/Users/niko/Downloads/project/model/gad_base.py:273)

```python
self.attention = nn.Sequential(
    nn.Conv2d(channels, channels, kernel_size=3, padding=1),
    nn.Sigmoid(),
)
self.feature = nn.Sequential(
    nn.Conv2d(channels, channels, kernel_size=3, padding=1),
    nn.PReLU(num_parameters=channels),
)
```

技术解释：

- `attention` 生成 0-1 权重。
- `feature` 生成变换后的特征。

```python
return self.attention(x) * self.feature(x)
```

技术解释：用 attention 控制 feature 的强弱。

通俗解释：

```text
先判断哪里重要，再把不重要的地方压低。
```

## `EITF`

源码位置：[model/gad_base.py:291](/Users/niko/Downloads/project/model/gad_base.py:291)

EITF 用于 DSM-guided local RGB similarity modulation。

```python
target_size = dsm_features.shape[-2:]
rgb_lr = F.interpolate(rgb_features, size=target_size, ...)
```

技术解释：

- RGB feature 先 resize 到 DSM feature 尺寸。
- 这样才能逐像素比较局部 patch。

```python
rgb_patches = F.unfold(rgb_lr, kernel_size=3, padding=1)
dsm_patches = F.unfold(dsm_features, kernel_size=3, padding=1)
```

技术解释：

- `unfold` 把每个位置周围 3x3 patch 拉成向量。
- RGB 和 DSM 都变成局部 patch 表示。

通俗解释：

```text
不是只看单个像素，而是看每个像素周围 3x3 小邻域。
```

```python
similarity = (F.normalize(rgb_patches, dim=1) * F.normalize(dsm_patches, dim=1)).sum(dim=1)
```

技术解释：

- 对 RGB patch 和 DSM patch 做归一化后点乘。
- 得到局部相似度。

通俗解释：

```text
如果 RGB 局部纹理和 DSM 局部变化相似，相似度就高。
```

```python
rgb_modulated = self.rgb_projection(rgb_lr * (1.0 + similarity))
```

技术解释：用相似度增强 RGB feature。

```python
rgb_modulated = F.interpolate(rgb_modulated, size=rgb_features.shape[-2:], ...)
```

技术解释：再 resize 回原 RGB feature 尺寸。

## `MMAB`

源码位置：[model/gad_base.py:330](/Users/niko/Downloads/project/model/gad_base.py:330)

MMAB 用 joint statistics 做 RGB/DSM channel gating。

```python
hidden_channels = max(8, (2 * channels) // reduction)
```

技术解释：融合后通道数是 `2*channels`，再按 reduction 降维，但最少 8。

```python
dsm_hr = F.interpolate(dsm_features, size=rgb_features.shape[-2:], ...)
fused_features = self.squeeze(torch.cat([dsm_hr, rgb_features], dim=1))
```

技术解释：

- DSM feature resize 到 RGB feature 尺寸。
- 拼接后用卷积压缩。

```python
mean = F.adaptive_avg_pool2d(fused_features, output_size=1)
variance = (fused_features - mean).square().mean(dim=(2, 3), keepdim=True)
joint_statistics = mean + variance
```

技术解释：

- `mean` 表示整体平均响应。
- `variance` 表示空间变化强度。
- 两者相加作为通道统计。

通俗解释：

```text
这个模块不仅看“这个通道平均强不强”，还看“这个通道空间变化大不大”。
```

```python
depth_gate = self.depth_excitation(joint_statistics)
rgb_gate = self.rgb_excitation(joint_statistics)
return 0.5 * depth_gate * dsm_hr, 0.5 * rgb_gate * rgb_features, depth_gate, rgb_gate
```

技术解释：

- 生成 depth gate 和 RGB gate。
- 输出被 gate 加权后的 DSM/RGB feature。
- 乘 0.5 控制融合幅度。

## `RGBDSMFeatureFusion`

源码位置：[model/gad_base.py:368](/Users/niko/Downloads/project/model/gad_base.py:368)

这个模块在 `LocalRefinementNet` 前融合显式 RGB--DSM 关系。

构造：

```python
self.rgb_stem = nn.Sequential(...)
self.dsm_stem = nn.Sequential(...)
self.rgb_gate = GateConv2D(feature_channels)
self.dsm_gate = GateConv2D(feature_channels)
self.eitf = EITF(feature_channels)
self.mmab = MMAB(feature_channels, reduction=reduction)
```

技术解释：

- RGB 和 DSM 先各自投影到 `feature_channels`。
- 各自经过 gate。
- EITF 建立局部 RGB/DSM 相似性。
- MMAB 做联合统计 gating。

forward：

```python
rgb_features = self.rgb_gate(self.rgb_stem(rgb))
dsm_features = self.dsm_gate(self.dsm_stem(dsm_lr))
```

技术解释：RGB 和 LR DSM 分别变成 feature。

```python
rgb_features, similarity = self.eitf(rgb_features, dsm_features)
dsm_features, rgb_features, depth_gate, rgb_gate = self.mmab(dsm_features, rgb_features)
```

技术解释：

- EITF 用 DSM 局部结构调制 RGB。
- MMAB 再生成两个模态的通道 gate。

```python
cross_features = self.cross_projection(torch.cat([dsm_features, rgb_features], dim=1))
```

技术解释：拼接后压回主 feature 通道。

```python
fusion_gate = self.fusion_gate(torch.cat([base_features, cross_features], dim=1))
fused_features = base_features + fusion_gate * cross_features
```

技术解释：

- `base_features` 是原始 guide feature。
- `cross_features` 是显式 RGB--DSM 融合特征。
- `fusion_gate` 决定加入多少 cross feature。

通俗解释：

```text
这不是简单替换原特征，而是在原特征上加一份经过 gate 控制的 RGB--DSM 交互信息。
```

aux：

```python
'rgb_dsm_similarity'
'rgb_dsm_depth_gate'
'rgb_dsm_rgb_gate'
'rgb_dsm_fusion_gate'
```

这些用于可视化或调试。

## `GADBase`

源码位置：[model/gad_base.py:420](/Users/niko/Downloads/project/model/gad_base.py:420)

这是整个模型的外层入口。

### `__init__`

主要参数：

```python
feature_extractor='UNet'
Npre=8000
Ntrain=1024
guide_channels=4
use_refinement_net=True
refinement_channels=FEATURE_DIM
refinement_blocks=4
refinement_only=False
boundary_refinement=False
use_cross_modal_fusion=False
semantic_modulation=False
```

技术解释：

- `feature_extractor` 控制是否用 UNet 提取 guide features。
- `Npre/Ntrain` 控制 diffusion 迭代次数。
- `use_refinement_net` 控制是否启用 `LocalRefinementNet`。
- `refinement_only` 控制是否跳过 diffusion。
- `boundary_refinement` 和 `semantic_modulation` 传给 `LocalRefinementNet`。

feature extractor 选择：

```python
if feature_extractor=='none':
    self.feature_extractor = None
    self.Ntrain = 0
    self.register_buffer('logk', torch.log(torch.tensor(0.03)))
    feature_channels = guide_channels + DEPTH_INPUT_CHANNELS
```

技术解释：

- 不用深度 feature extractor。
- 直接用 `[guide, DSM]` 作为 feature。
- `logk` 注册成 buffer，不参与训练。
- feature 通道是 `4 + 1 = 5`。

通俗解释：

```text
这是传统/简化模式：不训练 UNet，只用原始 guide 和 DSM。
```

```python
elif feature_extractor=='UNet':
    self.feature_extractor = torch.nn.Sequential(
        torch.nn.Upsample(scale_factor=1, mode='bicubic'),
        smp.Unet('resnet50', classes=FEATURE_DIM, in_channels=guide_channels + DEPTH_INPUT_CHANNELS),
        torch.nn.Identity())
    self.logk = torch.nn.Parameter(torch.log(torch.tensor(0.03)))
    feature_channels = FEATURE_DIM
```

技术解释：

- 输入通道是 guide 4 通道 + DSM 1 通道。
- 输出通道是 64。
- `logk` 是可训练参数。

通俗解释：

```text
UNet 模式会学习一张 dense feature map。
这张 feature map 后面既能给 refinement 用，也能给 diffusion 算边缘传播系数。
```

cross-modal fusion：

```python
if use_cross_modal_fusion:
    if guide_channels < 3:
        raise ValueError(...)
    self.cross_modal_fusion = RGBDSMFeatureFusion(...)
```

技术解释：cross-modal fusion 至少需要 RGB 三通道。

refinement net：

```python
if use_refinement_net:
    self.refinement_net = LocalRefinementNet(...)
```

技术解释：按配置构造 `LocalRefinementNet`，保持参数兼容。

### `forward`

源码位置：[model/gad_base.py:486](/Users/niko/Downloads/project/model/gad_base.py:486)

```python
guide = sample['guide']
y_bicubic = sample['y_bicubic'].clone()
guide_feats = self.extract_features(y_bicubic, guide.clone())
```

技术解释：

- 从 sample 取 guide 和 bicubic DSM。
- `clone()` 避免后续原地操作影响 sample。
- `extract_features` 生成 dense feature。

通俗解释：

```text
模型先拿到粗 DSM 和引导图，再把它们变成可以用于预测 residual 的特征。
```

cross-modal fusion：

```python
if self.cross_modal_fusion is not None:
    guide_feats, cross_modal_aux = self.cross_modal_fusion(
        base_features=guide_feats,
        rgb=guide[:, :3],
        dsm_lr=sample['source'],
    )
    aux.update(cross_modal_aux)
```

技术解释：

- 使用 guide 前 3 个通道作为 RGB。
- 使用 `source` 作为 LR DSM。
- 融合后的特征覆盖 `guide_feats`。
- aux 保存可视化字段。

refinement 输入选择：

```python
if self.semantic_modulation:
    self._validate_semantic_sample(sample)
    refinement_boundary_prior = sample['semantic_boundary_prior']
    semantic_masks = sample['semantic_masks']
    semantic_valid_mask = sample['semantic_valid_mask']
else:
    refinement_boundary_prior = sample.get('boundary_prior')
    semantic_masks = None
    semantic_valid_mask = None
```

技术解释：

- 开 semantic modulation 时，用 semantic boundary prior 和 semantic masks。
- 不开时，用普通 boundary prior。

通俗解释：

```text
语义模式和普通模式的边界来源不同：
语义模式更相信 building/road 类别边界；
普通模式使用旧的边缘图。
```

调用 refinement：

```python
y_init, refinement_residual, refinement_aux = self.refinement_net(...)
```

技术解释：

- `y_init` 是 refinement 后 DSM。
- `refinement_residual` 是预测的 residual。
- `refinement_aux` 是中间结果。

```python
aux['refinement_residual'] = refinement_residual
aux.update(refinement_aux)
```

技术解释：把 refinement 结果加入输出字典。

```python
if 'boundary_gate' in refinement_aux:
    boundary_gate = refinement_aux['boundary_gate']
    aux['fusion_gates'] = torch.cat([1.0 - boundary_gate, boundary_gate], dim=1)
    aux['fusion_gate_modalities'] = ('flat_residual', 'edge_residual')
```

技术解释：

- 如果 boundary refinement 开启，就把 flat/edge 两个分支的权重整理成 `fusion_gates`。
- 评估脚本可以用它保存 gate 可视化。

refinement-only：

```python
if self.refinement_only:
    return {**{'y_pred': y_init}, **aux}
```

技术解释：跳过 diffusion，直接把 refinement DSM 作为最终预测。

通俗解释：

```text
你现在主要训练的就是这个路径。
LocalRefinementNet 的输出就是最终结果。
```

diffusion：

```python
y_pred, diffusion_aux = self.diffuse(...)
aux.update(diffusion_aux)
```

技术解释：如果不是 refinement-only，则对 `y_init` 继续扩散细化。

```python
if self.refinement_net is not None:
    aux['y_refined'] = y_init
return {**{'y_pred': y_pred}, **aux}
```

技术解释：

- 最终输出必须有 `y_pred`。
- 如果走过 refinement，也保留 `y_refined` 供 loss 或分析使用。

## `_validate_semantic_sample`

源码位置：[model/gad_base.py:544](/Users/niko/Downloads/project/model/gad_base.py:544)

这个函数在开启 semantic modulation 时防止数据不符合语义 schema。

```python
schema_valid = sample.get('semantic_schema_valid')
if schema_valid is None:
    raise ValueError(...)
```

技术解释：数据集必须提供 schema 检查字段。

```python
if not schema_valid.bool().all().item():
    raise ValueError(...)
```

技术解释：batch 中只要有一个样本 schema 不合法，就报错。

```python
missing = [
    key for key in ('semantic_masks', 'semantic_valid_mask', 'semantic_boundary_prior')
    if key not in sample
]
```

技术解释：检查 semantic modulation 必需字段是否齐全。

通俗解释：

```text
开语义调制前，必须确认数据里真的有 building/road 语义，而不是旧标签或缺字段。
```

## `extract_features`

源码位置：[model/gad_base.py:570](/Users/niko/Downloads/project/model/gad_base.py:570)

```python
if self.feature_extractor is None:
    return torch.cat([guide, img], 1)
```

技术解释：无 UNet 时直接拼接 guide 和 DSM。

```python
return self.feature_extractor(torch.cat([guide, img-img.mean((1,2,3), keepdim=True)], 1))
```

技术解释：

- 输入是 guide 加上中心化后的 DSM。
- `img.mean((1,2,3), keepdim=True)` 对每个 batch 样本求 DSM 全局均值。
- `img - mean` 去掉绝对高度偏置。

通俗解释：

```text
UNet 不直接看原始绝对高度，而是看“相对平均高度的起伏”。
这样不同区域整体海拔不同，也不会让特征提取器太受影响。
```

## `diffuse`

源码位置：[model/gad_base.py:575](/Users/niko/Downloads/project/model/gad_base.py:575)

```python
if guide_feats is None:
    guide_feats = self.extract_features(img, guide)
```

技术解释：如果外部没传 feature，就现场提取。

```python
cv, ch = c(guide_feats, K=K)
```

技术解释：

- 根据 guide feature 计算纵向和横向 diffusion 系数。
- `K` 控制边缘敏感度。

通俗解释：

```text
feature 变化大的地方可能是边界，diffusion 不应该轻易跨过去。
feature 平滑的地方可以更大胆地传播。
```

```python
if self.Npre>0:
    with torch.no_grad():
        Npre = randrange(self.Npre) if train else self.Npre
        for t in range(Npre):
            img = diffuse_step(cv, ch, img, l=l)
```

技术解释：

- 前 `Npre` 次 diffusion 不记录梯度，省显存。
- 训练时随机迭代次数，增加扰动。
- 推理时固定 `Npre`。

```python
if self.Ntrain>0:
    for t in range(self.Ntrain):
        img = diffuse_step(cv, ch, img, l=l)
```

技术解释：后 `Ntrain` 次 diffusion 参与梯度计算。

```python
return img, {"cv": cv, "ch": ch}
```

技术解释：返回 diffusion 后的 DSM 和 diffusion 系数。

## `c`

源码位置：[model/gad_base.py:600](/Users/niko/Downloads/project/model/gad_base.py:600)

```python
cv = g(torch.unsqueeze(torch.mean(torch.abs(I[:,:,1:,:] - I[:,:,:-1,:]), 1), 1), K)
ch = g(torch.unsqueeze(torch.mean(torch.abs(I[:,:,:,1:] - I[:,:,:,:-1]), 1), 1), K)
```

技术解释：

- `I[:,:,1:,:] - I[:,:,:-1,:]` 计算上下相邻 feature 差，得到纵向边缘强度。
- `I[:,:,:,1:] - I[:,:,:,:-1]` 计算左右相邻 feature 差，得到横向边缘强度。
- 对通道求平均。
- 传给 `g` 转成 diffusion coefficient。

通俗解释：

```text
这两行是在问：
上下相邻位置的特征差多少？
左右相邻位置的特征差多少？
差得越大，越像边界，扩散越应该被抑制。
```

## `g`

源码位置：[model/gad_base.py:607](/Users/niko/Downloads/project/model/gad_base.py:607)

```python
return 1.0 / (1.0 + (torch.abs((x*x))/(K*K)))
```

技术解释：

- Perona-Malik 风格边缘函数。
- `x` 越大，输出越小。
- `K` 越大，对边缘越不敏感。

通俗解释：

```text
如果特征差很小，返回接近 1，允许扩散。
如果特征差很大，返回接近 0，阻止跨边界扩散。
```

## `diffuse_step`

源码位置：[model/gad_base.py:612](/Users/niko/Downloads/project/model/gad_base.py:612)

```python
@torch.jit.script
def diffuse_step(cv, ch, I, l: float=0.24):
```

技术解释：用 TorchScript 编译这个函数，提高循环 diffusion 的执行效率。

```python
dv = I[:,:,1:,:] - I[:,:,:-1,:]
dh = I[:,:,:,1:] - I[:,:,:,:-1]
```

技术解释：

- `dv` 是上下相邻 DSM 高度差。
- `dh` 是左右相邻 DSM 高度差。

```python
tv = l * cv * dv
I[:,:,1:,:] -= tv
I[:,:,:-1,:] += tv
```

技术解释：

- `tv` 是纵向传播量。
- 高的一侧减一点，低的一侧加一点，实现平滑。
- `cv` 控制传播强度。

```python
th = l * ch * dh
I[:,:,:,1:] -= th
I[:,:,:,:-1] += th
```

技术解释：横向做同样的传播。

通俗解释：

```text
diffusion 像是在相邻像素之间“倒水”：
高度差越大，理论上倒得越多；
但如果 guide feature 认为这里是边界，cv/ch 会变小，水就倒不过去。
```

## 当前模型输出字段

`GADBase.forward()` 最终一定返回：

```text
y_pred
```

根据配置，还可能包含：

```text
refinement_residual
flat_residual
edge_residual
boundary_gate
learned_boundary_gate
boundary_prior
fusion_gates
fusion_gate_modalities
semantic_features
semantic_masks
semantic_valid_mask
semantic_block_gates
rgb_dsm_similarity
rgb_dsm_depth_gate
rgb_dsm_rgb_gate
rgb_dsm_fusion_gate
y_refined
cv
ch
```

当前 `--refinement-only` 时，不会输出 `cv/ch`，因为 diffusion 被跳过。

