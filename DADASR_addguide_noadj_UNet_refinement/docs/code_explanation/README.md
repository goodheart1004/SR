# DSM 超分项目代码说明书总览

本文档目录用于解释当前项目的核心训练、数据、模型、损失和评估代码。这里不是源码注释的简单复制，而是按实际运行流程说明每个文件、每个函数、关键代码段在做什么，以及它们在 DSM 超分任务里的意义。

## 文档清单

1. [01_data_processed_dsm.md](./01_data_processed_dsm.md)
   - 解释 `data/processed_dsm.py`
   - 重点说明数据目录如何匹配、HR/LR DSM 如何裁剪、`y_bicubic` 如何生成、mask 如何生成、SAM3 语义输入和边界先验如何生成。

2. [02_model_gad_base.md](./02_model_gad_base.md)
   - 解释 `model/gad_base.py`
   - 重点说明 `GADBase`、`LocalRefinementNet`、语义调制、boundary residual heads、RGB--DSM cross-modal fusion 和 diffusion。

3. [03_losses.md](./03_losses.md)
   - 解释 `losses.py`
   - 重点说明 masked RMSE/L1/MSE 的计算，以及为什么只在有效 DSM 像素上算 loss。

4. [04_run_train.md](./04_run_train.md)
   - 解释 `run_train.py`
   - 重点说明训练器如何构建数据、模型、优化器，如何训练、验证、保存和恢复 checkpoint。

5. [05_run_eval.md](./05_run_eval.md)
   - 解释 `run_eval.py`
   - 重点说明评估脚本如何恢复训练参数、加载 checkpoint、计算指标、保存误差热力图和 gating 可视化。

## 当前项目的主流程

训练时的主流程可以理解为：

```text
ProcessedData_scale10
  -> ProcessedDSMDataset 读取 tif
  -> 生成 sample 字典
  -> GADBase.forward(sample)
  -> 输出 y_pred
  -> losses.get_loss(output, sample)
  -> optimizer 反向传播
```

更具体地说，`ProcessedDSMDataset` 每次返回一个样本，里面有：

```text
guide: RGB + adapter_guide，共 4 通道
y: 高分辨率 DSM，训练监督目标
source: 低分辨率 DSM
y_bicubic: source 上采样到 HR 尺寸后的初始 DSM
mask_hr: HR DSM 的有效区域
boundary_prior: 从 SAM3/label/adapter guide 推出来的边界先验
semantic_masks: building/road 两类语义 mask
semantic_valid_mask: 语义有效区域
semantic_boundary_prior: 类别边界先验
```

模型 `GADBase` 的主路径可以理解为：

```text
y_bicubic + guide
  -> feature_extractor 提取 dense feature
  -> 可选 RGBDSMFeatureFusion 融合 RGB 和 LR DSM 关系
  -> LocalRefinementNet 预测 residual
  -> y_init = y_bicubic + residual
  -> 如果 refinement_only=True，直接 y_pred = y_init
  -> 否则继续 diffusion，得到最终 y_pred
```

当前你主要训练的是：

```text
--refinement-only
```

因此最终预测核心就是 `LocalRefinementNet`，它负责直接从 `y_bicubic` 和引导特征中预测 DSM residual。

## 当前 LocalRefinementNet 的定位

当前 `LocalRefinementNet` 已经被改成 U-Net residual decoder。它不是直接从零预测 DSM，而是预测：

```python
residual = model(features, initial_dsm, ...)
refined_dsm = initial_dsm + residual
```

通俗理解：

```text
initial_dsm 是 bicubic 放大的粗 DSM。
LocalRefinementNet 学的是：粗 DSM 哪些地方应该加高、哪些地方应该降低、哪些边缘应该更锐利。
```

这比直接预测完整 DSM 更稳，因为模型只需要学习误差修正。

## 文件之间的依赖关系

```text
run_train.py
  imports:
    arguments.train_parser
    data.ProcessedDSMDataset
    model.GADBase
    losses.get_loss
    utils.new_log / seed_all / to_device

run_eval.py
  imports:
    arguments.train_parser
    data.ProcessedDSMDataset
    model.GADBase
    losses.get_loss

model/gad_base.py
  contains:
    GADBase
    LocalRefinementNet
    RGBDSMFeatureFusion
    diffusion functions

data/processed_dsm.py
  contains:
    ProcessedDSMDataset

losses.py
  contains:
    get_loss and masked loss helpers
```

## 推荐阅读顺序

如果你想理解代码如何训练，按这个顺序读：

1. `01_data_processed_dsm.md`
2. `02_model_gad_base.md`
3. `03_losses.md`
4. `04_run_train.md`
5. `05_run_eval.md`

如果你只关心模型结构，直接看：

1. `02_model_gad_base.md`
2. `03_losses.md`

如果你只关心运行脚本，看：

1. `04_run_train.md`
2. `05_run_eval.md`

## 当前训练命令和代码对应关系

你给出的训练命令：

```bash
python run_train.py \
  --save-dir results \
  --data-dir ProcessedData_scale10 \
  --batch-size 8 \
  --crop-size 250 \
  --scaling 10 \
  --feature-extractor UNet \
  --use-refinement-net \
  --boundary-refinement \
  --cross-modal-fusion \
  --refinement-only \
  --refinement-channels 96 \
  --refinement-blocks 8 \
  --num-epochs 150 \
  --lr 1e-4 \
  --gradient-clip 0.1
```

对应含义是：

```text
ProcessedDSMDataset:
  使用 ProcessedData_scale10，HR crop 为 250，LR-HR scale 为 10。

GADBase:
  feature_extractor='UNet'
  use_refinement_net=True
  boundary_refinement=True
  use_cross_modal_fusion=True
  refinement_only=True

LocalRefinementNet:
  hidden channels = 96
  bottleneck residual blocks = 8

Trainer:
  batch_size=8
  num_epochs=150
  lr=1e-4
  gradient_clip=0.1
```

## 关键张量形状

默认 `crop_size=250`、`scaling=10` 时，一个样本大致是：

```text
source:      [1, 25, 25]      LR DSM crop
y:           [1, 250, 250]    HR DSM target
y_bicubic:   [1, 250, 250]    LR DSM bicubic 到 HR
guide:       [4, 250, 250]    RGB 3 通道 + adapter 1 通道
mask_hr:     [1, 250, 250]
```

batch 后：

```text
guide:       [B, 4, 250, 250]
y_bicubic:   [B, 1, 250, 250]
y:           [B, 1, 250, 250]
```

如果使用 `feature_extractor='UNet'`：

```text
guide_feats: [B, 64, 250, 250]
```

如果使用你当前的 `LocalRefinementNet`：

```text
e1:          [B, C, 250, 250]
e2:          [B, C, 125, 125]
bottleneck:  [B, C, 63, 63]
d2:          [B, C, 125, 125]
d1:          [B, C, 250, 250]
residual:    [B, 1, 250, 250]
y_pred:      [B, 1, 250, 250]
```

这里 bottleneck 是 `63x63`，不是 `62x62`，因为 stride-2 conv 的输出尺寸对奇偶数会向上取整。decoder 里用 `size=e2.shape[-2:]` 和 `size=e1.shape[-2:]`，所以不会因为 `250` 不是 4 的倍数而 shape mismatch。

