# `run_train.py` 代码说明书

源码位置：[run_train.py](/Users/niko/Downloads/project/run_train.py)

这个文件是训练入口。它负责把数据、模型、loss、优化器和日志系统组织起来。

通俗地说，`run_train.py` 就是训练总调度员：

```text
读参数
-> 准备数据
-> 创建模型
-> 创建优化器
-> 每个 epoch 训练
-> 定期验证
-> 保存 checkpoint
```

## 顶部导入

```python
import argparse
import os
import time
from collections import defaultdict
```

技术解释：

- `argparse.Namespace` 用于类型标注。
- `os` 用于路径、文件判断。
- `time` 用于统计训练耗时。
- `defaultdict` 用于累积训练/验证指标。

```python
import numpy as np
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
```

技术解释：

- `numpy` 用于 `np.nan`、`np.inf`。
- `torch` 是训练主体。
- `optim.Adam` 是优化器。
- `clip_grad_norm_` 做梯度裁剪。
- `DataLoader` 加载数据。
- `SummaryWriter` 写 TensorBoard 日志。
- `tqdm` 显示进度条。

```python
from arguments import train_parser
from data import ProcessedDSMDataset
from losses import get_loss
from model import GADBase
from utils import new_log, seed_all, to_device
```

技术解释：

- `train_parser` 定义命令行参数。
- `ProcessedDSMDataset` 提供训练/验证样本。
- `get_loss` 计算损失。
- `GADBase` 是主模型。
- `new_log` 创建实验目录，`seed_all` 固定随机种子，`to_device` 把 sample 移到 GPU/CPU。

## `Trainer`

源码位置：[run_train.py:21](/Users/niko/Downloads/project/run_train.py:21)

`Trainer` 把训练所需状态封装到一个类中。

它保存：

```text
args
dataloaders
device
model
optimizer
scheduler
writer
epoch/iter
best_rmse_loss
```

## `Trainer.__init__`

源码位置：[run_train.py:23](/Users/niko/Downloads/project/run_train.py:23)

```python
self.args = args
self.dataloaders = self.get_dataloaders(args)
```

技术解释：

- 保存参数。
- 根据参数创建 train/val dataloaders。

通俗解释：

```text
先把命令行参数记住，再把数据管道搭好。
```

```python
seed_all(args.seed)
self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

技术解释：

- 固定随机种子，保证训练尽量可复现。
- 如果有 CUDA 就用 GPU，否则用 CPU。

通俗解释：

```text
seed_all 是为了减少“这次训练和上次训练结果差很多”的随机性。
device 是决定模型跑在哪里。
```

### 构建模型

```python
self.model = GADBase(
    args.feature_extractor,
    Npre=args.Npre,
    Ntrain=args.Ntrain,
    guide_channels=ProcessedDSMDataset.guide_channels,
    use_refinement_net=args.use_refinement_net,
    refinement_channels=args.refinement_channels,
    refinement_blocks=args.refinement_blocks,
    refinement_only=args.refinement_only,
    boundary_refinement=args.boundary_refinement,
    semantic_modulation=args.semantic_modulation,
    semantic_channels=args.semantic_channels,
    use_cross_modal_fusion=args.use_cross_modal_fusion,
    cross_modal_reduction=args.cross_modal_reduction,
).to(self.device)
```

技术解释：

- 用训练参数创建 `GADBase`。
- `guide_channels` 从数据集类读取，当前是 4。
- `.to(self.device)` 把模型参数移到 GPU/CPU。

通俗解释：

```text
命令行里的 --refinement-channels、--boundary-refinement、--cross-modal-fusion 等参数，
最后都是在这里传给 GADBase 的。
```

### 创建实验目录和 TensorBoard

```python
self.experiment_folder, self.args.expN, self.args.randN = new_log(
    os.path.join(args.save_dir, 'DSM'),
    args
)
```

技术解释：

- 在 `save_dir/DSM` 下创建新的 experiment 目录。
- 同时把 `args.csv` 写进去。

通俗解释：

```text
每次训练都会有自己的实验文件夹，里面保存参数、日志和 checkpoint。
```

```python
self.writer = SummaryWriter(log_dir=self.experiment_folder)
```

技术解释：TensorBoard writer 会把训练/验证指标写到实验目录。

### 创建优化器和 scheduler

```python
if not args.no_opt:
    self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.w_decay)
    self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
else:
    self.optimizer = None
    self.scheduler = None
```

技术解释：

- 默认使用 Adam。
- `weight_decay` 是 L2 正则。
- `StepLR` 每隔 `lr_step` 个 epoch 把学习率乘以 `lr_gamma`。
- `--no-opt` 时不训练，只跑 forward/loss。

通俗解释：

```text
optimizer 负责真正改模型参数。
scheduler 负责训练一段时间后慢慢调小学习率。
```

### 初始化训练状态

```python
self.epoch = 0
self.iter = 0
self.train_stats = defaultdict(lambda: np.nan)
self.val_stats = defaultdict(lambda: np.nan)
self.best_rmse_loss = np.inf
```

技术解释：

- `epoch` 当前训练轮数。
- `iter` 当前 batch 步数。
- `train_stats/val_stats` 保存指标。
- `best_rmse_loss` 用于保存 best checkpoint。

```python
if args.resume is not None:
    self.resume(path=args.resume)
```

技术解释：如果命令行指定 checkpoint，就恢复训练状态。

## `__del__`

源码位置：[run_train.py:70](/Users/niko/Downloads/project/run_train.py:70)

```python
writer = getattr(self, 'writer', None)
if writer is not None:
    writer.close()
```

技术解释：对象销毁时关闭 TensorBoard writer。

通俗解释：

```text
训练结束后把日志文件句柄关掉，避免日志没写完整。
```

## `train`

源码位置：[run_train.py:75](/Users/niko/Downloads/project/run_train.py:75)

这是总训练循环。

```python
with tqdm(range(self.epoch, self.args.num_epochs), leave=True) as tnr:
```

技术解释：

- 从当前 epoch 跑到 `num_epochs`。
- 如果 resume，`self.epoch` 可能不是 0。

```python
for _ in tnr:
    self.train_epoch(tnr)
```

技术解释：每轮调用一次 `train_epoch`。

```python
if (self.epoch + 1) % self.args.val_every_n_epochs == 0:
    self.validate()
```

技术解释：每隔指定 epoch 做一次验证。

```python
if self.args.save_model in ['last', 'both']:
    self.save_model('last')
```

技术解释：根据参数保存 last checkpoint。

```python
if self.args.lr_scheduler == 'step' and not self.args.no_opt:
    self.scheduler.step()
    self.writer.add_scalar('log_lr', np.log10(self.scheduler.get_last_lr()[0]), self.epoch)
```

技术解释：

- 如果使用 step scheduler，就更新学习率。
- 记录 `log10(lr)` 到 TensorBoard。

通俗解释：

```text
学习率通常很小，比如 1e-4。
记录 log_lr 更容易在图上看变化。
```

```python
self.epoch += 1
```

技术解释：一个 epoch 完成后计数加一。

## `train_epoch`

源码位置：[run_train.py:93](/Users/niko/Downloads/project/run_train.py:93)

这是一个 epoch 内的 batch 训练逻辑。

```python
self.train_stats = defaultdict(float)
self.model.train()
```

技术解释：

- 清空训练统计。
- `model.train()` 开启训练模式，例如 BatchNorm/Dropout 会按训练行为运行。

```python
log_interval = min(self.args.logstep_train, len(self.dataloaders['train']))
```

技术解释：日志间隔不能超过 dataloader 长度。

### 遍历 batch

```python
for i, sample in enumerate(inner_tnr):
    sample = to_device(sample, self.device)
```

技术解释：

- 从 dataloader 取 batch。
- 把 batch 中 tensor 移到 GPU/CPU。

通俗解释：

```text
DataLoader 给的是 CPU tensor。
模型在 GPU 上时，sample 也必须放到 GPU。
```

```python
if not self.args.no_opt:
    self.optimizer.zero_grad()
```

技术解释：清空上一 batch 的梯度。

通俗解释：

```text
PyTorch 默认梯度会累加。
每个 batch 训练前必须清零，否则梯度会混在一起。
```

```python
output = self.model(sample, train=True)
loss, loss_dict = get_loss(output, sample, self.args.loss)
```

技术解释：

- 前向传播得到 `output`。
- 计算 loss 和日志指标。

```python
if torch.isnan(loss):
    raise RuntimeError('detected NaN loss')
```

技术解释：防止 NaN loss 继续训练污染模型。

```python
for key, value in loss_dict.items():
    self.train_stats[key] += value.detach().cpu().item() if torch.is_tensor(value) else value
```

技术解释：

- 把当前 batch 的各项指标加到累计统计里。
- tensor 指标先 detach、移到 CPU、转成 Python 数字。

### 反向传播和参数更新

```python
if self.epoch > 0 or not self.args.skip_first:
```

技术解释：

- 如果设置 `--skip-first`，第一个 epoch 只 forward/log，不优化。
- 否则正常训练。

```python
loss.backward()
```

技术解释：反向传播，计算每个参数的梯度。

```python
if self.args.gradient_clip > 0.:
    clip_grad_norm_(self.model.parameters(), self.args.gradient_clip)
```

技术解释：

- 梯度裁剪限制梯度范数。
- 防止梯度爆炸。

通俗解释：

```text
如果某次 batch 让梯度特别大，模型参数可能被一脚踢飞。
gradient_clip 相当于给这一步更新限速。
```

```python
self.optimizer.step()
```

技术解释：根据梯度更新模型参数。

```python
self.iter += 1
```

技术解释：全局 batch 步数加一。

### 日志记录

```python
if (i + 1) % log_interval == 0:
    self.train_stats = {key: value / log_interval for key, value in self.train_stats.items()}
```

技术解释：每隔若干 batch，求平均指标。

```python
inner_tnr.set_postfix(training_rmse=self.train_stats['rmse_loss'])
```

技术解释：进度条显示当前平均 RMSE。

```python
for key, value in self.train_stats.items():
    self.writer.add_scalar('train/' + key, value, self.iter)
```

技术解释：写 TensorBoard 训练指标。

```python
self.train_stats = defaultdict(float)
```

技术解释：记录完后清零，准备下一个 log interval。

## `validate`

源码位置：[run_train.py:143](/Users/niko/Downloads/project/run_train.py:143)

验证逻辑和训练类似，但不反向传播。

```python
self.val_stats = defaultdict(float)
self.model.eval()
```

技术解释：

- 清空验证统计。
- `model.eval()` 切换到评估模式。

```python
with torch.no_grad():
```

技术解释：验证不需要梯度，省显存并加速。

```python
for sample in tqdm(self.dataloaders['val'], leave=False):
    sample = to_device(sample, self.device)
    output = self.model(sample)
    _, loss_dict = get_loss(output, sample, self.args.loss)
```

技术解释：

- 遍历验证集。
- 前向传播。
- 计算指标，但不使用 loss 反向传播。

```python
self.val_stats = {key: value / len(self.dataloaders['val']) for key, value in self.val_stats.items()}
```

技术解释：把所有 batch 累计指标取平均。

```python
if self.val_stats['rmse_loss'] < self.best_rmse_loss:
    self.best_rmse_loss = self.val_stats['rmse_loss']
    if self.args.save_model in ['best', 'both']:
        self.save_model('best')
```

技术解释：

- 如果当前验证 RMSE 比历史最好更低，就更新 best。
- 根据参数保存 best checkpoint。

通俗解释：

```text
best_model.pth 保存的是验证集表现最好的模型，不一定是最后一轮模型。
```

## `get_dataloaders`

源码位置：[run_train.py:169](/Users/niko/Downloads/project/run_train.py:169)

```python
data_args = {
    'crop_size': args.crop_size,
    'in_memory': args.in_memory,
    'max_rotation_angle': args.max_rotation,
    'do_horizontal_flip': not args.no_flip,
    'scaling': args.scaling
}
```

技术解释：整理传给 `ProcessedDSMDataset` 的参数。

```python
datasets = {
    'train': ProcessedDSMDataset(args.data_dir, **data_args, split='train', crop_deterministic=False),
    'val': ProcessedDSMDataset(args.data_dir, **data_args, split='val', crop_deterministic=True),
}
```

技术解释：

- 训练集随机裁剪。
- 验证集固定裁剪。

通俗解释：

```text
训练要多样性，所以随机裁。
验证要稳定性，所以固定裁。
```

```python
DataLoader(..., shuffle=True, drop_last=False)
```

训练 DataLoader：

- `shuffle=True` 打乱样本顺序。
- `drop_last=False` 不丢最后不足 batch 的样本。

验证 DataLoader：

- `shuffle=False` 保持顺序。
- 也不丢最后 batch。

## `save_model`

源码位置：[run_train.py:200](/Users/niko/Downloads/project/run_train.py:200)

```python
checkpoint = {
    'model': self.model.state_dict(),
    'epoch': self.epoch + 1,
    'iter': self.iter,
    'best_rmse_loss': self.best_rmse_loss,
}
```

技术解释：

- `model.state_dict()` 保存模型参数。
- `epoch` 保存下一次恢复时应该从哪一轮继续。
- `iter` 保存全局 step。
- `best_rmse_loss` 保存历史最好验证 RMSE。

```python
if not self.args.no_opt:
    checkpoint['optimizer'] = self.optimizer.state_dict()
    checkpoint['scheduler'] = self.scheduler.state_dict()
```

技术解释：保存优化器和学习率调度器状态，方便完整恢复训练。

```python
torch.save(checkpoint, os.path.join(self.experiment_folder, f'{prefix}_model.pth'))
```

技术解释：

- `prefix='best'` 时保存 `best_model.pth`。
- `prefix='last'` 时保存 `last_model.pth`。

## `resume`

源码位置：[run_train.py:212](/Users/niko/Downloads/project/run_train.py:212)

```python
if not os.path.isfile(path):
    raise RuntimeError(...)
```

技术解释：checkpoint 路径不存在就报错。

```python
checkpoint = torch.load(path)
self.model.load_state_dict(checkpoint['model'])
```

技术解释：读取 checkpoint 并恢复模型参数。

```python
if not self.args.no_opt:
    self.optimizer.load_state_dict(checkpoint['optimizer'])
    self.scheduler.load_state_dict(checkpoint['scheduler'])
```

技术解释：恢复 optimizer 和 scheduler 状态。

```python
self.epoch = checkpoint['epoch']
self.iter = checkpoint['iter']
self.best_rmse_loss = checkpoint.get('best_rmse_loss', np.inf)
```

技术解释：恢复训练进度和历史 best。

通俗解释：

```text
resume 不是只加载模型权重。
它还会接着原来的学习率、optimizer 动量、epoch 和 best 指标继续跑。
```

## 主入口

源码位置：[run_train.py:228](/Users/niko/Downloads/project/run_train.py:228)

```python
if __name__ == '__main__':
    args = train_parser.parse_args()
    print(train_parser.format_values())
```

技术解释：

- 当你直接运行 `python run_train.py` 时执行。
- 解析命令行参数并打印参数配置。

```python
trainer = Trainer(args)
since = time.time()
trainer.train()
time_elapsed = time.time() - since
print('Training completed in ...')
```

技术解释：

- 创建 Trainer。
- 记录开始时间。
- 启动训练。
- 打印总耗时。

## 你的训练命令在本文件中的流向

例如：

```bash
--refinement-only
```

进入：

```python
GADBase(..., refinement_only=args.refinement_only)
```

然后影响：

```python
GADBase.forward()
```

使模型跳过 diffusion，直接用 `LocalRefinementNet` 输出作为 `y_pred`。

再例如：

```bash
--gradient-clip 0.1
```

进入：

```python
clip_grad_norm_(self.model.parameters(), self.args.gradient_clip)
```

控制每个 batch 反向传播后的梯度最大范数。

## 训练中一个 batch 的完整顺序

```text
DataLoader 取 sample
-> sample 移到 GPU
-> optimizer.zero_grad()
-> model(sample, train=True)
-> get_loss(output, sample)
-> loss.backward()
-> clip_grad_norm_
-> optimizer.step()
-> 记录 loss_dict
```

这就是训练脚本的核心循环。

